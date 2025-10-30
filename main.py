#!/usr/bin/env python3
"""
Mnemosyne - Memory, Governance & Orchestration Agent

Central coordinator for the content pipeline:
- Orchestrates workflow (Aletheia → IRIS → Erebus → Kairos)
- Schedules pipeline execution
- Tracks cross-agent memory and state
- Enforces quality/voice consistency
- Reports system status

Completion Detection:
- Synchronous HTTP calls (agents return when complete)
- Future: Webhooks for async operations
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Import dashboard and email notifier
from dashboard import Dashboard
from email_notifier import notifier

# Import consolidated agents
from agents.iris import IrisAgent, IdeaInput as IrisIdeaInput, DraftRequest
from agents.erebus import ErebusAgent, DraftInput as ErebusDraftInput
from agents.aletheia import AletheiaAgent
from agents.kairos import KairosAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Mnemosyne - Memory & Governance",
    description="Central coordinator: orchestration, scheduling, memory, governance, status",
    version="1.0.0"
)

# Data directory
data_dir_env = os.getenv("MNEMOSYNE_DATA_DIR")
if data_dir_env:
    DATA_DIR = Path(data_dir_env).expanduser()
else:
    DATA_DIR = Path.home() / ".mnemosyne"

# Directories
PIPELINE_DIR = DATA_DIR / "pipeline"
SCHEDULE_DIR = DATA_DIR / "schedule"
MEMORY_DIR = DATA_DIR / "memory"

PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# Agent URLs
AGENTS = {
    "aletheia": "http://localhost:8001",
    "iris": "http://localhost:8002",
    "erebus": "http://localhost:8003",
    "kairos": "http://localhost:8004"
}

# Quality thresholds (governance)
QUALITY_THRESHOLDS = {
    "max_ai_likelihood": 0.25,
    "max_voice_deviation": 0.35
}

# Initialize dashboard
dashboard = Dashboard(DATA_DIR)

# Initialize consolidated agents
logger.info("Initializing consolidated agents...")
aletheia = AletheiaAgent(DATA_DIR)
iris = IrisAgent(DATA_DIR)
erebus = ErebusAgent(DATA_DIR)
kairos = KairosAgent(DATA_DIR)
logger.info("✓ All agents initialized (consolidated single-process deployment)")


# ============================================================================
# Models
# ============================================================================

class ScheduleConfig(BaseModel):
    """Schedule configuration"""
    enabled: bool = True
    interval_hours: int = 6  # Run discovery every 6 hours
    pipeline_enabled: bool = True  # Run full pipeline
    pipeline_interval_hours: int = 24  # Run pipeline daily
    num_ideas: int = 3
    review_required: bool = True
    # Legacy fields for backwards compatibility
    daily_time: Optional[str] = None
    days_of_week: Optional[List[int]] = None

class PipelineJob(BaseModel):
    """Pipeline job"""
    job_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    num_ideas: int
    current_step: Optional[str] = None
    results: Optional[Dict] = None

class TriggerRequest(BaseModel):
    """Manual trigger"""
    num_ideas: int = 3
    review_required: bool = True
    reason: Optional[str] = "manual"


# ============================================================================
# State Management
# ============================================================================

class MnemosyneState:
    """Manages memory, schedule, and jobs"""

    def __init__(self):
        self.jobs: Dict[str, PipelineJob] = {}
        self.schedule: Optional[ScheduleConfig] = None
        self.load_schedule()

    def load_schedule(self):
        """Load schedule configuration"""
        config_file = SCHEDULE_DIR / "schedule.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.schedule = ScheduleConfig(**json.load(f))
        else:
            self.schedule = ScheduleConfig()
            self.save_schedule()

    def save_schedule(self):
        """Save schedule configuration"""
        with open(SCHEDULE_DIR / "schedule.json", 'w') as f:
            json.dump(self.schedule.dict(), f, indent=2)

    def create_job(self, num_ideas: int, review_required: bool, reason: str) -> PipelineJob:
        """Create new job"""
        job = PipelineJob(
            job_id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status="queued",
            num_ideas=num_ideas
        )
        self.jobs[job.job_id] = job
        logger.info(f"Created job: {job.job_id} ({reason})")
        return job

    def update_job(self, job_id: str, **kwargs):
        """Update job"""
        if job_id in self.jobs:
            for k, v in kwargs.items():
                setattr(self.jobs[job_id], k, v)

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        """Get job"""
        return self.jobs.get(job_id)


state = MnemosyneState()


# ============================================================================
# Pipeline Orchestration
# ============================================================================

async def run_pipeline_job(job_id: str):
    """
    Run pipeline job - orchestrates all agents in sequence

    Completion Detection: Synchronous HTTP calls
    - When agent returns response = work completed ✓
    - Simple, reliable, works immediately
    """
    job = state.get_job(job_id)
    if not job:
        return

    try:
        logger.info(f"Starting job {job_id}")
        state.update_job(job_id, status="running", started_at=datetime.now().isoformat())

        results = {
            "run_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "job_id": job_id,
            "started_at": datetime.now().isoformat(),
            "num_ideas": job.num_ideas,
            "ideas_processed": [],
            "successes": 0,
            "failures": 0,
            "metrics": {
                "avg_ai_likelihood": 0.0,
                "avg_voice_deviation": 0.0,
                "quality_passed": 0,
                "quality_failed": 0
            }
        }

        async with aiohttp.ClientSession() as session:
            # Get ideas from discovery
            ideas = await get_ideas(session, job.num_ideas)

            for idx, idea in enumerate(ideas, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing Idea {idx}/{len(ideas)}: {idea.get('title', '')[:50]}...")
                logger.info(f"{'='*60}")

                try:
                    result = await process_idea(session, idea, state.schedule.review_required)
                    results["ideas_processed"].append(result)

                    if result["status"] == "success":
                        results["successes"] += 1
                        if "cleaned" in result:
                            results["metrics"]["avg_ai_likelihood"] += result["cleaned"]["ai_likelihood_after"]
                            results["metrics"]["avg_voice_deviation"] += result["cleaned"]["voice_deviation"]

                            if (result["cleaned"]["ai_likelihood_after"] <= QUALITY_THRESHOLDS["max_ai_likelihood"] and
                                result["cleaned"]["voice_deviation"] <= QUALITY_THRESHOLDS["max_voice_deviation"]):
                                results["metrics"]["quality_passed"] += 1
                            else:
                                results["metrics"]["quality_failed"] += 1
                    else:
                        results["failures"] += 1

                except Exception as e:
                    logger.error(f"Failed idea {idx}: {e}")
                    results["ideas_processed"].append({"idea": idea, "status": "error", "error": str(e)})
                    results["failures"] += 1

            # Calculate averages
            if results["successes"] > 0:
                results["metrics"]["avg_ai_likelihood"] /= results["successes"]
                results["metrics"]["avg_voice_deviation"] /= results["successes"]

        results["completed_at"] = datetime.now().isoformat()

        # Save results
        save_pipeline_results(results)

        # Update job
        state.update_job(job_id, status="completed", completed_at=datetime.now().isoformat(), results=results)

        logger.info(f"✅ Job {job_id} completed: {results['successes']} successes, {results['failures']} failures")

        # Send email notification with pipeline stats
        notifier.send_pipeline_stats(results)

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
        state.update_job(job_id, status="failed", completed_at=datetime.now().isoformat())

        # Send error notification
        notifier.send_error_notification(
            error_type="Pipeline Failure",
            error_message=str(e),
            context={"job_id": job_id, "timestamp": datetime.now().isoformat()}
        )


async def process_idea(session: aiohttp.ClientSession, idea: Dict, review_required: bool) -> Dict:
    """
    Process single idea through pipeline

    Now using direct function calls instead of HTTP (consolidated agents)
    """
    result = {"idea": idea, "status": "processing"}

    try:
        # Step 1: IRIS Outline
        logger.info("  [1/4] IRIS: Generating outline...")
        idea_input = IrisIdeaInput(
            title=idea.get("title", ""),
            content=idea.get("snippet", ""),
            url=idea.get("url"),
            source=idea.get("source"),
            score=idea.get("score")
        )
        outline = iris.generate_outline(idea_input)
        result["outline"] = outline.dict()
        logger.info(f"  ✓ Outline: {outline.outline_id}")

        # Step 2: IRIS Draft
        logger.info("  [2/4] IRIS: Generating draft...")
        draft_request = DraftRequest(
            outline_id=outline.outline_id,
            idea=idea_input,
            target_length=800
        )
        draft = iris.generate_draft(draft_request)
        result["draft"] = draft.dict()
        logger.info(f"  ✓ Draft: {draft.draft_id} ({draft.word_count} words)")

        # Step 3: Erebus Clean
        logger.info("  [3/4] Erebus: Cleaning...")
        draft_input = ErebusDraftInput(
            draft_id=draft.draft_id,
            title=draft.title,
            content=draft.content,
            word_count=draft.word_count,
            metadata=draft.metadata
        )
        cleaned = erebus.clean_draft(draft_input)
        result["cleaned"] = cleaned.dict()
        logger.info(f"  ✓ Cleaned: {cleaned.cleaned_id}")
        logger.info(f"    AI: {cleaned.ai_likelihood_before:.2f} → {cleaned.ai_likelihood_after:.2f}")
        logger.info(f"    Voice: {cleaned.voice_deviation:.2f}")

        # Governance: Check quality
        quality_pass = (cleaned.ai_likelihood_after <= QUALITY_THRESHOLDS["max_ai_likelihood"] and
                       cleaned.voice_deviation <= QUALITY_THRESHOLDS["max_voice_deviation"])

        if not quality_pass:
            logger.warning("  ⚠️  Quality check failed")

        # Step 4: Kairos Queue
        if review_required or not quality_pass:
            logger.info("  [4/4] Saving for review...")
            result["queued"] = {"status": "pending_review"}
        else:
            logger.info("  [4/4] Kairos: Queueing...")
            queued_result = kairos.queue_for_publishing({
                "draft_id": draft.draft_id,
                "cleaned_id": cleaned.cleaned_id,
                "title": cleaned.title,
                "cleaned_content": cleaned.cleaned_content,
                "metadata": cleaned.metadata
            })
            result["queued"] = queued_result

        result["status"] = "success"
        logger.info(f"  ✅ Complete")

    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")
        result["status"] = "error"
        result["error"] = str(e)

    return result


# ============================================================================
# Agent Calls (Completion = Response Received)
# ============================================================================

async def get_ideas(session: aiohttp.ClientSession, limit: int) -> List[Dict]:
    """Get ideas from discovery"""
    ideas_file = DATA_DIR / "daily-ideas.json"
    if ideas_file.exists():
        with open(ideas_file, 'r') as f:
            return json.load(f).get("ideas", [])[:limit]
    return []


async def call_iris_outline(session: aiohttp.ClientSession, idea: Dict) -> Dict:
    """Call IRIS to create outline - returns when complete"""
    async with session.post(
        f"{AGENTS['iris']}/v1/outlines",
        json={
            "title": idea.get("title", ""),
            "content": idea.get("content", ""),
            "source_url": idea.get("source_url") or idea.get("url"),
            "score": idea.get("score", 0.5)
        },
        timeout=aiohttp.ClientTimeout(total=60)
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def call_iris_draft(session: aiohttp.ClientSession, outline: Dict, idea: Dict) -> Dict:
    """Call IRIS to create draft - returns when complete"""
    async with session.post(
        f"{AGENTS['iris']}/v1/drafts",
        json={
            "outline_id": outline.get("outline_id"),
            "idea": {
                "title": idea.get("title", ""),
                "source_url": idea.get("source_url") or idea.get("url"),
                "score": idea.get("score", 0.5)
            },
            "target_length": 800
        },
        timeout=aiohttp.ClientTimeout(total=90)
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def call_erebus_clean(session: aiohttp.ClientSession, draft: Dict) -> Dict:
    """Call Erebus to clean draft - returns when complete"""
    async with session.post(
        f"{AGENTS['erebus']}/v1/clean",
        json={
            "draft_id": draft.get("draft_id"),
            "title": draft.get("title"),
            "content": draft.get("content"),
            "word_count": draft.get("word_count"),
            "metadata": draft.get("metadata", {})
        },
        timeout=aiohttp.ClientTimeout(total=90)
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def call_kairos_queue(session: aiohttp.ClientSession, cleaned: Dict) -> Dict:
    """Call Kairos to queue - returns when complete"""
    # For now, save locally
    queue_dir = DATA_DIR / "queue"
    queue_dir.mkdir(exist_ok=True)
    queue_file = queue_dir / f"queued_{cleaned['cleaned_id']}.json"
    with open(queue_file, 'w') as f:
        json.dump({"queued_at": datetime.now().isoformat(), "cleaned": cleaned}, f, indent=2)
    return {"status": "queued", "queue_id": queue_file.stem}


def save_pipeline_results(results: Dict):
    """Save pipeline results"""
    results_file = PIPELINE_DIR / f"{results['run_id']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


# ============================================================================
# Scheduler (APScheduler - more reliable than cron)
# ============================================================================

# Global scheduler
scheduler = AsyncIOScheduler()

async def pre_run_health_check() -> bool:
    """
    Check all critical agents are healthy before starting pipeline

    Returns:
        True if all critical agents are healthy, False otherwise
    """
    critical_agents = ["iris", "erebus"]  # Must be up for pipeline to run
    optional_agents = ["kairos"]  # Can be down

    logger.info("🔍 Running pre-run health check...")

    # Check all agents
    statuses = {}
    for agent_name, url in AGENTS.items():
        status = await check_agent_health(url, agent_name)
        statuses[agent_name] = status

    # Check critical agents
    failed_agents = []
    for agent in critical_agents:
        if agent in statuses and statuses[agent]["status"] != "healthy":
            failed_agents.append(agent)
            logger.error(f"❌ Critical agent {agent} is {statuses[agent]['status']}")

    if failed_agents:
        # Send alert email
        notifier.send_error_notification(
            error_type="Pre-Run Health Check Failed",
            error_message=f"Cannot start pipeline - critical agents are down: {', '.join(failed_agents)}",
            context={"agent_statuses": statuses}
        )
        return False

    logger.info("✅ Pre-run health check passed")
    return True

async def run_discovery():
    """
    Run discovery service (Aletheia) to find new ideas

    Calls the discovery service which runs Perplexity searches across all topics
    """
    try:
        logger.info("🔍 Running scheduled discovery...")

        # Call discovery service (runs locally as a script)
        import subprocess
        discovery_script = Path(__file__).parent.parent / "services" / "scheduler" / "run_discovery.sh"

        if discovery_script.exists():
            result = subprocess.run(
                [str(discovery_script)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Discovery completed successfully")
            else:
                logger.error(f"❌ Discovery failed: {result.stderr}")
        else:
            logger.warning(f"Discovery script not found at {discovery_script}")

    except Exception as e:
        logger.error(f"Discovery error: {e}", exc_info=True)

async def run_scheduled_pipeline():
    """Run scheduled pipeline job with pre-run health check"""
    try:
        if not state.schedule.pipeline_enabled:
            return

        # Pre-run health check
        if not await pre_run_health_check():
            logger.error("❌ Pipeline cancelled - health check failed")
            return

        logger.info("📅 Running scheduled pipeline...")
        job = state.create_job(state.schedule.num_ideas, state.schedule.review_required, "scheduled")
        await run_pipeline_job(job.job_id)

    except Exception as e:
        logger.error(f"Scheduled pipeline error: {e}", exc_info=True)
        notifier.send_error_notification(
            error_type="Scheduled Pipeline Error",
            error_message=str(e),
            context={"timestamp": datetime.now().isoformat()}
        )

def start_scheduler():
    """Initialize and start APScheduler"""
    if not scheduler.running:
        # Schedule discovery every N hours
        if state.schedule.enabled and state.schedule.interval_hours > 0:
            scheduler.add_job(
                run_discovery,
                trigger=IntervalTrigger(hours=state.schedule.interval_hours),
                id='discovery',
                name=f'Discovery (every {state.schedule.interval_hours}h)',
                replace_existing=True
            )
            logger.info(f"📅 Scheduled discovery: every {state.schedule.interval_hours} hours")

        # Schedule pipeline every N hours
        if state.schedule.pipeline_enabled and state.schedule.pipeline_interval_hours > 0:
            scheduler.add_job(
                run_scheduled_pipeline,
                trigger=IntervalTrigger(hours=state.schedule.pipeline_interval_hours),
                id='pipeline',
                name=f'Pipeline (every {state.schedule.pipeline_interval_hours}h)',
                replace_existing=True
            )
            logger.info(f"📅 Scheduled pipeline: every {state.schedule.pipeline_interval_hours} hours")

        scheduler.start()
        logger.info("✅ Scheduler started")

def update_scheduler():
    """Update scheduler with new configuration"""
    # Remove existing jobs
    if scheduler.get_job('discovery'):
        scheduler.remove_job('discovery')
    if scheduler.get_job('pipeline'):
        scheduler.remove_job('pipeline')

    # Re-add with new config
    if state.schedule.enabled and state.schedule.interval_hours > 0:
        scheduler.add_job(
            run_discovery,
            trigger=IntervalTrigger(hours=state.schedule.interval_hours),
            id='discovery',
            name=f'Discovery (every {state.schedule.interval_hours}h)',
            replace_existing=True
        )
        logger.info(f"📅 Updated discovery schedule: every {state.schedule.interval_hours} hours")

    if state.schedule.pipeline_enabled and state.schedule.pipeline_interval_hours > 0:
        scheduler.add_job(
            run_scheduled_pipeline,
            trigger=IntervalTrigger(hours=state.schedule.pipeline_interval_hours),
            id='pipeline',
            name=f'Pipeline (every {state.schedule.pipeline_interval_hours}h)',
            replace_existing=True
        )
        logger.info(f"📅 Updated pipeline schedule: every {state.schedule.pipeline_interval_hours} hours")


# ============================================================================
# Status Reporting (collects from all agents)
# ============================================================================

async def check_agent_health(url: str, name: str) -> Dict:
    """Check agent health"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/healthz", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return {"agent": name, "status": "healthy", "url": url}
                return {"agent": name, "status": "unhealthy", "url": url}
    except:
        return {"agent": name, "status": "down", "url": url}


@app.get("/v1/status")
async def get_system_status():
    """
    Get complete system status

    Mnemosyne collects stats from all agents
    """
    # Check agent health
    agents = []
    for name, url in AGENTS.items():
        health = await check_agent_health(url, name)
        agents.append(health)

    # Pipeline stats
    pipeline_files = list(PIPELINE_DIR.glob("pipeline_*.json"))
    total_runs = len(pipeline_files)

    # Recent jobs
    recent_jobs = [j.dict() for j in sorted(state.jobs.values(), key=lambda x: x.job_id, reverse=True)[:5]]

    return {
        "timestamp": datetime.now().isoformat(),
        "system_status": "healthy" if all(a["status"] == "healthy" for a in agents) else "degraded",
        "agents": agents,
        "pipeline": {
            "total_runs": total_runs,
            "active_jobs": len([j for j in state.jobs.values() if j.status == "running"])
        },
        "recent_jobs": recent_jobs,
        "schedule": state.schedule.dict() if state.schedule else {}
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup():
    """Start scheduler"""
    logger.info("Mnemosyne starting - Memory, Governance & Orchestration")
    start_scheduler()


@app.get("/healthz")
def health_check():
    return {"status": "ok", "agent": "mnemosyne", "role": "coordinator"}


@app.get("/v1/schedule")
def get_schedule():
    return state.schedule.dict() if state.schedule else {}


@app.put("/v1/schedule")
def update_schedule(config: ScheduleConfig):
    state.schedule = config
    state.save_schedule()
    update_scheduler()  # Restart scheduler with new config
    return {"status": "updated", "message": "Schedule updated and scheduler restarted"}


@app.post("/v1/trigger")
async def trigger_pipeline(req: TriggerRequest, bg: BackgroundTasks):
    job = state.create_job(req.num_ideas, req.review_required, req.reason)
    bg.add_task(run_pipeline_job, job.job_id)
    return {"status": "triggered", "job_id": job.job_id}


@app.get("/v1/jobs")
def list_jobs():
    return {"jobs": [j.dict() for j in sorted(state.jobs.values(), key=lambda x: x.job_id, reverse=True)[:20]]}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = state.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.dict()


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """
    Web-based command console dashboard

    Returns HTML interface showing:
    - System health and agent statuses
    - Pipeline statistics
    - Schedule configuration with controls
    - Recent jobs
    - Error summary
    """
    # Collect agent statuses
    agents = []
    for name, url in AGENTS.items():
        health = await check_agent_health(url, name)
        agents.append(health)

    # Get schedule config
    schedule = state.schedule.dict() if state.schedule else {}

    # Get recent jobs
    recent_jobs = [j.dict() for j in sorted(state.jobs.values(), key=lambda x: x.job_id, reverse=True)[:10]]

    # Get dashboard data
    data = dashboard.get_dashboard_data(agents, schedule, recent_jobs)

    # Generate HTML
    html = dashboard.generate_html_dashboard(data)

    return html


@app.post("/v1/discovery/trigger")
async def trigger_discovery(bg: BackgroundTasks):
    """
    Manually trigger discovery run

    Runs discovery service to find new ideas from configured sources
    """
    bg.add_task(run_discovery)
    return {"status": "triggered", "message": "Discovery started in background"}


@app.get("/v1/discovery/stats")
async def get_discovery_stats():
    """Get discovery statistics"""
    stats = dashboard._get_discovery_stats()
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
