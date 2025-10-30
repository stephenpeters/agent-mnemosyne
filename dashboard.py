#!/usr/bin/env python3
"""
Mnemosyne Command Console & Dashboard

Web-based dashboard served by agent-mnemosyne showing:
- System status and health
- Agent statuses (heartbeat monitoring)
- Recent pipeline runs
- Scheduler configuration and controls
- Error logs and alerts
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Dashboard:
    """Generate dashboard data and HTML for Mnemosyne"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.logs_dir = data_dir / "logs"
        self.pipeline_dir = data_dir / "pipeline"
        self.memory_dir = data_dir / "memory"

    def get_dashboard_data(self, agent_statuses: List[Dict], schedule: Dict, recent_jobs: List[Dict]) -> Dict:
        """
        Collect all dashboard data

        Args:
            agent_statuses: List of agent health check results
            schedule: Current schedule configuration
            recent_jobs: Recent pipeline jobs

        Returns:
            Dict with all dashboard data
        """
        # Calculate system health
        system_health = self._calculate_system_health(agent_statuses, recent_jobs)

        # Get pipeline statistics
        pipeline_stats = self._get_pipeline_stats()

        # Get error summary
        error_summary = self._get_error_summary()

        # Get discovery stats
        discovery_stats = self._get_discovery_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "agents": agent_statuses,
            "schedule": schedule,
            "pipeline_stats": pipeline_stats,
            "recent_jobs": recent_jobs,
            "discovery_stats": discovery_stats,
            "error_summary": error_summary
        }

    def _calculate_system_health(self, agent_statuses: List[Dict], recent_jobs: List[Dict]) -> Dict:
        """Calculate overall system health score"""

        health = {
            "score": 100,
            "status": "healthy",
            "color": "#28a745",
            "issues": []
        }

        # Check agent statuses
        down_agents = [a for a in agent_statuses if a["status"] == "down"]
        unhealthy_agents = [a for a in agent_statuses if a["status"] == "unhealthy"]

        if len(down_agents) >= 3:
            health["score"] -= 40
            health["issues"].append(f"{len(down_agents)} agents are down")
        elif len(down_agents) >= 1:
            health["score"] -= 20
            health["issues"].append(f"{len(down_agents)} agent(s) down: {', '.join([a['agent'] for a in down_agents])}")

        if unhealthy_agents:
            health["score"] -= 10
            health["issues"].append(f"{len(unhealthy_agents)} agent(s) unhealthy")

        # Check recent pipeline runs
        if recent_jobs:
            failed_recent = [j for j in recent_jobs[:3] if j.get("status") == "failed"]
            if len(failed_recent) >= 2:
                health["score"] -= 25
                health["issues"].append("Multiple recent pipeline runs failed")
            elif failed_recent:
                health["score"] -= 10
                health["issues"].append("Recent pipeline run failed")

        # Determine status and color
        if health["score"] >= 80:
            health["status"] = "healthy"
            health["color"] = "#28a745"
        elif health["score"] >= 60:
            health["status"] = "degraded"
            health["color"] = "#ffc107"
        else:
            health["status"] = "critical"
            health["color"] = "#dc3545"

        return health

    def _get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics from saved runs"""

        stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_ideas_processed": 0,
            "avg_success_rate": 0.0,
            "avg_ai_likelihood": 0.0,
            "avg_voice_deviation": 0.0
        }

        if not self.pipeline_dir.exists():
            return stats

        # Read all pipeline result files
        pipeline_files = sorted(self.pipeline_dir.glob("pipeline_*.json"), reverse=True)
        stats["total_runs"] = len(pipeline_files)

        if not pipeline_files:
            return stats

        successful = 0
        total_success_rate = 0
        total_ai_likelihood = 0
        total_voice_deviation = 0
        count_with_metrics = 0

        for pf in pipeline_files:
            try:
                with open(pf, 'r') as f:
                    data = json.load(f)

                    successes = data.get("successes", 0)
                    failures = data.get("failures", 0)
                    total = successes + failures

                    if total > 0:
                        stats["total_ideas_processed"] += total
                        success_rate = successes / total
                        total_success_rate += success_rate

                        if success_rate > 0.8:
                            successful += 1

                    # Get metrics
                    metrics = data.get("metrics", {})
                    if metrics.get("avg_ai_likelihood") is not None:
                        total_ai_likelihood += metrics["avg_ai_likelihood"]
                        total_voice_deviation += metrics.get("avg_voice_deviation", 0)
                        count_with_metrics += 1

            except Exception as e:
                logger.error(f"Error reading pipeline file {pf}: {e}")

        stats["successful_runs"] = successful
        stats["failed_runs"] = stats["total_runs"] - successful

        if stats["total_runs"] > 0:
            stats["avg_success_rate"] = total_success_rate / stats["total_runs"]

        if count_with_metrics > 0:
            stats["avg_ai_likelihood"] = total_ai_likelihood / count_with_metrics
            stats["avg_voice_deviation"] = total_voice_deviation / count_with_metrics

        return stats

    def _get_error_summary(self) -> List[Dict]:
        """Get error summary from logs"""

        errors = defaultdict(int)
        error_details = []

        if not self.logs_dir.exists():
            return error_details

        # Check discovery logs
        log_files = sorted(self.logs_dir.glob("discovery_*.log"), reverse=True)[:7]

        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if "ERROR" in line or "error:" in line.lower():
                            # Categorize error
                            if "401" in line:
                                error_type = "Authentication Error (401)"
                            elif "403" in line:
                                error_type = "Forbidden/Quota Error (403)"
                            elif "timeout" in line.lower():
                                error_type = "Timeout Error"
                            elif "scoring" in line.lower():
                                error_type = "Scoring Error"
                            else:
                                error_type = "Other Error"

                            errors[error_type] += 1

                            if len(error_details) < 20:
                                error_details.append({
                                    "date": datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d"),
                                    "type": error_type,
                                    "message": line.strip()[:200]
                                })
            except Exception as e:
                logger.error(f"Error reading log {log_file}: {e}")

        # Convert to list sorted by count
        return [{"type": k, "count": v} for k, v in sorted(errors.items(), key=lambda x: x[1], reverse=True)]

    def _get_discovery_stats(self) -> Dict:
        """Get discovery run statistics"""

        stats = {
            "total_ideas_discovered": 0,
            "last_discovery_run": None,
            "discovery_runs_7d": 0,
            "avg_ideas_per_run": 0
        }

        # Count ideas in daily-ideas.json
        daily_ideas_file = self.data_dir / "daily-ideas.json"
        if daily_ideas_file.exists():
            try:
                with open(daily_ideas_file, 'r') as f:
                    data = json.load(f)
                    stats["total_ideas_discovered"] = data.get("total_discovered", 0)
                    ideas = data.get("ideas", [])
                    if ideas and isinstance(ideas[0], dict):
                        last_discovered = ideas[0].get("discovered_at")
                        if last_discovered:
                            stats["last_discovery_run"] = last_discovered[:19]
            except Exception as e:
                logger.error(f"Error reading daily-ideas.json: {e}")

        # Count discovery runs in last 7 days
        if self.logs_dir.exists():
            cutoff = datetime.now() - timedelta(days=7)
            log_files = [f for f in self.logs_dir.glob("discovery_*.log")
                        if datetime.fromtimestamp(f.stat().st_mtime) > cutoff]
            stats["discovery_runs_7d"] = len(log_files)

            if stats["discovery_runs_7d"] > 0 and stats["total_ideas_discovered"] > 0:
                stats["avg_ideas_per_run"] = stats["total_ideas_discovered"] / stats["discovery_runs_7d"]

        return stats

    def generate_html_dashboard(self, data: Dict) -> str:
        """
        Generate HTML dashboard

        Args:
            data: Dashboard data from get_dashboard_data()

        Returns:
            HTML string
        """

        system_health = data["system_health"]
        agents = data["agents"]
        schedule = data["schedule"]
        pipeline_stats = data["pipeline_stats"]
        recent_jobs = data["recent_jobs"]
        discovery_stats = data["discovery_stats"]
        error_summary = data["error_summary"]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Mnemosyne Command Console</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #2E3440 0%, #3B4252 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        h1 {{
            font-size: 36px;
            margin-bottom: 10px;
            color: white;
        }}
        .subtitle {{
            color: rgba(255,255,255,0.9);
            font-size: 16px;
        }}
        .health-indicator {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 16px;
            margin-left: 15px;
            background: {system_health['color']};
            color: white;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: #2a2a2a;
            padding: 24px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            border: 1px solid #3a3a3a;
        }}
        .card h2 {{
            font-size: 20px;
            margin-bottom: 16px;
            color: #ffffff;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}
        .metric-big {{
            font-size: 48px;
            font-weight: 700;
            margin: 15px 0;
            background: linear-gradient(135deg, #2E3440 0%, #3B4252 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .metric-label {{
            font-size: 14px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #3a3a3a;
        }}
        .stat:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            color: #999;
            font-size: 14px;
        }}
        .stat-value {{
            font-weight: 600;
            color: #e0e0e0;
            font-size: 16px;
        }}
        .agent-status {{
            display: flex;
            align-items: center;
            padding: 12px;
            background: #333;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .agent-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 12px;
        }}
        .status-healthy {{ background: #28a745; }}
        .status-degraded {{ background: #ffc107; }}
        .status-down {{ background: #dc3545; }}
        .progress-bar {{
            width: 100%;
            height: 10px;
            background: #3a3a3a;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 15px;
        }}
        .progress-fill {{
            height: 100%;
            background: {system_health['color']};
            transition: width 0.3s;
        }}
        .issue-list {{
            list-style: none;
            padding: 0;
            margin-top: 15px;
        }}
        .issue-item {{
            padding: 10px 15px;
            background: rgba(255, 193, 7, 0.1);
            border-left: 3px solid #ffc107;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 14px;
            color: #ffc107;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            text-align: left;
            padding: 12px;
            background: #333;
            font-weight: 600;
            color: #999;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #3a3a3a;
            color: #e0e0e0;
        }}
        tr:hover {{
            background: #333;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-success {{ background: #28a745; color: white; }}
        .badge-error {{ background: #dc3545; color: white; }}
        .badge-running {{ background: #007bff; color: white; }}
        .badge-queued {{ background: #6c757d; color: white; }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
        .controls {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }}
        button {{
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #764ba2;
        }}
        button.danger {{
            background: #dc3545;
        }}
        button.danger:hover {{
            background: #c82333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Mnemosyne Command Console</h1>
            <span class="health-indicator">{system_health['status'].upper()}</span>
            <div class="subtitle">Autonomous Content Pipeline - Memory & Governance</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {system_health['score']}%"></div>
            </div>
            <div class="timestamp">Last updated: {now} • Auto-refresh every 30s</div>
        </header>

        <!-- System Health Overview -->
        <div class="grid">
            <div class="card">
                <h2>⚡ System Health</h2>
                <div class="metric-big">{system_health['score']}/100</div>
                <div class="metric-label">Health Score</div>
                {self._render_issues_html(system_health['issues'])}
            </div>

            <div class="card">
                <h2>🤖 Agent Status</h2>
                {self._render_agents_html(agents)}
            </div>

            <div class="card">
                <h2>📊 Pipeline Statistics</h2>
                <div class="stat">
                    <span class="stat-label">Total Runs</span>
                    <span class="stat-value">{pipeline_stats['total_runs']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Success Rate</span>
                    <span class="stat-value">{pipeline_stats['avg_success_rate']*100:.1f}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Ideas Processed</span>
                    <span class="stat-value">{pipeline_stats['total_ideas_processed']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg AI Likelihood</span>
                    <span class="stat-value">{pipeline_stats['avg_ai_likelihood']:.3f}</span>
                </div>
            </div>

            <div class="card">
                <h2>🔍 Discovery Status</h2>
                <div class="stat">
                    <span class="stat-label">Total Ideas</span>
                    <span class="stat-value">{discovery_stats['total_ideas_discovered']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Runs (7 days)</span>
                    <span class="stat-value">{discovery_stats['discovery_runs_7d']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Per Run</span>
                    <span class="stat-value">{discovery_stats['avg_ideas_per_run']:.1f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Last Run</span>
                    <span class="stat-value">{discovery_stats['last_discovery_run'] or 'N/A'}</span>
                </div>
            </div>
        </div>

        <!-- Schedule Configuration -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>⏰ Schedule Configuration</h2>
            <div class="grid">
                <div>
                    <div class="stat">
                        <span class="stat-label">Discovery Interval</span>
                        <span class="stat-value">Every {schedule.get('interval_hours', 'N/A')} hours</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Pipeline Interval</span>
                        <span class="stat-value">Every {schedule.get('pipeline_interval_hours', 'N/A')} hours</span>
                    </div>
                </div>
                <div>
                    <div class="stat">
                        <span class="stat-label">Enabled</span>
                        <span class="stat-value">{'✓ Yes' if schedule.get('enabled') else '✗ No'}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Review Required</span>
                        <span class="stat-value">{'✓ Yes' if schedule.get('review_required') else '✗ No'}</span>
                    </div>
                </div>
            </div>
            <div class="controls">
                <button onclick="triggerPipeline()">▶️ Trigger Pipeline Now</button>
                <button onclick="triggerDiscovery()">🔍 Run Discovery</button>
                <button class="danger" onclick="pauseScheduler()">⏸ Pause Scheduler</button>
            </div>
        </div>

        <!-- Recent Jobs -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>📋 Recent Pipeline Jobs</h2>
            {self._render_jobs_table_html(recent_jobs)}
        </div>

        <!-- Error Summary -->
        <div class="card">
            <h2>⚠️ Error Summary (Last 7 Days)</h2>
            {self._render_errors_html(error_summary)}
        </div>
    </div>

    <script>
        function triggerPipeline() {{
            fetch('/v1/trigger', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{num_ideas: 3, review_required: true}})
            }})
            .then(r => r.json())
            .then(data => alert('Pipeline triggered: ' + data.job_id))
            .catch(e => alert('Error: ' + e));
        }}

        function triggerDiscovery() {{
            fetch('/v1/discovery/trigger', {{method: 'POST'}})
            .then(r => r.json())
            .then(data => alert('Discovery triggered'))
            .catch(e => alert('Error: ' + e));
        }}

        function pauseScheduler() {{
            if (confirm('Pause the scheduler?')) {{
                fetch('/v1/schedule', {{
                    method: 'PUT',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{...{json.dumps(schedule)}, enabled: false}})
                }})
                .then(() => {{alert('Scheduler paused'); location.reload();}})
                .catch(e => alert('Error: ' + e));
            }}
        }}
    </script>
</body>
</html>"""
        return html

    def _render_issues_html(self, issues: List[str]) -> str:
        if not issues:
            return '<div style="color: #28a745; margin-top: 16px; font-size: 14px;">✓ No issues detected</div>'

        html = '<ul class="issue-list">'
        for issue in issues:
            html += f'<li class="issue-item">⚠ {issue}</li>'
        html += '</ul>'
        return html

    def _render_agents_html(self, agents: List[Dict]) -> str:
        html = ''
        for agent in agents:
            status = agent['status']
            dot_class = f"status-{status}" if status in ['healthy', 'down'] else "status-degraded"
            html += f'''
            <div class="agent-status">
                <div class="agent-dot {dot_class}"></div>
                <div style="flex: 1;">
                    <strong>{agent['agent'].upper()}</strong>
                    <div style="font-size: 12px; color: #666;">{agent['url']}</div>
                </div>
                <span style="color: {'#28a745' if status == 'healthy' else '#dc3545'};">{status}</span>
            </div>
            '''
        return html

    def _render_jobs_table_html(self, jobs: List[Dict]) -> str:
        if not jobs:
            return '<p style="color: #666;">No recent jobs</p>'

        html = '<table><thead><tr><th>Job ID</th><th>Status</th><th>Started</th><th>Ideas</th></tr></thead><tbody>'

        for job in jobs[:10]:
            status = job.get('status', 'unknown')
            badge_class = f"badge-{status}" if status in ['success', 'error', 'running', 'queued'] else "badge-queued"
            started = job.get('started_at', 'N/A')
            if started and len(started) > 19:
                started = started[:19]

            html += f'''<tr>
                <td>{job.get('job_id', 'N/A')}</td>
                <td><span class="{badge_class} status-badge">{status}</span></td>
                <td>{started}</td>
                <td>{job.get('num_ideas', 'N/A')}</td>
            </tr>'''

        html += '</tbody></table>'
        return html

    def _render_errors_html(self, errors: List[Dict]) -> str:
        if not errors:
            return '<p style="color: #28a745;">✓ No errors in the last 7 days</p>'

        html = '<table><thead><tr><th>Error Type</th><th>Count</th></tr></thead><tbody>'
        for error in errors[:10]:
            html += f'''<tr>
                <td>{error['type']}</td>
                <td><strong style="color: #dc3545;">{error['count']}</strong></td>
            </tr>'''
        html += '</tbody></table>'
        return html
