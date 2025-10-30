#!/usr/bin/env python3
"""
Kairos Agent - Scheduling & Publishing

Consolidated into agent-mnemosyne for single-process deployment.
Schedules and publishes content to LinkedIn (MVP: file-based queue).
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class QueuedContent(BaseModel):
    """Content queued for publishing"""
    queue_id: str
    draft_id: Optional[str] = None
    cleaned_id: Optional[str] = None
    title: str
    content: str
    status: str = "queued"  # queued, scheduled, published, failed
    queued_at: str
    scheduled_for: Optional[str] = None
    published_at: Optional[str] = None
    metadata: Dict = {}


# ============================================================================
# Kairos Agent
# ============================================================================

class KairosAgent:
    """
    Kairos - Scheduling & Publishing Agent

    Schedules and publishes content to LinkedIn.
    MVP: Simple file-based queue.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize Kairos agent

        Args:
            data_dir: Path to Mnemosyne data directory
        """
        self.data_dir = data_dir
        self.queue_dir = data_dir / "queue"
        self.published_dir = data_dir / "published"

        # Ensure directories exist
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.published_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✓ Kairos agent initialized")

    def queue_for_publishing(self, content: Dict) -> Dict:
        """
        Queue content for publishing

        Args:
            content: Content dict with title, content, and metadata

        Returns:
            Dict with queue_id and status
        """
        queue_id = f"queue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        queued_content = QueuedContent(
            queue_id=queue_id,
            draft_id=content.get("draft_id"),
            cleaned_id=content.get("cleaned_id"),
            title=content.get("title", "Untitled"),
            content=content.get("cleaned_content", content.get("content", "")),
            status="queued",
            queued_at=datetime.now().isoformat(),
            metadata=content.get("metadata", {})
        )

        # Save to queue directory
        queue_path = self.queue_dir / f"{queue_id}.json"
        with open(queue_path, 'w') as f:
            json.dump(queued_content.dict(), f, indent=2)

        logger.info(f"✓ Queued for publishing: {queue_id}")

        return {
            "queue_id": queue_id,
            "status": "queued",
            "queued_at": queued_content.queued_at
        }

    def list_queue(self, limit: int = 20) -> Dict:
        """List items in publishing queue"""
        queued = sorted(self.queue_dir.glob("queue_*.json"), reverse=True)[:limit]

        items = []
        for f in queued:
            with open(f, 'r') as file:
                data = json.load(file)
                items.append({
                    "queue_id": data.get("queue_id"),
                    "title": data.get("title"),
                    "status": data.get("status"),
                    "queued_at": data.get("queued_at")
                })

        return {
            "total": len(list(self.queue_dir.glob("queue_*.json"))),
            "returned": len(items),
            "items": items
        }

    def get_queued_item(self, queue_id: str) -> Optional[Dict]:
        """Retrieve specific queued item by ID"""
        queue_path = self.queue_dir / f"{queue_id}.json"
        if not queue_path.exists():
            return None

        with open(queue_path, 'r') as f:
            return json.load(f)

    def mark_as_published(self, queue_id: str, linkedin_post_id: Optional[str] = None) -> bool:
        """
        Mark queued content as published

        Args:
            queue_id: Queue item ID
            linkedin_post_id: LinkedIn post ID (optional)

        Returns:
            True if successful
        """
        queue_path = self.queue_dir / f"{queue_id}.json"
        if not queue_path.exists():
            logger.error(f"Queue item not found: {queue_id}")
            return False

        # Load queued content
        with open(queue_path, 'r') as f:
            content = json.load(f)

        # Update status
        content["status"] = "published"
        content["published_at"] = datetime.now().isoformat()
        if linkedin_post_id:
            content["linkedin_post_id"] = linkedin_post_id

        # Move to published directory
        published_path = self.published_dir / f"{queue_id}.json"
        with open(published_path, 'w') as f:
            json.dump(content, f, indent=2)

        # Remove from queue
        queue_path.unlink()

        logger.info(f"✓ Marked as published: {queue_id}")

        return True

    def list_published(self, limit: int = 20) -> Dict:
        """List recently published content"""
        published = sorted(self.published_dir.glob("queue_*.json"), reverse=True)[:limit]

        items = []
        for f in published:
            with open(f, 'r') as file:
                data = json.load(file)
                items.append({
                    "queue_id": data.get("queue_id"),
                    "title": data.get("title"),
                    "published_at": data.get("published_at"),
                    "linkedin_post_id": data.get("linkedin_post_id")
                })

        return {
            "total": len(list(self.published_dir.glob("queue_*.json"))),
            "returned": len(items),
            "items": items
        }
