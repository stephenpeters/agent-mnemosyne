"""
Aletheia - Idea Discovery Agent

Finds and scores content ideas from multiple sources.
Consolidated into agent-mnemosyne for single-process deployment.
"""

from .agent import (
    AletheiaAgent,
    WebSearcher
)

__all__ = [
    'AletheiaAgent',
    'WebSearcher'
]
