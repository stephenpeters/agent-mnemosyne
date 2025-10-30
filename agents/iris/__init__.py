"""
IRIS - Drafting & Composition Agent

Transforms ideas into structured outlines and authentic drafts.
Consolidated into agent-mnemosyne for single-process deployment.
"""

from .agent import (
    IrisAgent,
    VoicePrint,
    IdeaInput,
    OutlineSection,
    OutlineResponse,
    DraftRequest,
    DraftResponse
)

__all__ = [
    'IrisAgent',
    'VoicePrint',
    'IdeaInput',
    'OutlineSection',
    'OutlineResponse',
    'DraftRequest',
    'DraftResponse'
]
