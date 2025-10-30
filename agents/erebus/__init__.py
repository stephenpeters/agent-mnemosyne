"""
Erebus - Authenticity & Post-Processing Agent

Removes AI fingerprints and reintroduces human irregularity.
Consolidated into agent-mnemosyne for single-process deployment.
"""

from .agent import (
    ErebusAgent,
    AIDetector,
    LinguisticPerturber,
    VoiceBaseline,
    DraftInput,
    CleanedResponse
)

__all__ = [
    'ErebusAgent',
    'AIDetector',
    'LinguisticPerturber',
    'VoiceBaseline',
    'DraftInput',
    'CleanedResponse'
]
