#!/usr/bin/env python3
"""
IRIS Agent - Drafting & Composition

Consolidated into agent-mnemosyne for single-process deployment.
Generates structured outlines and authentic drafts with VoicePrint.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class IdeaInput(BaseModel):
    """Input idea from discovery"""
    title: str
    content: Optional[str] = None
    source_url: Optional[str] = None
    url: Optional[str] = None
    context: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None


class OutlineSection(BaseModel):
    """Section in outline"""
    heading: str
    key_points: List[str]


class OutlineResponse(BaseModel):
    """Structured outline response"""
    outline_id: str
    idea_title: str
    hook: str
    sections: List[OutlineSection]
    closing: str
    word_count_estimate: int
    created_at: str


class DraftRequest(BaseModel):
    """Request for draft generation"""
    outline_id: Optional[str] = None
    idea: IdeaInput
    target_length: int = Field(default=800, ge=400, le=1300)
    include_hashtags: bool = False


class DraftResponse(BaseModel):
    """Generated draft response"""
    draft_id: str
    outline_id: Optional[str] = None
    title: str
    content: str
    word_count: int
    voice_score: float
    created_at: str
    metadata: Dict


# ============================================================================
# VoicePrint System
# ============================================================================

class VoicePrint:
    """Manages voice parameters and authenticity"""

    def __init__(self, voiceprint_path: Path):
        self.voiceprint_path = voiceprint_path
        self.params = self._load_voiceprint()

    def _load_voiceprint(self) -> Dict:
        """Load VoicePrint from JSON"""
        if not self.voiceprint_path.exists():
            logger.warning(f"VoicePrint not found at {self.voiceprint_path}, using defaults")
            return self._get_default_voiceprint()

        with open(self.voiceprint_path, 'r') as f:
            return json.load(f)

    def _get_default_voiceprint(self) -> Dict:
        """Default VoicePrint parameters"""
        return {
            "voice_parameters": {
                "tone_markers": {
                    "analytical": 0.80,
                    "conversational": 0.65,
                    "technical": 0.75
                }
            },
            "structure_preferences": {
                "body_pattern": "define_contrast_synthesize_project"
            }
        }

    def get_voice_prompt(self) -> str:
        """Generate voice prompt for Claude"""
        vp = self.params.get("voice_parameters", {})
        sp = self.params.get("structure_preferences", {})
        examples = self.params.get("example_snippets", {})

        prompt = f"""# Voice Parameters

**Tone**: Analytical ({vp.get('tone_markers', {}).get('analytical', 0.8)*100:.0f}%), Conversational ({vp.get('tone_markers', {}).get('conversational', 0.65)*100:.0f}%), Technical ({vp.get('tone_markers', {}).get('technical', 0.75)*100:.0f}%)

**Writing Patterns**:
- Use specific numbers and concrete examples
- Make clear assertions, avoid hedging
- Prefer active voice
- Mix short punchy sentences with longer analytical ones

**Common Phrases**: {', '.join(vp.get('common_phrases', [])[:3])}

**Structure**: {sp.get('body_pattern', 'define_contrast_synthesize_project')}

**Example Snippets**:
- Opening: "{examples.get('opening', '')}"
- Analytical: "{examples.get('analytical', '')}"
- Conversational: "{examples.get('conversational', '')}"
- Closing: "{examples.get('closing', '')}"
"""
        return prompt


# ============================================================================
# IRIS Agent
# ============================================================================

class IrisAgent:
    """
    IRIS - Drafting & Composition Agent

    Generates outlines and drafts with VoicePrint authenticity.
    Consolidated into single process for direct function calls.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize IRIS agent

        Args:
            data_dir: Path to Mnemosyne data directory
        """
        self.data_dir = data_dir
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.voiceprint = VoicePrint(data_dir / "voiceprint.json")
        self.drafts_dir = data_dir / "drafts"
        self.outlines_dir = data_dir / "outlines"

        # Ensure directories exist
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.outlines_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✓ IRIS agent initialized")

    def generate_outline(self, idea: IdeaInput) -> OutlineResponse:
        """
        Generate structured outline from idea

        Args:
            idea: Input idea from discovery

        Returns:
            OutlineResponse with structured outline

        Structure: Define → Contrast → Synthesize → Project
        """
        # Get content URL
        content_url = idea.url or idea.source_url or ""
        content_summary = idea.content or idea.context or ""

        # Build outline prompt
        prompt = f"""You are IRIS, a drafting agent that creates structured outlines for LinkedIn posts.

{self.voiceprint.get_voice_prompt()}

# Task
Create a structured outline for a LinkedIn post about this idea:

**Title**: {idea.title}
**Source**: {content_url}
**Summary**: {content_summary[:500]}

# Outline Structure (Define → Contrast → Synthesize → Project)

1. **Hook** - Opening line that grabs attention (question or insight)
2. **Define** - What is this? Core concept/development
3. **Contrast** - How is this different? What's changing?
4. **Synthesize** - Why does this matter? Implications
5. **Project** - What's next? Actionable takeaway
6. **Closing** - Clear call-to-action or thought-provoking statement

# Requirements
- Hook must be punchy and specific
- Each section needs 2-4 concrete key points
- Use specific numbers where available
- Maintain analytical + conversational tone
- Target: 800 words final draft

Output format (JSON):
{{
  "hook": "...",
  "sections": [
    {{"heading": "Define", "key_points": ["...", "..."]}},
    {{"heading": "Contrast", "key_points": ["...", "..."]}},
    {{"heading": "Synthesize", "key_points": ["...", "..."]}},
    {{"heading": "Project", "key_points": ["...", "..."]}}
  ],
  "closing": "..."
}}
"""

        logger.info(f"Generating outline for: {idea.title}")

        try:
            # Call Claude
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response
            # Debug: log the raw response
            logger.debug(f"Raw message object: {message}")
            logger.debug(f"Content type: {type(message.content)}")
            logger.debug(f"Content: {message.content}")

            # Handle different response formats
            if isinstance(message.content, list) and len(message.content) > 0:
                content_block = message.content[0]
                if hasattr(content_block, 'text'):
                    response_text = content_block.text
                else:
                    response_text = str(content_block)
            else:
                response_text = str(message.content)

            logger.debug(f"Extracted response_text: {response_text[:500]}")

            # Try to extract JSON (Claude might wrap it in markdown)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            logger.debug(f"Extracted json_str: {json_str[:500] if json_str else 'EMPTY!'}")

            if not json_str or json_str.isspace():
                raise ValueError(f"Empty JSON string extracted from response. Full response: {response_text}")

            outline_data = json.loads(json_str)

            # Create outline response
            outline_id = f"outline_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            sections = [
                OutlineSection(
                    heading=section["heading"],
                    key_points=section["key_points"]
                )
                for section in outline_data["sections"]
            ]

            outline = OutlineResponse(
                outline_id=outline_id,
                idea_title=idea.title,
                hook=outline_data["hook"],
                sections=sections,
                closing=outline_data["closing"],
                word_count_estimate=800,
                created_at=datetime.now().isoformat()
            )

            # Save outline
            outline_path = self.outlines_dir / f"{outline_id}.json"
            with open(outline_path, 'w') as f:
                json.dump(outline.dict(), f, indent=2)

            logger.info(f"✓ Created outline: {outline_id}")

            return outline

        except Exception as e:
            logger.error(f"Error generating outline: {e}", exc_info=True)
            raise

    def generate_draft(self, request: DraftRequest) -> DraftResponse:
        """
        Generate full draft from outline or idea

        Args:
            request: Draft generation request with idea and parameters

        Returns:
            DraftResponse with complete draft content
        """
        # Load outline if provided
        outline = None
        if request.outline_id:
            outline_path = self.outlines_dir / f"{request.outline_id}.json"
            if outline_path.exists():
                with open(outline_path, 'r') as f:
                    outline = json.load(f)

        # Build draft prompt
        if outline:
            outline_text = f"""
**Hook**: {outline['hook']}

{chr(10).join([f"**{section['heading']}**:{chr(10)}- " + chr(10) + "- ".join(section['key_points']) for section in outline['sections']])}

**Closing**: {outline['closing']}
"""
        else:
            outline_text = "No outline provided - generate structure as you write"

        content_url = request.idea.url or request.idea.source_url or ""
        content_summary = request.idea.content or request.idea.context or ""

        prompt = f"""You are IRIS, a drafting agent that creates authentic LinkedIn posts.

{self.voiceprint.get_voice_prompt()}

# Task
Write a complete LinkedIn post based on this outline and source material.

**Topic**: {request.idea.title}
**Source**: {content_url}
**Summary**: {content_summary[:500]}

# Outline
{outline_text}

# Requirements
- Target length: {request.target_length} words
- Follow Define → Contrast → Synthesize → Project structure
- Use specific numbers and concrete examples
- Mix short punchy sentences with longer analytical ones
- Make clear assertions, avoid hedging language
- Include 1-2 relevant hashtags at end ONLY if requested: {request.include_hashtags}
- Write in active voice
- Maintain analytical + conversational tone

# Format
- Start with the hook (no title/heading)
- Use paragraph breaks for readability (blank lines)
- NO bullet points or numbered lists in the final draft
- End with actionable closing

Write the complete post now:
"""

        logger.info(f"Generating draft for: {request.idea.title}")

        try:
            # Call Claude
            message = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            draft_content = message.content[0].text.strip()
            word_count = len(draft_content.split())

            # Calculate basic voice score (simplified)
            voice_score = 0.85  # Placeholder - could add NLP analysis

            # Create draft response
            draft_id = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            draft = DraftResponse(
                draft_id=draft_id,
                outline_id=request.outline_id,
                title=request.idea.title,
                content=draft_content,
                word_count=word_count,
                voice_score=voice_score,
                created_at=datetime.now().isoformat(),
                metadata={
                    "source_url": content_url,
                    "target_length": request.target_length,
                    "model": "claude-sonnet-4-20250514"
                }
            )

            # Save draft
            draft_path = self.drafts_dir / f"{draft_id}.json"
            with open(draft_path, 'w') as f:
                json.dump(draft.dict(), f, indent=2)

            logger.info(f"✓ Created draft: {draft_id} ({word_count} words, voice score: {voice_score:.2f})")

            return draft

        except Exception as e:
            logger.error(f"Error generating draft: {e}", exc_info=True)
            raise

    def get_draft(self, draft_id: str) -> Optional[Dict]:
        """Retrieve specific draft by ID"""
        draft_path = self.drafts_dir / f"{draft_id}.json"
        if not draft_path.exists():
            return None

        with open(draft_path, 'r') as f:
            return json.load(f)

    def get_outline(self, outline_id: str) -> Optional[Dict]:
        """Retrieve specific outline by ID"""
        outline_path = self.outlines_dir / f"{outline_id}.json"
        if not outline_path.exists():
            return None

        with open(outline_path, 'r') as f:
            return json.load(f)

    def list_drafts(self, limit: int = 20) -> Dict:
        """List recent drafts"""
        drafts = sorted(self.drafts_dir.glob("draft_*.json"), reverse=True)[:limit]

        return {
            "total": len(list(self.drafts_dir.glob("draft_*.json"))),
            "returned": len(drafts),
            "drafts": [
                {
                    "draft_id": f.stem,
                    "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                }
                for f in drafts
            ]
        }
