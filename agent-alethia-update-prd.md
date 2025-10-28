PRD: Contextual Memory & Refresh System for Aletheia and Mnemosyne

Document Version: 1.0

Author: [System Design Lead, Athena Core Project]

Date: 28 Oct 2025

Status: Draft for Implementation

⸻

1. Product Overview

Purpose:
To enable Aletheia (the ideation and reflection agent) to maintain meaningful conversational and conceptual context across sessions, while Mnemosyne (the long-term memory archive) provides persistent storage, summarization, and decay logic.

The result: a cognitively coherent system that remembers recent and relevant ideas, forgets gracefully, and operates even when partially disconnected.

⸻

2. Goals & Success Criteria

Goal	Description	Success Metric
Context Continuity	Retain conversational and thematic context across sessions	Aletheia accurately recalls prior themes and rejected ideas 95% of time
Graceful Forgetting	Decay and summarise older context into 90-day summaries	No stale or irrelevant context in active session cache
Resilient Operation	Operate normally if Mnemosyne API unreachable	Zero downtime, with context_confidence score reported
Daily Refresh Cycle	Maintain updated digests every 24 hours	Refresh completes successfully within <2 min on local hardware
Transparent Memory State	Log and display memory summaries for debugging	Context summaries available on demand via API or UI command


⸻

3. Scope

In Scope
	•	Aletheia’s short-term cache (30 days) and mid-term summary (90 days) logic
	•	Mnemosyne’s API interface for context retrieval and update
	•	Daily refresh daemon with graceful fallback
	•	Context decay and summarization functions
	•	Logging and health metrics

Out of Scope
	•	Cross-agent synchronization with Athena or Hermes
	•	Enterprise integrations (Averus / Sybil / Minerva)
	•	Multi-user tenancy
	•	Real-time collaborative context sharing

⸻

4. Feature Requirements

4.1 Memory Tiers

Layer	Owner	Lifetime	Description
Short-Term Cache	Aletheia	≤30 days	Stores active conversation threads, current embeddings, local vector context
Mid-Term Summary	Aletheia	30–90 days	Stores thematic digests of recurring ideas, accepted/rejected content
Long-Term Archive	Mnemosyne	Indefinite	Stores summarized and raw semantic history, embeddings, and metadata


⸻

4.2 Daily Context Refresh Cycle

Step	Description	Frequency	Dependencies
1	Pull latest summaries via GET /mnemosyne/context	Daily @ 02:00	Network availability
2	Prune expired short-term cache (>30 days)	Daily	Local only
3	Generate new mid-term summaries (30–90 days)	Daily	Local only
4	Push updates via POST /mnemosyne/update	Daily	Network availability
5	Re-index embeddings and update weights	Daily	Local only

Fallback Rule:
If Mnemosyne API fails, continue in context-compromised mode using cached digest. Log warning, queue updates for retry.

⸻

4.3 Context Decay Rules

Type	Mechanism	Trigger
Temporal Decay	Delete cache items >30 days	Daily refresh
Summary Rotation	Convert mid-term (30–90 days) to summary nodes and archive to Mnemosyne	90-day rotation
Relevance Pruning	Drop embeddings with cosine similarity <0.5 vs current focus	On refresh
Reinforcement	Reset decay timer for accessed items	On session use


⸻

4.4 API Specification

GET /mnemosyne/context

Fetch relevant digests for current focus.

{
  "agent": "aletheia",
  "time_window_days": 90,
  "topics": ["liquidity", "agentic commerce"],
  "context_summary": [
    {
      "topic": "intelligent liquidity",
      "summary": "User frames topics around corporate treasury optimization.",
      "weight": 0.86,
      "embedding_id": "liq-123"
    }
  ]
}

POST /mnemosyne/update

Push accepted/rejected ideas and summaries.

{
  "session_id": "uuid",
  "accepted_ideas": ["liquidity-as-signal"],
  "rejected_ideas": ["AI moralism narrative"],
  "summary_text": "Prefers applied framing around stablecoin liquidity.",
  "timestamp": "2025-10-28T09:32Z"
}


⸻

4.5 Error Handling

Failure	Behaviour	Severity
API timeout	Continue with cached digest, mark	

