# Mnemosyne – Memory & Governance Layer

**Mission**: Maintain cross-agent memory, track voice drift, and enforce governance rules for data and tone consistency.

## Core Capabilities

- Persist embeddings of all generated and published content
- Serve contextual data for new drafts and ideas
- Monitor voice deviation and topic balance
- Manage keys, credentials, and access scopes
- Track system performance and learning metrics

## API Endpoints

- `POST /v1/context` - Retrieve contextual data for content generation
- `POST /v1/learn` - Persist new content and update memory
- `GET /v1/metrics/voice` - Voice drift monitoring
- `GET /v1/metrics/performance` - System performance analytics
- `GET /healthz` - Health check endpoint

## Key Deliverables

- `mnemosyne_corpus` schema for content memory
- Embedding and retrieval system
- Voice drift tracking algorithms
- Governance and access control framework

## Dependencies

- Python 3.11+
- FastAPI, uvicorn
- Postgres + pgvector for semantic search
- Cloud storage (S3-compatible) for long-term archives
- Cloud KMS + Secret Manager for key management

## Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the service
uvicorn main:app --reload --port 8005

# Install additional dependencies
pip install <package>
pip freeze > requirements.txt
```

## Data Flow

Mnemosyne is the central memory and governance layer for all agents:

```
        Aletheia → IRIS → Erebus → Kairos
            ↓       ↓       ↓        ↓
                  Mnemosyne
                (Memory & Learning)
```

## Memory System

Mnemosyne maintains:
- **Content corpus**: Embeddings of all drafts and published content
- **Voice baseline**: Authentic voice parameters and patterns
- **Context history**: Related ideas and topic clusters
- **Performance data**: Engagement metrics and timing patterns

## Governance Functions

- **Key rotation**: Monthly credential refresh
- **Audit trail**: Track all content transformations
- **Access control**: Role-based permissions
- **Data encryption**: At-rest and in-transit security
- **Voice monitoring**: Detect and alert on drift

## Quality Targets

- Voice deviation: < 0.35 from baseline
- Context relevance: > 0.80 similarity for suggestions
- System uptime: 99.9%
- Audit completeness: 100% of transformations logged

## Related Repositories

- [agent-aletheia](https://github.com/stephenpeters/agent-aletheia) - Provides ideas for context checking
- [agent-iris](https://github.com/stephenpeters/agent-iris) - Uses voice baseline for drafting
- [agent-erebus](https://github.com/stephenpeters/agent-erebus) - Compares against style baseline
- [agent-kairos](https://github.com/stephenpeters/agent-kairos) - Feeds performance data
- [agent-sdk](https://github.com/stephenpeters/agent-sdk) - Shared schemas and utilities
