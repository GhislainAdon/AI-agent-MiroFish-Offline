<div align="center">

<img src="./static/image/mirofish-offline-banner.png" alt="MiroFish Offline" width="100%"/>

# MiroFish-Offline

**Fully local fork of [MiroFish](https://github.com/666ghj/MiroFish) — no cloud APIs required. English UI.**

*A multi-agent swarm intelligence engine that simulates public opinion, market sentiment, and social dynamics. Entirely on your hardware.*

[![GitHub Stars](https://img.shields.io/github/stars/nikmcfly/MiroFish-Offline?style=flat-square&color=DAA520)](https://github.com/nikmcfly/MiroFish-Offline/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/nikmcfly/MiroFish-Offline?style=flat-square)](https://github.com/nikmcfly/MiroFish-Offline/network)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## What is this?

MiroFish is a multi-agent simulation engine: upload any document (press release, policy draft, financial report), and it generates hundreds of AI agents with unique personalities that simulate the public reaction on social media. Posts, arguments, opinion shifts — hour by hour.

The [original MiroFish](https://github.com/666ghj/MiroFish) was built for the Chinese market (Chinese UI, Zep Cloud for knowledge graphs, DashScope API). This fork makes it **fully local and fully English**:

| Original MiroFish | MiroFish-Offline |
|---|---|
| Chinese UI | **English UI** (1,000+ strings translated) |
| Zep Cloud (graph memory) | **Neo4j Community Edition 5.15** |
| DashScope / OpenAI API (LLM) | **Ollama** (qwen2.5, llama3, etc.) |
| Zep Cloud embeddings | **nomic-embed-text** via Ollama |
| Cloud API keys required | **Zero cloud dependencies** |

## Workflow

1. **Construction du graphe** — Extrait les entites (personnes, entreprises, evenements) et les relations depuis vos documents. Construit un graphe de connaissances avec memoire individuelle et collective via Neo4j.
2. **Preparation de la simulation** — Lit les entites du graphe, genere les profils/personas des agents, puis produit la configuration OASIS adaptee au scenario.
3. **Execution Twitter/Reddit** — Lance la simulation sur les plateformes sociales simulees: publications, reponses, controverses, changements d'opinion. Le systeme suit en temps reel l'evolution du sentiment, la propagation des sujets et les dynamiques d'influence.
4. **Rapport** — Un ReportAgent analyse l'environnement post-simulation, interroge un groupe cible d'agents, recherche des preuves dans le graphe de connaissances et genere une analyse structuree.
5. **Interaction** — Permet de dialoguer avec les agents simules ou avec le ReportAgent pour comprendre les reactions, les memoires et les raisonnements issus de la simulation.

## Screenshot

<div align="center">
<img src="./static/image/mirofish-offline-screenshot.jpg" alt="MiroFish Offline — English UI" width="100%"/>
</div>

## Quick Start

### Prerequisites

- Docker Desktop & Docker Compose (recommended), **or**
- Python 3.11+, Node.js 18+, Neo4j 5.15+, Ollama

### Option A: Docker + Ollama local (recommended for Windows/Mac)

This setup runs the MiroFish app and Neo4j in Docker containers, while using your **existing local Ollama** installation. This avoids GPU passthrough issues on Windows and lets you reuse models you've already downloaded.

**1. Install Ollama on your host machine**

Download from [ollama.com](https://ollama.com/download) and install. Then pull the required models:

```bash
ollama pull glm4:9b             # LLM (or llama3.2:1b for lighter setup)
ollama pull nomic-embed-text    # Embeddings (768d)
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

**2. Clone and configure**

```bash
git clone https://github.com/GhislainAdon/AI-agent-MiroFish-Offline.git
cd AI-agent-MiroFish-Offline
```

Create a `.env` file with these settings (pointing to your local Ollama via `host.docker.internal`):

```bash
# LLM Configuration
LLM_API_KEY=ollama
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL_NAME=glm4:9b

# Neo4j (container-to-container communication)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish

# Embeddings
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://host.docker.internal:11434

# OASIS / CAMEL-AI
OPENAI_API_KEY=ollama
OPENAI_API_BASE_URL=http://host.docker.internal:11434/v1
```

> **Note:** Replace `glm4:9b` with any Ollama model you have: `llama3.2:1b` (lightweight), `qwen2.5:14b`, `mistral`, etc.

**3. Start with Docker Compose**

```bash
docker compose up --build -d
```

This will:
- Build the MiroFish app image (Flask backend + Vue frontend)
- Start a Neo4j 5.18 database container
- Connect to your local Ollama via `host.docker.internal`

**4. Verify everything is running**

```bash
# Check containers
docker ps

# Check logs
docker logs mirofish-offline

# Test Ollama connectivity from inside the container
docker exec mirofish-offline curl -s http://host.docker.internal:11434/api/tags
```

Open **http://localhost:3000** — that's it.

| Service | URL | Description |
|---------|-----|-------------|
| MiroFish UI | http://localhost:3000 | Main application |
| Backend API | http://localhost:5001 | Flask REST API |
| Neo4j Browser | http://localhost:7474 | Database admin (neo4j/mirofish) |
| Ollama | http://localhost:11434 | LLM server (on host) |

**5. Stop / restart**

```bash
docker compose down       # Stop all containers
docker compose up -d      # Restart
docker compose logs -f    # Follow logs
```

### Option B: Full Docker (with containerized Ollama + NVIDIA GPU)

For Linux systems with NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
git clone https://github.com/GhislainAdon/AI-agent-MiroFish-Offline.git
cd AI-agent-MiroFish-Offline
cp .env.example .env

# Start all services (Neo4j, Ollama, MiroFish)
docker compose up -d

# Pull the required models into the containerized Ollama
docker exec mirofish-ollama ollama pull qwen2.5:32b
docker exec mirofish-ollama ollama pull nomic-embed-text
```

> **Windows users:** Option A is recommended. Docker Desktop on Windows does not easily support NVIDIA GPU passthrough for containers.

### Option C: Manual (no Docker)

**1. Start Neo4j**

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/mirofish \
  neo4j:5.18-community
```

**2. Start Ollama & pull models**

```bash
ollama serve &
ollama pull glm4:9b          # LLM (or any compatible model)
ollama pull nomic-embed-text  # Embeddings (768d)
```

**3. Configure & run backend**

```bash
cp .env.example .env
# Edit .env — use localhost URLs (not host.docker.internal)

cd backend
pip install -r requirements.txt
python run.py
```

**4. Run frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Running Tests

The project includes unit tests for the backend. Run them from the `backend/` directory:

```bash
cd backend
uv run pytest tests/ -v
```

Tests cover: LLM client, task manager, entity reader, file parser, embedding service, and Flask app factory.

## Configuration

All settings are in `.env` (copy from `.env.example`):

```bash
# LLM — points to local Ollama (OpenAI-compatible API)
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2.5:32b

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish

# Embeddings
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434
```

Works with any OpenAI-compatible API — swap Ollama for Claude, GPT, or any other provider by changing `LLM_BASE_URL` and `LLM_API_KEY`.

## Architecture

This fork introduces a clean abstraction layer between the application and the graph database:

```
┌─────────────────────────────────────────┐
│              Flask API                   │
│  graph.py  simulation.py  report.py     │
└──────────────┬──────────────────────────┘
               │ app.extensions['neo4j_storage']
┌──────────────▼──────────────────────────┐
│           Service Layer                  │
│  EntityReader  GraphToolsService         │
│  GraphMemoryUpdater  ReportAgent         │
└──────────────┬──────────────────────────┘
               │ storage: GraphStorage
┌──────────────▼──────────────────────────┐
│         GraphStorage (abstract)          │
│              │                            │
│    ┌─────────▼─────────┐                │
│    │   Neo4jStorage     │                │
│    │  ┌───────────────┐ │                │
│    │  │ EmbeddingService│ ← Ollama       │
│    │  │ NERExtractor   │ ← Ollama LLM   │
│    │  │ SearchService  │ ← Hybrid search │
│    │  └───────────────┘ │                │
│    └───────────────────┘                │
└─────────────────────────────────────────┘
               │
        ┌──────▼──────┐
        │  Neo4j CE   │
        │  5.15       │
        └─────────────┘
```

**Key design decisions:**

- `GraphStorage` is an abstract interface — swap Neo4j for any other graph DB by implementing one class
- Dependency injection via Flask `app.extensions` — no global singletons
- Hybrid search: 0.7 × vector similarity + 0.3 × BM25 keyword search
- Synchronous NER/RE extraction via local LLM (replaces Zep's async episodes)
- All original dataclasses and LLM tools (InsightForge, Panorama, Agent Interviews) preserved

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 10 GB (14b model) | 24 GB (32b model) |
| Disk | 20 GB | 50 GB |
| CPU | 4 cores | 8+ cores |

CPU-only mode works but is significantly slower for LLM inference. For lighter setups, use `qwen2.5:14b` or `qwen2.5:7b`.

## Use Cases

- **PR crisis testing** — simulate the public reaction to a press release before publishing
- **Trading signal generation** — feed financial news and observe simulated market sentiment
- **Policy impact analysis** — test draft regulations against simulated public response
- **Creative experiments** — someone fed it a classical Chinese novel with a lost ending; the agents wrote a narratively consistent conclusion

## License

AGPL-3.0 — same as the original MiroFish project. See [LICENSE](./LICENSE).

## Credits & Attribution

This is a modified fork of [MiroFish](https://github.com/666ghj/MiroFish) by [666ghj](https://github.com/666ghj), originally supported by [Shanda Group](https://www.shanda.com/). The simulation engine is powered by [OASIS](https://github.com/camel-ai/oasis) from the CAMEL-AI team.

**Modifications in this fork:**
- Backend migrated from Zep Cloud to local Neo4j CE 5.15 + Ollama
- Entire frontend translated from Chinese to English (20 files, 1,000+ strings)
- All Zep references replaced with Neo4j across the UI
- Rebranded to MiroFish Offline
