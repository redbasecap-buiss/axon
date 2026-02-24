# 🧠 Axon

**A tiny brain that never stops learning.**

Axon is a lightweight, continuously self-learning knowledge engine written in pure Rust. It crawls the web, extracts entities and relations, builds a knowledge graph, and answers your questions — all with zero ML frameworks, minimal compute, and a single SQLite file as its brain.

## Philosophy

> Knowledge isn't static. A brain that stops learning is already dead.

Axon treats knowledge like memory: facts have **confidence** that grows when reinforced and **decays** when neglected. Feed it URLs, let it crawl, and watch it build understanding over time. Old, unreinforced facts fade away — just like real memory.

## Quick Start

```bash
# Build
cargo build --release

# Feed it a URL
axon feed https://en.wikipedia.org/wiki/Rust_(programming_language)

# Ask questions
axon ask "Rust programming"

# See what it knows about something
axon about "Rust"

# Show related entities
axon related "Mozilla"

# See recently learned facts
axon recent

# Show top knowledge areas
axon topics

# View brain statistics
axon stats

# Watch a URL for changes
axon watch https://news.ycombinator.com

# Crawl known sources
axon crawl --max-pages 20

# Forget old, low-confidence facts
axon forget --threshold 0.1 --min-age-days 30

# Fuzzy search (typo-tolerant)
axon fuzzy "Einstien"            # finds "Einstein"
axon fuzzy "Motzart" --distance 2

# Ingest local markdown files
axon ingest notes.md
axon ingest ./docs/ --recursive

# Import URLs from sitemaps
axon sitemap https://example.com --max 200

# Detect contradictions in the knowledge graph
axon contradictions

# Run as daemon (continuously learn)
axon daemon --interval 30m
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Web Crawler │────▶│ Text Process │────▶│ Knowledge Graph  │
│  (reqwest)   │     │   (NLP)      │     │   (SQLite)       │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                   │
                                          ┌────────┴────────┐
                                          │  Query Engine    │
                                          │  (ask/about/etc) │
                                          └─────────────────┘
```

### Knowledge Graph (SQLite)

- **Entities**: Named things with types, confidence scores, and access counts
- **Relations**: Subject → predicate → object triples with provenance
- **Facts**: Key-value pairs attached to entities
- **Frontier**: Crawl queue with priority and change detection

### Text Processing (Pure Rust, no ML)

- Sentence splitting & tokenization
- TF-IDF keyword extraction
- Entity extraction: capitalized phrases, years, URLs
- Relation extraction: subject-verb-object patterns
- Deduplication via Levenshtein distance

### Incremental Learning

- **Reinforcement**: Seeing something again increases confidence
- **Temporal decay**: Unused knowledge fades over time
- **Forgetting**: Prune low-confidence old facts to keep the brain lean
- **Change detection**: Content hashing to learn only from new/changed pages

### Web Crawler

- Respects `robots.txt`
- Rate limiting with politeness delays
- Crawl frontier with priority queue
- Content hash-based change detection
- **Sitemap parsing** — auto-discover URLs from sitemap.xml and sitemap indexes
- **Markdown ingestion** — feed local .md files and directories

### Fuzzy Search

- Levenshtein distance with Damerau transposition
- Auto-calibrated edit distance based on query length
- Substring and word-level matching
- Bigram similarity for phrase matching
- Typo-tolerant entity lookup across CLI, API, and queries

### Contradiction Detection

- Detects conflicting facts for the same entity/key
- Severity levels: HARD (equal confidence), SOFT (one dominant), MINOR (numeric rounding)
- Boolean contradiction detection (true/false, alive/dead, etc.)
- Numeric closeness analysis (within 5% = minor)

## Configuration

For daemon mode, create an `axon.toml`:

```toml
[crawl]
max_pages = 20
```

```bash
axon daemon --interval 1h --config axon.toml
```

## Dependencies

Pure Rust, minimal footprint:

- `reqwest` — HTTP client (rustls-tls, no OpenSSL)
- `scraper` — HTML parsing
- `rusqlite` — SQLite (bundled, zero system deps)
- `clap` — CLI parsing
- `chrono` — Time handling
- `tokio` — Async runtime
- `texting_robots` — robots.txt parsing

## 🌐 REST API

```bash
axon serve --port 8080
```

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/api/ask?q=...` | GET | Query the knowledge graph |
| `/api/stats` | GET | Brain statistics |
| `/api/topics` | GET | Top entities |
| `/api/feed` | POST | Feed a URL `{"url": "..."}` |
| `/api/fuzzy?q=...&distance=2` | GET | Fuzzy entity search |
| `/api/contradictions` | GET | Detect contradictions |
| `/api/about?name=...` | GET | Everything known about an entity |

## 🔬 PROMETHEUS — Automated Scientific Discovery

PROMETHEUS is axon's discovery engine. It finds patterns in the knowledge graph, detects gaps, generates hypotheses, and validates them — automated scientific reasoning over your data.

### What it does

1. **Pattern Discovery** — Frequent subgraph mining, co-occurrence analysis, temporal sequence detection, statistical anomaly finding
2. **Gap Detection** — Structural holes (missing links between connected entities), type-based gaps (entities missing expected relations), analogy detection
3. **Hypothesis Engine** — Generates testable hypotheses from discovered gaps, scores confidence, checks for contradictions, builds full reasoning chains
4. **Validation** — Cross-references hypotheses against existing knowledge, updates confidence based on evidence
5. **Meta-learning** — Tracks which discovery patterns lead to confirmed hypotheses, adjusts pattern weights over time

### Commands

```bash
# Run full discovery pipeline
axon discover
axon discover --format json
axon discover --format markdown

# List hypotheses
axon hypotheses
axon hypotheses --status proposed

# Explain a hypothesis (full reasoning chain)
axon explain 42
```

### Data model

- **Hypothesis**: subject-predicate-object triple with confidence, evidence for/against, reasoning chain, status (proposed → testing → confirmed/rejected)
- **Pattern**: recurring structural motif with type, frequency, involved entities
- **Discovery**: confirmed hypothesis with evidence sources
- **Pattern weights**: meta-learning — tracks confirmation/rejection rates per pattern type

All stored in SQLite alongside the existing brain tables.

## License

MIT
