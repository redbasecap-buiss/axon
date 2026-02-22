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

## License

MIT
