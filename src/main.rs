mod config;
mod crawler;
mod db;
#[allow(dead_code)]
mod embeddings;
mod export;
mod graph;
mod nlp;
pub mod plugin;
pub mod prometheus;
mod query;
mod server;
mod tui;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "axon",
    version,
    about = "A tiny brain that never stops learning."
)]
struct Cli {
    /// Path to the brain database
    #[arg(long, default_value = "axon.db")]
    db: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Feed a URL to learn from
    Feed { url: String },
    /// Crawl known sources and discover new pages
    Crawl {
        /// Maximum pages to crawl
        #[arg(long, default_value = "10")]
        max_pages: usize,
    },
    /// Watch a URL for changes
    Watch { url: String },
    /// Ask a question
    Ask { question: String },
    /// Show everything known about an entity
    About { entity: String },
    /// Show related entities
    Related { entity: String },
    /// Show recently learned facts
    Recent {
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Show top knowledge areas
    Topics {
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Forget low-confidence old facts
    Forget {
        /// Minimum confidence threshold (facts below this are pruned)
        #[arg(long, default_value = "0.1")]
        threshold: f64,
        /// Minimum age in days before considering for pruning
        #[arg(long, default_value = "30")]
        min_age_days: i64,
    },
    /// Find semantically similar entities
    Similar {
        entity: String,
        /// Number of results
        #[arg(long, default_value = "10")]
        k: usize,
    },
    /// Auto-cluster entities by similarity
    Cluster,
    /// Show brain statistics
    Stats,
    /// Browse the knowledge graph interactively in the terminal
    Browse,
    /// Find shortest path between two entities
    Path { from: String, to: String },
    /// Show top entities by PageRank
    Rank {
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Export the knowledge graph
    Export {
        /// Output format: json, mermaid, dot, csv
        #[arg(long, default_value = "json")]
        format: String,
    },
    /// Deduplicate near-duplicate entities
    Dedup,
    /// Run PROMETHEUS discovery pipeline — find patterns and generate hypotheses
    Discover {
        /// Output format: text, json, markdown
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// List current hypotheses
    Hypotheses {
        /// Filter by status: proposed, testing, confirmed, rejected
        #[arg(long)]
        status: Option<String>,
    },
    /// Explain a hypothesis — show full reasoning chain
    Explain {
        /// Hypothesis ID
        hypothesis_id: i64,
    },
    /// Generate default config at ~/.axon/config.toml
    Init,
    /// Launch HTTP API server
    Serve {
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
    },
    /// Run as daemon, continuously crawling and learning
    Daemon {
        /// Interval between crawl cycles (e.g. 30m, 1h)
        #[arg(long, default_value = "30m")]
        interval: String,
        /// Config file path
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

fn parse_duration(s: &str) -> Result<std::time::Duration, String> {
    let s = s.trim();
    if let Some(mins) = s.strip_suffix('m') {
        let n: u64 = mins.parse().map_err(|e| format!("{e}"))?;
        Ok(std::time::Duration::from_secs(n * 60))
    } else if let Some(hrs) = s.strip_suffix('h') {
        let n: u64 = hrs.parse().map_err(|e| format!("{e}"))?;
        Ok(std::time::Duration::from_secs(n * 3600))
    } else if let Some(secs) = s.strip_suffix('s') {
        let n: u64 = secs.parse().map_err(|e| format!("{e}"))?;
        Ok(std::time::Duration::from_secs(n))
    } else {
        Err(format!("Invalid duration: {s}. Use e.g. 30m, 1h, 60s"))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Handle init before loading config (it creates the config file).
    if matches!(cli.command, Commands::Init) {
        let path = config::Config::write_default()?;
        println!("✅ Default config written to {}", path.display());
        return Ok(());
    }

    let cfg = config::Config::load()?;
    let db_path = if cli.db == std::path::Path::new("axon.db") {
        PathBuf::from(&cfg.brain_path)
    } else {
        cli.db.clone()
    };
    let brain = db::Brain::open(&db_path)?;

    match cli.command {
        Commands::Feed { url } => {
            println!("🧠 Feeding: {url}");
            let text = crawler::fetch_and_extract(&url).await?;
            let extracted = nlp::process_text(&text, &url);
            let (entities, relations, facts) = brain.learn(&extracted)?;
            println!("   Learned {entities} entities, {relations} relations, {facts} facts");
        }
        Commands::Crawl { max_pages } => {
            println!("🕷️  Crawling (max {max_pages} pages)...");
            let total = crawler::crawl(&brain, max_pages).await?;
            println!("   Crawled {total} pages");
        }
        Commands::Watch { url } => {
            println!("👁️  Watching: {url}");
            brain.add_to_frontier(&url, 1)?;
            println!(
                "   Added to watch list. Use `axon crawl` or `axon daemon` to check for updates."
            );
        }
        Commands::Ask { question } => {
            let results = query::ask(&brain, &question)?;
            if results.is_empty() {
                println!("🤷 I don't know anything about that yet. Try feeding me some URLs!");
            } else {
                println!("💡 Here's what I know:\n");
                for r in results {
                    println!("  {r}");
                }
            }
        }
        Commands::About { entity } => {
            let info = query::about(&brain, &entity)?;
            if info.is_empty() {
                println!("🤷 I don't know about \"{entity}\" yet.");
            } else {
                println!("📖 About \"{entity}\":\n");
                for line in info {
                    println!("  {line}");
                }
            }
        }
        Commands::Related { entity } => {
            let related = query::related(&brain, &entity)?;
            if related.is_empty() {
                println!("🤷 No related entities found for \"{entity}\".");
            } else {
                println!("🔗 Related to \"{entity}\":\n");
                for r in related {
                    println!("  {r}");
                }
            }
        }
        Commands::Recent { limit } => {
            let facts = query::recent(&brain, limit)?;
            if facts.is_empty() {
                println!("🤷 No facts learned yet. Try `axon feed <url>`!");
            } else {
                println!("🕐 Recently learned:\n");
                for f in facts {
                    println!("  {f}");
                }
            }
        }
        Commands::Topics { limit } => {
            let topics = query::topics(&brain, limit)?;
            if topics.is_empty() {
                println!("🤷 No topics yet. Feed me some URLs!");
            } else {
                println!("📊 Top topics:\n");
                for t in topics {
                    println!("  {t}");
                }
            }
        }
        Commands::Similar { entity, k } => {
            let results = embeddings::find_similar(&brain, &entity, k)?;
            if results.is_empty() {
                println!("🤷 No similar entities found for \"{entity}\".");
            } else {
                println!("🔮 Similar to \"{entity}\":\n");
                for (name, score) in results {
                    println!("  {score:.3}  {name}");
                }
            }
        }
        Commands::Cluster => {
            let clusters = embeddings::cluster_entities(&brain)?;
            if clusters.is_empty() {
                println!("🤷 No entities to cluster.");
            } else {
                println!("🧬 Entity clusters:\n");
                for (id, members) in clusters {
                    println!("  Cluster {id}: {}", members.join(", "));
                }
            }
        }
        Commands::Forget {
            threshold,
            min_age_days,
        } => {
            let pruned = brain.forget(threshold, min_age_days)?;
            println!("🧹 Forgot {pruned} low-confidence facts");
        }
        Commands::Stats => {
            let stats = brain.stats()?;
            println!("🧠 Brain Statistics:\n");
            println!("  Entities:  {}", stats.entity_count);
            println!("  Relations: {}", stats.relation_count);
            println!("  Facts:     {}", stats.fact_count);
            println!("  Sources:   {}", stats.source_count);
            println!("  DB size:   {}", stats.db_size);
        }
        Commands::Browse => {
            tui::run(&db_path)?;
        }
        Commands::Path { from, to } => match graph::shortest_path(&brain, &from, &to)? {
            Some(path) => {
                let formatted = graph::format_path(&brain, &path)?;
                println!("🔗 Path ({} hops): {formatted}", path.len() - 1);
            }
            None => println!("🤷 No path found between \"{from}\" and \"{to}\"."),
        },
        Commands::Rank { limit } => {
            let scores = graph::pagerank(&brain, 0.85, 30)?;
            let mut ranked: Vec<_> = scores.into_iter().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.truncate(limit);
            if ranked.is_empty() {
                println!("🤷 No entities to rank.");
            } else {
                println!("📊 Top entities by PageRank:\n");
                for (i, (id, score)) in ranked.iter().enumerate() {
                    let name = brain
                        .get_entity_by_id(*id)?
                        .map(|e| e.name)
                        .unwrap_or_else(|| format!("#{id}"));
                    println!("  {}. {name} ({score:.4})", i + 1);
                }
            }
        }
        Commands::Export { format } => {
            let output = match format.as_str() {
                "json" => export::to_json(&brain)?,
                "mermaid" => export::to_mermaid(&brain)?,
                "dot" => export::to_dot(&brain)?,
                "csv" => export::to_csv(&brain)?,
                "json-ld" => export::to_json_ld(&brain)?,
                "graphml" => export::to_graphml(&brain)?,
                _ => {
                    eprintln!(
                        "Unknown format: {format}. Use: json, mermaid, dot, csv, json-ld, graphml"
                    );
                    std::process::exit(1);
                }
            };
            println!("{output}");
        }
        Commands::Dedup => {
            let merged = graph::merge_near_duplicates(&brain)?;
            if merged.is_empty() {
                println!("✅ No near-duplicates found.");
            } else {
                println!("🔗 Merged {} duplicate pairs:\n", merged.len());
                for (keep, removed) in &merged {
                    println!("  \"{removed}\" → \"{keep}\"");
                }
            }
        }
        Commands::Discover { format } => {
            let p = prometheus::Prometheus::new(&brain)?;
            let report = p.discover()?;
            match format.as_str() {
                "json" => println!("{}", p.report_json(&report)),
                "markdown" | "md" => println!("{}", p.report_markdown(&report)),
                _ => {
                    println!("🔬 PROMETHEUS Discovery Report\n");
                    println!("{}\n", report.summary);
                    if !report.patterns_found.is_empty() {
                        println!("Patterns found: {}", report.patterns_found.len());
                        for pat in &report.patterns_found {
                            println!(
                                "  [{:>15}] (freq: {:>3}) {}",
                                pat.pattern_type.as_str(),
                                pat.frequency,
                                pat.description
                            );
                        }
                        println!();
                    }
                    if !report.hypotheses_generated.is_empty() {
                        println!(
                            "Hypotheses generated: {}",
                            report.hypotheses_generated.len()
                        );
                        for h in report.hypotheses_generated.iter().take(30) {
                            println!(
                                "  [{:.2}] {} {} {} — {}",
                                h.confidence,
                                h.subject,
                                h.predicate,
                                h.object,
                                h.status.as_str()
                            );
                        }
                        if report.hypotheses_generated.len() > 30 {
                            println!("  ... and {} more", report.hypotheses_generated.len() - 30);
                        }
                        println!();
                    }
                    // Knowledge frontiers
                    let frontiers = p.find_knowledge_frontiers().unwrap_or_default();
                    if !frontiers.is_empty() {
                        println!("Knowledge frontiers:");
                        for (etype, count, avg, reason) in frontiers.iter().take(10) {
                            println!(
                                "  📡 {} ({} entities, {:.1} avg rels): {}",
                                etype, count, avg, reason
                            );
                        }
                        println!();
                    }
                    // Cross-domain gaps
                    let cross_gaps = p.find_cross_domain_gaps().unwrap_or_default();
                    if !cross_gaps.is_empty() {
                        println!("Cross-domain gaps:");
                        for (a, b, reason) in cross_gaps.iter().take(10) {
                            println!("  🔗 [{}] ↔ [{}]: {}", a.join(", "), b.join(", "), reason);
                        }
                        println!();
                    }
                    // Crawl suggestions
                    let suggestions = p.suggest_crawl_topics().unwrap_or_default();
                    if !suggestions.is_empty() {
                        println!("Suggested topics to crawl:");
                        for (topic, reason) in suggestions.iter().take(10) {
                            println!("  🌐 {}: {}", topic, reason);
                        }
                    }
                }
            }
        }
        Commands::Hypotheses { status } => {
            let p = prometheus::Prometheus::new(&brain)?;
            let filter = status.map(|s| prometheus::HypothesisStatus::from_str(&s));
            let hyps = p.list_hypotheses(filter)?;
            if hyps.is_empty() {
                println!("🤷 No hypotheses yet. Run `axon discover` first!");
            } else {
                println!("📋 Hypotheses ({}):\n", hyps.len());
                for h in &hyps {
                    println!(
                        "  #{:<4} [{:.2}] {} {} {} — {}",
                        h.id,
                        h.confidence,
                        h.subject,
                        h.predicate,
                        h.object,
                        h.status.as_str()
                    );
                }
            }
        }
        Commands::Explain { hypothesis_id } => {
            let p = prometheus::Prometheus::new(&brain)?;
            match p.explain(hypothesis_id)? {
                Some(explanation) => println!("{}", explanation),
                None => println!("🤷 Hypothesis #{} not found.", hypothesis_id),
            }
        }
        Commands::Init => unreachable!(),
        Commands::Serve { port } => {
            server::run_server(db_path, port).await?;
        }
        Commands::Daemon {
            interval,
            config: config_path,
        } => {
            let dcfg = if let Some(p) = config_path {
                config::Config::load_from(&p)?
            } else {
                cfg.clone()
            };
            let dur = if interval != "30m" {
                parse_duration(&interval).map_err(|e| anyhow::anyhow!(e))?
            } else {
                std::time::Duration::from_secs(dcfg.daemon.interval_secs)
            };
            let max_pages = dcfg.crawl.max_depth;
            println!(
                "🤖 Daemon mode: crawling every {}s, max {max_pages} depth",
                dur.as_secs()
            );
            loop {
                println!(
                    "\n--- Crawl cycle at {} ---",
                    chrono::Local::now().format("%H:%M:%S")
                );
                match crawler::crawl(&brain, max_pages).await {
                    Ok(n) => println!("   Crawled {n} pages"),
                    Err(e) => eprintln!("   Error: {e}"),
                }
                // Apply decay
                let decayed = brain.apply_decay(0.01)?;
                if decayed > 0 {
                    println!("   Decayed {decayed} items");
                }
                tokio::time::sleep(dur).await;
            }
        }
    }
    Ok(())
}
