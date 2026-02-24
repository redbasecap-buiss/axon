mod chat;
mod config;
pub mod criticality;
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
pub mod reasoning;
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
    /// Interactive chat REPL — ask questions conversationally
    Chat,
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
    /// Run Self-Organized Criticality analysis — full brain health report
    Criticality,
    /// Meta-cognitive self-assessment — domain balance, staleness, velocity
    Introspect,
    /// Abductive reasoning — explain an observation
    Abduce {
        /// Observation to explain (e.g. "Newton and Leibniz both connected to Calculus")
        observation: String,
    },
    /// Predict which entities are most likely to gain new connections
    Predict,
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
            match chat::answer_question(&brain, &question)? {
                Some(answer) => print!("{}", chat::format_answer(&answer)),
                None => println!("🤷 I don't know anything about that yet. Try feeding me some URLs!"),
            }
        }
        Commands::Chat => {
            chat::run_chat(&brain).map_err(|e| anyhow::anyhow!("{}", e))?;
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
            // Filter noise types from ranking display
            let noise_types: std::collections::HashSet<&str> = [
                "phrase",
                "source",
                "url",
                "relative_date",
                "number_unit",
                "date",
                "year",
                "currency",
                "email",
                "compound_noun",
            ]
            .iter()
            .copied()
            .collect();
            let mut filtered = Vec::new();
            for (id, score) in &ranked {
                if let Some(e) = brain.get_entity_by_id(*id)? {
                    if !noise_types.contains(e.entity_type.as_str()) && e.name.len() >= 2 {
                        filtered.push((*id, *score, e.name, e.entity_type));
                    }
                }
                if filtered.len() >= limit {
                    break;
                }
            }
            if filtered.is_empty() {
                println!("🤷 No entities to rank.");
            } else {
                println!("📊 Top entities by PageRank:\n");
                for (i, (_id, score, name, etype)) in filtered.iter().enumerate() {
                    println!("  {}. {name} [{etype}] ({score:.4})", i + 1);
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
                    // Fuzzy duplicates
                    let dupes = p.find_fuzzy_duplicates().unwrap_or_default();
                    if !dupes.is_empty() {
                        println!("Fuzzy duplicate candidates:");
                        for (a, b, sim, action) in dupes.iter().take(15) {
                            println!("  🔀 [{:.2}] '{}' ↔ '{}' ({})", sim, a, b, action);
                        }
                        if dupes.len() > 15 {
                            println!("  ... and {} more", dupes.len() - 15);
                        }
                        println!();
                    }
                    // Topic coverage
                    let topics = p.topic_coverage_analysis().unwrap_or_default();
                    if !topics.is_empty() {
                        println!("Topic coverage by source domain:");
                        for (domain, entities, rels, density, assessment) in topics.iter().take(10)
                        {
                            println!(
                                "  📊 {} — {} entities, {} internal rels, density {:.4}: {}",
                                domain, entities, rels, density, assessment
                            );
                        }
                        println!();
                    }
                    // Enrichment targets
                    let enrichment = p.rank_enrichment_targets(10).unwrap_or_default();
                    if !enrichment.is_empty() {
                        println!("Top enrichment targets (need deeper crawling):");
                        for (name, etype, score, reason) in &enrichment {
                            println!("  🎯 [{:.3}] {} ({}): {}", score, name, etype, reason);
                        }
                        println!();
                    }
                    // Predicate & type entropy
                    if let Ok((pred_ent, top_pred, top_frac)) =
                        crate::graph::predicate_entropy(&brain)
                    {
                        if let Ok((type_ent, num_types, top_type, type_frac)) =
                            crate::graph::type_entropy(&brain)
                        {
                            println!("Graph information content:");
                            println!(
                                "  📐 Predicate entropy: {:.2} bits (top: '{}' at {:.0}%)",
                                pred_ent,
                                top_pred,
                                top_frac * 100.0
                            );
                            println!("  📐 Type entropy: {:.2} bits across {} types (dominant: '{}' at {:.0}%)", type_ent, num_types, top_type, type_frac * 100.0);
                            println!();
                        }
                    }
                    // Strategy ROI
                    if let Ok(roi) = p.strategy_roi() {
                        if !roi.is_empty() {
                            println!("Strategy ROI (confirmations / total generated):");
                            for (strategy, total, confirmed, _rejected, roi_val, rec) in
                                roi.iter().take(10)
                            {
                                println!(
                                    "  📈 {} — {}/{} ({:.0}%): {}",
                                    strategy,
                                    confirmed,
                                    total,
                                    roi_val * 100.0,
                                    rec
                                );
                            }
                            println!();
                        }
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
        Commands::Criticality => {
            println!("⚡ Running Self-Organized Criticality analysis...\n");
            let report = criticality::criticality_report(&brain)?;
            println!("{}", criticality::format_criticality_report(&report));
        }
        Commands::Introspect => {
            println!("🧠 Running meta-cognitive self-assessment...\n");
            let mc = criticality::introspect(&brain)?;
            println!("{}", criticality::format_introspection(&mc));
        }
        Commands::Abduce { observation } => {
            println!("🔍 Abductive reasoning: \"{}\"\n", observation);
            let hypothesis = criticality::abduce(&brain, &observation)?;
            println!("Observation: {}", hypothesis.observation);
            println!("Confidence: {:.2}", hypothesis.confidence);
            println!(
                "Candidate explanations: {}\n",
                hypothesis.candidate_explanations.len()
            );
            if let Some(best) = &hypothesis.best_explanation {
                println!("Best explanation:");
                println!("  {}", best.summary);
                println!(
                    "  Parsimony: {:.2} | Coverage: {:.2} | Consistency: {:.2}",
                    best.parsimony_score, best.coverage_score, best.consistency_score
                );
            }
            for (i, expl) in hypothesis.candidate_explanations.iter().enumerate().skip(1) {
                if i > 5 {
                    break;
                }
                println!("\nAlternative {}:", i);
                println!("  {}", expl.summary);
                println!("  Score: {:.3}", expl.combined_score);
            }
        }
        Commands::Predict => {
            println!("🔮 Predicting next discoveries...\n");
            let predictions = criticality::predict_next_discovery(&brain)?;
            if predictions.is_empty() {
                println!("🤷 Not enough data to make predictions.");
            } else {
                println!(
                    "Top {} entities likely to gain new connections:\n",
                    predictions.len()
                );
                for (i, p) in predictions.iter().enumerate() {
                    println!(
                        "  {}. {} [{}] — score: {:.3}",
                        i + 1,
                        p.entity_name,
                        p.entity_type,
                        p.predicted_score
                    );
                    println!("     {}", p.reason);
                }
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
