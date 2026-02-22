mod config;
mod crawler;
mod db;
mod nlp;
pub mod plugin;
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
    /// Show brain statistics
    Stats,
    /// Browse the knowledge graph interactively in the terminal
    Browse,
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
