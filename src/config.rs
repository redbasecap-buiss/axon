use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Path to the brain database file.
    #[serde(default = "default_brain_path")]
    pub brain_path: String,

    #[serde(default)]
    pub crawl: CrawlConfig,

    #[serde(default)]
    pub daemon: DaemonConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlConfig {
    /// Delay between requests to the same host (ms).
    #[serde(default = "default_politeness_delay_ms")]
    pub politeness_delay_ms: u64,

    /// Maximum link depth when crawling.
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,

    /// User-Agent header.
    #[serde(default = "default_user_agent")]
    pub user_agent: String,

    /// Maximum concurrent HTTP requests.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Seconds between crawl cycles.
    #[serde(default = "default_interval_secs")]
    pub interval_secs: u64,
}

// ── defaults ──

fn default_brain_path() -> String {
    "axon.db".to_string()
}
fn default_politeness_delay_ms() -> u64 {
    1000
}
fn default_max_depth() -> usize {
    3
}
fn default_user_agent() -> String {
    "axon/0.2 (knowledge-engine; +https://github.com/redbasecap-buiss/axon)".to_string()
}
fn default_max_concurrent() -> usize {
    4
}
fn default_interval_secs() -> u64 {
    1800
}

impl Default for Config {
    fn default() -> Self {
        Self {
            brain_path: default_brain_path(),
            crawl: CrawlConfig::default(),
            daemon: DaemonConfig::default(),
        }
    }
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            politeness_delay_ms: default_politeness_delay_ms(),
            max_depth: default_max_depth(),
            user_agent: default_user_agent(),
            max_concurrent: default_max_concurrent(),
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            interval_secs: default_interval_secs(),
        }
    }
}

impl Config {
    /// Default config file path: `~/.axon/config.toml`.
    pub fn default_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".axon")
            .join("config.toml")
    }

    /// Load config from the default path, falling back to defaults if the file
    /// does not exist.
    pub fn load() -> Result<Self> {
        Self::load_from(&Self::default_path())
    }

    /// Load config from an explicit path (falls back to defaults when missing).
    pub fn load_from(path: &Path) -> Result<Self> {
        if path.exists() {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading config {}", path.display()))?;
            let cfg: Config =
                toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
            Ok(cfg)
        } else {
            Ok(Config::default())
        }
    }

    /// Write the default config to `~/.axon/config.toml`, creating the
    /// directory if needed. Returns the path written.
    pub fn write_default() -> Result<PathBuf> {
        let path = Self::default_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let cfg = Config::default();
        let text = toml::to_string_pretty(&cfg).context("serialising default config")?;
        std::fs::write(&path, &text).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips() {
        let cfg = Config::default();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let parsed: Config = toml::from_str(&text).unwrap();
        assert_eq!(parsed.crawl.max_depth, cfg.crawl.max_depth);
        assert_eq!(parsed.daemon.interval_secs, cfg.daemon.interval_secs);
    }

    #[test]
    fn missing_file_gives_defaults() {
        let cfg = Config::load_from(Path::new("/tmp/__axon_nonexistent__")).unwrap();
        assert_eq!(cfg.crawl.politeness_delay_ms, 1000);
    }
}
