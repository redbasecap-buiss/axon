#![allow(dead_code)]
//! PROMETHEUS — Automated Scientific Discovery Engine
//!
//! Pattern discovery, gap detection, hypothesis generation, validation,
//! and meta-learning over the axon knowledge graph.

use chrono::Utc;
use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::db::Brain;

// ---------------------------------------------------------------------------
// Noise filtering
// ---------------------------------------------------------------------------

/// Entity types to exclude from discovery (low signal).
const NOISE_TYPES: &[&str] = &[
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
];

/// Predicates too generic to drive discovery.
const GENERIC_PREDICATES: &[&str] = &[
    "is", "are", "was", "were", "has", "have", "had", "be", "been", "do", "does", "did",
];

/// High-value entity types for focused discovery.
const HIGH_VALUE_TYPES: &[&str] = &[
    "person",
    "organization",
    "place",
    "concept",
    "technology",
    "company",
    "product",
    "event",
];

/// Minimum confidence threshold for meaningful hypothesis confirmation.
const CONFIRMATION_THRESHOLD: f64 = 0.7;
/// Auto-reject threshold.
const REJECTION_THRESHOLD: f64 = 0.15;

/// Stopword entity names that sneak through NLP extraction.
const NOISE_NAMES: &[&str] = &[
    "however",
    "this",
    "that",
    "what",
    "which",
    "these",
    "those",
    "where",
    "when",
    "there",
    "here",
    "also",
    "other",
    "such",
    "some",
    "many",
    "most",
    "more",
    "very",
    "just",
    "only",
    "even",
    "still",
    "already",
    "often",
    "never",
    "because",
    "although",
    "therefore",
    "moreover",
    "furthermore",
    "nevertheless",
    "nonetheless",
    "whereas",
    "meanwhile",
    "otherwise",
    "accordingly",
    "consequently",
    "hence",
    "thus",
    "thereby",
    "pages",
    "adding",
    "besides",
    "journal",
    "pdf",
    "time",
    "see",
    "new",
    "used",
    "using",
    "based",
    "known",
    "called",
    "named",
    "including",
    "currently",
    "recently",
    "today",
    "later",
    "isbn",
    "doi",
    "vol",
    "chapter",
    "section",
    "figure",
    "table",
    "appendix",
    "note",
    "notes",
    "references",
    "bibliography",
    "list",
    "index",
    "contents",
    "abstract",
    "introduction",
    "conclusion",
    "discussion",
    "results",
    "methods",
    "analysis",
    "review",
    "surveys",
    "commentary",
    "proceedings",
    "during",
    "resting",
    "within",
    "between",
    "through",
    "about",
    "above",
    "below",
    "after",
    "before",
    "early",
    "modern",
    "several",
    "various",
    "certain",
    "another",
    "following",
    "previous",
    "former",
    "latter",
    "both",
    "each",
    "every",
    "first",
    "second",
    "third",
    "last",
    "next",
    "major",
    "minor",
    "resources",
    "papers",
    "encounters",
    "bits",
    "college",
    "nature",
    "online",
    "links",
    "related",
    "archives",
    "works",
    "press",
    "images",
    "media",
    "events",
    "topics",
    "articles",
    "reports",
    "documents",
    "files",
    "records",
    "entries",
    "items",
    "details",
    "options",
    "updates",
    "features",
    "tools",
    "services",
    "further",
    "reading",
    "external",
    "official",
    "website",
    "webpage",
    "homepage",
    "galleries",
    "collections",
    "publications",
    "editions",
];

fn is_noise_type(t: &str) -> bool {
    NOISE_TYPES.contains(&t)
}

fn is_generic_predicate(p: &str) -> bool {
    GENERIC_PREDICATES.contains(&p)
}

/// Verb-heavy words that signal a sentence fragment rather than a real entity name.
const FRAGMENT_VERBS: &[&str] = &[
    "would",
    "could",
    "should",
    "became",
    "become",
    "produced",
    "succeeded",
    "incurred",
    "controlled",
    "founded",
    "built",
    "wrote",
    "studied",
];

/// Check if an entity name looks like noise (stopword, too short, numeric, etc.)
fn is_noise_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    if NOISE_NAMES.contains(&lower.as_str()) {
        return true;
    }
    // Single word under 3 chars
    if !name.contains(' ') && name.len() < 3 {
        return true;
    }
    // Mostly non-alphabetic (URLs, references, etc.)
    let alpha_count = name.chars().filter(|c| c.is_alphabetic()).count();
    let total = name.len().max(1);
    if alpha_count < total / 2 {
        return true;
    }
    // Starts with common fragment indicators
    let lower_trimmed = lower.trim();
    if lower_trimmed.starts_with("pdf ")
        || lower_trimmed.starts_with("the original")
        || lower_trimmed.starts_with("archived")
        || lower_trimmed.starts_with("because ")
        || lower_trimmed.starts_with("although ")
        || lower_trimmed.starts_with("therefore ")
        || lower_trimmed.starts_with("whereas ")
        || lower_trimmed.starts_with("furthermore ")
        || lower_trimmed.starts_with("moreover ")
        || lower_trimmed.ends_with(" because")
        || lower_trimmed.ends_with(" although")
    {
        return true;
    }
    // Long multi-word names that look like sentence fragments (>6 words)
    let word_count = lower_trimmed.split_whitespace().count();
    if word_count > 6 {
        return true;
    }
    // Names starting with adverbs/conjunctions — NLP extraction errors (e.g. "Eventually Pierre", "Conversely West Berlin")
    if word_count >= 2 {
        let first_word = lower_trimmed.split_whitespace().next().unwrap_or("");
        let adverb_starts = [
            "eventually",
            "conversely",
            "subsequently",
            "additionally",
            "alternatively",
            "approximately",
            "essentially",
            "historically",
            "immediately",
            "increasingly",
            "particularly",
            "primarily",
            "significantly",
            "specifically",
            "traditionally",
            "ultimately",
            "apparently",
            "presumably",
            "supposedly",
            "meanwhile",
            "similarly",
            "likewise",
            "initially",
            "originally",
            "typically",
            "generally",
            "basically",
            "naturally",
            "notably",
            "merely",
            "largely",
            "partly",
            "partly",
        ];
        if adverb_starts.contains(&first_word) {
            return true;
        }
    }
    // Names starting with gerunds/verbs — NLP extraction errors (e.g. "Using Hilbert", "Subscribe Soviet")
    if word_count >= 2 {
        let first_word = lower_trimmed.split_whitespace().next().unwrap_or("");
        let verb_starts = [
            "using",
            "including",
            "according",
            "following",
            "resulting",
            "subscribe",
            "devastated",
            "introducing",
            "presenting",
            "featuring",
            "announcing",
            "celebrating",
            "exploring",
            "examining",
            "investigating",
            "analyzing",
            "discovering",
            "revealing",
            "surviving",
            "conquering",
            "defeating",
            "invading",
            "occupying",
            "liberating",
            "capturing",
            "completing",
            "completions",
            "completeness",
            "containing",
            "considering",
            "regarding",
            "concerning",
            "involving",
            "requiring",
            "providing",
            "offering",
            "describing",
            "explaining",
            "demonstrating",
            "establishing",
            "developing",
            "producing",
            "creating",
            "building",
            "making",
            "taking",
            "giving",
            "getting",
            "having",
            "being",
            "doing",
            "going",
            "coming",
            "keeping",
            "leaving",
            "putting",
            "running",
            "setting",
            "turning",
            "working",
            "playing",
            "moving",
            "living",
            "starting",
            "beginning",
            "opening",
            "closing",
            "ending",
            "finishing",
            "applying",
            "measuring",
            "calculating",
            "computing",
            "processing",
            "representing",
            "connecting",
            "combining",
            "comparing",
            "assuming",
        ];
        if verb_starts.contains(&first_word) {
            return true;
        }
    }
    // Names containing verbs typical of sentence fragments
    if word_count > 3 {
        let words: Vec<&str> = lower_trimmed.split_whitespace().collect();
        let verb_count = words.iter().filter(|w| FRAGMENT_VERBS.contains(w)).count();
        if verb_count >= 1 {
            return true;
        }
    }
    // Citation-like patterns: contains parentheses with years or publisher info
    if (lower_trimmed.contains("(") && lower_trimmed.contains(")"))
        || lower_trimmed.contains(" pp ")
        || lower_trimmed.contains("pp.")
    {
        return true;
    }
    // Starts with lowercase (likely a sentence fragment, not a proper entity)
    if let Some(first) = name.chars().next() {
        if first.is_lowercase() && word_count > 2 {
            return true;
        }
    }
    // Possessive forms as standalone entities (e.g. "Newton's", "Switzerland's") — noise
    if (lower_trimmed.ends_with("'s") || lower_trimmed.ends_with("'s")) && word_count <= 2 {
        return true;
    }
    // Entities ending with common suffix noise
    if lower_trimmed.ends_with(" et al") || lower_trimmed.ends_with(" et al.") {
        return true;
    }
    // Entities that are just adjectives/demonyms (e.g. "German-speaking", "British", "Jewish")
    let demonym_suffixes = ["ish", "ian", "ese", "ean", "speaking"];
    if word_count == 1
        && demonym_suffixes.iter().any(|s| lower_trimmed.ends_with(s))
        && lower_trimmed.len() < 20
    {
        return true;
    }
    // Entities that look like duplicated words (e.g. "Zurich Airport Zurich Airport")
    if word_count >= 4 {
        let words: Vec<&str> = lower_trimmed.split_whitespace().collect();
        let half = words.len() / 2;
        if words[..half] == words[half..half * 2] {
            return true;
        }
    }
    // Capitalized generic words that aren't real entities
    let generic_caps = [
        "history",
        "theory",
        "revolution",
        "state",
        "prize",
        "meanwhile",
        "buried",
        "research",
        "finally",
        "published",
        "development",
        "example",
        "general",
        "original",
        "national",
        "international",
        "government",
        "society",
        "university",
        "institute",
        "foundation",
        "academy",
        "department",
        "commentary",
        "surveys",
        "critique",
        "conflict",
        "sport",
        "civilization",
        "tradition",
        "empire",
        "kingdom",
        "dynasty",
        "republic",
        "province",
        "strom",
        "like",
        "daughter",
        "airport",
        "cathedral",
        "city",
        "county",
        "church",
        "reformed",
        "orthodox",
        "catholic",
        "association",
        "mathematical",
        "henry",
        "phillips",
        "admiral",
        "director",
        "minimum",
        "maximum",
        "define",
        "storms",
        "formulas",
        "candidates",
        "prejudice",
        "birthplace",
        "novels",
        "slaughter",
        "inferno",
        "scotsman",
        "century",
        "decades",
        "period",
        "chapter",
        "volume",
        "edition",
        "series",
        "region",
        "area",
        "system",
        "process",
        "method",
        "approach",
        "model",
        "structure",
        "function",
        "principle",
        "concept",
        "element",
        "factor",
        "feature",
        "aspect",
        "issue",
        "problem",
        "solution",
        "result",
        "effect",
        "impact",
        "role",
        "type",
        "form",
        "kind",
        "level",
        "degree",
        "range",
        "rate",
        "size",
        "number",
        "amount",
        "value",
        "point",
        "source",
        "basis",
        "terms",
        "means",
        "end",
        "part",
        "side",
        "case",
        "fact",
        "idea",
        "view",
        "sense",
    ];
    if word_count == 1 && generic_caps.contains(&lower_trimmed) {
        return true;
    }
    // Multi-word names ending with generic nouns that aren't real entities
    let trailing_generic = [
        "resources",
        "papers",
        "encounters",
        "bits",
        "links",
        "archives",
        "works",
        "images",
        "media",
        "articles",
        "reports",
        "documents",
        "files",
        "records",
        "entries",
        "items",
        "details",
        "options",
        "updates",
        "features",
        "tools",
        "services",
        "publications",
        "editions",
        "galleries",
        "collections",
    ];
    if word_count >= 2 {
        let last_word = lower_trimmed.split_whitespace().last().unwrap_or("");
        if trailing_generic.contains(&last_word) {
            return true;
        }
    }
    // Names ending with "Journal", "During", "Resting", "Commentary" etc. — citation/fragment noise
    let trailing_noise = [
        "journal",
        "during",
        "resting",
        "commentary",
        "surveys",
        "proceedings",
        "magazine",
        "review",
        "bulletin",
        "newsletter",
        "gazette",
        "digest",
        "quarterly",
        "annual",
        "monthly",
        "weekly",
    ];
    if word_count >= 2 {
        let last_word = lower_trimmed.split_whitespace().last().unwrap_or("");
        if trailing_noise.contains(&last_word) {
            return true;
        }
    }
    // Names containing "Journal" anywhere (journal titles extracted as entities)
    if lower_trimmed.contains("journal") {
        return true;
    }
    // Names containing academic publication patterns
    if lower_trimmed.contains("monthly notices") || lower_trimmed.contains("physical review") {
        return true;
    }
    // Names containing 4-digit years (citation/reference fragments like "Ramanujan Journal 1997 1")
    let year_re = lower_trimmed.split_whitespace().any(|w| {
        w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()) && w.starts_with(['1', '2'])
    });
    if year_re && word_count >= 3 {
        return true;
    }
    // Roman numeral suffixes without real content (e.g. "Michael V", "Morgan I")
    let roman_numerals = [
        "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii",
    ];
    if word_count == 2 {
        let words: Vec<&str> = lower_trimmed.split_whitespace().collect();
        if roman_numerals.contains(&words[1]) && words[0].len() <= 10 {
            // But allow real names like "Henry VIII" — only filter if first word isn't a common first name
            // For now, be conservative and keep this as a soft signal
            // combined with other noise indicators
        }
    }
    // "Like" suffix patterns (e.g. "German Like")
    if word_count >= 2 && lower_trimmed.ends_with(" like") {
        return true;
    }
    // Names that are mostly numbers with some text (e.g. "1997 1", "Vol 23")
    let digit_count = lower_trimmed.chars().filter(|c| c.is_ascii_digit()).count();
    if digit_count > 0 && digit_count as f64 / lower_trimmed.len().max(1) as f64 > 0.4 {
        return true;
    }
    // Names starting with "Sir" followed by a single letter (e.g. "Sir I")
    if lower_trimmed.starts_with("sir ") && word_count <= 2 {
        return true;
    }
    // Names containing single-letter words interspersed (citation fragments like "Symanzik K Schrödinger")
    if word_count >= 3 {
        let words: Vec<&str> = lower_trimmed.split_whitespace().collect();
        let single_letter_count = words.iter().filter(|w| w.len() == 1).count();
        if single_letter_count >= 1 && single_letter_count as f64 / word_count as f64 >= 0.25 {
            return true;
        }
    }
    // Names that are just titles/honorifics
    let title_only = ["sir", "mr", "mrs", "ms", "dr", "prof", "lord", "lady"];
    if word_count == 1 && title_only.contains(&lower_trimmed) {
        return true;
    }
    false
}

/// Filter entity IDs to only meaningful ones (non-noise type, reasonable name, quality).
fn meaningful_ids(brain: &Brain) -> Result<HashSet<i64>> {
    let entities = brain.all_entities()?;
    Ok(entities
        .iter()
        .filter(|e| {
            !is_noise_type(&e.entity_type)
                && e.name.len() <= 80
                && e.name.len() >= 2
                && !is_noise_name(&e.name)
        })
        .map(|e| e.id)
        .collect())
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisStatus {
    Proposed,
    Testing,
    Confirmed,
    Rejected,
}

impl HypothesisStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Testing => "testing",
            Self::Confirmed => "confirmed",
            Self::Rejected => "rejected",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "testing" => Self::Testing,
            "confirmed" => Self::Confirmed,
            "rejected" => Self::Rejected,
            _ => Self::Proposed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: i64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub evidence_for: Vec<String>,
    pub evidence_against: Vec<String>,
    pub reasoning_chain: Vec<String>,
    pub status: HypothesisStatus,
    pub discovered_at: String,
    pub pattern_source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    CoOccurrence,
    StructuralHole,
    TypeGap,
    Analogy,
    TemporalSequence,
    FrequentSubgraph,
}

impl PatternType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CoOccurrence => "co_occurrence",
            Self::StructuralHole => "structural_hole",
            Self::TypeGap => "type_gap",
            Self::Analogy => "analogy",
            Self::TemporalSequence => "temporal_sequence",
            Self::FrequentSubgraph => "frequent_subgraph",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "co_occurrence" => Self::CoOccurrence,
            "structural_hole" => Self::StructuralHole,
            "type_gap" => Self::TypeGap,
            "analogy" => Self::Analogy,
            "temporal_sequence" => Self::TemporalSequence,
            "frequent_subgraph" => Self::FrequentSubgraph,
            _ => Self::CoOccurrence,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: i64,
    pub pattern_type: PatternType,
    pub entities_involved: Vec<String>,
    pub frequency: i64,
    pub last_seen: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discovery {
    pub id: i64,
    pub hypothesis_id: i64,
    pub confirmed_at: String,
    pub evidence_sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternWeight {
    pub pattern_type: String,
    pub confirmations: i64,
    pub rejections: i64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub patterns_found: Vec<Pattern>,
    pub hypotheses_generated: Vec<Hypothesis>,
    pub gaps_detected: usize,
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Schema initialisation
// ---------------------------------------------------------------------------

pub fn init_prometheus_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_for TEXT NOT NULL DEFAULT '[]',
            evidence_against TEXT NOT NULL DEFAULT '[]',
            reasoning_chain TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'proposed',
            discovered_at TEXT NOT NULL,
            pattern_source TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            entities_involved TEXT NOT NULL DEFAULT '[]',
            frequency INTEGER NOT NULL DEFAULT 1,
            last_seen TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id INTEGER NOT NULL,
            confirmed_at TEXT NOT NULL,
            evidence_sources TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(id)
        );
        CREATE TABLE IF NOT EXISTS pattern_weights (
            pattern_type TEXT PRIMARY KEY,
            confirmations INTEGER NOT NULL DEFAULT 0,
            rejections INTEGER NOT NULL DEFAULT 0,
            weight REAL NOT NULL DEFAULT 1.0
        );
        CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
        CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
        ",
    )
}

// ---------------------------------------------------------------------------
// Prometheus engine
// ---------------------------------------------------------------------------

pub struct Prometheus<'a> {
    brain: &'a Brain,
}

impl<'a> Prometheus<'a> {
    pub fn new(brain: &'a Brain) -> Result<Self> {
        brain.with_conn(init_prometheus_schema)?;
        Ok(Self { brain })
    }

    // -----------------------------------------------------------------------
    // Pattern Discovery
    // -----------------------------------------------------------------------

    /// Find entities that share a source URL but have no direct relation.
    /// In sparse graphs, co-extraction from the same page is strong evidence of relatedness.
    pub fn find_source_co_occurrences(&self) -> Result<Vec<Pattern>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Group entities by source_url
        let mut source_entities: HashMap<String, HashSet<i64>> = HashMap::new();
        for r in &relations {
            if !r.source_url.is_empty() {
                if meaningful.contains(&r.subject_id) {
                    source_entities
                        .entry(r.source_url.clone())
                        .or_default()
                        .insert(r.subject_id);
                }
                if meaningful.contains(&r.object_id) {
                    source_entities
                        .entry(r.source_url.clone())
                        .or_default()
                        .insert(r.object_id);
                }
            }
        }

        // Build direct-connection set
        let mut connected: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            connected.insert(key);
        }

        // Find pairs sharing ≥2 sources but not directly connected
        let mut pair_sources: HashMap<(i64, i64), usize> = HashMap::new();
        for entities in source_entities.values() {
            let ids: Vec<i64> = entities.iter().copied().collect();
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let key = if ids[i] < ids[j] {
                        (ids[i], ids[j])
                    } else {
                        (ids[j], ids[i])
                    };
                    if !connected.contains(&key) {
                        *pair_sources.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut patterns = Vec::new();
        for ((a, b), count) in &pair_sources {
            if *count >= 1 {
                let a_name = self.entity_name(*a)?;
                let b_name = self.entity_name(*b)?;
                patterns.push(Pattern {
                    id: 0,
                    pattern_type: PatternType::CoOccurrence,
                    entities_involved: vec![a_name.clone(), b_name.clone()],
                    frequency: *count as i64,
                    last_seen: now_str(),
                    description: format!(
                        "{} and {} co-occur in {} source(s) but lack direct relation",
                        a_name, b_name, count
                    ),
                });
            }
        }
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        patterns.truncate(50);
        Ok(patterns)
    }

    /// Find co-occurring entity pairs using Jaccard similarity.
    /// Better than raw count for sparse graphs — normalizes by neighbourhood size.
    pub fn find_co_occurrences(&self, min_shared: usize) -> Result<Vec<Pattern>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;
        let mut neighbours: HashMap<i64, HashSet<i64>> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id) && meaningful.contains(&r.object_id) {
                neighbours
                    .entry(r.subject_id)
                    .or_default()
                    .insert(r.object_id);
                neighbours
                    .entry(r.object_id)
                    .or_default()
                    .insert(r.subject_id);
            }
        }
        // Only consider entities with ≥2 neighbours (otherwise can't share any)
        let ids: Vec<i64> = neighbours
            .iter()
            .filter(|(_, nb)| nb.len() >= min_shared.max(2))
            .map(|(&id, _)| id)
            .collect();
        // Cap at 2000 highest-degree entities to avoid O(n²) blowup
        let mut ids_sorted: Vec<(i64, usize)> =
            ids.iter().map(|&id| (id, neighbours[&id].len())).collect();
        ids_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        ids_sorted.truncate(2000);
        let ids: Vec<i64> = ids_sorted.into_iter().map(|(id, _)| id).collect();
        // Use inverted index: for each neighbour, list which candidates have it
        let mut inv: HashMap<i64, Vec<usize>> = HashMap::new();
        for (idx, &id) in ids.iter().enumerate() {
            for &nb in &neighbours[&id] {
                inv.entry(nb).or_default().push(idx);
            }
        }
        // Count shared neighbours via inverted index (avoids full O(n²))
        let mut pair_shared: HashMap<(usize, usize), usize> = HashMap::new();
        for posting in inv.values() {
            if posting.len() < 2 || posting.len() > 200 {
                continue; // skip hubs to avoid quadratic blowup within posting lists
            }
            for i in 0..posting.len() {
                for j in (i + 1)..posting.len() {
                    let key = (posting[i].min(posting[j]), posting[i].max(posting[j]));
                    *pair_shared.entry(key).or_insert(0) += 1;
                }
            }
        }
        let mut patterns = Vec::new();
        for ((i, j), shared) in &pair_shared {
            if *shared >= min_shared {
                let a = ids[*i];
                let b = ids[*j];
                let na = &neighbours[&a];
                let nb = &neighbours[&b];
                let union_size = na.union(nb).count();
                let jaccard = if union_size > 0 {
                    *shared as f64 / union_size as f64
                } else {
                    0.0
                };
                let a_name = self.entity_name(a)?;
                let b_name = self.entity_name(b)?;
                patterns.push(Pattern {
                    id: 0,
                    pattern_type: PatternType::CoOccurrence,
                    entities_involved: vec![a_name.clone(), b_name.clone()],
                    frequency: *shared as i64,
                    last_seen: now_str(),
                    description: format!(
                        "{} and {} share {} neighbours (Jaccard: {:.2})",
                        a_name, b_name, shared, jaccard
                    ),
                });
            }
        }
        // Sort by frequency descending for better results
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        patterns.truncate(100);
        Ok(patterns)
    }

    /// Find co-occurring entity pairs using Pointwise Mutual Information (PMI).
    /// PMI measures how much more likely two entities are to co-occur (share a source)
    /// than expected by chance. Better than Jaccard for sparse graphs because it
    /// accounts for entity frequency — rare entities co-occurring is more informative
    /// than common ones. PMI = log2(P(a,b) / (P(a) * P(b)))
    pub fn find_pmi_co_occurrences(&self, min_pmi: f64) -> Result<Vec<Pattern>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Build entity → set of source URLs
        let mut entity_sources: HashMap<i64, HashSet<String>> = HashMap::new();
        let mut all_sources: HashSet<String> = HashSet::new();
        for r in &relations {
            if !r.source_url.is_empty() {
                all_sources.insert(r.source_url.clone());
                if meaningful.contains(&r.subject_id) {
                    entity_sources
                        .entry(r.subject_id)
                        .or_default()
                        .insert(r.source_url.clone());
                }
                if meaningful.contains(&r.object_id) {
                    entity_sources
                        .entry(r.object_id)
                        .or_default()
                        .insert(r.source_url.clone());
                }
            }
        }

        let n = all_sources.len() as f64;
        if n < 2.0 {
            return Ok(vec![]);
        }

        // Build direct-connection set
        let mut connected: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            connected.insert(key);
        }

        // Only consider entities appearing in ≥2 sources (reduces noise + O(n²) cost)
        let mut candidates: Vec<(i64, &HashSet<String>)> = entity_sources
            .iter()
            .filter(|(_, sources)| sources.len() >= 2)
            .map(|(&id, sources)| (id, sources))
            .collect();
        // Cap at 1000 entities with most sources to avoid O(n²) blowup
        candidates.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        candidates.truncate(1000);

        let mut patterns = Vec::new();
        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let (a, sa) = candidates[i];
                let (b, sb) = candidates[j];
                let key = if a < b { (a, b) } else { (b, a) };
                if connected.contains(&key) {
                    continue;
                }
                let joint = sa.intersection(sb).count() as f64;
                if joint < 1.0 {
                    continue;
                }
                let p_ab = joint / n;
                let p_a = sa.len() as f64 / n;
                let p_b = sb.len() as f64 / n;
                let pmi = (p_ab / (p_a * p_b)).log2();
                if pmi >= min_pmi {
                    let a_name = self.entity_name(a)?;
                    let b_name = self.entity_name(b)?;
                    patterns.push(Pattern {
                        id: 0,
                        pattern_type: PatternType::CoOccurrence,
                        entities_involved: vec![a_name.clone(), b_name.clone()],
                        frequency: joint as i64,
                        last_seen: now_str(),
                        description: format!(
                            "{} and {} have PMI={:.2} ({} shared sources) — statistically surprising co-occurrence",
                            a_name, b_name, pmi, joint as i64
                        ),
                    });
                }
            }
        }
        patterns.sort_by(|a, b| {
            // Sort by PMI (embedded in description), approximate by frequency for now
            b.frequency.cmp(&a.frequency)
        });
        patterns.truncate(30);
        Ok(patterns)
    }

    /// Find entity clusters: groups of entities connected by the same predicate type.
    /// Useful for discovering thematic clusters in sparse graphs.
    pub fn find_entity_clusters(&self) -> Result<Vec<Pattern>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;
        // Group entities by predicate they participate in
        let mut pred_entities: HashMap<String, HashSet<i64>> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id) {
                pred_entities
                    .entry(r.predicate.clone())
                    .or_default()
                    .insert(r.subject_id);
            }
            if meaningful.contains(&r.object_id) {
                pred_entities
                    .entry(r.predicate.clone())
                    .or_default()
                    .insert(r.object_id);
            }
        }
        let mut patterns = Vec::new();
        for (pred, entities) in &pred_entities {
            if entities.len() >= 3 && !is_generic_predicate(pred) {
                let names: Vec<String> = entities
                    .iter()
                    .take(5)
                    .filter_map(|&id| self.entity_name(id).ok())
                    .collect();
                patterns.push(Pattern {
                    id: 0,
                    pattern_type: PatternType::CoOccurrence,
                    entities_involved: names,
                    frequency: entities.len() as i64,
                    last_seen: now_str(),
                    description: format!(
                        "Predicate '{}' connects {} entities — thematic cluster",
                        pred,
                        entities.len()
                    ),
                });
            }
        }
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        Ok(patterns)
    }

    /// Find knowledge frontiers: entity types that appear in many entities
    /// but have disproportionately few relations (growth potential).
    pub fn find_knowledge_frontiers(&self) -> Result<Vec<(String, usize, f64, String)>> {
        let density = crate::graph::knowledge_density(self.brain)?;
        let mut frontiers: Vec<(String, usize, f64, String)> = Vec::new();
        for (etype, (count, avg_rels)) in &density {
            if is_noise_type(etype) || *count < 2 {
                continue;
            }
            // Low density = high frontier potential
            if *avg_rels < 2.0 {
                let reason = if *avg_rels < 0.5 {
                    format!(
                        "CRITICAL: {} '{}' entities, only {:.1} avg relations — near-zero connectivity",
                        count, etype, avg_rels
                    )
                } else {
                    format!(
                        "{} '{}' entities with {:.1} avg relations — underexplored",
                        count, etype, avg_rels
                    )
                };
                frontiers.push((etype.clone(), *count, *avg_rels, reason));
            }
        }
        frontiers.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(frontiers)
    }

    /// Find frequent subgraph patterns — recurring predicate patterns around entities.
    pub fn find_frequent_subgraphs(&self, min_freq: usize) -> Result<Vec<Pattern>> {
        let relations = self.brain.all_relations()?;
        // Count predicate pair motifs: (pred_out, pred_in) around a common entity
        let mut motifs: HashMap<(String, String), Vec<(i64, i64, i64)>> = HashMap::new();
        let mut outgoing: HashMap<i64, Vec<(String, i64)>> = HashMap::new();
        for r in &relations {
            outgoing
                .entry(r.subject_id)
                .or_default()
                .push((r.predicate.clone(), r.object_id));
        }
        // For each entity with 2+ outgoing edges, pair predicates
        for (eid, edges) in &outgoing {
            for i in 0..edges.len() {
                for j in (i + 1)..edges.len() {
                    let key = if edges[i].0 <= edges[j].0 {
                        (edges[i].0.clone(), edges[j].0.clone())
                    } else {
                        (edges[j].0.clone(), edges[i].0.clone())
                    };
                    motifs
                        .entry(key)
                        .or_default()
                        .push((*eid, edges[i].1, edges[j].1));
                }
            }
        }
        let mut patterns = Vec::new();
        for ((p1, p2), instances) in &motifs {
            if instances.len() >= min_freq && !is_generic_predicate(p1) && !is_generic_predicate(p2)
            {
                let example_names: Vec<String> = instances
                    .iter()
                    .take(3)
                    .filter_map(|(e, _, _)| self.entity_name(*e).ok())
                    .collect();
                patterns.push(Pattern {
                    id: 0,
                    pattern_type: PatternType::FrequentSubgraph,
                    entities_involved: example_names,
                    frequency: instances.len() as i64,
                    last_seen: now_str(),
                    description: format!(
                        "Motif ({}, {}) appears {} times",
                        p1,
                        p2,
                        instances.len()
                    ),
                });
            }
        }
        Ok(patterns)
    }

    /// Find temporal patterns — predicates that often appear in sequence by learned_at.
    pub fn find_temporal_patterns(&self, min_freq: usize) -> Result<Vec<Pattern>> {
        let relations = self.brain.all_relations()?;
        // Group by subject, sort by learned_at, look at consecutive predicate pairs
        let mut by_subject: HashMap<i64, Vec<(String, String)>> = HashMap::new();
        for r in &relations {
            by_subject
                .entry(r.subject_id)
                .or_default()
                .push((r.predicate.clone(), r.learned_at.to_string()));
        }
        let mut seq_count: HashMap<(String, String), i64> = HashMap::new();
        for (_sid, mut preds) in by_subject {
            preds.sort_by(|a, b| a.1.cmp(&b.1));
            for w in preds.windows(2) {
                *seq_count
                    .entry((w[0].0.clone(), w[1].0.clone()))
                    .or_insert(0) += 1;
            }
        }
        let mut patterns = Vec::new();
        for ((p1, p2), count) in &seq_count {
            if *count >= min_freq as i64 && !is_generic_predicate(p1) && !is_generic_predicate(p2) {
                patterns.push(Pattern {
                    id: 0,
                    pattern_type: PatternType::TemporalSequence,
                    entities_involved: vec![p1.clone(), p2.clone()],
                    frequency: *count,
                    last_seen: now_str(),
                    description: format!(
                        "Predicate '{}' often followed by '{}' ({} times)",
                        p1, p2, count
                    ),
                });
            }
        }
        Ok(patterns)
    }

    /// Find predicate chain patterns: A→p1→B→p2→C suggests a compound relation A→(p1∘p2)→C.
    /// E.g., "Einstein" →born_in→ "Germany" →located_in→ "Europe" suggests "Einstein" associated_with "Europe".
    /// Returns patterns with the chain predicates and involved entities.
    pub fn find_predicate_chains(&self, min_freq: usize) -> Result<Vec<Pattern>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Build outgoing edges: entity_id → [(predicate, target_id)]
        let mut outgoing: HashMap<i64, Vec<(String, i64)>> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id)
                && meaningful.contains(&r.object_id)
                && !is_generic_predicate(&r.predicate)
            {
                outgoing
                    .entry(r.subject_id)
                    .or_default()
                    .push((r.predicate.clone(), r.object_id));
            }
        }

        // Count predicate chain motifs: (p1, p2) → frequency
        let mut chain_count: HashMap<(String, String), Vec<(i64, i64, i64)>> = HashMap::new();
        for (&a, edges_a) in &outgoing {
            for (p1, b) in edges_a {
                if let Some(edges_b) = outgoing.get(b) {
                    for (p2, c) in edges_b {
                        if *c != a && p1 != p2 {
                            let key = (p1.clone(), p2.clone());
                            chain_count.entry(key).or_default().push((a, *b, *c));
                        }
                    }
                }
            }
        }

        let mut patterns = Vec::new();
        for ((p1, p2), instances) in &chain_count {
            if instances.len() >= min_freq {
                let example_names: Vec<String> = instances
                    .iter()
                    .take(3)
                    .filter_map(|(a, b, c)| {
                        let an = self.entity_name(*a).ok()?;
                        let bn = self.entity_name(*b).ok()?;
                        let cn = self.entity_name(*c).ok()?;
                        Some(format!("{}→{}→{}", an, bn, cn))
                    })
                    .collect();
                patterns.push(Pattern {
                    id: 0,
                    pattern_type: PatternType::FrequentSubgraph,
                    entities_involved: example_names,
                    frequency: instances.len() as i64,
                    last_seen: now_str(),
                    description: format!(
                        "Chain pattern ({} → {}) appears {} times — transitive relation candidate",
                        p1,
                        p2,
                        instances.len()
                    ),
                });
            }
        }
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        patterns.truncate(30);
        Ok(patterns)
    }

    /// Generate hypotheses from predicate chains: if A→p1→B→p2→C occurs frequently,
    /// propose that A is transitively related to C.
    pub fn generate_hypotheses_from_chains(&self) -> Result<Vec<Hypothesis>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        let mut outgoing: HashMap<i64, Vec<(String, i64)>> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id)
                && meaningful.contains(&r.object_id)
                && !is_generic_predicate(&r.predicate)
            {
                outgoing
                    .entry(r.subject_id)
                    .or_default()
                    .push((r.predicate.clone(), r.object_id));
            }
        }

        // Build direct connection set
        let mut connected: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            connected.insert(key);
        }

        // Transitive predicate pairs that suggest compound relations
        let transitive_chains: &[(&str, &str, &str)] = &[
            ("born_in", "located_in", "associated_with"),
            ("located_in", "located_in", "located_in"),
            ("part_of", "part_of", "part_of"),
            ("member_of", "part_of", "associated_with"),
            ("works_at", "located_in", "associated_with"),
            ("founded_by", "born_in", "associated_with"),
            ("created_by", "member_of", "associated_with"),
            ("headquartered_in", "located_in", "operates_in"),
        ];

        let mut hypotheses = Vec::new();
        for (&a, edges_a) in &outgoing {
            for (p1, b) in edges_a {
                if let Some(edges_b) = outgoing.get(b) {
                    for (p2, c) in edges_b {
                        if *c == a {
                            continue;
                        }
                        let key = if a < *c { (a, *c) } else { (*c, a) };
                        if connected.contains(&key) {
                            continue;
                        }
                        // Check if this chain matches a known transitive pattern
                        for &(cp1, cp2, new_pred) in transitive_chains {
                            if p1 == cp1 && p2 == cp2 {
                                let a_name = self.entity_name(a)?;
                                let b_name = self.entity_name(*b)?;
                                let c_name = self.entity_name(*c)?;
                                hypotheses.push(Hypothesis {
                                    id: 0,
                                    subject: a_name.clone(),
                                    predicate: new_pred.to_string(),
                                    object: c_name.clone(),
                                    confidence: 0.5,
                                    evidence_for: vec![format!(
                                        "Chain: {} →{}→ {} →{}→ {}",
                                        a_name, p1, b_name, p2, c_name
                                    )],
                                    evidence_against: vec![],
                                    reasoning_chain: vec![
                                        format!("{} {} {}", a_name, p1, b_name),
                                        format!("{} {} {}", b_name, p2, c_name),
                                        format!(
                                            "Transitive chain ({} → {}) implies {} {} {}",
                                            p1, p2, a_name, new_pred, c_name
                                        ),
                                    ],
                                    status: HypothesisStatus::Proposed,
                                    discovered_at: now_str(),
                                    pattern_source: "predicate_chain".to_string(),
                                });
                                if hypotheses.len() >= 50 {
                                    return Ok(hypotheses);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(hypotheses)
    }

    /// Find statistical anomalies — predicates that are surprisingly absent for certain entities.
    pub fn find_anomalies(&self) -> Result<Vec<Pattern>> {
        // For each entity_type, collect which predicates are common (>50%)
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut type_entities: HashMap<String, Vec<i64>> = HashMap::new();
        for e in &entities {
            if e.entity_type != "source" {
                type_entities
                    .entry(e.entity_type.clone())
                    .or_default()
                    .push(e.id);
            }
        }
        let mut entity_preds: HashMap<i64, HashSet<String>> = HashMap::new();
        for r in &relations {
            entity_preds
                .entry(r.subject_id)
                .or_default()
                .insert(r.predicate.clone());
        }
        let mut patterns = Vec::new();
        for (etype, eids) in &type_entities {
            if eids.len() < 3 {
                continue;
            }
            // Count predicate prevalence
            let mut pred_count: HashMap<String, usize> = HashMap::new();
            for eid in eids {
                if let Some(preds) = entity_preds.get(eid) {
                    for p in preds {
                        *pred_count.entry(p.clone()).or_insert(0) += 1;
                    }
                }
            }
            let threshold = (eids.len() as f64 * 0.5).ceil() as usize;
            for (pred, count) in &pred_count {
                if *count >= threshold {
                    // Find entities missing this predicate
                    for eid in eids {
                        let has = entity_preds.get(eid).is_some_and(|s| s.contains(pred));
                        if !has {
                            let name = self.entity_name(*eid)?;
                            patterns.push(Pattern {
                                id: 0,
                                pattern_type: PatternType::TypeGap,
                                entities_involved: vec![name.clone(), pred.clone(), etype.clone()],
                                frequency: 1,
                                last_seen: now_str(),
                                description: format!(
                                    "Entity '{}' of type '{}' lacks predicate '{}' which {}/{} peers have",
                                    name, etype, pred, count, eids.len()
                                ),
                            });
                        }
                    }
                }
            }
        }
        Ok(patterns)
    }

    // -----------------------------------------------------------------------
    // Gap Detection
    // -----------------------------------------------------------------------

    /// Find structural holes: A→B, A→C, B→D, C→D but B↛C.
    /// Filters to meaningful entity types only.
    pub fn find_structural_holes(&self) -> Result<Vec<(String, String)>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;
        let mut adj: HashMap<i64, HashSet<i64>> = HashMap::new();
        for r in &relations {
            // Include edge if at least one endpoint is meaningful
            if meaningful.contains(&r.subject_id) || meaningful.contains(&r.object_id) {
                adj.entry(r.subject_id).or_default().insert(r.object_id);
                adj.entry(r.object_id).or_default().insert(r.subject_id);
            }
        }
        let mut holes = Vec::new();
        let mut seen: HashSet<(i64, i64)> = HashSet::new();
        for (&a, a_nb) in &adj {
            // Skip high-degree nodes to avoid O(d²) blowup
            if a_nb.len() > 50 {
                continue;
            }
            let a_list: Vec<i64> = a_nb.iter().copied().collect();
            for i in 0..a_list.len() {
                for j in (i + 1)..a_list.len() {
                    let b = a_list[i];
                    let c = a_list[j];
                    let b_connected_c = adj.get(&b).is_some_and(|s| s.contains(&c));
                    if !b_connected_c {
                        // Check if B and C share another neighbour besides A
                        let b_nb = adj.get(&b).cloned().unwrap_or_default();
                        let c_nb = adj.get(&c).cloned().unwrap_or_default();
                        let shared: usize = b_nb.intersection(&c_nb).filter(|&&x| x != a).count();
                        if shared > 0 {
                            let key = if b < c { (b, c) } else { (c, b) };
                            if seen.insert(key) {
                                let b_name = self.entity_name(b)?;
                                let c_name = self.entity_name(c)?;
                                holes.push((b_name, c_name));
                            }
                        }
                    }
                }
            }
            if holes.len() >= 200 {
                break;
            }
        }
        Ok(holes)
    }

    /// Type-based gaps: entities of type X that lack a predicate Y which most peers have.
    pub fn find_type_gaps(&self) -> Result<Vec<(String, String, String)>> {
        let patterns = self.find_anomalies()?;
        let mut gaps = Vec::new();
        for p in patterns {
            if p.pattern_type == PatternType::TypeGap && p.entities_involved.len() >= 3 {
                gaps.push((
                    p.entities_involved[0].clone(), // entity
                    p.entities_involved[1].clone(), // predicate
                    p.entities_involved[2].clone(), // type
                ));
            }
        }
        Ok(gaps)
    }

    /// Analogy detection: "A is to B as C is to ?" — entities with parallel relation structures.
    pub fn find_analogies(&self) -> Result<Vec<(String, String, String, String, String)>> {
        let relations = self.brain.all_relations()?;
        // Group relations by predicate
        let mut by_pred: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
        for r in &relations {
            by_pred
                .entry(r.predicate.clone())
                .or_default()
                .push((r.subject_id, r.object_id));
        }
        // For each entity pair sharing ≥2 predicates, look for analogy gaps
        let mut pair_preds: HashMap<(i64, i64), HashSet<String>> = HashMap::new();
        for (pred, pairs) in &by_pred {
            for &(s, o) in pairs {
                pair_preds.entry((s, o)).or_default().insert(pred.clone());
            }
        }
        let mut analogies = Vec::new();
        let pairs: Vec<((i64, i64), HashSet<String>)> = pair_preds.into_iter().collect();
        for i in 0..pairs.len() {
            for j in (i + 1)..pairs.len() {
                let ((a, b), preds_ab) = &pairs[i];
                let ((c, d), preds_cd) = &pairs[j];
                if a == c || b == d || a == d || b == c {
                    continue;
                }
                let shared: HashSet<&String> = preds_ab.intersection(preds_cd).collect();
                let only_ab: Vec<&String> = preds_ab.difference(preds_cd).collect();
                let only_cd: Vec<&String> = preds_cd.difference(preds_ab).collect();
                if shared.len() >= 2 && (!only_ab.is_empty() || !only_cd.is_empty()) {
                    let a_name = self.entity_name(*a)?;
                    let b_name = self.entity_name(*b)?;
                    let c_name = self.entity_name(*c)?;
                    let d_name = self.entity_name(*d)?;
                    let missing = if !only_ab.is_empty() {
                        format!("{} may also {} {}", c_name, only_ab[0], d_name)
                    } else {
                        format!("{} may also {} {}", a_name, only_cd[0], b_name)
                    };
                    analogies.push((a_name, b_name, c_name, d_name, missing));
                    if analogies.len() >= 50 {
                        return Ok(analogies);
                    }
                }
            }
        }
        Ok(analogies)
    }

    // -----------------------------------------------------------------------
    // Hypothesis Engine
    // -----------------------------------------------------------------------

    /// Generate hypotheses from structural holes.
    pub fn generate_hypotheses_from_holes(&self) -> Result<Vec<Hypothesis>> {
        let holes = self.find_structural_holes()?;
        let mut hypotheses = Vec::new();
        for (b, c) in &holes {
            let h = Hypothesis {
                id: 0,
                subject: b.clone(),
                predicate: "related_to".to_string(),
                object: c.clone(),
                confidence: 0.4,
                evidence_for: vec![format!(
                    "Structural hole: {} and {} share common neighbours but are not directly connected",
                    b, c
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!("Found structural hole between {} and {}", b, c),
                    "Both entities share common neighbours".to_string(),
                    "Missing direct link suggests potential relation".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "structural_hole".to_string(),
            };
            hypotheses.push(h);
        }
        Ok(hypotheses)
    }

    /// Generate hypotheses from type-based gaps.
    pub fn generate_hypotheses_from_type_gaps(&self) -> Result<Vec<Hypothesis>> {
        let gaps = self.find_type_gaps()?;
        let mut hypotheses = Vec::new();
        for (entity, predicate, etype) in &gaps {
            let h = Hypothesis {
                id: 0,
                subject: entity.clone(),
                predicate: predicate.clone(),
                object: "?".to_string(),
                confidence: 0.5,
                evidence_for: vec![format!(
                    "Most entities of type '{}' have predicate '{}'",
                    etype, predicate
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!("Entity '{}' is of type '{}'", entity, etype),
                    format!("Most '{}' entities have predicate '{}'", etype, predicate),
                    format!("'{}' lacks this predicate — likely a gap", entity),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "type_gap".to_string(),
            };
            hypotheses.push(h);
        }
        Ok(hypotheses)
    }

    /// Generate hypotheses from shared-object patterns: if A→pred→X and B→pred→X,
    /// maybe A and B are related.
    /// Filters out generic predicates and caps per-entity generation to avoid hub explosion.
    pub fn generate_hypotheses_from_shared_objects(&self) -> Result<Vec<Hypothesis>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;
        // Group by (predicate, object_id) → list of subject_ids
        // Only require subjects to be meaningful (objects can be any type)
        // Skip generic predicates that create too many spurious connections
        let skip_preds: HashSet<&str> = [
            "contributed_to",
            "associated_with",
            "references",
            "related_concept",
            "relevant_to",
            "works_on",
            "related_to",
            "is",
            "has",
            "was",
        ]
        .into_iter()
        .collect();
        let mut groups: HashMap<(String, i64), Vec<i64>> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id) && !skip_preds.contains(r.predicate.as_str()) {
                groups
                    .entry((r.predicate.clone(), r.object_id))
                    .or_default()
                    .push(r.subject_id);
            }
        }
        let mut hypotheses = Vec::new();
        for ((pred, obj_id), subjects) in &groups {
            if subjects.len() < 2 || subjects.len() > 10 {
                continue;
            }
            let obj_name = self.entity_name(*obj_id)?;
            // For each pair of subjects, propose a relationship
            for i in 0..subjects.len().min(5) {
                for j in (i + 1)..subjects.len().min(5) {
                    let a = self.entity_name(subjects[i])?;
                    let b = self.entity_name(subjects[j])?;
                    hypotheses.push(Hypothesis {
                        id: 0,
                        subject: a.clone(),
                        predicate: "related_to".to_string(),
                        object: b.clone(),
                        confidence: 0.45,
                        evidence_for: vec![format!(
                            "Both {} and {} share '{}' relationship to '{}'",
                            a, b, pred, obj_name
                        )],
                        evidence_against: vec![],
                        reasoning_chain: vec![
                            format!("{} {} {}", a, pred, obj_name),
                            format!("{} {} {}", b, pred, obj_name),
                            format!("Shared object suggests {} and {} may be related", a, b),
                        ],
                        status: HypothesisStatus::Proposed,
                        discovered_at: now_str(),
                        pattern_source: "shared_object".to_string(),
                    });
                }
            }
        }
        Ok(hypotheses)
    }

    /// Generate hypotheses from source co-occurrence: entities extracted from the same
    /// source page likely have a semantic relationship.
    pub fn generate_hypotheses_from_source_co_occurrence(&self) -> Result<Vec<Hypothesis>> {
        let source_patterns = self.find_source_co_occurrences()?;
        let mut hypotheses = Vec::new();
        for p in source_patterns.iter().take(30) {
            if p.entities_involved.len() >= 2 {
                let a = &p.entities_involved[0];
                let b = &p.entities_involved[1];
                hypotheses.push(Hypothesis {
                    id: 0,
                    subject: a.clone(),
                    predicate: "related_to".to_string(),
                    object: b.clone(),
                    confidence: 0.35 + (p.frequency as f64 * 0.1).min(0.3),
                    evidence_for: vec![format!(
                        "Co-extracted from {} shared source(s)",
                        p.frequency
                    )],
                    evidence_against: vec![],
                    reasoning_chain: vec![
                        format!("{} and {} appear in the same source document(s)", a, b),
                        "Co-occurrence in source material suggests semantic relation".to_string(),
                    ],
                    status: HypothesisStatus::Proposed,
                    discovered_at: now_str(),
                    pattern_source: "source_co_occurrence".to_string(),
                });
            }
        }
        Ok(hypotheses)
    }

    /// Analyze predicate diversity — identify over-reliance on generic predicates.
    /// Returns (total_rels, generic_count, diverse_count, diversity_ratio).
    pub fn predicate_diversity(&self) -> Result<(usize, usize, usize, f64)> {
        let relations = self.brain.all_relations()?;
        let total = relations.len();
        let generic = relations
            .iter()
            .filter(|r| is_generic_predicate(&r.predicate))
            .count();
        let diverse = total - generic;
        let ratio = if total > 0 {
            diverse as f64 / total as f64
        } else {
            0.0
        };
        Ok((total, generic, diverse, ratio))
    }

    /// Generate hypotheses from analogy patterns (A:B :: C:? ).
    pub fn generate_hypotheses_from_analogies(&self) -> Result<Vec<Hypothesis>> {
        let analogies = self.find_analogies()?;
        let mut hypotheses = Vec::new();
        for (a, b, c, d, missing) in analogies.iter().take(20) {
            hypotheses.push(Hypothesis {
                id: 0,
                subject: c.clone(),
                predicate: "analogous_to".to_string(),
                object: format!("{} (via {}-{} analogy)", missing, a, b),
                confidence: 0.35,
                evidence_for: vec![format!(
                    "Analogy: {} is to {} as {} is to {} — {}",
                    a, b, c, d, missing
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!(
                        "{} and {} share relation structure with {} and {}",
                        a, b, c, d
                    ),
                    format!("Gap suggests: {}", missing),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "analogy".to_string(),
            });
        }
        Ok(hypotheses)
    }

    /// Find island entities: meaningful entities with zero relations (knowledge gaps).
    pub fn find_island_entities(&self) -> Result<Vec<(String, String)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }
        let mut islands = Vec::new();
        for e in &entities {
            if !is_noise_type(&e.entity_type) && !connected.contains(&e.id) {
                islands.push((e.name.clone(), e.entity_type.clone()));
            }
        }
        Ok(islands)
    }

    /// Check if a hypothesis contradicts known facts.
    pub fn check_contradiction(&self, hypothesis: &Hypothesis) -> Result<bool> {
        // Check if the exact opposite relation exists
        let subj = self.brain.get_entity_by_name(&hypothesis.subject)?;
        let obj = self.brain.get_entity_by_name(&hypothesis.object)?;
        if let (Some(s), Some(_o)) = (subj, obj) {
            let rels = self.brain.get_relations_for(s.id)?;
            for (sname, pred, oname, _conf) in &rels {
                // Check for contradiction predicates
                let contradicts = is_contradicting_predicate(&hypothesis.predicate, pred);
                if contradicts
                    && ((sname == &hypothesis.subject && oname == &hypothesis.object)
                        || (sname == &hypothesis.object && oname == &hypothesis.subject))
                {
                    return Ok(true);
                }
            }
            // Check singleton facts
            let facts = self.brain.get_facts_for(s.id)?;
            for f in &facts {
                if f.key == hypothesis.predicate && f.value != hypothesis.object && f.value != "?" {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Score a hypothesis based on available evidence, community co-membership,
    /// and k-core depth.
    pub fn score_hypothesis(&self, hypothesis: &mut Hypothesis) -> Result<f64> {
        let mut score = 0.5_f64;
        // Boost for more evidence_for
        score += hypothesis.evidence_for.len() as f64 * 0.1;
        // Penalty for evidence_against
        score -= hypothesis.evidence_against.len() as f64 * 0.15;
        // Check for contradiction
        if self.check_contradiction(hypothesis)? {
            score -= 0.3;
            hypothesis
                .evidence_against
                .push("Contradicts existing knowledge".to_string());
        }

        // Note: community and k-core boosting is done in batch via
        // boost_hypotheses_with_graph_structure() to avoid recomputing per-hypothesis.

        // Apply pattern weight
        let weight = self.get_pattern_weight(&hypothesis.pattern_source)?;
        score *= weight;
        score = score.clamp(0.0, 1.0);
        hypothesis.confidence = score;
        Ok(score)
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    pub fn save_hypothesis(&self, h: &Hypothesis) -> Result<i64> {
        self.brain.with_conn(|conn| {
            conn.execute(
                "INSERT INTO hypotheses (subject, predicate, object, confidence, evidence_for, evidence_against, reasoning_chain, status, discovered_at, pattern_source)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    h.subject,
                    h.predicate,
                    h.object,
                    h.confidence,
                    serde_json::to_string(&h.evidence_for).unwrap_or_default(),
                    serde_json::to_string(&h.evidence_against).unwrap_or_default(),
                    serde_json::to_string(&h.reasoning_chain).unwrap_or_default(),
                    h.status.as_str(),
                    h.discovered_at,
                    h.pattern_source,
                ],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn save_pattern(&self, p: &Pattern) -> Result<i64> {
        self.brain.with_conn(|conn| {
            conn.execute(
                "INSERT INTO patterns (pattern_type, entities_involved, frequency, last_seen, description)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    p.pattern_type.as_str(),
                    serde_json::to_string(&p.entities_involved).unwrap_or_default(),
                    p.frequency,
                    p.last_seen,
                    p.description,
                ],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn get_hypothesis(&self, id: i64) -> Result<Option<Hypothesis>> {
        self.brain.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, subject, predicate, object, confidence, evidence_for, evidence_against, reasoning_chain, status, discovered_at, pattern_source
                 FROM hypotheses WHERE id = ?1",
            )?;
            let mut rows = stmt.query_map(params![id], |row| {
                Ok(parse_hypothesis_row(row))
            })?;
            match rows.next() {
                Some(Ok(h)) => Ok(Some(h)),
                _ => Ok(None),
            }
        })
    }

    pub fn list_hypotheses(&self, status: Option<HypothesisStatus>) -> Result<Vec<Hypothesis>> {
        self.brain.with_conn(|conn| {
            let (sql, param): (String, Option<String>) = match status {
                Some(s) => (
                    "SELECT id, subject, predicate, object, confidence, evidence_for, evidence_against, reasoning_chain, status, discovered_at, pattern_source FROM hypotheses WHERE status = ?1 ORDER BY confidence DESC".to_string(),
                    Some(s.as_str().to_string()),
                ),
                None => (
                    "SELECT id, subject, predicate, object, confidence, evidence_for, evidence_against, reasoning_chain, status, discovered_at, pattern_source FROM hypotheses ORDER BY confidence DESC".to_string(),
                    None,
                ),
            };
            let mut stmt = conn.prepare(&sql)?;
            let rows = if let Some(ref p) = param {
                stmt.query_map(params![p], |row| Ok(parse_hypothesis_row(row)))?
                    .collect::<Result<Vec<_>>>()?
            } else {
                stmt.query_map([], |row| Ok(parse_hypothesis_row(row)))?
                    .collect::<Result<Vec<_>>>()?
            };
            Ok(rows)
        })
    }

    pub fn update_hypothesis_status(&self, id: i64, status: HypothesisStatus) -> Result<()> {
        self.brain.with_conn(|conn| {
            conn.execute(
                "UPDATE hypotheses SET status = ?1 WHERE id = ?2",
                params![status.as_str(), id],
            )?;
            Ok(())
        })
    }

    pub fn save_discovery(&self, hypothesis_id: i64, evidence_sources: &[String]) -> Result<i64> {
        self.brain.with_conn(|conn| {
            conn.execute(
                "INSERT INTO discoveries (hypothesis_id, confirmed_at, evidence_sources) VALUES (?1, ?2, ?3)",
                params![
                    hypothesis_id,
                    now_str(),
                    serde_json::to_string(evidence_sources).unwrap_or_default(),
                ],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate a hypothesis against existing knowledge.
    pub fn validate_hypothesis(&self, h: &mut Hypothesis) -> Result<()> {
        // Search for supporting/contradicting evidence in the graph
        let subj = self.brain.search_entities(&h.subject)?;
        let _obj = if h.object != "?" {
            self.brain.search_entities(&h.object)?
        } else {
            vec![]
        };

        // Check if the relation already exists (confirms the hypothesis)
        for s in &subj {
            let rels = self.brain.get_relations_for(s.id)?;
            for (sname, pred, oname, conf) in &rels {
                if (pred == &h.predicate || predicates_similar(pred, &h.predicate))
                    && (h.object == "?" || oname == &h.object || sname == &h.object)
                {
                    h.evidence_for.push(format!(
                        "Found relation: {} {} {} (confidence: {:.2})",
                        sname, pred, oname, conf
                    ));
                    h.confidence = (h.confidence + 0.2).min(1.0);
                }
            }
        }

        // Check for contradictions
        if self.check_contradiction(h)? {
            h.evidence_against
                .push("Contradicts existing knowledge".to_string());
            h.confidence = (h.confidence - 0.3).max(0.0);
        }

        // Update status based on confidence
        if h.confidence >= 0.8 {
            h.status = HypothesisStatus::Confirmed;
        } else if h.confidence <= 0.1 {
            h.status = HypothesisStatus::Rejected;
        } else {
            h.status = HypothesisStatus::Testing;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Meta-learning
    // -----------------------------------------------------------------------

    pub fn get_pattern_weight(&self, pattern_type: &str) -> Result<f64> {
        self.brain.with_conn(|conn| {
            let result: std::result::Result<f64, _> = conn.query_row(
                "SELECT weight FROM pattern_weights WHERE pattern_type = ?1",
                params![pattern_type],
                |row| row.get(0),
            );
            Ok(result.unwrap_or(1.0))
        })
    }

    pub fn record_outcome(&self, pattern_type: &str, confirmed: bool) -> Result<()> {
        self.brain.with_conn(|conn| {
            conn.execute(
                "INSERT INTO pattern_weights (pattern_type, confirmations, rejections, weight)
                 VALUES (?1, ?2, ?3, 1.0)
                 ON CONFLICT(pattern_type) DO UPDATE SET
                    confirmations = confirmations + ?2,
                    rejections = rejections + ?3,
                    weight = CAST(confirmations + ?2 AS REAL) / MAX(1, confirmations + ?2 + rejections + ?3)",
                params![
                    pattern_type,
                    if confirmed { 1 } else { 0 },
                    if confirmed { 0 } else { 1 },
                ],
            )?;
            Ok(())
        })
    }

    pub fn get_pattern_weights(&self) -> Result<Vec<PatternWeight>> {
        self.brain.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT pattern_type, confirmations, rejections, weight FROM pattern_weights ORDER BY weight DESC",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(PatternWeight {
                    pattern_type: row.get(0)?,
                    confirmations: row.get(1)?,
                    rejections: row.get(2)?,
                    weight: row.get(3)?,
                })
            })?;
            rows.collect()
        })
    }

    /// Get calibrated initial confidence for a pattern source.
    /// Uses historical confirmation rate as Bayesian prior, with smoothing.
    /// New/unknown sources get 0.5 (neutral), proven sources get boosted,
    /// unreliable sources get suppressed.
    pub fn calibrated_confidence(&self, pattern_source: &str, base: f64) -> Result<f64> {
        let weight = self.get_pattern_weight(pattern_source)?;
        // Blend base confidence with historical weight using Bayesian update.
        // Weight of 1.0 = no data (neutral), so don't adjust.
        // Weight close to 0.0 = unreliable source, suppress.
        // Weight close to 1.0 = no data or very good.
        let weights = self.get_pattern_weights()?;
        let source_data = weights.iter().find(|w| w.pattern_type == pattern_source);
        let total_observations = source_data
            .map(|w| w.confirmations + w.rejections)
            .unwrap_or(0);

        if total_observations < 3 {
            // Too few data points — use base confidence
            return Ok(base);
        }
        // Bayesian blend: shift base toward observed weight
        let alpha = (total_observations as f64 / (total_observations as f64 + 10.0)).min(0.7);
        let calibrated = base * (1.0 - alpha) + weight * alpha;
        Ok(calibrated.clamp(0.05, 0.95))
    }

    /// Deduplicate hypotheses by entity pair: if multiple hypotheses exist about
    /// the same (subject, object) pair, keep only the highest-confidence one.
    /// Returns count of hypotheses removed.
    pub fn dedup_hypotheses_by_pair(&self) -> Result<usize> {
        let hyps = self.list_hypotheses(None)?;
        let mut by_pair: HashMap<(String, String), Vec<(i64, f64, String)>> = HashMap::new();

        for h in &hyps {
            // Normalize pair order
            let pair = if h.subject <= h.object {
                (h.subject.clone(), h.object.clone())
            } else {
                (h.object.clone(), h.subject.clone())
            };
            by_pair.entry(pair).or_default().push((
                h.id,
                h.confidence,
                h.status.as_str().to_string(),
            ));
        }

        let mut removed = 0usize;
        for (_pair, mut entries) in by_pair {
            if entries.len() < 2 {
                continue;
            }
            // Sort: confirmed first, then by confidence descending
            entries.sort_by(|a, b| {
                let status_ord = |s: &str| -> i32 {
                    match s {
                        "confirmed" => 0,
                        "testing" => 1,
                        "proposed" => 2,
                        _ => 3,
                    }
                };
                status_ord(&a.2)
                    .cmp(&status_ord(&b.2))
                    .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            // Keep the best, remove the rest
            for &(id, _, _) in &entries[1..] {
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM hypotheses WHERE id = ?1", params![id])?;
                    Ok(())
                })?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Track discovery velocity: count of new patterns, hypotheses, and confirmations
    /// per discovery run. Persists to a tracking table for trend analysis.
    pub fn track_discovery_velocity(
        &self,
        patterns: usize,
        hypotheses: usize,
        confirmed: usize,
        rejected: usize,
    ) -> Result<()> {
        self.brain.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS discovery_velocity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT NOT NULL,
                    patterns_found INTEGER NOT NULL,
                    hypotheses_generated INTEGER NOT NULL,
                    confirmed INTEGER NOT NULL,
                    rejected INTEGER NOT NULL
                );"
            )?;
            conn.execute(
                "INSERT INTO discovery_velocity (run_at, patterns_found, hypotheses_generated, confirmed, rejected) VALUES (datetime('now'), ?1, ?2, ?3, ?4)",
                params![patterns as i64, hypotheses as i64, confirmed as i64, rejected as i64],
            )?;
            Ok(())
        })
    }

    /// Get discovery velocity trend (recent runs).
    pub fn get_velocity_trend(&self, limit: usize) -> Result<Vec<(String, i64, i64, i64, i64)>> {
        self.brain.with_conn(|conn| {
            let exists: bool = conn.query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='discovery_velocity'",
                [],
                |row| row.get(0),
            )?;
            if !exists {
                return Ok(vec![]);
            }
            let mut stmt = conn.prepare(
                "SELECT run_at, patterns_found, hypotheses_generated, confirmed, rejected FROM discovery_velocity ORDER BY id DESC LIMIT ?1"
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?, row.get::<_, i64>(4)?))
            })?;
            rows.collect::<Result<Vec<_>>>()
        })
    }

    /// Discovery score: how many confirmed hypotheses involve each entity.
    pub fn discovery_scores(&self) -> Result<HashMap<String, i64>> {
        let confirmed = self.list_hypotheses(Some(HypothesisStatus::Confirmed))?;
        let mut scores: HashMap<String, i64> = HashMap::new();
        for h in &confirmed {
            *scores.entry(h.subject.clone()).or_insert(0) += 1;
            if h.object != "?" {
                *scores.entry(h.object.clone()).or_insert(0) += 1;
            }
        }
        Ok(scores)
    }

    // -----------------------------------------------------------------------
    // Full Discovery Pipeline
    // -----------------------------------------------------------------------

    /// Run the full discovery pipeline: patterns → gaps → hypotheses → validation.
    pub fn discover(&self) -> Result<DiscoveryReport> {
        let mut all_patterns = Vec::new();
        let mut all_hypotheses = Vec::new();

        // 1. Pattern discovery (adaptive thresholds based on graph density)
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;
        let meaningful_rels: usize = relations
            .iter()
            .filter(|r| meaningful.contains(&r.subject_id) || meaningful.contains(&r.object_id))
            .count();
        // Sparse graphs need lower thresholds
        let co_threshold = if meaningful_rels < 50 { 1 } else { 2 };
        let subgraph_threshold = if meaningful_rels < 100 { 1 } else { 2 };

        let co_occ = self.find_co_occurrences(co_threshold)?;
        all_patterns.extend(co_occ);

        let source_co = self.find_source_co_occurrences()?;
        all_patterns.extend(source_co);

        let clusters = self.find_entity_clusters()?;
        all_patterns.extend(clusters);

        let subgraphs = self.find_frequent_subgraphs(subgraph_threshold)?;
        all_patterns.extend(subgraphs);

        let temporal = self.find_temporal_patterns(2)?;
        all_patterns.extend(temporal);

        let anomalies = self.find_anomalies()?;
        all_patterns.extend(anomalies);

        let pmi_patterns = self.find_pmi_co_occurrences(1.0)?;
        all_patterns.extend(pmi_patterns);

        let chains = self.find_predicate_chains(2)?;
        all_patterns.extend(chains);

        // Deduplicate and save patterns
        self.dedup_patterns(&mut all_patterns);
        for p in &all_patterns {
            let _ = self.save_pattern(p);
        }

        // 2. Gap detection & hypothesis generation (adaptive: skip low-weight strategies)
        let hole_weight = self.get_pattern_weight("structural_hole")?;
        if hole_weight >= 0.05 {
            let hole_hyps = self.generate_hypotheses_from_holes()?;
            all_hypotheses.extend(hole_hyps);
        }

        let gap_hyps = self.generate_hypotheses_from_type_gaps()?;
        all_hypotheses.extend(gap_hyps);

        let shared_weight = self.get_pattern_weight("shared_object")?;
        if shared_weight >= 0.05 {
            let shared_hyps = self.generate_hypotheses_from_shared_objects()?;
            all_hypotheses.extend(shared_hyps);
        }

        let source_weight = self.get_pattern_weight("source_co_occurrence")?;
        if source_weight >= 0.05 {
            let source_hyps = self.generate_hypotheses_from_source_co_occurrence()?;
            all_hypotheses.extend(source_hyps);
        } else {
            // Strategy has proven unreliable — skip but note it
            all_patterns.push(Pattern {
                id: 0,
                pattern_type: PatternType::CoOccurrence,
                entities_involved: vec![],
                frequency: 0,
                last_seen: now_str(),
                description: format!(
                    "source_co_occurrence strategy skipped (weight {:.3} < 0.05)",
                    source_weight
                ),
            });
        }

        // 2b. Analogy-based hypotheses
        let analogy_hyps = self.generate_hypotheses_from_analogies()?;
        all_hypotheses.extend(analogy_hyps);

        // 2c. Hub-spoke hypotheses: hubs should connect to nearby same-type entities
        let hub_weight = self.get_pattern_weight("hub_spoke")?;
        if hub_weight >= 0.10 {
            let hub_hyps = self.generate_hypotheses_from_hubs()?;
            all_hypotheses.extend(hub_hyps);
        }

        // 2d. Community bridge hypotheses
        let bridge_hyps = self.generate_hypotheses_from_community_bridges()?;
        all_hypotheses.extend(bridge_hyps);

        // 2e. Predicate chain hypotheses
        let chain_weight = self.get_pattern_weight("predicate_chain")?;
        if chain_weight >= 0.05 {
            let chain_hyps = self.generate_hypotheses_from_chains()?;
            all_hypotheses.extend(chain_hyps);
        }

        // 2f. Near-miss connection hypotheses (entities with many indirect paths but no direct edge)
        let near_miss_weight = self.get_pattern_weight("near_miss")?;
        if near_miss_weight >= 0.05 {
            let near_miss_hyps = self.generate_hypotheses_from_near_misses()?;
            all_hypotheses.extend(near_miss_hyps);
        }

        // 2g. Adamic-Adar link prediction (topology-based)
        let aa_weight = self.get_pattern_weight("adamic_adar")?;
        if aa_weight >= 0.05 {
            let aa_hyps = self.generate_hypotheses_from_adamic_adar()?;
            all_hypotheses.extend(aa_hyps);
        }

        // 2h. Resource Allocation link prediction (better for sparse graphs)
        let ra_weight = self.get_pattern_weight("resource_allocation")?;
        if ra_weight >= 0.05 {
            let ra_hyps = self.generate_hypotheses_from_resource_allocation()?;
            all_hypotheses.extend(ra_hyps);
        }

        // 2i. Type-aware link prediction (boosts same-type entity pairs)
        let ta_weight = self.get_pattern_weight("type_affinity")?;
        if ta_weight >= 0.05 {
            let ta_hyps = self.generate_hypotheses_from_type_affinity()?;
            all_hypotheses.extend(ta_hyps);
        }

        // 2j. Neighborhood overlap link prediction (robust to degree imbalance)
        let no_weight = self.get_pattern_weight("neighborhood_overlap")?;
        if no_weight >= 0.05 {
            let no_hyps = self.generate_hypotheses_from_neighborhood_overlap()?;
            all_hypotheses.extend(no_hyps);
        }

        // 2k. Triadic closure (open triads with ≥2 mutual neighbors)
        let tc_weight = self.get_pattern_weight("triadic_closure")?;
        if tc_weight >= 0.05 {
            let tc_hyps = self.generate_hypotheses_from_triadic_closure()?;
            all_hypotheses.extend(tc_hyps);
        }

        // 2l. Semantic fingerprint similarity (entities sharing same predicate-object patterns)
        let sf_weight = self.get_pattern_weight("semantic_fingerprint")?;
        if sf_weight >= 0.05 {
            let sf_hyps = self.generate_hypotheses_from_semantic_similarity()?;
            all_hypotheses.extend(sf_hyps);
        }

        // 3. Island entities as gaps
        let islands = self.find_island_entities()?;

        let gaps_detected = all_hypotheses.len() + islands.len();

        // 4. Deduplicate hypotheses against existing DB, filter noise, then score and validate
        all_hypotheses.retain(|h| {
            // Skip hypotheses involving noise entities
            if is_noise_name(&h.subject) || (h.object != "?" && is_noise_name(&h.object)) {
                return false;
            }
            // Skip hypotheses with very long entity names (likely sentence fragments)
            if h.subject.len() > 60 || h.object.len() > 60 {
                return false;
            }
            // Skip substring hypotheses — one name contains the other (merge candidates, not discoveries)
            if h.object != "?" {
                let sl = h.subject.to_lowercase();
                let ol = h.object.to_lowercase();
                if sl.contains(&ol) || ol.contains(&sl) {
                    return false;
                }
            }
            // Skip single-word generic entities as hypothesis subjects
            if !h.subject.contains(' ') && h.subject.len() < 10 {
                let lower = h.subject.to_lowercase();
                let generics = [
                    "church",
                    "city",
                    "court",
                    "sea",
                    "bay",
                    "lake",
                    "river",
                    "bridge",
                    "federal",
                    "state",
                    "power",
                    "steam",
                    "monster",
                    "alice",
                    "grace",
                    "forest",
                    "desert",
                    "county",
                    "district",
                    "island",
                    "port",
                    "cape",
                    "berkeley",
                    "paris",
                    "london",
                    "berlin",
                    "vienna",
                    "zurich",
                    "monthly",
                    "reform",
                    "precision",
                    "regime",
                    "ancient",
                ];
                if generics.contains(&lower.as_str()) {
                    return false;
                }
            }
            // Skip hypotheses where both entities share the same first word (likely noise variants)
            // e.g., "French Army" ↔ "French Revolution" — these share a prefix, not a discovery
            if h.object != "?" {
                let s_first = h.subject.split_whitespace().next().unwrap_or("");
                let o_first = h.object.split_whitespace().next().unwrap_or("");
                if !s_first.is_empty() && s_first == o_first {
                    return false;
                }
            }
            // Skip hypotheses where both entities share the same last word (categorical, not insightful)
            // e.g., "Caroline Islands" ↔ "Faroe Islands" — same-suffix entities cluster trivially
            if h.object != "?" {
                let s_last = h.subject.split_whitespace().last().unwrap_or("");
                let o_last = h.object.split_whitespace().last().unwrap_or("");
                if !s_last.is_empty() && s_last.len() > 3 && s_last == o_last {
                    return false;
                }
            }
            !self
                .hypothesis_exists(&h.subject, &h.predicate, &h.object)
                .unwrap_or(true)
        });
        for h in all_hypotheses.iter_mut().take(200) {
            self.score_hypothesis(h)?;
            self.validate_hypothesis(h)?;
        }
        // Batch-boost using graph structure (communities + k-cores) — computed once
        let boost_slice = if all_hypotheses.len() > 200 {
            &mut all_hypotheses[..200]
        } else {
            &mut all_hypotheses[..]
        };
        let _ = self.boost_hypotheses_with_graph_structure(boost_slice);
        for h in all_hypotheses.iter().take(200) {
            let _ = self.save_hypothesis(h);
        }

        let confirmed = all_hypotheses
            .iter()
            .filter(|h| h.status == HypothesisStatus::Confirmed)
            .count();
        let rejected = all_hypotheses
            .iter()
            .filter(|h| h.status == HypothesisStatus::Rejected)
            .count();

        // Cross-domain gap analysis
        let cross_gaps = self.find_cross_domain_gaps().unwrap_or_default();
        let frontiers = self.find_knowledge_frontiers().unwrap_or_default();
        let bridges = self.find_bridge_entities().unwrap_or_default();
        let island_clusters = self.cluster_islands_for_crawl().unwrap_or_default();

        // Apply calibrated confidence to all hypotheses before scoring
        for h in all_hypotheses.iter_mut() {
            if let Ok(cal) = self.calibrated_confidence(&h.pattern_source, h.confidence) {
                h.confidence = cal;
            }
        }

        // Decay old hypotheses (meta-learning)
        let decayed = self.decay_old_hypotheses(7).unwrap_or(0);

        // Prune stale low-confidence hypotheses (>14 days old)
        let pruned = self.prune_stale_hypotheses(14).unwrap_or(0);

        // Auto-resolve existing testing hypotheses
        let (auto_confirmed, auto_rejected) = self.auto_resolve_hypotheses().unwrap_or((0, 0));

        // Find connectable islands
        let connectable = self.find_connectable_islands().unwrap_or_default();

        // Prioritize gaps
        let gap_priorities = self.prioritize_gaps().unwrap_or_default();

        // Predicate diversity analysis
        let (total_rels, _generic_rels, diverse_rels, div_ratio) =
            self.predicate_diversity().unwrap_or((0, 0, 0, 0.0));

        // Exact-name deduplication (catches case-only duplicates)
        let exact_deduped = self.dedup_exact_name_matches().unwrap_or(0);

        // Reverse containment island merge (island "Euler" → "Leonhard Euler")
        let reverse_merged = self.reverse_containment_island_merge().unwrap_or(0);

        // Bulk reject stale testing hypotheses (>5 days, confidence < 0.5)
        let bulk_rejected = self.bulk_reject_stale_testing(5, 0.5).unwrap_or(0);

        // Clean up fragment hypotheses (single-word subject+object = NLP noise, not discoveries)
        let fragment_cleaned = self.cleanup_fragment_hypotheses().unwrap_or(0);

        // Fuzzy duplicate detection + auto-merge high-confidence dupes
        let fuzzy_dupes = self.find_fuzzy_duplicates().unwrap_or_default();
        let merge_candidates = fuzzy_dupes.iter().filter(|d| d.3 == "merge").count();
        let auto_merged = self.auto_merge_duplicates(&fuzzy_dupes).unwrap_or(0);

        // Topic coverage
        let topic_coverage = self.topic_coverage_analysis().unwrap_or_default();
        let sparse_topics = topic_coverage.iter().filter(|t| t.3 < 0.01).count();

        // Name subsumption detection (abbreviated entity forms)
        let subsumptions = self.find_name_subsumptions().unwrap_or_default();

        // Fix misclassified entity types
        let types_fixed = self.fix_entity_types().unwrap_or(0);

        // Infer entity types from neighborhood context
        let types_inferred = self.infer_types_from_neighborhood().unwrap_or(0);

        // Purge noise entities (cleanup)
        let purged = self.purge_noise_entities().unwrap_or(0);

        // Bulk quality cleanup (aggressive isolated noise removal)
        let bulk_cleaned = self.bulk_quality_cleanup().unwrap_or(0);

        // Deep island cleanup (remove low-confidence isolated entities without facts)
        let deep_cleaned = self.deep_island_cleanup(0.6).unwrap_or(0);

        // Predicate normalization (reduce "is" overuse)
        let normalized = self.normalize_predicates().unwrap_or(0);

        // Island reconnection
        let reconnected = self.reconnect_islands().unwrap_or(0);

        // Fact-based relation inference
        let fact_inferred = self.infer_relations_from_facts().unwrap_or(0);

        // High-confidence prefix variant merge (cross-type, 3x degree threshold)
        // Run BEFORE compound decomposition to avoid degree inflation
        let hc_prefix_merged = self.merge_high_confidence_prefix_variants().unwrap_or(0);

        // Entity name cross-referencing
        let name_crossrefs = self.crossref_entity_names().unwrap_or(0);

        // Compound entity decomposition (break "Ada Lovelace Building" → named_after → "Ada Lovelace")
        let (compound_rels, compound_merged) = self.decompose_compound_entities().unwrap_or((0, 0));

        // Split concatenated entity names ("Caucasus Crimea Balkans" → components)
        let (concat_rels, concat_cleaned) = self.split_concatenated_entities().unwrap_or((0, 0));

        // Prefix island consolidation (merge "Euler" island → "Leonhard Euler" connected)
        let prefix_merged = self.consolidate_prefix_islands().unwrap_or(0);

        // Suffix-strip island merge ("Ada Lovelace WIRED" → "Ada Lovelace")
        let suffix_merged = self.suffix_strip_island_merge().unwrap_or(0);

        // Word-overlap island merging (aggressive: merge "Marie Curie Avenue" → "Marie Curie")
        let word_merged = self.word_overlap_island_merge(0.6).unwrap_or(0);

        // Fragment island purging (remove single-word fragments like "Lovelace", "Hopper")
        let fragment_purged = self.purge_fragment_islands().unwrap_or(0);

        // Prefix-strip island merge ("Christy Grace Hopper" → "Grace Hopper" by stripping leading words)
        let prefix_strip_merged = self.prefix_strip_island_merge().unwrap_or(0);

        // Name variant merge: merge titled/suffixed variants into canonical forms
        // ("Professor Claude Shannon" → "Claude Shannon", "Claude Shannon Time" → "Claude Shannon")
        let name_variants_merged = self.merge_name_variants().unwrap_or(0);

        // Auto-consolidation: merge high-scoring entity pairs from consolidation analysis
        let auto_consolidated = self.auto_consolidate_entities(0.65).unwrap_or(0);

        // Dissolve single-word name fragment hubs ("Charles" → "Charles Babbage", etc.)
        let fragments_dissolved = self.dissolve_name_fragment_hubs().unwrap_or(0);

        // Multi-pass convergence: repeat merge strategies until no more progress
        let mut convergence_merges = 0usize;
        for _pass in 0..3 {
            let mut pass_merges = 0usize;
            pass_merges += self.dedup_exact_name_matches().unwrap_or(0);
            pass_merges += self.suffix_strip_island_merge().unwrap_or(0);
            pass_merges += self.consolidate_prefix_islands().unwrap_or(0);
            pass_merges += self.word_overlap_island_merge(0.6).unwrap_or(0);
            pass_merges += self.reverse_containment_island_merge().unwrap_or(0);
            pass_merges += self.purge_fragment_islands().unwrap_or(0);
            pass_merges += self.prefix_strip_island_merge().unwrap_or(0);
            pass_merges += self.merge_name_variants().unwrap_or(0);
            pass_merges += self.aggressive_prefix_dedup().unwrap_or(0);
            pass_merges += self.merge_high_confidence_prefix_variants().unwrap_or(0);
            pass_merges += self.dissolve_name_fragment_hubs().unwrap_or(0);
            pass_merges += self.strip_leading_adjectives().unwrap_or(0);
            if pass_merges == 0 {
                break;
            }
            convergence_merges += pass_merges;
        }

        // Merge connected entities where one name contains the other
        let containment_merged = self.merge_connected_containment().unwrap_or(0);

        // Aggressive same-type prefix deduplication
        let aggressive_deduped = self.aggressive_prefix_dedup().unwrap_or(0);

        // Purge generic single-word islands (adverbs, adjectives, citation surnames)
        let generic_purged = self.purge_generic_single_word_islands().unwrap_or(0);

        // Purge multi-word island noise (fragments, misclassifications)
        let multiword_purged = self.purge_multiword_island_noise().unwrap_or(0);

        // Purge fragment island entities (NLP extraction errors)
        let fragment_islands_purged = self.purge_fragment_island_entities().unwrap_or(0);

        // Purge single-word concept islands (adjectives, common nouns, non-English stubs)
        let concept_islands_purged = self.purge_single_word_concept_islands().unwrap_or(0);

        // Purge mistyped person islands (multi-word "person" entities that aren't people)
        let mistyped_person_purged = self.purge_mistyped_person_islands().unwrap_or(0);

        // Fix country-concatenation entities (e.g., "Netherlands Oskar Klein" → merge into "Oskar Klein")
        let country_concat_fixed = self.fix_country_concatenation_entities().unwrap_or(0);

        // Merge prefix-noise entities (e.g., "Devastated Tim Berners-Lee" → "Tim Berners-Lee")
        let prefix_noise_merged = self.merge_prefix_noise_entities().unwrap_or(0);

        // Token-based island reconnection (TF-IDF shared token matching)
        let token_reconnected = self.reconnect_islands_by_tokens().unwrap_or(0);

        // Name-containment island reconnection (substring matching within same type)
        let name_containment_reconnected =
            self.reconnect_islands_by_name_containment().unwrap_or(0);

        // Single-word island reconnection (match isolated single-word entities to connected multi-word entities)
        let single_word_reconnected = self.reconnect_single_word_islands().unwrap_or(0);

        // Refine generic predicates using entity type pairs
        let predicates_refined = self.refine_associated_with().unwrap_or(0);
        let contributed_refined = self.refine_contributed_to().unwrap_or(0);

        // Promote mature testing hypotheses (>3 days, confidence >= 0.65)
        let promoted = self.promote_mature_hypotheses(3, 0.65).unwrap_or(0);

        // Hypothesis pair deduplication (prevent bloat from multiple runs)
        let pair_deduped = self.dedup_hypotheses_by_pair().unwrap_or(0);

        // K-core analysis: find the dense backbone
        let (max_k, core_members) =
            crate::graph::densest_core(self.brain, 3).unwrap_or((0, vec![]));

        // Save graph snapshot for trend tracking
        let _snapshot_id = crate::graph::save_graph_snapshot(self.brain).unwrap_or(0);

        // Track discovery velocity
        let _ = self.track_discovery_velocity(
            all_patterns.len(),
            all_hypotheses.len(),
            confirmed,
            rejected,
        );

        // Get trend comparison if previous snapshots exist
        let trend_line = {
            let snapshots = crate::graph::get_graph_snapshots(self.brain, 2).unwrap_or_default();
            if snapshots.len() >= 2 {
                format!(
                    ", trend: {}",
                    crate::graph::format_trend(&snapshots[0], &snapshots[1])
                )
            } else {
                String::new()
            }
        };

        let summary = format!(
            "Discovered {} patterns, generated {} new hypotheses ({} confirmed, {} rejected), \
             auto-resolved {} existing ({}✓ {}✗), {} island entities, {} meaningful relations, \
             {} cross-domain gaps, {} knowledge frontiers, {} bridge entities, {} connectable islands, \
             {} prioritized gaps, {} decayed, {} pruned, {} island clusters, \
             predicate diversity: {}/{} ({:.0}% diverse), \
             {} exact-name deduped, {} reverse-containment merged, {} bulk-rejected stale, \
             {} fuzzy duplicates ({} auto-merge, {} auto-merged), {} sparse topic domains, \
             {} name subsumptions found, {} types fixed, {} types inferred from neighborhood, {} noise entities purged, \
             {} bulk cleaned, {} deep cleaned, {} predicates normalized, {} islands reconnected, \
             {} fact-inferred relations, {} name cross-references, \
             {} compound relations + {} compound merged, {} concat split ({}r/{}c), {} prefix-merged, {} suffix-merged, {} word-overlap merged, \
             {} fragment-purged, {} prefix-strip merged, {} name-variants merged, {} auto-consolidated, \
             {} fragment-hubs dissolved, {} hc-prefix merged, {} convergence-pass merges, \
             {} connected-containment merged, {} aggressive-prefix deduped, \
             {} generic islands purged, {} multiword noise purged, {} fragment islands purged, {} concept islands purged, {} mistyped-person purged, {} country-concat fixed, {} prefix-noise merged, {} token-reconnected, {} name-containment reconnected, {} single-word reconnected, {} predicates refined, {} contributed_to refined, {} hypotheses promoted, \
             {} fragment hypotheses cleaned, \
             {} hypothesis pairs deduped, k-core: k={} with {} entities in dense backbone{}",
            all_patterns.len(),
            all_hypotheses.len(),
            confirmed,
            rejected,
            auto_confirmed + auto_rejected,
            auto_confirmed,
            auto_rejected,
            islands.len(),
            meaningful_rels,
            cross_gaps.len(),
            frontiers.len(),
            bridges.len(),
            connectable.len(),
            gap_priorities.len(),
            decayed,
            pruned,
            island_clusters.len(),
            diverse_rels,
            total_rels,
            div_ratio * 100.0,
            exact_deduped,
            reverse_merged,
            bulk_rejected,
            fuzzy_dupes.len(),
            merge_candidates,
            auto_merged,
            sparse_topics,
            subsumptions.len(),
            types_fixed,
            types_inferred,
            purged,
            bulk_cleaned,
            deep_cleaned,
            normalized,
            reconnected,
            fact_inferred,
            name_crossrefs,
            compound_rels,
            compound_merged,
            concat_rels + concat_cleaned,
            concat_rels,
            concat_cleaned,
            prefix_merged,
            suffix_merged,
            word_merged,
            fragment_purged,
            prefix_strip_merged,
            name_variants_merged,
            auto_consolidated,
            fragments_dissolved,
            hc_prefix_merged,
            convergence_merges,
            containment_merged,
            aggressive_deduped,
            generic_purged,
            multiword_purged,
            fragment_islands_purged,
            concept_islands_purged,
            mistyped_person_purged,
            country_concat_fixed,
            prefix_noise_merged,
            token_reconnected,
            name_containment_reconnected,
            single_word_reconnected,
            predicates_refined,
            contributed_refined,
            promoted,
            fragment_cleaned,
            pair_deduped,
            max_k,
            core_members.len(),
            trend_line,
        );

        Ok(DiscoveryReport {
            patterns_found: all_patterns,
            hypotheses_generated: all_hypotheses,
            gaps_detected,
            summary,
        })
    }

    // -----------------------------------------------------------------------
    // Cross-domain gap detection
    // -----------------------------------------------------------------------

    /// Find disconnected clusters that likely should be connected based on entity types.
    pub fn find_cross_domain_gaps(&self) -> Result<Vec<(Vec<String>, Vec<String>, String)>> {
        let components = crate::graph::connected_components(self.brain)?;
        if components.len() < 2 {
            return Ok(vec![]);
        }
        let entities = self.brain.all_entities()?;
        let id_to_entity: HashMap<i64, &crate::db::Entity> =
            entities.iter().map(|e| (e.id, e)).collect();

        // For each component, collect entity types (excluding noise types)
        let noise_types: HashSet<&str> = ["phrase", "source", "url"].iter().copied().collect();
        let mut component_types: Vec<(Vec<String>, HashMap<String, usize>)> = Vec::new();
        for comp in &components {
            // Skip tiny components (likely noise fragments)
            if comp.len() < 3 {
                continue;
            }
            let mut types: HashMap<String, usize> = HashMap::new();
            let mut names: Vec<String> = Vec::new();
            for &id in comp {
                if let Some(e) = id_to_entity.get(&id) {
                    if !noise_types.contains(e.entity_type.as_str()) && !is_noise_name(&e.name) {
                        *types.entry(e.entity_type.clone()).or_insert(0) += 1;
                        if names.len() < 5 && !is_noise_name(&e.name) {
                            names.push(e.name.clone());
                        }
                    }
                }
            }
            if !types.is_empty() && names.len() >= 2 {
                component_types.push((names, types));
            }
        }

        // Find pairs of components sharing high-value entity types
        let skip_types: HashSet<&str> = ["unknown", "phrase", "source", "url"]
            .iter()
            .copied()
            .collect();
        let mut gaps = Vec::new();
        for i in 0..component_types.len().min(20) {
            for j in (i + 1)..component_types.len().min(20) {
                let (names_i, types_i) = &component_types[i];
                let (names_j, types_j) = &component_types[j];
                let shared: Vec<String> = types_i
                    .keys()
                    .filter(|t| types_j.contains_key(*t) && !skip_types.contains(t.as_str()))
                    .cloned()
                    .collect();
                if !shared.is_empty() {
                    gaps.push((
                        names_i.clone(),
                        names_j.clone(),
                        format!(
                            "Clusters share types [{}] but are disconnected",
                            shared.join(", ")
                        ),
                    ));
                }
            }
        }
        Ok(gaps)
    }

    /// Find entities that bridge multiple communities — potential cross-domain connectors.
    pub fn find_bridge_entities(&self) -> Result<Vec<(String, String, usize)>> {
        let components = crate::graph::connected_components(self.brain)?;
        if components.len() < 2 {
            return Ok(vec![]);
        }
        // Map entity → component index
        let mut entity_comp: HashMap<i64, usize> = HashMap::new();
        for (idx, comp) in components.iter().enumerate() {
            for &id in comp {
                entity_comp.insert(id, idx);
            }
        }
        // Find entities whose neighbours span multiple components (via relations)
        let relations = self.brain.all_relations()?;
        let mut entity_comps: HashMap<i64, HashSet<usize>> = HashMap::new();
        for r in &relations {
            if let Some(&comp) = entity_comp.get(&r.object_id) {
                entity_comps.entry(r.subject_id).or_default().insert(comp);
            }
            if let Some(&comp) = entity_comp.get(&r.subject_id) {
                entity_comps.entry(r.object_id).or_default().insert(comp);
            }
        }
        let mut bridges: Vec<(String, String, usize)> = Vec::new();
        for (eid, comps) in &entity_comps {
            if comps.len() >= 2 {
                let name = self.entity_name(*eid)?;
                let etype = self
                    .brain
                    .get_entity_by_id(*eid)?
                    .map(|e| e.entity_type)
                    .unwrap_or_default();
                if !is_noise_name(&name) && !is_noise_type(&etype) {
                    bridges.push((name, etype, comps.len()));
                }
            }
        }
        bridges.sort_by(|a, b| b.2.cmp(&a.2));
        bridges.truncate(20);
        Ok(bridges)
    }

    /// Decay confidence of old unconfirmed hypotheses (meta-learning cleanup).
    pub fn decay_old_hypotheses(&self, max_age_days: i64) -> Result<usize> {
        let cutoff = (Utc::now() - chrono::Duration::days(max_age_days))
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let count = self.brain.with_conn(|conn| {
            let updated = conn.execute(
                "UPDATE hypotheses SET confidence = confidence * 0.8 WHERE status IN ('proposed', 'testing') AND discovered_at < ?1",
                params![cutoff],
            )?;
            Ok(updated)
        })?;
        Ok(count)
    }

    /// Suggest topics to crawl based on knowledge gaps.
    pub fn suggest_crawl_topics(&self) -> Result<Vec<(String, String)>> {
        let mut suggestions = Vec::new();

        // 1. Entity types with low knowledge density (skip noise types)
        let density = crate::graph::knowledge_density(self.brain)?;
        for (etype, (count, avg)) in &density {
            if *count >= 3 && *avg < 0.5 && !is_noise_type(etype) && *etype != "unknown" {
                suggestions.push((
                    etype.clone(),
                    format!(
                        "Type '{}' has {} entities but only {:.1} avg relations — needs enrichment",
                        etype, count, avg
                    ),
                ));
            }
        }

        // 2. Entities involved in hypotheses with object "?"
        let hyps = self.list_hypotheses(Some(HypothesisStatus::Proposed))?;
        for h in hyps.iter().take(10) {
            if h.object == "?" {
                suggestions.push((
                    h.subject.clone(),
                    format!(
                        "Hypothesis: '{}' likely has '{}' but value unknown",
                        h.subject, h.predicate
                    ),
                ));
            }
        }

        // 3. High-centrality entities with few facts
        let pr = crate::graph::pagerank(self.brain, 0.85, 20)?;
        let mut ranked: Vec<(i64, f64)> = pr.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (id, score) in ranked.iter().take(20) {
            if let Some(e) = self.brain.get_entity_by_id(*id)? {
                if e.entity_type == "phrase" || e.entity_type == "source" {
                    continue;
                }
                let facts = self.brain.get_facts_for(*id)?;
                let rels = self.brain.get_relations_for(*id)?;
                if facts.len() + rels.len() < 3 {
                    suggestions.push((
                        e.name.clone(),
                        format!(
                            "High-rank entity ({:.4}) '{}' has only {} facts+relations",
                            score,
                            e.name,
                            facts.len() + rels.len()
                        ),
                    ));
                }
            }
        }

        Ok(suggestions)
    }

    /// Rank connected entities by enrichment priority: which entities would benefit
    /// most from deeper crawling? Combines PageRank importance, low fact/relation
    /// count, and high-value entity type into a single score.
    /// Returns (entity_name, entity_type, score, reason) sorted by priority.
    pub fn rank_enrichment_targets(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String, f64, String)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        // Only consider connected entities
        let mut connected: HashSet<i64> = HashSet::new();
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // PageRank for importance weighting
        let pr = crate::graph::pagerank(self.brain, 0.85, 20)?;

        // Type value weights
        let type_weight = |t: &str| -> f64 {
            match t {
                "person" => 1.5,
                "concept" => 1.3,
                "organization" => 1.2,
                "place" => 1.0,
                "technology" => 1.4,
                "event" => 1.1,
                _ => 0.5,
            }
        };

        let mut targets: Vec<(String, String, f64, String)> = Vec::new();
        for e in &entities {
            if !connected.contains(&e.id) || is_noise_type(&e.entity_type) || is_noise_name(&e.name)
            {
                continue;
            }
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            let rank = pr.get(&e.id).copied().unwrap_or(0.0);
            let facts = self.brain.get_facts_for(e.id)?.len();

            // Enrichment score: high importance + low knowledge = high priority
            // Entities with high PageRank but few connections/facts need enrichment most
            let sparsity = 1.0 / (1.0 + deg as f64 + facts as f64);
            let importance = rank * 10000.0; // Normalize PageRank
            let tw = type_weight(&e.entity_type);
            let score = importance * sparsity * tw;

            if score > 0.001 {
                let reason = format!(
                    "PageRank {:.4}, {} relations, {} facts — {} type",
                    rank, deg, facts, e.entity_type
                );
                targets.push((e.name.clone(), e.entity_type.clone(), score, reason));
            }
        }

        targets.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        targets.truncate(limit);
        Ok(targets)
    }

    /// Detect multi-word island entities that are likely NLP extraction artifacts.
    /// Patterns: "Noun Verb" fragments, reversed "Surname Firstname" citation format,
    /// entities containing common academic terms mixed with proper nouns.
    /// Returns count of entities purged.
    pub fn purge_multiword_island_noise(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Words that signal sentence fragments when appearing in entity names
        let fragment_signals: HashSet<&str> = [
            "quiz",
            "film",
            "printed",
            "feature",
            "mosaics",
            "explorations",
            "mission",
            "lexicon",
            "revue",
            "etablierte",
            "kirchen",
            "lycée",
            "recipients",
            "glance",
            "visualize",
            "shrieking",
            "servile",
            "thrilling",
            "arrives",
            "vessel",
            "resting",
            "gefolgschaft",
            "hintergrund",
            "überblick",
        ]
        .iter()
        .copied()
        .collect();

        // Non-person type indicators wrongly classified as persons
        let not_person_patterns: &[&str] =
            &["age", "bsd", "hat", "kong", "ship", "hms", "uss", "iss"];

        let mut purged = 0usize;
        for e in &entities {
            if connected.contains(&e.id) {
                continue;
            }
            let facts = self.brain.get_facts_for(e.id)?;
            if !facts.is_empty() {
                continue;
            }
            let lower = e.name.to_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();

            let should_purge = if words.len() >= 2 {
                // Check for fragment signal words
                words.iter().any(|w| fragment_signals.contains(w))
            } else {
                false
            };

            // Purge entities with names containing non-ASCII quote artifacts
            let has_artifact = e.name.contains("Has-") && e.name.contains("Ayş");

            if should_purge || has_artifact {
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                    Ok(())
                })?;
                purged += 1;
            }

            // Fix type misclassification: "Bronze Age", "Red Hat", etc. aren't persons
            if e.entity_type == "person" && words.len() == 2 {
                let last = words[1];
                if not_person_patterns.contains(&last) {
                    let new_type = if last == "age" {
                        "concept"
                    } else if last == "hat" || last == "bsd" {
                        "technology"
                    } else {
                        "concept"
                    };
                    self.brain.with_conn(|conn| {
                        conn.execute(
                            "UPDATE entities SET entity_type = ?1 WHERE id = ?2",
                            params![new_type, e.id],
                        )?;
                        Ok(())
                    })?;
                }
            }
        }
        Ok(purged)
    }

    /// Purge island entities whose names contain mixed-case fragments indicating
    /// NLP extraction errors (e.g. "Derivatives Several", "CEO Calista Redmond",
    /// "Heimgartner Susanna Zürich"). These are sentence fragments, not real entities.
    /// Also fixes "State X" person misclassifications for connected entities.
    /// Returns count of entities removed.
    pub fn purge_fragment_island_entities(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Words that signal extraction fragments when appearing in entity names
        let fragment_words: HashSet<&str> = [
            "several",
            "various",
            "multiple",
            "numerous",
            "certain",
            "particular",
            "respective",
            "notable",
            "significant",
            "prominent",
            "renowned",
            "alleged",
            "supposed",
            "purported",
            "attempted",
            "failed",
            "increasingly",
            "particularly",
            "especially",
            "approximately",
            "subsequently",
            "simultaneously",
            "predominantly",
            "primarily",
        ]
        .iter()
        .copied()
        .collect();

        // Title prefixes that when combined with names create fragments
        let title_prefixes: &[&str] = &[
            "ceo",
            "cfo",
            "cto",
            "coo",
            "vp",
            "svp",
            "evp",
            "director",
            "chairman",
            "president",
            "secretary",
            "minister",
            "ambassador",
            "governor",
            "senator",
            "professor",
            "dr",
            "dean",
            "provost",
        ];

        let mut purged = 0usize;
        for e in &entities {
            if connected.contains(&e.id) {
                continue;
            }
            let facts = self.brain.get_facts_for(e.id)?;
            if !facts.is_empty() {
                continue;
            }
            let lower = e.name.to_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();
            if words.len() < 2 {
                continue;
            }

            let should_purge =
                // Contains fragment signal words
                words.iter().any(|w| fragment_words.contains(w))
                // Starts with a title prefix (e.g. "CEO Calista Redmond")
                || title_prefixes.contains(&words[0])
                // "State X" patterns that are secretary-of-state fragments
                || (words[0] == "state" && words.len() >= 2 && e.entity_type == "person")
                // Names with 4+ words where first word is lowercase-looking role
                || (words.len() >= 4 && words[0].len() <= 6 && e.entity_type == "person"
                    && !words[0].chars().next().unwrap_or('a').is_uppercase()
                    && e.name.chars().next().unwrap_or('a').is_lowercase());

            if should_purge {
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                    Ok(())
                })?;
                purged += 1;
            }
        }
        Ok(purged)
    }

    /// Purge single-word concept/unknown islands that are clearly noise (adjectives, common nouns,
    /// non-English words without context). These are NLP extraction artifacts that add no knowledge.
    /// Returns count of entities purged.
    pub fn purge_single_word_concept_islands(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        let mut purged = 0usize;
        for e in &entities {
            if connected.contains(&e.id) {
                continue;
            }
            let facts = self.brain.get_facts_for(e.id)?;
            if !facts.is_empty() {
                continue;
            }
            let name = e.name.trim();
            let words: Vec<&str> = name.split_whitespace().collect();
            let lower = name.to_lowercase();

            let should_purge = match words.len() {
                1 => {
                    // Single-word concepts/unknowns that are just adjectives or common nouns
                    let is_concept_or_unknown =
                        e.entity_type == "concept" || e.entity_type == "unknown";
                    let too_generic = lower.len() <= 12
                        && (lower.ends_with("ic")
                            || lower.ends_with("al")
                            || lower.ends_with("ous")
                            || lower.ends_with("ive")
                            || lower.ends_with("ment")
                            || lower.ends_with("tion")
                            || lower.ends_with("ness")
                            || lower.ends_with("ung") // German suffix
                            || lower.ends_with("keit") // German suffix
                            || lower.ends_with("schaft") // German suffix
                            || lower.ends_with("ing")
                            || lower.ends_with("ity")
                            || lower.ends_with("ism")
                            || lower.ends_with("ure")
                            || lower.ends_with("ary")
                            || lower.ends_with("ery")
                            || lower.ends_with("ory"));
                    // Also purge plural common nouns as concepts
                    let is_plural_common = is_concept_or_unknown
                        && lower.ends_with('s')
                        && !lower.ends_with("ss")
                        && lower.len() <= 10
                        && lower.chars().next().is_some_and(|c| c.is_lowercase());
                    // Common English words that are clearly not standalone entities
                    let common_non_entities = [
                        "inside",
                        "outside",
                        "suppose",
                        "choose",
                        "twelve",
                        "tribute",
                        "spirit",
                        "prior",
                        "interest",
                        "denote",
                        "wisdom",
                        "resting",
                        "highest-ever",
                        "prevent",
                        "protect",
                        "produce",
                        "promote",
                        "propose",
                        "promise",
                        "provide",
                        "pursue",
                        "receive",
                        "reduce",
                        "remain",
                        "remove",
                        "repeat",
                        "replace",
                        "require",
                        "resolve",
                        "respond",
                        "restore",
                        "reveal",
                        "select",
                        "separate",
                        "suggest",
                        "support",
                        "survive",
                        "threat",
                        "toward",
                        "unlike",
                        "within",
                        "without",
                        "among",
                        "beneath",
                        "beside",
                        "beyond",
                        "despite",
                        "except",
                        "thought",
                        "enough",
                        "perhaps",
                        "rather",
                        "though",
                        "unless",
                        "whether",
                        "almost",
                        "simply",
                        "indeed",
                        "merely",
                        "hardly",
                        "likely",
                        "mainly",
                        "mostly",
                        "nearly",
                        "partly",
                        "quite",
                        "truly",
                        "fully",
                        "deeply",
                        "highly",
                        "widely",
                        "closely",
                        "directly",
                        "exactly",
                        "rapidly",
                        "slowly",
                        "roughly",
                        "slightly",
                        "somewhat",
                        "largely",
                        "entirely",
                        "possibly",
                        "hole",
                        "sign",
                        "myth",
                        "tell",
                        "swin",
                    ];
                    let is_common_non_entity =
                        is_concept_or_unknown && common_non_entities.contains(&lower.as_str());
                    // Only purge generic-looking single words for concepts
                    (is_concept_or_unknown && too_generic && lower.len() < 15)
                        || is_plural_common
                        || is_common_non_entity
                }
                2 | 3 => {
                    // Multi-word fragments: "Seventh Through", "Open Zihintpause Pause"
                    let has_ordinal = words[0].to_lowercase().ends_with("th")
                        && [
                            "four", "fif", "six", "seven", "eigh", "nin", "ten", "eleven", "twelf",
                        ]
                        .iter()
                        .any(|p| words[0].to_lowercase().starts_with(p));
                    let has_prep_end = ["through", "about", "during", "within", "between"]
                        .contains(&words.last().unwrap_or(&"").to_lowercase().as_str());
                    // "Scholarship Est", "Golden Peaches" for non-place types
                    let ends_with_noise = ["est", "pause", "peaches"]
                        .contains(&words.last().unwrap_or(&"").to_lowercase().as_str());
                    (has_ordinal && has_prep_end)
                        || ends_with_noise
                        || (e.entity_type == "person"
                            && words.len() == 2
                            && ["arts", "open", "scholarship"]
                                .contains(&words[0].to_lowercase().as_str()))
                }
                _ => false,
            };

            if should_purge {
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                    Ok(())
                })?;
                purged += 1;
            }
        }
        Ok(purged)
    }

    /// Cluster island entities by type and name similarity to suggest batch crawl targets.
    pub fn cluster_islands_for_crawl(&self) -> Result<Vec<(String, Vec<String>)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }
        // Group islands by type
        let mut type_islands: HashMap<String, Vec<String>> = HashMap::new();
        for e in &entities {
            if !connected.contains(&e.id)
                && !is_noise_type(&e.entity_type)
                && !is_noise_name(&e.name)
            {
                type_islands
                    .entry(e.entity_type.clone())
                    .or_default()
                    .push(e.name.clone());
            }
        }
        let mut clusters: Vec<(String, Vec<String>)> = type_islands
            .into_iter()
            .filter(|(_, names)| names.len() >= 2)
            .map(|(t, mut names)| {
                names.sort();
                names.truncate(20);
                (t, names)
            })
            .collect();
        clusters.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        Ok(clusters)
    }

    /// Find islands that could plausibly connect to existing hub entities.
    /// Uses name substring matching and type affinity to suggest links.
    pub fn find_connectable_islands(&self) -> Result<Vec<(String, String, String, f64)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        // Build connected set and degree map
        let mut connected: HashSet<i64> = HashSet::new();
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Collect hubs (connected entities with degree >= 2)
        let hubs: Vec<&crate::db::Entity> = entities
            .iter()
            .filter(|e| {
                connected.contains(&e.id)
                    && degree.get(&e.id).copied().unwrap_or(0) >= 2
                    && !is_noise_type(&e.entity_type)
                    && !is_noise_name(&e.name)
            })
            .collect();

        // Collect high-value islands
        let islands: Vec<&crate::db::Entity> = entities
            .iter()
            .filter(|e| {
                !connected.contains(&e.id)
                    && HIGH_VALUE_TYPES.contains(&e.entity_type.as_str())
                    && !is_noise_name(&e.name)
                    && e.name.len() >= 3
            })
            .collect();

        let mut suggestions = Vec::new();

        // For each island, check if any hub name contains it or vice versa
        for island in islands.iter().take(500) {
            let island_lower = island.name.to_lowercase();
            let island_words: HashSet<&str> = island_lower.split_whitespace().collect();
            if island_words.is_empty() {
                continue;
            }

            for hub in &hubs {
                let hub_lower = hub.name.to_lowercase();
                let hub_words: HashSet<&str> = hub_lower.split_whitespace().collect();

                // Check word overlap (at least one significant shared word)
                let shared: Vec<&&str> = island_words
                    .intersection(&hub_words)
                    .filter(|w| w.len() >= 4)
                    .collect();

                if !shared.is_empty() {
                    let overlap_ratio =
                        shared.len() as f64 / island_words.len().max(hub_words.len()) as f64;
                    let confidence = 0.3 + overlap_ratio * 0.4;
                    let reason = format!(
                        "Island '{}' ({}) shares words [{}] with hub '{}' ({})",
                        island.name,
                        island.entity_type,
                        shared.iter().map(|s| **s).collect::<Vec<_>>().join(", "),
                        hub.name,
                        hub.entity_type
                    );
                    suggestions.push((island.name.clone(), hub.name.clone(), reason, confidence));
                }
            }
        }

        // Sort by confidence descending
        suggestions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(50);
        Ok(suggestions)
    }

    /// Compute a priority score for each knowledge gap to guide crawling.
    /// Higher score = more important to fill.
    pub fn prioritize_gaps(&self) -> Result<Vec<(String, String, f64)>> {
        let mut gap_scores: Vec<(String, String, f64)> = Vec::new();

        // 1. Islands of high-value types get base priority
        let islands = self.find_island_entities()?;
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for (_, etype) in &islands {
            *type_counts.entry(etype.clone()).or_insert(0) += 1;
        }

        // Types with many islands are systematic gaps
        for (etype, count) in &type_counts {
            if *count >= 5 && HIGH_VALUE_TYPES.contains(&etype.as_str()) {
                let score = (*count as f64).ln() * 2.0;
                gap_scores.push((
                    etype.clone(),
                    format!(
                        "{} disconnected '{}' entities — systematic gap",
                        count, etype
                    ),
                    score,
                ));
            }
        }

        // 2. Hypotheses with object "?" (unknown targets)
        let hyps = self.list_hypotheses(Some(HypothesisStatus::Proposed))?;
        let unknown_count = hyps.iter().filter(|h| h.object == "?").count();
        if unknown_count > 0 {
            gap_scores.push((
                "unknown_relations".to_string(),
                format!("{} hypotheses have unknown objects", unknown_count),
                (unknown_count as f64).ln() * 1.5,
            ));
        }

        // 3. Cross-domain gaps (disconnected clusters)
        let cross_gaps = self.find_cross_domain_gaps()?;
        if !cross_gaps.is_empty() {
            gap_scores.push((
                "cross_domain".to_string(),
                format!(
                    "{} disconnected cluster pairs sharing entity types",
                    cross_gaps.len()
                ),
                (cross_gaps.len() as f64).ln() * 3.0,
            ));
        }

        gap_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(gap_scores)
    }

    /// Auto-confirm or reject hypotheses based on graph evidence.
    /// More aggressive than validate_hypothesis — checks transitive paths.
    pub fn auto_resolve_hypotheses(&self) -> Result<(usize, usize)> {
        let mut hyps = self.list_hypotheses(Some(HypothesisStatus::Testing))?;
        let mut confirmed = 0usize;
        let mut rejected = 0usize;

        for h in hyps.iter_mut() {
            // Check if a path exists between subject and object (evidence of relation)
            if h.object != "?" {
                // Reject single-word fragment pairs (NLP noise like "Grace" → "Hopper")
                if !h.subject.contains(' ')
                    && h.subject.len() <= 12
                    && !h.object.contains(' ')
                    && h.object.len() <= 12
                {
                    self.update_hypothesis_status(h.id, HypothesisStatus::Rejected)?;
                    self.record_outcome(&h.pattern_source, false)?;
                    rejected += 1;
                    continue;
                }

                // Skip substring pairs — these are merge candidates, not discoveries
                let sl = h.subject.to_lowercase();
                let ol = h.object.to_lowercase();
                if sl.contains(&ol) || ol.contains(&sl) {
                    self.update_hypothesis_status(h.id, HypothesisStatus::Rejected)?;
                    self.record_outcome(&h.pattern_source, false)?;
                    rejected += 1;
                    continue;
                }

                let path = crate::graph::shortest_path(self.brain, &h.subject, &h.object)?;
                if let Some(p) = &path {
                    // Only confirm via path if it's a direct connection (len=2) or
                    // a 2-hop path (len=3) with high base confidence
                    let path_boost = match p.len() {
                        2 => 0.3,                         // direct connection is strong evidence
                        3 if h.confidence >= 0.5 => 0.15, // 2-hop with existing confidence
                        _ => 0.0,                         // longer paths are too weak
                    };
                    if path_boost > 0.0 {
                        let new_conf = (h.confidence + path_boost).min(1.0);
                        if new_conf >= CONFIRMATION_THRESHOLD {
                            self.update_hypothesis_status(h.id, HypothesisStatus::Confirmed)?;
                            self.record_outcome(&h.pattern_source, true)?;
                            confirmed += 1;
                            continue;
                        }
                    }
                }

                // Check if entities even exist with relations
                let subj = self.brain.get_entity_by_name(&h.subject)?;
                let obj = self.brain.get_entity_by_name(&h.object)?;
                match (subj, obj) {
                    (None, _) | (_, None) => {
                        // Entity deleted or never existed — reject
                        self.update_hypothesis_status(h.id, HypothesisStatus::Rejected)?;
                        self.record_outcome(&h.pattern_source, false)?;
                        rejected += 1;
                    }
                    _ => {
                        // Check for contradictions
                        if self.check_contradiction(h)? {
                            self.update_hypothesis_status(h.id, HypothesisStatus::Rejected)?;
                            self.record_outcome(&h.pattern_source, false)?;
                            rejected += 1;
                        }
                    }
                }
            }
        }
        Ok((confirmed, rejected))
    }

    /// Batch-boost hypotheses using graph structure (communities + k-cores).
    /// Computes expensive graph metrics once, then applies to all hypotheses.
    pub fn boost_hypotheses_with_graph_structure(
        &self,
        hypotheses: &mut [Hypothesis],
    ) -> Result<()> {
        // Compute once
        let communities = crate::graph::louvain_communities(self.brain)?;
        let cores = crate::graph::k_core_decomposition(self.brain)?;

        // Build name→id cache
        let entities = self.brain.all_entities()?;
        let name_to_id: HashMap<String, i64> = entities
            .iter()
            .map(|e| (e.name.to_lowercase(), e.id))
            .collect();

        for h in hypotheses.iter_mut() {
            if h.object == "?" {
                continue;
            }
            let s_id = name_to_id.get(&h.subject.to_lowercase()).copied();
            let o_id = name_to_id.get(&h.object.to_lowercase()).copied();

            if let (Some(sid), Some(oid)) = (s_id, o_id) {
                // Community co-membership boost
                if let (Some(&cs), Some(&co)) = (communities.get(&sid), communities.get(&oid)) {
                    if cs == co {
                        h.confidence = (h.confidence + 0.15).min(1.0);
                        h.evidence_for.push(
                            "Subject and object belong to the same graph community".to_string(),
                        );
                    }
                }
                // K-core depth boost
                let s_core = cores.get(&sid).copied().unwrap_or(0);
                let o_core = cores.get(&oid).copied().unwrap_or(0);
                let min_core = s_core.min(o_core);
                if min_core >= 2 {
                    h.confidence = (h.confidence + 0.05 * min_core as f64).min(1.0);
                    h.evidence_for.push(format!(
                        "Both entities in {}-core (deeply embedded)",
                        min_core
                    ));
                }
            }
        }
        Ok(())
    }

    /// Check if a hypothesis already exists (dedup across runs).
    fn hypothesis_exists(&self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        self.brain.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM hypotheses WHERE subject = ?1 AND predicate = ?2 AND object = ?3",
                params![subject, predicate, object],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })
    }

    /// Deduplicate patterns: merge identical descriptions, increment frequency.
    fn dedup_patterns(&self, patterns: &mut Vec<Pattern>) {
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut deduped: Vec<Pattern> = Vec::new();
        for p in patterns.drain(..) {
            if let Some(&idx) = seen.get(&p.description) {
                deduped[idx].frequency += p.frequency;
            } else {
                seen.insert(p.description.clone(), deduped.len());
                deduped.push(p);
            }
        }
        *patterns = deduped;
    }

    // -----------------------------------------------------------------------
    // Explain
    // -----------------------------------------------------------------------

    /// Build a full explanation for a hypothesis.
    pub fn explain(&self, hypothesis_id: i64) -> Result<Option<String>> {
        let h = self.get_hypothesis(hypothesis_id)?;
        match h {
            None => Ok(None),
            Some(h) => {
                let mut lines = Vec::new();
                lines.push(format!("# Hypothesis #{}", h.id));
                lines.push(format!(
                    "**Claim:** {} {} {}",
                    h.subject, h.predicate, h.object
                ));
                lines.push(format!(
                    "**Status:** {} | **Confidence:** {:.2}",
                    h.status.as_str(),
                    h.confidence
                ));
                lines.push(format!("**Source pattern:** {}", h.pattern_source));
                lines.push(format!("**Discovered:** {}", h.discovered_at));
                lines.push(String::new());
                lines.push("## Reasoning Chain".to_string());
                for (i, step) in h.reasoning_chain.iter().enumerate() {
                    lines.push(format!("{}. {}", i + 1, step));
                }
                lines.push(String::new());
                if !h.evidence_for.is_empty() {
                    lines.push("## Evidence For".to_string());
                    for e in &h.evidence_for {
                        lines.push(format!("- ✅ {}", e));
                    }
                    lines.push(String::new());
                }
                if !h.evidence_against.is_empty() {
                    lines.push("## Evidence Against".to_string());
                    for e in &h.evidence_against {
                        lines.push(format!("- ❌ {}", e));
                    }
                }
                Ok(Some(lines.join("\n")))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Output formats
    // -----------------------------------------------------------------------

    pub fn report_json(&self, report: &DiscoveryReport) -> String {
        serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".to_string())
    }

    /// Find likely duplicate entities using word-overlap similarity.
    /// Returns (entity_a, entity_b, similarity, suggested_action).
    /// Much better than Levenshtein for multi-word entity names like
    /// "Swiss Federal Institute" vs "Federal Institute of Switzerland".
    pub fn find_fuzzy_duplicates(&self) -> Result<Vec<(String, String, f64, String)>> {
        let entities = self.brain.all_entities()?;
        let meaningful: Vec<&crate::db::Entity> = entities
            .iter()
            .filter(|e| {
                !is_noise_type(&e.entity_type) && !is_noise_name(&e.name) && e.name.len() >= 3
            })
            .collect();

        // Build word sets for each entity (lowercase, skip tiny words)
        let word_sets: Vec<(i64, &str, &str, HashSet<String>)> = meaningful
            .iter()
            .map(|e| {
                let words: HashSet<String> = e
                    .name
                    .to_lowercase()
                    .split_whitespace()
                    .filter(|w| w.len() >= 3)
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                    .filter(|w| !w.is_empty())
                    .collect();
                (e.id, e.name.as_str(), e.entity_type.as_str(), words)
            })
            .collect();

        let mut duplicates = Vec::new();

        // Only compare entities of the same type, limit comparisons
        let mut by_type: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, (_, _, etype, _)) in word_sets.iter().enumerate() {
            by_type.entry(etype).or_default().push(idx);
        }

        for indices in by_type.values() {
            // Skip huge groups to avoid O(n²) explosion
            if indices.len() > 2000 {
                continue;
            }
            for (pos_i, &idx_a) in indices.iter().enumerate() {
                let (_, name_a, _, words_a) = &word_sets[idx_a];
                if words_a.is_empty() {
                    continue;
                }
                for &idx_b in &indices[(pos_i + 1)..] {
                    let (_, name_b, _, words_b) = &word_sets[idx_b];
                    if words_b.is_empty() {
                        continue;
                    }

                    let intersection = words_a.intersection(words_b).count();
                    if intersection == 0 {
                        continue;
                    }
                    let union = words_a.union(words_b).count();
                    let jaccard = intersection as f64 / union as f64;

                    // High overlap = likely duplicate
                    if jaccard >= 0.5 && intersection >= 2 {
                        // Auto-merge if very high Jaccard OR if Jaccard ≥ 0.7 and
                        // names have similar length (avoids merging "X" into "X Y Z")
                        let len_ratio = name_a.len().min(name_b.len()) as f64
                            / name_a.len().max(name_b.len()) as f64;
                        let action = if jaccard >= 0.8 || (jaccard >= 0.7 && len_ratio >= 0.6) {
                            "merge".to_string()
                        } else {
                            "review".to_string()
                        };
                        duplicates.push((name_a.to_string(), name_b.to_string(), jaccard, action));
                    }

                    // Also check containment (one name is subset of another)
                    let contained = intersection as f64 / words_a.len().min(words_b.len()) as f64;
                    if contained >= 0.9 && jaccard < 0.5 && words_a.len().min(words_b.len()) >= 2 {
                        duplicates.push((
                            name_a.to_string(),
                            name_b.to_string(),
                            contained * 0.7, // lower confidence for containment
                            "review-containment".to_string(),
                        ));
                    }
                }
            }
        }

        duplicates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        duplicates.truncate(100);
        Ok(duplicates)
    }

    /// Auto-merge fuzzy duplicates with high confidence.
    /// Keeps the shorter/cleaner name, merges the longer/noisier one into it.
    pub fn auto_merge_duplicates(&self, dupes: &[(String, String, f64, String)]) -> Result<usize> {
        let mut merged = 0usize;
        for (name_a, name_b, _sim, action) in dupes {
            if action != "merge" {
                continue;
            }
            let entity_a = self.brain.get_entity_by_name(name_a)?;
            let entity_b = self.brain.get_entity_by_name(name_b)?;
            if let (Some(a), Some(b)) = (entity_a, entity_b) {
                // Keep the shorter name (usually cleaner)
                let (keep, remove) = if a.name.len() <= b.name.len() {
                    (a, b)
                } else {
                    (b, a)
                };
                self.brain.merge_entities(remove.id, keep.id)?;
                merged += 1;
            }
        }
        Ok(merged)
    }

    /// Analyze topic coverage: group entities by source URL domain and measure
    /// how well each topic area is connected internally.
    pub fn topic_coverage_analysis(&self) -> Result<Vec<(String, usize, usize, f64, String)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        // Build entity→source mapping from relations
        let mut entity_sources: HashMap<i64, HashSet<String>> = HashMap::new();
        for r in &relations {
            if !r.source_url.is_empty() {
                // Extract domain from URL
                let domain = extract_domain(&r.source_url);
                entity_sources
                    .entry(r.subject_id)
                    .or_default()
                    .insert(domain.clone());
                entity_sources
                    .entry(r.object_id)
                    .or_default()
                    .insert(domain);
            }
        }

        // Also check source_url on entities themselves (from facts)
        let facts_sources: HashMap<i64, HashSet<String>> = {
            let mut m: HashMap<i64, HashSet<String>> = HashMap::new();
            for e in &entities {
                let facts = self.brain.get_facts_for(e.id)?;
                for f in &facts {
                    if !f.source_url.is_empty() {
                        let domain = extract_domain(&f.source_url);
                        m.entry(e.id).or_default().insert(domain);
                    }
                }
            }
            m
        };

        // Merge
        for (eid, sources) in facts_sources {
            entity_sources.entry(eid).or_default().extend(sources);
        }

        // Group entities by domain
        let mut domain_entities: HashMap<String, HashSet<i64>> = HashMap::new();
        for (eid, domains) in &entity_sources {
            for d in domains {
                domain_entities.entry(d.clone()).or_default().insert(*eid);
            }
        }

        // For each domain: count entities, count internal relations, compute density
        let meaningful = meaningful_ids(self.brain)?;
        let mut coverage: Vec<(String, usize, usize, f64, String)> = Vec::new();

        for (domain, eids) in &domain_entities {
            let meaningful_count = eids.iter().filter(|id| meaningful.contains(id)).count();
            if meaningful_count < 2 {
                continue;
            }

            // Count relations between entities in this domain
            let mut internal_rels = 0usize;
            for r in &relations {
                if eids.contains(&r.subject_id) && eids.contains(&r.object_id) {
                    internal_rels += 1;
                }
            }

            let max_possible = meaningful_count * (meaningful_count - 1) / 2;
            let density = if max_possible > 0 {
                internal_rels as f64 / max_possible as f64
            } else {
                0.0
            };

            let assessment = if density < 0.01 {
                "sparse — needs more crawling".to_string()
            } else if density < 0.1 {
                "moderate — some connections exist".to_string()
            } else {
                "well-connected".to_string()
            };

            coverage.push((
                domain.clone(),
                meaningful_count,
                internal_rels,
                density,
                assessment,
            ));
        }

        coverage.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by entity count
        Ok(coverage)
    }

    /// Graph evolution metrics: compare current state against prior discovery runs.
    /// Tracks growth rate, connectivity improvement, and hypothesis hit rate.
    pub fn evolution_metrics(&self) -> Result<HashMap<String, f64>> {
        let mut metrics: HashMap<String, f64> = HashMap::new();

        // Hypothesis success rate by pattern type
        let weights = self.get_pattern_weights()?;
        for w in &weights {
            metrics.insert(format!("hit_rate_{}", w.pattern_type), w.weight);
        }

        // Overall hypothesis stats
        let all_hyps = self.list_hypotheses(None)?;
        let total = all_hyps.len() as f64;
        let confirmed = all_hyps
            .iter()
            .filter(|h| h.status == HypothesisStatus::Confirmed)
            .count() as f64;
        let rejected = all_hyps
            .iter()
            .filter(|h| h.status == HypothesisStatus::Rejected)
            .count() as f64;

        if total > 0.0 {
            metrics.insert("hypothesis_confirmation_rate".into(), confirmed / total);
            metrics.insert("hypothesis_rejection_rate".into(), rejected / total);
        }

        // Island ratio (lower is better)
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }
        let meaningful = meaningful_ids(self.brain)?;
        let meaningful_islands = meaningful
            .iter()
            .filter(|id| !connected.contains(id))
            .count();
        if !meaningful.is_empty() {
            metrics.insert(
                "island_ratio".into(),
                meaningful_islands as f64 / meaningful.len() as f64,
            );
        }

        // Predicate diversity
        let (_, _, _, div_ratio) = self.predicate_diversity()?;
        metrics.insert("predicate_diversity".into(), div_ratio);

        // Entity count and relation count
        metrics.insert("total_entities".into(), entities.len() as f64);
        metrics.insert("total_relations".into(), relations.len() as f64);
        metrics.insert("meaningful_entities".into(), meaningful.len() as f64);

        Ok(metrics)
    }

    /// Find entities that are likely abbreviated forms of other entities.
    /// E.g., "Ada" ↔ "Ada Lovelace", "RISC-V" ↔ "RISC-V Foundation".
    /// Returns (short_name, full_name, entity_type, confidence).
    pub fn find_name_subsumptions(&self) -> Result<Vec<(String, String, String, f64)>> {
        let entities = self.brain.all_entities()?;
        let meaningful: Vec<&crate::db::Entity> = entities
            .iter()
            .filter(|e| {
                !is_noise_type(&e.entity_type) && !is_noise_name(&e.name) && e.name.len() >= 2
            })
            .collect();

        // Group by type for efficient comparison
        let mut by_type: HashMap<&str, Vec<&crate::db::Entity>> = HashMap::new();
        for e in &meaningful {
            by_type.entry(&e.entity_type).or_default().push(e);
        }

        let mut subsumptions = Vec::new();
        for group in by_type.values() {
            if group.len() > 3000 {
                continue; // skip huge groups
            }
            for i in 0..group.len() {
                let short = group[i];
                let short_lower = short.name.to_lowercase();
                let short_words: Vec<&str> = short_lower.split_whitespace().collect();
                if short_words.len() > 3 {
                    continue; // only look at short names as "abbreviations"
                }
                for j in 0..group.len() {
                    if i == j {
                        continue;
                    }
                    let full = group[j];
                    let full_lower = full.name.to_lowercase();
                    let full_words: Vec<&str> = full_lower.split_whitespace().collect();
                    // Short name must be shorter
                    if short_words.len() >= full_words.len() {
                        continue;
                    }
                    // Check if short name starts full name (e.g. "Ada" starts "Ada Lovelace")
                    if full_lower.starts_with(&short_lower)
                        && full_lower.len() > short_lower.len() + 1
                    {
                        let conf = 0.6 + (short_words.len() as f64 * 0.1).min(0.3);
                        subsumptions.push((
                            short.name.clone(),
                            full.name.clone(),
                            short.entity_type.clone(),
                            conf,
                        ));
                    }
                    // Also check suffix match: "Elwood Shannon" ends "Claude Elwood Shannon"
                    else if full_lower.ends_with(&short_lower)
                        && full_lower.len() > short_lower.len() + 1
                        && short_words.len() >= 2
                    {
                        let conf = 0.55 + (short_words.len() as f64 * 0.1).min(0.3);
                        subsumptions.push((
                            short.name.clone(),
                            full.name.clone(),
                            short.entity_type.clone(),
                            conf,
                        ));
                    }
                }
            }
        }
        subsumptions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        subsumptions.truncate(100);
        Ok(subsumptions)
    }

    /// Purge noise entities from the brain — entities matching noise filters
    /// that have zero or very few relations. Returns count of purged entities.
    pub fn purge_noise_entities(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        let mut purged = 0usize;
        for e in &entities {
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            // Purge isolated or near-isolated noise; also purge zero-degree entities
            // with very low confidence
            let is_noise = is_noise_name(&e.name) || is_noise_type(&e.entity_type);
            let is_low_value = deg == 0 && e.confidence < 0.3 && e.name.len() < 4;
            // Aggressive: purge isolated entities that look like citation fragments
            let is_citation_fragment = deg == 0 && looks_like_citation(&e.name);
            // For entities matching noise name patterns, purge up to degree 3
            // (many noise entities accumulate relations through automated discovery)
            let noise_threshold = if is_noise_name(&e.name) { 3 } else { 1 };
            if (deg <= noise_threshold && is_noise) || is_low_value || is_citation_fragment {
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                    conn.execute(
                        "DELETE FROM relations WHERE subject_id = ?1 OR object_id = ?1",
                        params![e.id],
                    )?;
                    Ok(())
                })?;
                purged += 1;
            }
        }
        Ok(purged)
    }

    /// Bulk quality cleanup: remove isolated entities that are clearly extraction noise.
    /// More aggressive than purge_noise_entities — targets patterns common in Wikipedia extraction.
    /// Returns count of entities removed.
    pub fn bulk_quality_cleanup(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        let mut removed = 0usize;
        for e in &entities {
            if connected.contains(&e.id) {
                continue; // keep all connected entities
            }
            let should_remove = is_extraction_noise(&e.name, &e.entity_type);
            if should_remove {
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                    Ok(())
                })?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Normalize generic predicates to more specific ones using context.
    /// E.g., "Person X is Concept Y" → "Person X instance_of Concept Y".
    /// Returns count of relations updated.
    pub fn normalize_predicates(&self) -> Result<usize> {
        let relations = self.brain.all_relations()?;
        let entities = self.brain.all_entities()?;
        let id_to_type: HashMap<i64, String> = entities
            .iter()
            .map(|e| (e.id, e.entity_type.clone()))
            .collect();

        let mut updated = 0usize;
        for r in &relations {
            if !is_generic_predicate(&r.predicate) {
                continue;
            }
            let subj_type = id_to_type
                .get(&r.subject_id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let obj_type = id_to_type
                .get(&r.object_id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");

            let new_pred = match (r.predicate.as_str(), subj_type, obj_type) {
                ("is", "person", "concept") => Some("instance_of"),
                ("is", "person", "organization") => Some("member_of"),
                ("is", "organization", "concept") => Some("classified_as"),
                ("is", "place", "concept") => Some("classified_as"),
                ("is", "concept", "concept") => Some("subclass_of"),
                ("is", _, "place") => Some("located_in"),
                ("is", "technology", "concept") => Some("classified_as"),
                ("is", "product", "concept") => Some("classified_as"),
                ("has", "person", "concept") => Some("possesses"),
                ("has", "organization", "concept") => Some("features"),
                ("has", "technology", "concept") => Some("features"),
                ("has", "place", "concept") => Some("features"),
                ("was", "person", "concept") => Some("formerly"),
                ("was", "person", "organization") => Some("formerly_at"),
                ("was", "organization", "concept") => Some("formerly"),
                ("are", "concept", "concept") => Some("subclass_of"),
                ("are", _, "concept") => Some("classified_as"),
                ("were", "person", "organization") => Some("formerly_at"),
                ("were", _, "concept") => Some("formerly"),
                ("had", "person", "concept") => Some("formerly_possessed"),
                ("had", "organization", "concept") => Some("formerly_had"),
                ("do", _, _) | ("does", _, _) | ("did", _, _) => Some("performs"),
                _ => None,
            };

            if let Some(np) = new_pred {
                self.brain.with_conn(|conn| {
                    conn.execute(
                        "UPDATE relations SET predicate = ?1 WHERE subject_id = ?2 AND object_id = ?3 AND predicate = ?4",
                        params![np, r.subject_id, r.object_id, r.predicate],
                    )?;
                    Ok(())
                })?;
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Reconnect island entities to the main graph using exact name matching.
    /// If an island entity's name appears as a substring of a connected entity
    /// (or vice versa), create a "same_as" or "related_to" relation.
    /// Returns count of new connections made.
    pub fn reconnect_islands(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build lookup: lowercase name → connected entity id
        let mut name_to_connected: HashMap<String, i64> = HashMap::new();
        for e in &entities {
            if connected.contains(&e.id)
                && !is_noise_type(&e.entity_type)
                && !is_noise_name(&e.name)
            {
                name_to_connected.insert(e.name.to_lowercase(), e.id);
            }
        }

        // Find islands with exact name matches to connected entities
        let mut reconnected = 0usize;
        for e in &entities {
            if connected.contains(&e.id) || is_noise_type(&e.entity_type) || is_noise_name(&e.name)
            {
                continue;
            }
            let lower = e.name.to_lowercase();
            // Exact match with a connected entity (different ID, same name)
            if let Some(&hub_id) = name_to_connected.get(&lower) {
                if hub_id != e.id {
                    // Merge: island → hub
                    self.brain.merge_entities(e.id, hub_id)?;
                    reconnected += 1;
                    continue;
                }
            }
            // Containment match: if island name is a prefix of a connected entity's name
            // and they share the same type, merge (e.g. "Ada" → "Ada Lovelace" when both are person)
            if e.name.len() >= 4 && !e.entity_type.is_empty() && e.entity_type != "unknown" {
                let mut best_match: Option<(i64, usize)> = None;
                for (cname, &cid) in &name_to_connected {
                    if cname.starts_with(&lower) && cname.len() > lower.len() + 1 {
                        // Check same type
                        if let Some(ce) = self.brain.get_entity_by_id(cid)? {
                            if ce.entity_type == e.entity_type {
                                let score = lower.len();
                                if best_match.is_none() || score > best_match.unwrap().1 {
                                    best_match = Some((cid, score));
                                }
                            }
                        }
                    }
                }
                if let Some((hub_id, _)) = best_match {
                    self.brain.merge_entities(e.id, hub_id)?;
                    reconnected += 1;
                }
            }
        }
        Ok(reconnected)
    }

    /// Compute entity importance scores combining PageRank, degree, and type value.
    /// Returns top entities sorted by composite score.
    pub fn entity_importance(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String, f64, usize, f64)>> {
        let pr = crate::graph::pagerank(self.brain, 0.85, 20)?;
        let relations = self.brain.all_relations()?;
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        let entities = self.brain.all_entities()?;
        let mut scores: Vec<(String, String, f64, usize, f64)> = Vec::new();

        for e in &entities {
            if is_noise_type(&e.entity_type) || is_noise_name(&e.name) {
                continue;
            }
            let rank = pr.get(&e.id).copied().unwrap_or(0.0);
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            let type_boost = if HIGH_VALUE_TYPES.contains(&e.entity_type.as_str()) {
                1.5
            } else {
                1.0
            };
            let composite = (rank * 10000.0 + deg as f64) * type_boost;
            if composite > 0.0 {
                scores.push((e.name.clone(), e.entity_type.clone(), rank, deg, composite));
            }
        }

        scores.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        Ok(scores)
    }

    /// Find entity type distribution anomalies — types that are over/under-represented
    /// relative to their connectivity, suggesting systematic extraction bias.
    pub fn type_distribution_analysis(&self) -> Result<Vec<(String, usize, f64, String)>> {
        let entities = self.brain.all_entities()?;
        let meaningful = meaningful_ids(self.brain)?;
        let relations = self.brain.all_relations()?;

        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut type_connected: HashMap<String, usize> = HashMap::new();
        let mut connected_ids: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected_ids.insert(r.subject_id);
            connected_ids.insert(r.object_id);
        }

        for e in &entities {
            if meaningful.contains(&e.id) {
                *type_counts.entry(e.entity_type.clone()).or_insert(0) += 1;
                if connected_ids.contains(&e.id) {
                    *type_connected.entry(e.entity_type.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut analysis = Vec::new();
        for (etype, count) in &type_counts {
            if *count < 5 {
                continue;
            }
            let connected = type_connected.get(etype).copied().unwrap_or(0);
            let connectivity = connected as f64 / *count as f64;
            let assessment = if connectivity < 0.05 {
                format!(
                    "EXTRACTION NOISE: {}/{} connected ({:.0}%) — likely over-extracted",
                    connected,
                    count,
                    connectivity * 100.0
                )
            } else if connectivity < 0.2 {
                format!(
                    "SPARSE: {}/{} connected ({:.0}%) — needs enrichment",
                    connected,
                    count,
                    connectivity * 100.0
                )
            } else if connectivity > 0.8 {
                format!(
                    "HEALTHY: {}/{} connected ({:.0}%)",
                    connected,
                    count,
                    connectivity * 100.0
                )
            } else {
                format!(
                    "MODERATE: {}/{} connected ({:.0}%)",
                    connected,
                    count,
                    connectivity * 100.0
                )
            };
            analysis.push((etype.clone(), *count, connectivity, assessment));
        }
        analysis.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(analysis)
    }

    /// Fix misclassified entity types based on name heuristics.
    /// E.g., "Middle East" as person → place, "Deep Neural Networks" as organization → concept.
    /// Returns count of entities re-typed.
    pub fn fix_entity_types(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let mut fixed = 0usize;

        for e in &entities {
            let lower = e.name.to_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();
            let new_type = detect_correct_type(&lower, &words, &e.entity_type);
            if let Some(nt) = new_type {
                self.brain.with_conn(|conn| {
                    conn.execute(
                        "UPDATE entities SET entity_type = ?1 WHERE id = ?2",
                        params![nt, e.id],
                    )?;
                    Ok(())
                })?;
                fixed += 1;
            }
        }
        Ok(fixed)
    }

    /// Infer entity types from neighborhood: if an "unknown" entity is connected
    /// primarily to entities of a known type via specific predicates, we can infer its type.
    /// E.g., if X →born_in→ Y and Y is "unknown" but connected to persons, Y is likely a place.
    /// Returns count of types inferred.
    pub fn infer_types_from_neighborhood(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let id_to_type: HashMap<i64, String> = entities
            .iter()
            .map(|e| (e.id, e.entity_type.clone()))
            .collect();

        // Build per-entity: {neighbor_type → count} and {predicate_role → count}
        // predicate_role = "object_of:<pred>" or "subject_of:<pred>"
        let mut neighbor_types: HashMap<i64, HashMap<String, usize>> = HashMap::new();
        let mut pred_roles: HashMap<i64, HashMap<String, usize>> = HashMap::new();
        for r in &relations {
            if let Some(obj_type) = id_to_type.get(&r.object_id) {
                if obj_type != "unknown" {
                    *neighbor_types
                        .entry(r.subject_id)
                        .or_default()
                        .entry(obj_type.clone())
                        .or_insert(0) += 1;
                }
            }
            if let Some(subj_type) = id_to_type.get(&r.subject_id) {
                if subj_type != "unknown" {
                    *neighbor_types
                        .entry(r.object_id)
                        .or_default()
                        .entry(subj_type.clone())
                        .or_insert(0) += 1;
                }
            }
            *pred_roles
                .entry(r.subject_id)
                .or_default()
                .entry(format!("subj:{}", r.predicate))
                .or_insert(0) += 1;
            *pred_roles
                .entry(r.object_id)
                .or_default()
                .entry(format!("obj:{}", r.predicate))
                .or_insert(0) += 1;
        }

        // Predicate-based type inference rules
        let pred_type_rules: &[(&str, &str)] = &[
            ("obj:born_in", "place"),
            ("obj:located_in", "place"),
            ("obj:headquartered_in", "place"),
            ("obj:died_in", "place"),
            ("obj:capital_of", "place"),
            ("subj:born_in", "person"),
            ("subj:died_in", "person"),
            ("subj:authored", "person"),
            ("subj:invented", "person"),
            ("subj:discovered", "person"),
            ("subj:founded", "person"),
            ("obj:founded_by", "person"),
            ("obj:invented_by", "person"),
            ("obj:discovered_by", "person"),
            ("obj:developed_by", "person"),
            ("subj:subsidiary_of", "organization"),
            ("subj:member_of", "person"),
            ("obj:member_of", "organization"),
            ("subj:employs", "organization"),
            ("obj:instance_of", "concept"),
        ];

        let mut inferred = 0usize;
        for e in &entities {
            if e.entity_type != "unknown" {
                continue;
            }
            // Try predicate-based rules first
            let mut inferred_type: Option<&str> = None;
            if let Some(roles) = pred_roles.get(&e.id) {
                for &(role_pattern, target_type) in pred_type_rules {
                    if roles.get(role_pattern).copied().unwrap_or(0) >= 1 {
                        inferred_type = Some(target_type);
                        break;
                    }
                }
            }

            // Fall back to majority neighbor type if ≥3 neighbors of same type
            if inferred_type.is_none() {
                if let Some(ntypes) = neighbor_types.get(&e.id) {
                    if let Some((best_type, &count)) = ntypes.iter().max_by_key(|(_, &c)| c) {
                        if count >= 3 && !is_noise_type(best_type) {
                            // If mostly connected to persons, this might be a place/org/concept
                            // (persons cluster around shared contexts)
                            if best_type == "person" {
                                inferred_type = Some("concept"); // conservative
                            } else {
                                // Adopt the dominant neighbor type
                                inferred_type = Some(Box::leak(best_type.clone().into_boxed_str()));
                            }
                        }
                    }
                }
            }

            if let Some(new_type) = inferred_type {
                self.brain.with_conn(|conn| {
                    conn.execute(
                        "UPDATE entities SET entity_type = ?1 WHERE id = ?2",
                        params![new_type, e.id],
                    )?;
                    Ok(())
                })?;
                inferred += 1;
            }
        }
        Ok(inferred)
    }

    /// Compute entity quality scores: higher score = more valuable for discovery.
    /// Score based on: degree, type quality, name quality, fact count, community membership.
    /// Returns (entity_id, score) sorted descending.
    pub fn entity_quality_scores(&self, limit: usize) -> Result<Vec<(i64, String, f64)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        let mut scores: Vec<(i64, String, f64)> = Vec::new();
        for e in &entities {
            let mut score = 0.0_f64;
            let deg = degree.get(&e.id).copied().unwrap_or(0);

            // Degree contribution (log scale to avoid hub domination)
            score += (1.0 + deg as f64).ln() * 0.3;

            // Type quality
            if HIGH_VALUE_TYPES.contains(&e.entity_type.as_str()) {
                score += 0.3;
            } else if e.entity_type == "unknown" {
                score -= 0.2;
            } else if is_noise_type(&e.entity_type) {
                score -= 0.5;
            }

            // Name quality
            if is_noise_name(&e.name) {
                score -= 1.0;
            }
            if e.name.contains(' ') && e.name.len() >= 5 && e.name.len() <= 40 {
                score += 0.2; // multi-word, reasonable length
            }

            // Confidence
            score += e.confidence * 0.2;

            // Isolated penalty
            if deg == 0 {
                score -= 0.3;
            }

            scores.push((e.id, e.name.clone(), score));
        }
        scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        Ok(scores)
    }

    /// Deep island cleanup: remove isolated entities that provide no discovery value.
    /// More aggressive than bulk_quality_cleanup — removes ALL isolated entities
    /// with confidence below a threshold and no facts stored.
    /// Returns count removed.
    pub fn deep_island_cleanup(&self, min_confidence: f64) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        let mut removed = 0usize;
        for e in &entities {
            if connected.contains(&e.id) {
                continue;
            }
            // Keep high-confidence entities and those with stored facts
            if e.confidence >= min_confidence {
                continue;
            }
            let facts = self.brain.get_facts_for(e.id)?;
            if !facts.is_empty() {
                continue;
            }
            // Keep entities with well-known names (proper nouns, 2-3 word person names)
            if is_likely_real_entity(&e.name, &e.entity_type) {
                continue;
            }
            self.brain.with_conn(|conn| {
                conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                Ok(())
            })?;
            removed += 1;
        }
        Ok(removed)
    }

    pub fn report_markdown(&self, report: &DiscoveryReport) -> String {
        let mut md = String::new();
        md.push_str("# 🔬 PROMETHEUS Discovery Report\n\n");
        md.push_str(&format!("**Summary:** {}\n\n", report.summary));

        if !report.patterns_found.is_empty() {
            md.push_str("## Patterns Found\n\n");
            for p in &report.patterns_found {
                md.push_str(&format!(
                    "- **{}** (freq: {}): {}\n",
                    p.pattern_type.as_str(),
                    p.frequency,
                    p.description
                ));
            }
            md.push('\n');
        }

        if !report.hypotheses_generated.is_empty() {
            md.push_str("## Hypotheses Generated\n\n");
            for h in &report.hypotheses_generated {
                md.push_str(&format!(
                    "- [{:.2}] {} {} {} — *{}*\n",
                    h.confidence,
                    h.subject,
                    h.predicate,
                    h.object,
                    h.status.as_str()
                ));
            }
        }

        md
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Generate hypotheses from hub entities: if a high-degree entity of type T exists,
    /// suggest it should connect to nearby isolated entities of the same type or related types.
    pub fn generate_hypotheses_from_hubs(&self) -> Result<Vec<Hypothesis>> {
        let hubs = crate::graph::find_hubs(self.brain, 10)?;
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build adjacency for quick lookup
        let mut adj: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            adj.insert((r.subject_id, r.object_id));
            adj.insert((r.object_id, r.subject_id));
        }

        let id_to_entity: HashMap<i64, &crate::db::Entity> =
            entities.iter().map(|e| (e.id, e)).collect();

        let mut hypotheses = Vec::new();
        for (hub_id, _degree) in &hubs {
            let hub = match id_to_entity.get(hub_id) {
                Some(e) => e,
                None => continue,
            };
            if is_noise_type(&hub.entity_type) || is_noise_name(&hub.name) {
                continue;
            }

            // Find islands of the same type that might relate to this hub
            for e in &entities {
                if e.id == *hub_id
                    || !meaningful.contains(&e.id)
                    || connected.contains(&e.id)
                    || is_noise_name(&e.name)
                {
                    continue;
                }
                // Same type AND name word overlap (at least one shared word ≥4 chars)
                if e.entity_type == hub.entity_type {
                    let hub_words: HashSet<String> = hub
                        .name
                        .to_lowercase()
                        .split_whitespace()
                        .filter(|w| w.len() >= 4)
                        .map(|w| w.to_string())
                        .collect();
                    let e_words: HashSet<String> = e
                        .name
                        .to_lowercase()
                        .split_whitespace()
                        .filter(|w| w.len() >= 4)
                        .map(|w| w.to_string())
                        .collect();
                    let shared: Vec<&String> = hub_words.intersection(&e_words).collect();
                    if !shared.is_empty() {
                        hypotheses.push(Hypothesis {
                            id: 0,
                            subject: hub.name.clone(),
                            predicate: "related_to".to_string(),
                            object: e.name.clone(),
                            confidence: 0.3 + (shared.len() as f64 * 0.1).min(0.3),
                            evidence_for: vec![format!(
                                "Hub '{}' shares words [{}] with isolated '{}' (type '{}')",
                                hub.name,
                                shared
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", "),
                                e.name,
                                e.entity_type
                            )],
                            evidence_against: vec![],
                            reasoning_chain: vec![
                                format!(
                                    "'{}' is a hub entity of type '{}'",
                                    hub.name, hub.entity_type
                                ),
                                format!("'{}' is isolated but shares words with hub", e.name),
                                "Name overlap + hub-spoke pattern suggests likely relation"
                                    .to_string(),
                            ],
                            status: HypothesisStatus::Proposed,
                            discovered_at: now_str(),
                            pattern_source: "hub_spoke".to_string(),
                        });
                        if hypotheses.len() >= 100 {
                            return Ok(hypotheses);
                        }
                    }
                }
            }
        }
        Ok(hypotheses)
    }

    /// Fact-based relation inference: create relations from facts that reference other entities.
    /// E.g., entity "France" has fact "capital: Paris" and entity "Paris" exists → create relation.
    /// Returns count of new relations created.
    pub fn infer_relations_from_facts(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Build name→id lookup (case-insensitive)
        let mut name_to_id: HashMap<String, i64> = HashMap::new();
        for e in &entities {
            if meaningful.contains(&e.id) {
                name_to_id.insert(e.name.to_lowercase(), e.id);
            }
        }

        // Build existing relation set for dedup
        let relations = self.brain.all_relations()?;
        let mut existing: HashSet<(i64, String, i64)> = HashSet::new();
        for r in &relations {
            existing.insert((r.subject_id, r.predicate.clone(), r.object_id));
        }

        // Predicate mapping: fact key → relation predicate
        let key_to_pred: HashMap<&str, &str> = [
            ("capital", "has_capital"),
            ("headquarters", "headquartered_in"),
            ("founded_by", "founded_by"),
            ("creator", "created_by"),
            ("author", "authored_by"),
            ("inventor", "invented_by"),
            ("born_in", "born_in"),
            ("died_in", "died_in"),
            ("located_in", "located_in"),
            ("part_of", "part_of"),
            ("member_of", "member_of"),
            ("parent", "child_of"),
            ("subsidiary", "subsidiary_of"),
            ("language", "uses_language"),
            ("currency", "uses_currency"),
            ("president", "led_by"),
            ("ceo", "led_by"),
            ("founder", "founded_by"),
            ("nationality", "nationality"),
            ("country", "located_in"),
            ("continent", "located_in"),
            ("region", "located_in"),
            ("city", "located_in"),
        ]
        .iter()
        .cloned()
        .collect();

        let mut created = 0usize;
        for e in &entities {
            if !meaningful.contains(&e.id) {
                continue;
            }
            let facts = self.brain.get_facts_for(e.id)?;
            for f in &facts {
                let pred = key_to_pred
                    .get(f.key.to_lowercase().as_str())
                    .copied()
                    .unwrap_or("related_to");

                // Check if fact value matches an entity name
                let value_lower = f.value.to_lowercase();
                if let Some(&target_id) = name_to_id.get(&value_lower) {
                    if target_id != e.id && !existing.contains(&(e.id, pred.to_string(), target_id))
                    {
                        self.brain
                            .upsert_relation(e.id, pred, target_id, &f.source_url)?;
                        existing.insert((e.id, pred.to_string(), target_id));
                        created += 1;
                    }
                }
            }
        }
        Ok(created)
    }

    /// Entity name cross-reference: find entities whose names contain other entity names.
    /// E.g., "Swiss Federal Council" contains "Swiss" → link to Switzerland if it exists.
    /// Only matches high-value entities with names ≥ 5 chars to avoid false positives.
    /// Returns count of new relations created.
    pub fn crossref_entity_names(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Build candidate targets: meaningful entities with short-ish names (2-4 words)
        let targets: Vec<(i64, String)> = entities
            .iter()
            .filter(|e| {
                meaningful.contains(&e.id)
                    && !is_noise_name(&e.name)
                    && e.name.len() >= 5
                    && e.name.split_whitespace().count() <= 3
                    && HIGH_VALUE_TYPES.contains(&e.entity_type.as_str())
            })
            .map(|e| (e.id, e.name.clone()))
            .collect();

        // Build existing relation set
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            connected.insert(key);
        }

        let mut created = 0usize;
        for e in &entities {
            if !meaningful.contains(&e.id) || is_noise_name(&e.name) {
                continue;
            }
            let e_lower = e.name.to_lowercase();
            let e_words: Vec<&str> = e_lower.split_whitespace().collect();
            if e_words.len() < 2 {
                continue; // only multi-word entities can contain references
            }

            for (tid, tname) in &targets {
                if *tid == e.id {
                    continue;
                }
                let t_lower = tname.to_lowercase();
                // Check if entity name contains the target name as a word boundary match
                if e_lower.contains(&t_lower) && e_lower != t_lower {
                    // Verify word boundary (not just substring within a word)
                    let idx = e_lower.find(&t_lower).unwrap();
                    let before_ok = idx == 0
                        || e_lower.as_bytes()[idx - 1] == b' '
                        || e_lower.as_bytes()[idx - 1] == b'-';
                    let end = idx + t_lower.len();
                    let after_ok = end == e_lower.len()
                        || e_lower.as_bytes()[end] == b' '
                        || e_lower.as_bytes()[end] == b'-';

                    if before_ok && after_ok {
                        let key = if e.id < *tid {
                            (e.id, *tid)
                        } else {
                            (*tid, e.id)
                        };
                        if !connected.contains(&key) {
                            self.brain
                                .upsert_relation(e.id, "associated_with", *tid, "")?;
                            connected.insert(key);
                            created += 1;
                            if created >= 200 {
                                return Ok(created);
                            }
                        }
                    }
                }
            }
        }
        Ok(created)
    }

    /// Decompose compound entity names into base entity + qualifier.
    /// E.g., "Ada Lovelace Building" → relates to existing "Ada Lovelace" entity
    /// with predicate "named_after". Also handles patterns like:
    /// - "X Award" → X + Award concept
    /// - "X Institute" → X + organization  
    /// - "X Day" → X + event
    /// Returns count of new relations created and entities cleaned up.
    pub fn decompose_compound_entities(&self) -> Result<(usize, usize)> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build name→id lookup for connected entities (these are the "real" entities)
        let mut name_to_id: HashMap<String, i64> = HashMap::new();
        for e in &entities {
            if connected.contains(&e.id)
                && !is_noise_name(&e.name)
                && !is_noise_type(&e.entity_type)
            {
                name_to_id.insert(e.name.to_lowercase(), e.id);
            }
        }
        // Also include high-confidence islands
        for e in &entities {
            if !connected.contains(&e.id) && e.confidence >= 0.8 && !is_noise_name(&e.name) {
                name_to_id.entry(e.name.to_lowercase()).or_insert(e.id);
            }
        }

        // Qualifier suffixes that indicate compound entities
        let qualifier_map: &[(&str, &str)] = &[
            ("building", "named_after"),
            ("house", "named_after"),
            ("suite", "named_after"),
            ("center", "named_after"),
            ("centre", "named_after"),
            ("institute", "named_after"),
            ("foundation", "named_after"),
            ("award", "named_after"),
            ("prize", "named_after"),
            ("day", "named_after"),
            ("biography", "subject_of"),
            ("original", "variant_of"),
            ("countess", "title_of"),
            ("theorem", "attributed_to"),
            ("equation", "attributed_to"),
            ("principle", "attributed_to"),
            ("law", "attributed_to"),
            ("conjecture", "attributed_to"),
            ("paradox", "attributed_to"),
            ("constant", "attributed_to"),
            ("transform", "attributed_to"),
            ("distribution", "attributed_to"),
            ("method", "attributed_to"),
            ("algorithm", "attributed_to"),
            ("bridge", "named_after"),
            ("tower", "named_after"),
            ("park", "named_after"),
            ("university", "named_after"),
            ("college", "named_after"),
            ("school", "named_after"),
            ("museum", "named_after"),
            ("library", "named_after"),
            ("hospital", "named_after"),
            ("airport", "named_after"),
            ("station", "named_after"),
            ("square", "named_after"),
            ("street", "named_after"),
        ];

        let mut relations_created = 0usize;
        let mut entities_merged = 0usize;
        let mut existing_rels: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            existing_rels.insert((r.subject_id, r.object_id));
        }

        for e in &entities {
            if is_noise_name(&e.name) || is_noise_type(&e.entity_type) {
                continue;
            }
            let lower = e.name.to_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();
            if words.len() < 2 {
                continue;
            }

            // Check if the last word is a qualifier
            let last = *words.last().unwrap();
            let qualifier_pred = qualifier_map
                .iter()
                .find(|(q, _)| *q == last)
                .map(|(_, p)| *p);

            if let Some(pred) = qualifier_pred {
                // Try progressively shorter prefixes to find a matching base entity
                for prefix_len in (1..words.len()).rev() {
                    let prefix: String = words[..prefix_len].join(" ");
                    if let Some(&base_id) = name_to_id.get(&prefix) {
                        if base_id != e.id && !existing_rels.contains(&(e.id, base_id)) {
                            self.brain.upsert_relation(e.id, pred, base_id, "")?;
                            existing_rels.insert((e.id, base_id));
                            relations_created += 1;
                        }
                        break;
                    }
                }
            }

            // Also handle "X Y Z" where "Y Z" is a known entity (e.g., "Babbage Ada Lovelace")
            if words.len() >= 3 {
                for split in 1..words.len() {
                    let suffix: String = words[split..].join(" ");
                    if let Some(&base_id) = name_to_id.get(&suffix) {
                        if base_id != e.id && !existing_rels.contains(&(e.id, base_id)) {
                            self.brain
                                .upsert_relation(e.id, "references", base_id, "")?;
                            existing_rels.insert((e.id, base_id));
                            relations_created += 1;
                            break;
                        }
                    }
                }
            }
        }

        // Merge truly redundant island entities: if an island "X Qualifier" has the same
        // type as base entity "X" and the qualifier is just noise, merge into X
        let noise_qualifiers: HashSet<&str> =
            ["original", "wired", "founder"].iter().copied().collect();
        let entities_refreshed = self.brain.all_entities()?;
        for e in &entities_refreshed {
            if connected.contains(&e.id) {
                continue;
            }
            let lower = e.name.to_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();
            if words.len() < 2 {
                continue;
            }
            let last = *words.last().unwrap();
            if !noise_qualifiers.contains(last) {
                continue;
            }
            let prefix: String = words[..words.len() - 1].join(" ");
            if let Some(&base_id) = name_to_id.get(&prefix) {
                if base_id != e.id {
                    self.brain.merge_entities(e.id, base_id)?;
                    entities_merged += 1;
                }
            }
        }

        Ok((relations_created, entities_merged))
    }

    /// Aggressive island consolidation: for each island entity, check if its name
    /// is an exact prefix match of a connected entity (same type). If so, merge.
    /// Handles cases like "Euler" (island) → "Leonhard Euler" (connected).
    /// Returns count of merges.
    pub fn consolidate_prefix_islands(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build connected entity names (all types in one flat list for cross-type matching)
        let mut all_connected_names: Vec<(i64, String, String)> = Vec::new();
        for e in &entities {
            if connected.contains(&e.id)
                && !is_noise_name(&e.name)
                && !is_noise_type(&e.entity_type)
            {
                all_connected_names.push((e.id, e.name.to_lowercase(), e.entity_type.clone()));
            }
        }

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        // Compatible type pairs for cross-type merging
        let compatible_types = |a: &str, b: &str| -> bool {
            if a == b {
                return true;
            }
            matches!(
                (a, b),
                ("person", "concept")
                    | ("concept", "person")
                    | ("person", "organization")
                    | ("organization", "person")
                    | ("place", "concept")
                    | ("concept", "place")
                    | ("technology", "concept")
                    | ("concept", "technology")
                    | ("organization", "concept")
                    | ("concept", "organization")
            )
        };

        for e in &entities {
            if connected.contains(&e.id) || absorbed.contains(&e.id) {
                continue;
            }
            if is_noise_name(&e.name) || is_noise_type(&e.entity_type) || e.name.len() < 4 {
                continue;
            }
            let lower = e.name.to_lowercase();

            let mut best: Option<(i64, usize, bool)> = None; // (id, len, same_type)
            for (cid, cname, ctype) in &all_connected_names {
                if *cid == e.id {
                    continue;
                }
                // Island name is prefix of connected entity
                if cname.starts_with(&lower) && cname.len() > lower.len() {
                    let next_char = cname.as_bytes()[lower.len()];
                    if next_char == b' ' || next_char == b'-' {
                        let same_type = ctype == &e.entity_type;
                        if !same_type && !compatible_types(&e.entity_type, ctype) {
                            continue;
                        }
                        // Prefer same-type matches, then shorter names
                        let dominated = best.is_some_and(|(_, blen, bsame)| {
                            (same_type && !bsame) || (same_type == bsame && cname.len() < blen)
                        });
                        if best.is_none() || dominated {
                            best = Some((*cid, cname.len(), same_type));
                        }
                    }
                }
            }
            if let Some((target_id, _, _)) = best {
                self.brain.merge_entities(e.id, target_id)?;
                absorbed.insert(e.id);
                merged += 1;
            }
        }
        Ok(merged)
    }

    /// Aggressive island consolidation via word-overlap matching.
    /// For each isolated entity, find the best-matching connected entity of compatible type
    /// using Jaccard word similarity. If similarity ≥ threshold, merge.
    /// This handles cases like "Marie Curie Avenue" → merge into "Marie Curie".
    /// Returns count of merges performed.
    pub fn word_overlap_island_merge(&self, min_jaccard: f64) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build word sets for connected entities, grouped by type
        struct ConnectedInfo {
            id: i64,
            words: HashSet<String>,
            word_count: usize,
        }
        let mut connected_by_type: HashMap<String, Vec<ConnectedInfo>> = HashMap::new();
        for e in &entities {
            if !connected.contains(&e.id) || is_noise_name(&e.name) || is_noise_type(&e.entity_type)
            {
                continue;
            }
            let words: HashSet<String> = e
                .name
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() >= 3)
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| !w.is_empty())
                .collect();
            if words.is_empty() {
                continue;
            }
            let wc = words.len();
            connected_by_type
                .entry(e.entity_type.clone())
                .or_default()
                .push(ConnectedInfo {
                    id: e.id,
                    words,
                    word_count: wc,
                });
        }
        // Also allow cross-type matching for person→concept, person→place, etc.
        // Build a flat list for cross-type fallback
        let all_connected: Vec<&ConnectedInfo> =
            connected_by_type.values().flat_map(|v| v.iter()).collect();

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        for e in &entities {
            if connected.contains(&e.id) || absorbed.contains(&e.id) {
                continue;
            }
            if is_noise_name(&e.name) || is_noise_type(&e.entity_type) || e.name.len() < 4 {
                continue;
            }
            let island_words: HashSet<String> = e
                .name
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() >= 3)
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| !w.is_empty())
                .collect();
            if island_words.is_empty() {
                continue;
            }

            // First try same-type matches
            let candidates = connected_by_type.get(&e.entity_type);
            let mut best: Option<(i64, f64)> = None;

            if let Some(cands) = candidates {
                for c in cands {
                    let intersection = island_words.intersection(&c.words).count();
                    if intersection == 0 {
                        continue;
                    }
                    let union = island_words.union(&c.words).count();
                    let jaccard = intersection as f64 / union as f64;
                    // Also check containment: if island words are a superset/subset of connected
                    let containment =
                        intersection as f64 / island_words.len().min(c.word_count) as f64;
                    let score = jaccard.max(containment * 0.9); // containment slightly discounted
                    if score >= min_jaccard && (best.is_none() || score > best.unwrap().1) {
                        best = Some((c.id, score));
                    }
                }
            }

            // Cross-type fallback: only if island words fully contained in a connected entity
            if best.is_none() && island_words.len() >= 2 {
                for c in &all_connected {
                    let intersection = island_words.intersection(&c.words).count();
                    let containment = intersection as f64 / island_words.len() as f64;
                    if containment >= 0.95 && intersection >= 2 {
                        let jaccard =
                            intersection as f64 / island_words.union(&c.words).count() as f64;
                        if jaccard >= min_jaccard * 0.8 {
                            best = Some((c.id, jaccard));
                            break;
                        }
                    }
                }
            }

            if let Some((target_id, _score)) = best {
                if target_id != e.id {
                    self.brain.merge_entities(e.id, target_id)?;
                    absorbed.insert(e.id);
                    merged += 1;
                }
            }
        }
        Ok(merged)
    }

    /// Purge single-word island entities that are clearly fragments of known multi-word
    /// connected entities. E.g., "Lovelace" when "Ada Lovelace" exists connected,
    /// "Hopper" when "Grace Hopper" exists connected.
    /// Only purges when the single word appears in ≥2 connected entity names of its type.
    /// Returns count of entities removed.
    pub fn purge_fragment_islands(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build index: for each word, which connected entities contain it?
        let mut word_to_connected: HashMap<String, Vec<i64>> = HashMap::new();
        for e in &entities {
            if connected.contains(&e.id)
                && !is_noise_type(&e.entity_type)
                && !is_noise_name(&e.name)
                && e.name.contains(' ')
            {
                for word in e.name.split_whitespace() {
                    let lower = word.to_lowercase();
                    if lower.len() >= 3 {
                        word_to_connected.entry(lower).or_default().push(e.id);
                    }
                }
            }
        }

        let mut removed = 0usize;
        for e in &entities {
            if connected.contains(&e.id) {
                continue;
            }
            // Only target single-word islands
            if e.name.contains(' ') || e.name.len() < 3 {
                continue;
            }
            if is_noise_type(&e.entity_type) {
                continue;
            }
            let lower = e.name.to_lowercase();
            // Check if this word appears as a component of connected entities
            if let Some(matches) = word_to_connected.get(&lower) {
                if !matches.is_empty() {
                    // This single word is a fragment of at least one known entity
                    // Check if it has any facts worth keeping
                    let facts = self.brain.get_facts_for(e.id)?;
                    if facts.is_empty() {
                        // If there's exactly one match, merge into it
                        if matches.len() == 1 {
                            self.brain.merge_entities(e.id, matches[0])?;
                        } else {
                            // Multiple matches — just delete the fragment
                            self.brain.with_conn(|conn| {
                                conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                                Ok(())
                            })?;
                        }
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }

    /// Suffix-strip island merge: for island entities like "Ada Lovelace WIRED",
    /// strip trailing words one at a time and check if a connected entity exists with
    /// the shorter name. If so, merge. This handles NLP extractors that append context
    /// words to entity names.
    /// Returns count of merges performed.
    pub fn suffix_strip_island_merge(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build name→id index for connected entities (case-insensitive)
        let mut name_to_id: HashMap<String, i64> = HashMap::new();
        for e in &entities {
            if connected.contains(&e.id) && !is_noise_name(&e.name) {
                name_to_id.insert(e.name.to_lowercase(), e.id);
            }
        }

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        for e in &entities {
            if connected.contains(&e.id) || absorbed.contains(&e.id) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 2 {
                continue;
            }
            // Try stripping 1, 2, (up to half) trailing words
            let max_strip = (words.len() / 2).max(1).min(3);
            for strip in 1..=max_strip {
                if words.len() <= strip {
                    break;
                }
                let prefix: String = words[..words.len() - strip].join(" ");
                if prefix.len() < 3 {
                    break;
                }
                let lower_prefix = prefix.to_lowercase();
                if let Some(&target_id) = name_to_id.get(&lower_prefix) {
                    if target_id != e.id {
                        self.brain.merge_entities(e.id, target_id)?;
                        absorbed.insert(e.id);
                        merged += 1;
                        break;
                    }
                }
            }
        }
        Ok(merged)
    }

    /// Prefix-strip merge: for entities like "Christy Grace Hopper" or "Admiral Grace Hopper",
    /// strip leading words one at a time and check if a shorter-named entity exists.
    /// If so, merge the longer name into the shorter one. Works on both islands and
    /// connected entities (merges redundant variants into the canonical shorter form).
    /// Also strips trailing words as a second pass.
    /// Returns count of merges performed.
    pub fn prefix_strip_island_merge(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        // Build degree map for deciding which entity to keep
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Build name→(id, degree) index (case-insensitive)
        let mut name_to_info: HashMap<String, (i64, usize)> = HashMap::new();
        for e in &entities {
            if !is_noise_name(&e.name) && !is_noise_type(&e.entity_type) {
                let deg = degree.get(&e.id).copied().unwrap_or(0);
                let lower = e.name.to_lowercase();
                // Keep the entry with the highest degree
                let existing_deg = name_to_info.get(&lower).map(|(_, d)| *d).unwrap_or(0);
                if deg > existing_deg || !name_to_info.contains_key(&lower) {
                    name_to_info.insert(lower, (e.id, deg));
                }
            }
        }

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        eprintln!(
            "PREFIX_STRIP_DEBUG: entities count={}, name_to_info count={}",
            entities.len(),
            name_to_info.len()
        );
        let mut _checked = 0usize;

        for e in &entities {
            if absorbed.contains(&e.id) || is_noise_name(&e.name) || is_noise_type(&e.entity_type) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }
            _checked += 1;

            let my_deg = degree.get(&e.id).copied().unwrap_or(0);

            // Try stripping 1-2 leading words
            let max_strip = (words.len() / 2).max(1).min(2);
            let mut found = false;
            for strip in 1..=max_strip {
                if words.len() <= strip + 1 {
                    break; // result must be at least 2 words
                }
                let suffix: String = words[strip..].join(" ");
                let lower_suffix = suffix.to_lowercase();
                if let Some(&(target_id, target_deg)) = name_to_info.get(&lower_suffix) {
                    if target_id != e.id && target_deg >= my_deg {
                        // Only merge if target has more/equal connections (is more canonical)
                        self.brain.merge_entities(e.id, target_id)?;
                        absorbed.insert(e.id);
                        merged += 1;
                        found = true;
                        break;
                    }
                }
            }
            if found {
                continue;
            }

            // Also try stripping 1-2 trailing words (for patterns like "Blaise Pascal Chairs")
            for strip in 1..=max_strip {
                if words.len() <= strip + 1 {
                    break;
                }
                let prefix: String = words[..words.len() - strip].join(" ");
                let lower_prefix = prefix.to_lowercase();
                if let Some(&(target_id, target_deg)) = name_to_info.get(&lower_prefix) {
                    if target_id != e.id && target_deg >= my_deg {
                        self.brain.merge_entities(e.id, target_id)?;
                        absorbed.insert(e.id);
                        merged += 1;
                        break;
                    }
                }
            }
        }
        eprintln!(
            "PREFIX_STRIP_DEBUG: checked={}, merged={}",
            _checked, merged
        );
        Ok(merged)
    }

    /// Auto-consolidation: use entity_consolidation_candidates to find and merge
    /// entity pairs with high consolidation scores. Only merges pairs above
    /// `min_score` threshold to avoid false positives.
    /// Returns count of merges performed.
    pub fn auto_consolidate_entities(&self, min_score: f64) -> Result<usize> {
        let candidates = self.entity_consolidation_candidates(100)?;
        let mut merged = 0usize;
        let mut absorbed: HashSet<String> = HashSet::new();

        for (name_a, name_b, score, _reason) in &candidates {
            if *score < min_score {
                continue;
            }
            if absorbed.contains(name_a) || absorbed.contains(name_b) {
                continue;
            }
            let entity_a = self.brain.get_entity_by_name(name_a)?;
            let entity_b = self.brain.get_entity_by_name(name_b)?;
            if let (Some(a), Some(b)) = (entity_a, entity_b) {
                // Keep the shorter name (usually the canonical form)
                let (keep, remove) = if a.name.len() <= b.name.len() {
                    (a, b)
                } else {
                    (b, a)
                };
                self.brain.merge_entities(remove.id, keep.id)?;
                absorbed.insert(remove.name.clone());
                merged += 1;
            }
        }
        Ok(merged)
    }

    /// Merge name variants: merge entities whose names are title-prefixed or
    /// noise-suffixed versions of other entities. Works on ALL entities (not just
    /// islands), targeting patterns like:
    /// - "Professor Claude Shannon" → "Claude Shannon"
    /// - "Claude Shannon Time" → "Claude Shannon"
    /// - "Grace Hopper Admiral" → "Grace Hopper"
    /// - "Caliph Selim I" → "Selim I"
    /// Always merges into the entity with higher degree (more connections).
    /// Returns count of merges performed.
    pub fn merge_name_variants(&self) -> Result<usize> {
        // eprintln!("NAME_VARIANT_START: entering merge_name_variants");
        let entities = self.brain.all_entities()?;
        let cs_count = entities
            .iter()
            .filter(|e| e.name.contains("Claude Shannon"))
            .count();
        eprintln!(
            "NAME_VARIANT_START: got {} entities ({} contain 'Claude Shannon')",
            entities.len(),
            cs_count
        );
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Build name→(id, degree) index
        let mut name_to_info: HashMap<String, (i64, usize)> = HashMap::new();
        for e in &entities {
            if !is_noise_type(&e.entity_type) {
                let deg = degree.get(&e.id).copied().unwrap_or(0);
                let lower = e.name.to_lowercase();
                let existing_deg = name_to_info.get(&lower).map(|(_, d)| *d).unwrap_or(0);
                if deg > existing_deg || !name_to_info.contains_key(&lower) {
                    name_to_info.insert(lower, (e.id, deg));
                }
            }
        }

        // Title prefixes that should be stripped
        let title_prefixes: &[&str] = &[
            "professor",
            "prof",
            "dr",
            "sir",
            "lord",
            "lady",
            "king",
            "queen",
            "prince",
            "princess",
            "duke",
            "count",
            "countess",
            "baron",
            "admiral",
            "general",
            "colonel",
            "captain",
            "sultan",
            "caliph",
            "emperor",
            "empress",
            "president",
            "premier",
            "chancellor",
            "minister",
            "saint",
            "pope",
            "chaires",
            "boot",
        ];

        // Trailing noise words that NLP extractors commonly append
        let trailing_noise: &[&str] = &[
            "time",
            "created",
            "admiral",
            "original",
            "wired",
            "founder",
            "bits",
            "focus",
            "dead",
            "mass",
            "chairs",
            "accompagnées",
            "pensées",
            "even",
            "post",
            "hall",
            "suite",
            "center",
            "centre",
            "house",
            "building",
            "institute",
            "biography",
            "award",
            "day",
            "website",
            "frontiers",
            "countess",
            "working",
            "linux",
            "dm",
            "online",
            "resources",
            "imdb",
            "first",
            "start",
            "women",
            "past",
            "prestige",
            "no",
            "lwn",
            "there",
            "yesugei",
            "inc",
            "apsnews",
            "boingboing",
            "ratified",
            "ancien",
            "clark",
        ];

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        let mut iter_count = 0usize;
        for e in &entities {
            iter_count += 1;
            if e.name.contains("Claude Shannon") || e.name.contains("Professor Claude") {
                eprintln!(
                    "NAME_VARIANT_ITER[{}]: '{}' id={} absorbed={} noise_type={}",
                    iter_count,
                    e.name,
                    e.id,
                    absorbed.contains(&e.id),
                    is_noise_type(&e.entity_type)
                );
            }
            if absorbed.contains(&e.id) || is_noise_type(&e.entity_type) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if e.name.contains("Claude Shannon")
                || e.name.contains("Ada Lovelace")
                || e.name.contains("Professor Claude")
            {
                let last_w = words.last().map(|w| w.to_lowercase()).unwrap_or_default();
                let is_trailing = trailing_noise.contains(&last_w.as_str());
                let target = if words.len() >= 2 {
                    name_to_info.get(&words[..words.len() - 1].join(" ").to_lowercase())
                } else {
                    None
                };
                eprintln!(
                    "NAME_VARIANT_DEBUG: '{}' type='{}' words={} last='{}' is_trailing={} target={:?}",
                    e.name, e.entity_type, words.len(), last_w, is_trailing, target
                );
            }
            if words.len() < 2 {
                continue;
            }
            let my_deg = degree.get(&e.id).copied().unwrap_or(0);

            // 1. Strip title prefix — always merge into canonical (untitled) form
            if words.len() >= 2 {
                let first_lower = words[0].to_lowercase();
                if title_prefixes.contains(&first_lower.as_str()) {
                    let suffix: String = words[1..].join(" ");
                    let lower_suffix = suffix.to_lowercase();
                    if let Some(&(target_id, _target_deg)) = name_to_info.get(&lower_suffix) {
                        if target_id != e.id {
                            self.brain.merge_entities(e.id, target_id)?;
                            absorbed.insert(e.id);
                            merged += 1;
                            continue;
                        }
                    }
                }
            }

            // 2. Strip trailing noise word — always merge into canonical (shorter) form
            if words.len() >= 2 && !absorbed.contains(&e.id) {
                let last_lower = words.last().unwrap().to_lowercase();
                if trailing_noise.contains(&last_lower.as_str()) {
                    let prefix: String = words[..words.len() - 1].join(" ");
                    let lower_prefix = prefix.to_lowercase();
                    // eprintln!(
                    //     "TRAILING_STRIP: '{}' → trailing='{}' prefix='{}' lookup={:?}",
                    //     e.name,
                    //     last_lower,
                    //     lower_prefix,
                    //     name_to_info.get(&lower_prefix)
                    // );
                    if let Some(&(target_id, _target_deg)) = name_to_info.get(&lower_prefix) {
                        if target_id != e.id {
                            eprintln!(
                                "TRAILING_MERGE: merging '{}' (id={}) into target_id={}",
                                e.name, e.id, target_id
                            );
                            self.brain.merge_entities(e.id, target_id)?;
                            absorbed.insert(e.id);
                            merged += 1;
                            continue;
                        }
                    }
                }
            }

            // 3. Strip both leading AND trailing word for 4+ word entities
            if words.len() >= 4 {
                for lead in 1..=2 {
                    for trail in 1..=2 {
                        if lead + trail >= words.len() {
                            continue;
                        }
                        let core: String = words[lead..words.len() - trail].join(" ");
                        if core.split_whitespace().count() < 2 {
                            continue;
                        }
                        let lower_core = core.to_lowercase();
                        if let Some(&(target_id, target_deg)) = name_to_info.get(&lower_core) {
                            if target_id != e.id && target_deg > my_deg {
                                self.brain.merge_entities(e.id, target_id)?;
                                absorbed.insert(e.id);
                                merged += 1;
                                break;
                            }
                        }
                    }
                    if absorbed.contains(&e.id) {
                        break;
                    }
                }
            }
        }
        Ok(merged)
    }

    /// Prune stale hypotheses: reject testing hypotheses older than `max_days`
    /// whose confidence has decayed below the rejection threshold.
    /// Returns count of hypotheses pruned.
    pub fn prune_stale_hypotheses(&self, max_days: i64) -> Result<usize> {
        let cutoff = (Utc::now() - chrono::Duration::days(max_days))
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let count = self.brain.with_conn(|conn| {
            let updated = conn.execute(
                "UPDATE hypotheses SET status = 'rejected' \
                 WHERE status IN ('proposed', 'testing') \
                 AND discovered_at < ?1 \
                 AND confidence < ?2",
                params![cutoff, REJECTION_THRESHOLD],
            )?;
            Ok(updated)
        })?;
        Ok(count)
    }

    /// Aggressive hypothesis cleanup: reject all testing hypotheses older than `max_days`
    /// with confidence below `max_confidence`, regardless of the normal rejection threshold.
    /// This prevents the testing queue from growing unboundedly.
    /// Returns count of hypotheses rejected.
    /// Clean up fragment hypotheses: reject confirmed/testing hypotheses where both
    /// subject and object are single-word short strings (likely NLP extraction fragments
    /// like "Grace" → "Hopper" rather than real discoveries).
    pub fn cleanup_fragment_hypotheses(&self) -> Result<usize> {
        let count = self.brain.with_conn(|conn| {
            let updated = conn.execute(
                "UPDATE hypotheses SET status = 'rejected' \
                 WHERE status IN ('confirmed', 'testing') \
                 AND subject NOT LIKE '% %' AND length(subject) <= 12 \
                 AND object NOT LIKE '% %' AND length(object) <= 12 \
                 AND object != '?'",
                [],
            )?;
            Ok(updated)
        })?;
        Ok(count)
    }

    pub fn bulk_reject_stale_testing(&self, max_days: i64, max_confidence: f64) -> Result<usize> {
        let cutoff = (Utc::now() - chrono::Duration::days(max_days))
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let count = self.brain.with_conn(|conn| {
            let updated = conn.execute(
                "UPDATE hypotheses SET status = 'rejected' \
                 WHERE status = 'testing' \
                 AND discovered_at < ?1 \
                 AND confidence < ?2",
                params![cutoff, max_confidence],
            )?;
            Ok(updated)
        })?;
        // Record rejections for meta-learning
        if count > 0 {
            // Batch record — approximate by pattern source
            let _ = self.record_outcome("stale_bulk_reject", false);
        }
        Ok(count)
    }

    /// Reconnect islands using reverse containment: find islands whose name appears
    /// as a significant substring within a connected entity name.
    /// E.g., island "Euler" → connected "Euler's Method" or "Leonhard Euler".
    /// Unlike consolidate_prefix_islands (which checks if island is a prefix),
    /// this checks if the island name appears anywhere as a word boundary match.
    /// Returns count of merges performed.
    pub fn reverse_containment_island_merge(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build word index: word → list of connected entity IDs containing that word
        let mut word_to_entities: HashMap<String, Vec<(i64, String, String)>> = HashMap::new();
        for e in &entities {
            if !connected.contains(&e.id) || is_noise_name(&e.name) || is_noise_type(&e.entity_type)
            {
                continue;
            }
            for word in e.name.to_lowercase().split_whitespace() {
                if word.len() >= 4 {
                    word_to_entities.entry(word.to_string()).or_default().push((
                        e.id,
                        e.name.to_lowercase(),
                        e.entity_type.clone(),
                    ));
                }
            }
        }

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        for e in &entities {
            if connected.contains(&e.id) || absorbed.contains(&e.id) {
                continue;
            }
            if is_noise_name(&e.name) || is_noise_type(&e.entity_type) {
                continue;
            }
            let lower = e.name.to_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();

            // Only handle single-word or two-word islands
            if words.len() > 2 || words.is_empty() {
                continue;
            }

            // For single-word islands: find connected entities containing this word
            if words.len() == 1 && lower.len() >= 5 {
                if let Some(candidates) = word_to_entities.get(&lower) {
                    // Find best match: same type, shortest name (most specific)
                    let mut best: Option<(i64, usize)> = None;
                    for (cid, cname, ctype) in candidates {
                        if *cid == e.id {
                            continue;
                        }
                        // Must be multi-word (otherwise it's the same entity)
                        if !cname.contains(' ') {
                            continue;
                        }
                        let same_type = *ctype == e.entity_type;
                        let compatible = same_type
                            || matches!(
                                (e.entity_type.as_str(), ctype.as_str()),
                                ("concept", "person")
                                    | ("person", "concept")
                                    | ("concept", "place")
                                    | ("place", "concept")
                            );
                        if !compatible {
                            continue;
                        }
                        // Prefer same-type, then shorter names
                        let score =
                            if same_type { 10000 } else { 0 } + (1000 - cname.len().min(999));
                        if best.is_none() || score > best.unwrap().1 {
                            best = Some((*cid, score));
                        }
                    }
                    if let Some((target_id, _)) = best {
                        // Check the island has no facts worth preserving
                        let facts = self.brain.get_facts_for(e.id)?;
                        if facts.is_empty() {
                            self.brain.merge_entities(e.id, target_id)?;
                            absorbed.insert(e.id);
                            merged += 1;
                        }
                    }
                }
            }

            // For two-word islands: check if both words appear in a connected entity
            if words.len() == 2 && lower.len() >= 6 {
                let w0 = words[0].to_string();
                let w1 = words[1].to_string();
                if w0.len() < 4 || w1.len() < 4 {
                    continue;
                }
                let c0 = word_to_entities.get(&w0);
                let c1 = word_to_entities.get(&w1);
                if let (Some(list0), Some(list1)) = (c0, c1) {
                    // Find entities appearing in both word lists
                    let ids0: HashSet<i64> = list0.iter().map(|(id, _, _)| *id).collect();
                    for (cid, _cname, ctype) in list1 {
                        if ids0.contains(cid) && *cid != e.id {
                            let compatible = *ctype == e.entity_type
                                || matches!(
                                    (e.entity_type.as_str(), ctype.as_str()),
                                    ("concept", "person")
                                        | ("person", "concept")
                                        | ("concept", "place")
                                        | ("place", "concept")
                                );
                            if compatible {
                                let facts = self.brain.get_facts_for(e.id)?;
                                if facts.is_empty() {
                                    self.brain.merge_entities(e.id, *cid)?;
                                    absorbed.insert(e.id);
                                    merged += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(merged)
    }

    /// Deduplicate entities with identical lowercase names but different IDs.
    /// This catches exact duplicates that differ only in casing or whitespace.
    /// Returns count of merges performed.
    pub fn dedup_exact_name_matches(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let mut name_groups: HashMap<String, Vec<(i64, f64, usize)>> = HashMap::new();

        // Build relation degree for tie-breaking
        let relations = self.brain.all_relations()?;
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        for e in &entities {
            if is_noise_type(&e.entity_type) {
                continue;
            }
            let key = e.name.to_lowercase().trim().to_string();
            if key.len() < 2 {
                continue;
            }
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            name_groups
                .entry(key)
                .or_default()
                .push((e.id, e.confidence, deg));
        }

        let mut merged = 0usize;
        for (_name, mut group) in name_groups {
            if group.len() < 2 {
                continue;
            }
            // Sort: highest degree first, then highest confidence
            group.sort_by(|a, b| {
                b.2.cmp(&a.2)
                    .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            let keep_id = group[0].0;
            for &(remove_id, _, _) in &group[1..] {
                self.brain.merge_entities(remove_id, keep_id)?;
                merged += 1;
            }
        }
        Ok(merged)
    }

    /// Generate hypotheses from Louvain community bridges: entities in different
    /// communities but sharing a common neighbour likely have an indirect relationship.
    /// More precise than structural_hole because it respects community boundaries.
    pub fn generate_hypotheses_from_community_bridges(&self) -> Result<Vec<Hypothesis>> {
        let communities = crate::graph::louvain_communities(self.brain)?;
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Build adjacency
        let mut adj: HashMap<i64, HashSet<i64>> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id) && meaningful.contains(&r.object_id) {
                adj.entry(r.subject_id).or_default().insert(r.object_id);
                adj.entry(r.object_id).or_default().insert(r.subject_id);
            }
        }

        // Find pairs in different communities that share ≥2 common neighbours
        let mut hypotheses = Vec::new();
        let mut seen: HashSet<(i64, i64)> = HashSet::new();
        let ids: Vec<i64> = adj.keys().copied().collect();

        for &node in &ids {
            let _node_comm = communities.get(&node).copied().unwrap_or(usize::MAX);
            let neighbours = match adj.get(&node) {
                Some(n) => n,
                None => continue,
            };
            // For each pair of this node's neighbours in different communities
            let nb_list: Vec<i64> = neighbours.iter().copied().collect();
            for i in 0..nb_list.len() {
                for j in (i + 1)..nb_list.len() {
                    let a = nb_list[i];
                    let b = nb_list[j];
                    let ca = communities.get(&a).copied().unwrap_or(usize::MAX);
                    let cb = communities.get(&b).copied().unwrap_or(usize::MAX);
                    if ca == cb {
                        continue; // same community, already well-connected
                    }
                    // Check they're not directly connected
                    if adj.get(&a).is_some_and(|s| s.contains(&b)) {
                        continue;
                    }
                    let key = if a < b { (a, b) } else { (b, a) };
                    if !seen.insert(key) {
                        continue;
                    }
                    let a_name = self.entity_name(a)?;
                    let b_name = self.entity_name(b)?;
                    let bridge_name = self.entity_name(node)?;
                    hypotheses.push(Hypothesis {
                        id: 0,
                        subject: a_name.clone(),
                        predicate: "related_to".to_string(),
                        object: b_name.clone(),
                        confidence: 0.45,
                        evidence_for: vec![format!(
                            "Both connected to bridge entity '{}' but in different communities ({}≠{})",
                            bridge_name, ca, cb
                        )],
                        evidence_against: vec![],
                        reasoning_chain: vec![
                            format!("{} and {} share neighbour {}", a_name, b_name, bridge_name),
                            format!("They belong to different communities ({} vs {})", ca, cb),
                            "Cross-community bridge pattern suggests potential relation".to_string(),
                        ],
                        status: HypothesisStatus::Proposed,
                        discovered_at: now_str(),
                        pattern_source: "community_bridge".to_string(),
                    });
                    if hypotheses.len() >= 50 {
                        return Ok(hypotheses);
                    }
                }
            }
        }
        Ok(hypotheses)
    }

    /// Compute hypothesis quality metrics: distribution of confidence scores,
    /// pattern source effectiveness, and staleness analysis.
    pub fn hypothesis_quality_report(&self) -> Result<HashMap<String, f64>> {
        let all = self.list_hypotheses(None)?;
        let mut metrics: HashMap<String, f64> = HashMap::new();
        let total = all.len() as f64;
        if total == 0.0 {
            return Ok(metrics);
        }

        // Status distribution
        let mut status_counts: HashMap<String, usize> = HashMap::new();
        let mut source_counts: HashMap<String, (usize, f64)> = HashMap::new();
        let mut conf_sum = 0.0_f64;

        for h in &all {
            *status_counts
                .entry(h.status.as_str().to_string())
                .or_insert(0) += 1;
            let entry = source_counts
                .entry(h.pattern_source.clone())
                .or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += h.confidence;
            conf_sum += h.confidence;
        }

        metrics.insert("total_hypotheses".into(), total);
        metrics.insert("avg_confidence".into(), conf_sum / total);

        for (status, count) in &status_counts {
            metrics.insert(format!("status_{}", status), *count as f64);
        }

        // Average confidence per source
        for (source, (count, sum)) in &source_counts {
            metrics.insert(format!("source_{}_count", source), *count as f64);
            metrics.insert(format!("source_{}_avg_conf", source), sum / *count as f64);
        }

        Ok(metrics)
    }

    fn entity_name(&self, id: i64) -> Result<String> {
        Ok(self
            .brain
            .get_entity_by_id(id)?
            .map(|e| e.name)
            .unwrap_or_else(|| format!("#{}", id)))
    }

    // -----------------------------------------------------------------------
    // Predicate Specificity Analysis
    // -----------------------------------------------------------------------

    /// Analyze predicate quality: identify over-reliance on vague predicates
    /// and suggest more specific alternatives based on entity types and context.
    /// Returns (predicate, count, specificity_score, suggestion).
    pub fn predicate_specificity_analysis(&self) -> Result<Vec<(String, usize, f64, String)>> {
        let relations = self.brain.all_relations()?;
        let entities = self.brain.all_entities()?;
        let id_to_type: HashMap<i64, &str> = entities
            .iter()
            .map(|e| (e.id, e.entity_type.as_str()))
            .collect();

        // Count predicates and their subject/object type pairs
        let mut pred_count: HashMap<String, usize> = HashMap::new();
        let mut pred_type_pairs: HashMap<String, HashMap<(String, String), usize>> = HashMap::new();

        for r in &relations {
            *pred_count.entry(r.predicate.clone()).or_insert(0) += 1;
            let stype = id_to_type
                .get(&r.subject_id)
                .copied()
                .unwrap_or("unknown")
                .to_string();
            let otype = id_to_type
                .get(&r.object_id)
                .copied()
                .unwrap_or("unknown")
                .to_string();
            *pred_type_pairs
                .entry(r.predicate.clone())
                .or_default()
                .entry((stype, otype))
                .or_insert(0) += 1;
        }

        let total = relations.len();
        let mut results = Vec::new();

        // Vague predicates that could be more specific
        let vague_predicates: HashMap<&str, &[(&str, &str, &str)]> = HashMap::from([
            (
                "associated_with",
                &[
                    ("person", "place", "born_in / lived_in / worked_in"),
                    ("person", "organization", "member_of / founded / worked_at"),
                    (
                        "person",
                        "person",
                        "collaborated_with / influenced / mentored",
                    ),
                    ("person", "concept", "developed / studied / contributed_to"),
                    ("organization", "place", "headquartered_in / operates_in"),
                    (
                        "concept",
                        "concept",
                        "related_to / subclass_of / derived_from",
                    ),
                    ("technology", "person", "invented_by / developed_by"),
                    ("technology", "concept", "implements / based_on"),
                ][..],
            ),
            (
                "related_to",
                &[
                    ("person", "person", "collaborated_with / influenced"),
                    ("concept", "concept", "subclass_of / part_of / derived_from"),
                    ("place", "place", "borders / located_in / part_of"),
                ][..],
            ),
        ]);

        for (pred, count) in &pred_count {
            let dominance = *count as f64 / total.max(1) as f64;
            let specificity = if vague_predicates.contains_key(pred.as_str()) {
                // Vague predicate — lower specificity
                (1.0 - dominance).max(0.0)
            } else if is_generic_predicate(pred) {
                0.1
            } else {
                // Specific predicate — high specificity
                0.8 + (0.2 * (1.0 - dominance))
            };

            let suggestion = if let Some(suggestions) = vague_predicates.get(pred.as_str()) {
                // Find most common type pairs for this predicate
                if let Some(type_pairs) = pred_type_pairs.get(pred) {
                    let mut pairs: Vec<_> = type_pairs.iter().collect();
                    pairs.sort_by(|a, b| b.1.cmp(a.1));
                    let top_pair = &pairs[0].0;
                    // Find matching suggestion
                    suggestions
                        .iter()
                        .find(|(st, ot, _)| *st == top_pair.0 && *ot == top_pair.1)
                        .map(|(st, ot, sugg)| {
                            format!(
                                "For {}->{}: consider {} (affects {} rels)",
                                st, ot, sugg, pairs[0].1
                            )
                        })
                        .unwrap_or_else(|| {
                            format!(
                                "Top usage: {}->{} ({} times) — needs domain-specific predicate",
                                top_pair.0, top_pair.1, pairs[0].1
                            )
                        })
                } else {
                    "No type pair data".to_string()
                }
            } else {
                "OK — specific predicate".to_string()
            };

            results.push((pred.clone(), *count, specificity, suggestion));
        }

        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Compute a graph-wide predicate quality score (0-1).
    /// 0 = all vague predicates, 1 = all specific predicates.
    pub fn predicate_quality_score(&self) -> Result<f64> {
        let analysis = self.predicate_specificity_analysis()?;
        if analysis.is_empty() {
            return Ok(0.0);
        }
        let total_weighted: f64 = analysis.iter().map(|(_, c, s, _)| *c as f64 * s).sum();
        let total_count: f64 = analysis.iter().map(|(_, c, _, _)| *c as f64).sum();
        Ok(if total_count > 0.0 {
            total_weighted / total_count
        } else {
            0.0
        })
    }

    /// Suggest predicate refinements for "associated_with" relations based on
    /// the entity types of subject and object.
    /// Returns Vec of (relation_subject, relation_object, current_pred, suggested_pred, confidence).
    pub fn suggest_predicate_refinements(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String, String, String, f64)>> {
        let relations = self.brain.all_relations()?;
        let entities = self.brain.all_entities()?;
        let id_to_entity: HashMap<i64, &crate::db::Entity> =
            entities.iter().map(|e| (e.id, e)).collect();

        // Type-pair → suggested predicate mapping
        let refinement_map: HashMap<(&str, &str), (&str, f64)> = HashMap::from([
            (("person", "place"), ("born_in_or_lived_in", 0.5)),
            (("person", "organization"), ("affiliated_with", 0.6)),
            (("person", "person"), ("collaborated_with", 0.4)),
            (("person", "concept"), ("contributed_to", 0.5)),
            (("person", "technology"), ("developed", 0.5)),
            (("organization", "place"), ("headquartered_in", 0.5)),
            (("organization", "person"), ("employs_or_founded_by", 0.4)),
            (("technology", "person"), ("invented_by", 0.6)),
            (("technology", "concept"), ("implements", 0.5)),
            (("concept", "person"), ("developed_by", 0.5)),
            (("place", "place"), ("located_in", 0.6)),
            (("event", "place"), ("occurred_in", 0.7)),
            (("event", "person"), ("involved", 0.5)),
        ]);

        let mut suggestions = Vec::new();
        for r in &relations {
            if r.predicate != "associated_with" && r.predicate != "related_to" {
                continue;
            }
            let stype = id_to_entity
                .get(&r.subject_id)
                .map(|e| e.entity_type.as_str())
                .unwrap_or("unknown");
            let otype = id_to_entity
                .get(&r.object_id)
                .map(|e| e.entity_type.as_str())
                .unwrap_or("unknown");

            if let Some(&(suggested, confidence)) = refinement_map.get(&(stype, otype)) {
                let sname = id_to_entity
                    .get(&r.subject_id)
                    .map(|e| e.name.as_str())
                    .unwrap_or("?");
                let oname = id_to_entity
                    .get(&r.object_id)
                    .map(|e| e.name.as_str())
                    .unwrap_or("?");
                suggestions.push((
                    sname.to_string(),
                    oname.to_string(),
                    r.predicate.clone(),
                    suggested.to_string(),
                    confidence,
                ));
            }
            if suggestions.len() >= limit {
                break;
            }
        }
        Ok(suggestions)
    }

    // -----------------------------------------------------------------------
    // Entity Consolidation Scoring
    // -----------------------------------------------------------------------

    /// Score entity pairs for consolidation potential using multiple signals:
    /// name similarity, type match, shared neighbours, co-source frequency.
    /// Returns sorted list of (entity_a, entity_b, consolidation_score, reason).
    pub fn entity_consolidation_candidates(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String, f64, String)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Build connected set and neighbour map
        let mut connected: HashSet<i64> = HashSet::new();
        let mut neighbours: HashMap<i64, HashSet<i64>> = HashMap::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
            neighbours
                .entry(r.subject_id)
                .or_default()
                .insert(r.object_id);
            neighbours
                .entry(r.object_id)
                .or_default()
                .insert(r.subject_id);
        }

        // Source co-occurrence map
        let mut source_entities: HashMap<String, HashSet<i64>> = HashMap::new();
        for r in &relations {
            if !r.source_url.is_empty() {
                source_entities
                    .entry(r.source_url.clone())
                    .or_default()
                    .insert(r.subject_id);
                source_entities
                    .entry(r.source_url.clone())
                    .or_default()
                    .insert(r.object_id);
            }
        }

        // Focus on meaningful entities, preferring connected ones with island candidates
        let meaningful_entities: Vec<&crate::db::Entity> = entities
            .iter()
            .filter(|e| meaningful.contains(&e.id) && !is_noise_name(&e.name))
            .collect();

        // Index by first significant word for blocking (avoid O(n²))
        let mut word_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, e) in meaningful_entities.iter().enumerate() {
            for word in e.name.to_lowercase().split_whitespace() {
                if word.len() >= 4 {
                    word_index.entry(word.to_string()).or_default().push(idx);
                }
            }
        }

        let mut candidates: Vec<(String, String, f64, String)> = Vec::new();
        let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();

        for indices in word_index.values() {
            if indices.len() > 200 {
                continue; // Skip extremely common words
            }
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let idx_a = indices[i];
                    let idx_b = indices[j];
                    let pair = if idx_a < idx_b {
                        (idx_a, idx_b)
                    } else {
                        (idx_b, idx_a)
                    };
                    if !seen_pairs.insert(pair) {
                        continue;
                    }

                    let ea = meaningful_entities[idx_a];
                    let eb = meaningful_entities[idx_b];

                    // Must be same type
                    if ea.entity_type != eb.entity_type {
                        continue;
                    }

                    let mut score = 0.0_f64;
                    let mut reasons = Vec::new();

                    // 1. Name word overlap (Jaccard)
                    let words_a: HashSet<String> = ea
                        .name
                        .to_lowercase()
                        .split_whitespace()
                        .filter(|w| w.len() >= 3)
                        .map(|s| s.to_string())
                        .collect();
                    let words_b: HashSet<String> = eb
                        .name
                        .to_lowercase()
                        .split_whitespace()
                        .filter(|w| w.len() >= 3)
                        .map(|s| s.to_string())
                        .collect();
                    let intersection = words_a.intersection(&words_b).count();
                    let union = words_a.union(&words_b).count();
                    let jaccard = if union > 0 {
                        intersection as f64 / union as f64
                    } else {
                        0.0
                    };

                    if jaccard < 0.3 {
                        continue; // Not similar enough
                    }
                    score += jaccard * 0.4;
                    reasons.push(format!("name overlap {:.0}%", jaccard * 100.0));

                    // 2. Containment (one name is substring of the other)
                    let a_lower = ea.name.to_lowercase();
                    let b_lower = eb.name.to_lowercase();
                    if a_lower.contains(&b_lower) || b_lower.contains(&a_lower) {
                        score += 0.2;
                        reasons.push("name containment".to_string());
                    }

                    // 3. Shared neighbours
                    let na = neighbours.get(&ea.id).cloned().unwrap_or_default();
                    let nb = neighbours.get(&eb.id).cloned().unwrap_or_default();
                    let shared_nb = na.intersection(&nb).count();
                    if shared_nb > 0 {
                        score += (shared_nb as f64 * 0.1).min(0.2);
                        reasons.push(format!("{} shared neighbours", shared_nb));
                    }

                    // 4. Island bonus (one or both disconnected)
                    if !connected.contains(&ea.id) || !connected.contains(&eb.id) {
                        score += 0.1;
                        reasons.push("island entity".to_string());
                    }

                    // 5. Co-source frequency
                    let mut co_sources = 0usize;
                    for src_entities in source_entities.values() {
                        if src_entities.contains(&ea.id) && src_entities.contains(&eb.id) {
                            co_sources += 1;
                        }
                    }
                    if co_sources > 0 {
                        score += (co_sources as f64 * 0.05).min(0.15);
                        reasons.push(format!("{} co-sources", co_sources));
                    }

                    if score >= 0.4 {
                        candidates.push((
                            ea.name.clone(),
                            eb.name.clone(),
                            score,
                            reasons.join(", "),
                        ));
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);
        Ok(candidates)
    }

    /// Find entities that are "almost connected" — they share 2+ hop paths through
    /// the graph but no direct edge. These are high-value hypothesis targets because
    /// a direct relation likely exists but wasn't extracted.
    /// Returns (entity_a, entity_b, path_count, shortest_path_len, suggested_predicate).
    pub fn find_near_miss_connections(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String, usize, usize, String)>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        // Build adjacency for meaningful entities only
        let mut adj: HashMap<i64, HashSet<i64>> = HashMap::new();
        let mut direct: HashSet<(i64, i64)> = HashSet::new();
        let mut pred_map: HashMap<(i64, i64), String> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id) && meaningful.contains(&r.object_id) {
                adj.entry(r.subject_id).or_default().insert(r.object_id);
                adj.entry(r.object_id).or_default().insert(r.subject_id);
                let key = if r.subject_id < r.object_id {
                    (r.subject_id, r.object_id)
                } else {
                    (r.object_id, r.subject_id)
                };
                direct.insert(key);
                pred_map.insert(key, r.predicate.clone());
            }
        }

        // For each entity, do a 2-hop BFS to find entities reachable in exactly 2 hops
        // that share multiple 2-hop paths (strong indirect connection signal)
        let mut near_misses: Vec<(String, String, usize, usize, String)> = Vec::new();
        let entities: Vec<i64> = adj.keys().copied().collect();

        // Sample to avoid O(n²) on large graphs
        let sample_size = entities.len().min(200);
        let step = if entities.len() <= sample_size {
            1
        } else {
            entities.len() / sample_size
        };

        for &src in entities.iter().step_by(step).take(sample_size) {
            let src_nb = match adj.get(&src) {
                Some(s) => s,
                None => continue,
            };
            // Collect 2-hop targets with path counts
            let mut two_hop_count: HashMap<i64, usize> = HashMap::new();
            for &mid in src_nb {
                if let Some(mid_nb) = adj.get(&mid) {
                    for &target in mid_nb {
                        if target != src && !src_nb.contains(&target) {
                            let key = if src < target {
                                (src, target)
                            } else {
                                (target, src)
                            };
                            if !direct.contains(&key) {
                                *two_hop_count.entry(target).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }

            // Entities reachable via 3+ distinct 2-hop paths are strong candidates
            for (&target, &count) in &two_hop_count {
                if count >= 3 {
                    let src_name = self.entity_name(src)?;
                    let tgt_name = self.entity_name(target)?;
                    // Suggest predicate based on most common predicate in shared neighbourhood
                    let suggested = "related_to".to_string();
                    near_misses.push((src_name, tgt_name, count, 2, suggested));
                }
            }
        }

        near_misses.sort_by(|a, b| b.2.cmp(&a.2));
        near_misses.truncate(limit);
        Ok(near_misses)
    }

    /// Generate hypotheses from near-miss connections (entities with multiple
    /// indirect paths but no direct edge).
    /// Generate hypotheses from Adamic-Adar link prediction.
    /// Uses network topology to predict the most likely missing links.
    pub fn generate_hypotheses_from_adamic_adar(&self) -> Result<Vec<Hypothesis>> {
        let predictions = crate::graph::adamic_adar_predict(self.brain, 30)?;
        let mut hypotheses = Vec::new();
        for (a_id, b_id, score) in &predictions {
            let a_name = self.entity_name(*a_id)?;
            let b_name = self.entity_name(*b_id)?;
            if is_noise_name(&a_name) || is_noise_name(&b_name) {
                continue;
            }
            let confidence = (0.4 + score * 0.1).min(0.85);
            hypotheses.push(Hypothesis {
                id: 0,
                subject: a_name.clone(),
                predicate: "related_to".to_string(),
                object: b_name.clone(),
                confidence,
                evidence_for: vec![format!(
                    "Adamic-Adar score {:.3}: shared neighbors weighted by inverse log-degree",
                    score
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!(
                        "{} and {} share multiple well-connected neighbors",
                        a_name, b_name
                    ),
                    format!("Adamic-Adar link prediction score: {:.3}", score),
                    "High AA score strongly predicts missing links in real networks".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "adamic_adar".to_string(),
            });
        }
        Ok(hypotheses)
    }

    /// Generate hypotheses using Resource Allocation Index — better for sparse graphs.
    /// RA uses 1/degree instead of 1/ln(degree), giving more weight to exclusive shared neighbors.
    pub fn generate_hypotheses_from_resource_allocation(&self) -> Result<Vec<Hypothesis>> {
        let predictions = crate::graph::resource_allocation_predict(self.brain, 30)?;
        let mut hypotheses = Vec::new();
        for (a_id, b_id, score) in &predictions {
            let a_name = self.entity_name(*a_id)?;
            let b_name = self.entity_name(*b_id)?;
            if is_noise_name(&a_name) || is_noise_name(&b_name) {
                continue;
            }
            let confidence = (0.4 + score * 0.15).min(0.85);
            hypotheses.push(Hypothesis {
                id: 0,
                subject: a_name.clone(),
                predicate: "related_to".to_string(),
                object: b_name.clone(),
                confidence,
                evidence_for: vec![format!(
                    "Resource Allocation score {:.3}: shared neighbors weighted by inverse degree",
                    score
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!(
                        "{} and {} share neighbors with low degree (exclusive connections)",
                        a_name, b_name
                    ),
                    format!("RA link prediction score: {:.3}", score),
                    "RA outperforms Adamic-Adar in sparse knowledge graphs".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "resource_allocation".to_string(),
            });
        }
        Ok(hypotheses)
    }

    /// Generate hypotheses using type-aware link prediction — boosts same-type entity pairs.
    pub fn generate_hypotheses_from_type_affinity(&self) -> Result<Vec<Hypothesis>> {
        let predictions = crate::graph::type_aware_link_predict(self.brain, 30)?;
        let mut hypotheses = Vec::new();
        for (a_id, b_id, score) in &predictions {
            let a_name = self.entity_name(*a_id)?;
            let b_name = self.entity_name(*b_id)?;
            if is_noise_name(&a_name) || is_noise_name(&b_name) {
                continue;
            }
            let a_type = self
                .brain
                .get_entity_by_id(*a_id)?
                .map(|e| e.entity_type)
                .unwrap_or_default();
            let b_type = self
                .brain
                .get_entity_by_id(*b_id)?
                .map(|e| e.entity_type)
                .unwrap_or_default();
            let type_info = if a_type == b_type {
                format!("both {} type", a_type)
            } else {
                format!("{} + {} types", a_type, b_type)
            };
            let confidence = (0.4 + score * 0.08).min(0.85);
            hypotheses.push(Hypothesis {
                id: 0,
                subject: a_name.clone(),
                predicate: "related_to".to_string(),
                object: b_name.clone(),
                confidence,
                evidence_for: vec![format!(
                    "Type-aware CN score {:.3} ({}) — same-type entities sharing neighbors",
                    score, type_info
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!("{} and {} share common neighbors", a_name, b_name),
                    format!("Type affinity: {}", type_info),
                    "Same-type entities with shared neighbors are strongly correlated".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "type_affinity".to_string(),
            });
        }
        Ok(hypotheses)
    }

    /// Generate hypotheses using neighborhood overlap coefficient.
    /// Finds unconnected node pairs with high overlap in their neighborhoods.
    /// Overlap = |N(a) ∩ N(b)| / min(|N(a)|, |N(b)|) — robust to degree imbalance.
    pub fn generate_hypotheses_from_neighborhood_overlap(&self) -> Result<Vec<Hypothesis>> {
        let overlaps = crate::graph::neighborhood_overlap(self.brain, 0.3, 30)?;
        let mut hypotheses = Vec::new();
        for (a_id, b_id, score) in &overlaps {
            let a_name = self.entity_name(*a_id)?;
            let b_name = self.entity_name(*b_id)?;
            if is_noise_name(&a_name) || is_noise_name(&b_name) {
                continue;
            }
            let confidence = (0.45 + score * 0.4).min(0.90);
            hypotheses.push(Hypothesis {
                id: 0,
                subject: a_name.clone(),
                predicate: "related_to".to_string(),
                object: b_name.clone(),
                confidence,
                evidence_for: vec![format!(
                    "Neighborhood overlap coefficient {:.3} — strong structural similarity",
                    score
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!(
                        "{} and {} share a large fraction of their neighbors",
                        a_name, b_name
                    ),
                    format!("Overlap score: {:.2} (≥0.3 threshold)", score),
                    "High neighborhood overlap strongly predicts missing links".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "neighborhood_overlap".to_string(),
            });
        }
        Ok(hypotheses)
    }

    pub fn generate_hypotheses_from_near_misses(&self) -> Result<Vec<Hypothesis>> {
        let near_misses = self.find_near_miss_connections(30)?;
        let mut hypotheses = Vec::new();
        for (a, b, path_count, _path_len, pred) in &near_misses {
            hypotheses.push(Hypothesis {
                id: 0,
                subject: a.clone(),
                predicate: pred.clone(),
                object: b.clone(),
                confidence: 0.45 + (*path_count as f64 * 0.05).min(0.3),
                evidence_for: vec![format!(
                    "{} and {} connected via {} distinct 2-hop paths",
                    a, b, path_count
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!("{} indirect paths between {} and {}", path_count, a, b),
                    "Multiple indirect connections suggest a missing direct relation".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "near_miss".to_string(),
            });
        }
        Ok(hypotheses)
    }

    /// Triadic closure hypothesis generation: if A→B and B→C but not A→C,
    /// and A and C are of compatible types, hypothesize A→C.
    /// Weighted by number of distinct intermediaries (more paths = higher confidence).
    /// Generate hypotheses from semantic fingerprint similarity.
    /// Entities sharing the same (predicate, object) patterns are likely related
    /// even if not directly connected.
    pub fn generate_hypotheses_from_semantic_similarity(&self) -> Result<Vec<Hypothesis>> {
        let similar = crate::graph::semantic_fingerprint_similarity(self.brain, 2, 30)?;
        let meaningful = meaningful_ids(self.brain)?;
        let entities = self.brain.all_entities()?;
        let id_name: HashMap<i64, &str> =
            entities.iter().map(|e| (e.id, e.name.as_str())).collect();
        let id_type: HashMap<i64, &str> = entities
            .iter()
            .map(|e| (e.id, e.entity_type.as_str()))
            .collect();

        // Check direct connections
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            connected.insert(key);
        }

        let mut hypotheses = Vec::new();
        for (a, b, shared, jaccard) in &similar {
            if !meaningful.contains(a) || !meaningful.contains(b) {
                continue;
            }
            let key = if a < b { (*a, *b) } else { (*b, *a) };
            if connected.contains(&key) {
                continue;
            }
            let a_name = id_name.get(a).copied().unwrap_or("?");
            let b_name = id_name.get(b).copied().unwrap_or("?");
            if is_noise_name(a_name) || is_noise_name(b_name) {
                continue;
            }
            let a_type = id_type.get(a).copied().unwrap_or("?");
            let b_type = id_type.get(b).copied().unwrap_or("?");
            let predicate = match (a_type, b_type) {
                ("person", "person") => "contemporary_of",
                ("concept", "concept") => "related_concept",
                ("organization", "organization") => "partner_of",
                _ => "related_to",
            };
            hypotheses.push(Hypothesis {
                id: 0,
                subject: a_name.to_string(),
                predicate: predicate.to_string(),
                object: b_name.to_string(),
                confidence: 0.40 + (*jaccard * 0.4).min(0.4),
                evidence_for: vec![format!(
                    "Semantic fingerprint similarity: {} shared (pred,obj) patterns, Jaccard {:.2}",
                    shared, jaccard
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!(
                        "{} and {} share {} predicate-object patterns",
                        a_name, b_name, shared
                    ),
                    format!("Jaccard similarity: {:.2}", jaccard),
                    "Entities with similar relational patterns are likely related".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "semantic_fingerprint".to_string(),
            });
        }
        Ok(hypotheses)
    }

    pub fn generate_hypotheses_from_triadic_closure(&self) -> Result<Vec<Hypothesis>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;
        let entities = self.brain.all_entities()?;
        let id_name: HashMap<i64, &str> =
            entities.iter().map(|e| (e.id, e.name.as_str())).collect();
        let id_type: HashMap<i64, &str> = entities
            .iter()
            .map(|e| (e.id, e.entity_type.as_str()))
            .collect();

        // Build adjacency with predicate info
        let mut adj: HashMap<i64, HashSet<i64>> = HashMap::new();
        let mut edge_preds: HashMap<(i64, i64), String> = HashMap::new();
        for r in &relations {
            if !meaningful.contains(&r.subject_id) || !meaningful.contains(&r.object_id) {
                continue;
            }
            adj.entry(r.subject_id).or_default().insert(r.object_id);
            adj.entry(r.object_id).or_default().insert(r.subject_id);
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            edge_preds.entry(key).or_insert_with(|| r.predicate.clone());
        }

        // Direct edge set
        let direct: HashSet<(i64, i64)> = edge_preds.keys().copied().collect();

        // Find open triads: A-B-C where A and C are not connected
        // For efficiency, only consider nodes with degree 3..100
        let candidates: Vec<i64> = adj
            .iter()
            .filter(|(_, nb)| nb.len() >= 3 && nb.len() <= 100)
            .map(|(&id, _)| id)
            .collect();

        // Count intermediaries for each (A,C) pair
        let mut pair_intermediaries: HashMap<(i64, i64), Vec<i64>> = HashMap::new();
        for &b in &candidates {
            let nb: Vec<i64> = adj[&b].iter().copied().collect();
            // Cap to avoid O(n²) within large neighborhoods
            if nb.len() > 80 {
                continue;
            }
            for i in 0..nb.len() {
                for j in (i + 1)..nb.len() {
                    let a = nb[i].min(nb[j]);
                    let c = nb[i].max(nb[j]);
                    if !direct.contains(&(a, c)) {
                        pair_intermediaries.entry((a, c)).or_default().push(b);
                    }
                }
            }
        }

        // Only generate hypotheses for pairs with ≥2 intermediaries (strong triadic signal)
        let mut scored: Vec<((i64, i64), usize)> = pair_intermediaries
            .iter()
            .filter(|(_, intermediaries)| intermediaries.len() >= 2)
            .map(|(&pair, intermediaries)| (pair, intermediaries.len()))
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(50);

        let mut hypotheses = Vec::new();
        for ((a, c), count) in scored {
            let a_name = id_name.get(&a).copied().unwrap_or("?");
            let c_name = id_name.get(&c).copied().unwrap_or("?");
            let a_type = id_type.get(&a).copied().unwrap_or("?");
            let c_type = id_type.get(&c).copied().unwrap_or("?");

            if is_noise_name(a_name) || is_noise_name(c_name) {
                continue;
            }

            let predicate = match (a_type, c_type) {
                ("person", "person") => "contemporary_of",
                ("concept", "concept") => "related_concept",
                ("place", "place") => "associated_with",
                ("person", "organization") | ("organization", "person") => "affiliated_with",
                _ => "related_to",
            };

            let intermediary_names: Vec<String> = pair_intermediaries[&(a, c)]
                .iter()
                .take(5)
                .filter_map(|&id| id_name.get(&id).map(|n| n.to_string()))
                .collect();

            hypotheses.push(Hypothesis {
                id: 0,
                subject: a_name.to_string(),
                predicate: predicate.to_string(),
                object: c_name.to_string(),
                confidence: 0.40 + (count as f64 * 0.08).min(0.35),
                evidence_for: vec![format!(
                    "{} and {} share {} mutual connections: {}",
                    a_name,
                    c_name,
                    count,
                    intermediary_names.join(", ")
                )],
                evidence_against: vec![],
                reasoning_chain: vec![
                    format!("Triadic closure: {} mutual neighbors", count),
                    format!("Intermediaries: {}", intermediary_names.join(", ")),
                    "Open triads tend to close in real-world networks".to_string(),
                ],
                status: HypothesisStatus::Proposed,
                discovered_at: now_str(),
                pattern_source: "triadic_closure".to_string(),
            });
        }
        Ok(hypotheses)
    }

    /// Aggressively purge single-word island entities that are clearly generic English words,
    /// citation surnames, adverbs, or adjectives. These entities have zero relations and
    /// zero facts — they contribute nothing to the knowledge graph.
    ///
    /// Heuristic: a single-word island is "generic" if it matches common English word patterns
    /// Merge connected entities where one name fully contains the other.
    /// Targets patterns like "Byzantine Empire Christianity" connected to "Byzantine Empire"
    /// — the longer form is usually a sentence-fragment extraction, not a real entity.
    /// Only merges when the shorter form has higher or equal degree and names share the same type.
    /// Returns count of merges performed.
    pub fn merge_connected_containment(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        // Build adjacency and degree
        let mut adj: HashSet<(i64, i64)> = HashSet::new();
        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            adj.insert((r.subject_id, r.object_id));
            adj.insert((r.object_id, r.subject_id));
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Build name→entity index for meaningful entities only
        let meaningful: Vec<&crate::db::Entity> = entities
            .iter()
            .filter(|e| !is_noise_type(&e.entity_type) && e.name.len() >= 3)
            .collect();

        // Index by first word for efficient lookup
        let mut by_first_word: HashMap<String, Vec<&crate::db::Entity>> = HashMap::new();
        for e in &meaningful {
            if let Some(first) = e.name.split_whitespace().next() {
                by_first_word
                    .entry(first.to_lowercase())
                    .or_default()
                    .push(e);
            }
        }

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        for e in &meaningful {
            if absorbed.contains(&e.id) {
                continue;
            }
            let e_words: Vec<&str> = e.name.split_whitespace().collect();
            if e_words.len() < 2 {
                continue;
            }
            let first_lower = e_words[0].to_lowercase();
            // Find potential shorter forms that this entity's name contains
            if let Some(candidates) = by_first_word.get(&first_lower) {
                for &cand in candidates {
                    if cand.id == e.id || absorbed.contains(&cand.id) {
                        continue;
                    }
                    // Check: is one name contained in the other?
                    let (shorter, longer) = if cand.name.len() < e.name.len() {
                        (cand, *e)
                    } else if cand.name.len() > e.name.len() {
                        (*e, cand)
                    } else {
                        continue;
                    };
                    let shorter_lower = shorter.name.to_lowercase();
                    let longer_lower = longer.name.to_lowercase();
                    // The longer name must start with or end with the shorter name
                    if !longer_lower.starts_with(&shorter_lower)
                        && !longer_lower.ends_with(&shorter_lower)
                    {
                        continue;
                    }
                    // Must be same type, or shorter has 3x+ degree (NLP mistype)
                    let short_deg = degree.get(&shorter.id).copied().unwrap_or(0);
                    let long_deg = degree.get(&longer.id).copied().unwrap_or(0);
                    if shorter.entity_type != longer.entity_type && short_deg < long_deg.max(1) * 3
                    {
                        continue;
                    }
                    // Must be connected (directly related)
                    let connected_pair = adj.contains(&(shorter.id, longer.id));
                    if !connected_pair {
                        continue;
                    }
                    // Shorter form should have >= degree of longer form
                    if short_deg < long_deg {
                        continue;
                    }
                    // The extra words in the longer name should be "noise-like"
                    let shorter_words: Vec<&str> = shorter.name.split_whitespace().collect();
                    let longer_words: Vec<&str> = longer.name.split_whitespace().collect();
                    let extra_words: Vec<&&str> = longer_words
                        .iter()
                        .filter(|w| !shorter_words.contains(w))
                        .collect();
                    // At least one extra word, and all extra words should be short or generic
                    if extra_words.is_empty() {
                        continue;
                    }

                    // Merge longer into shorter
                    self.brain.merge_entities(longer.id, shorter.id)?;
                    absorbed.insert(longer.id);
                    merged += 1;
                }
            }
        }
        Ok(merged)
    }

    /// Aggressive same-type prefix deduplication: merge entities whose name starts
    /// with a known shorter entity name of the SAME type. Unlike suffix_strip which
    /// only handles islands, this works on all entities and doesn't require the
    /// extra words to be in a noise list. For example:
    /// - "Byzantine Empire Diocletian" (concept) → "Byzantine Empire" (concept)
    /// - "Emmy Noether APSNews" (person) → "Emmy Noether" (person)
    /// - "Ada Lovelace WIRED" (person) → "Ada Lovelace" (person)
    /// Only merges when the target (shorter name) has strictly more connections.
    pub fn aggressive_prefix_dedup(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Build (lowercase_name → (id, degree, word_count, type)) for entities with 2+ words
        let mut name_index: HashMap<String, (i64, usize, usize, String)> = HashMap::new();
        for e in &entities {
            if is_noise_type(&e.entity_type) {
                continue;
            }
            let lower = e.name.to_lowercase();
            let wc = lower.split_whitespace().count();
            if wc < 2 {
                continue;
            }
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            let existing = name_index.get(&lower);
            if existing.is_none() || existing.is_some_and(|(_, d, _, _)| deg > *d) {
                name_index.insert(lower, (e.id, deg, wc, e.entity_type.clone()));
            }
        }

        // For each entity with 3+ words, check if a 2-word prefix exists with same type
        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        for e in &entities {
            if absorbed.contains(&e.id) || is_noise_type(&e.entity_type) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }
            let my_deg = degree.get(&e.id).copied().unwrap_or(0);

            // Try progressively shorter prefixes (keep at least 2 words)
            for take in (2..words.len()).rev() {
                let prefix: String = words[..take].join(" ").to_lowercase();
                if let Some(&(target_id, target_deg, _, ref target_type)) = name_index.get(&prefix)
                {
                    if target_id == e.id || absorbed.contains(&target_id) {
                        continue;
                    }
                    // Type check: same type required, OR target has 3x+ degree
                    // (high-degree target with matching prefix = NLP mistyped variant)
                    if target_type != &e.entity_type && target_deg < my_deg.max(1) * 3 {
                        continue;
                    }
                    // Target must be more connected (canonical)
                    if target_deg <= my_deg && my_deg > 0 {
                        continue;
                    }
                    self.brain.merge_entities(e.id, target_id)?;
                    absorbed.insert(e.id);
                    merged += 1;
                    break;
                }
            }
        }
        Ok(merged)
    }

    /// Merge any entity whose name is "KnownEntity + suffix" into KnownEntity,
    /// regardless of type or connectivity, when KnownEntity has 5x+ the degree.
    /// This catches NLP artifacts like "Ada Lovelace WIRED", "Byzantine Empire Rhomaioi" etc.
    pub fn merge_high_confidence_prefix_variants(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Build index: lowercase name → (id, degree) for entities with 2+ words and degree >= 5
        let mut canonical: HashMap<String, (i64, usize)> = HashMap::new();
        for e in &entities {
            if is_noise_type(&e.entity_type) {
                continue;
            }
            let wc = e.name.split_whitespace().count();
            if wc < 2 {
                continue;
            }
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            if deg < 5 {
                continue;
            }
            let lower = e.name.to_lowercase();
            let existing_deg = canonical.get(&lower).map(|(_, d)| *d).unwrap_or(0);
            if deg > existing_deg {
                canonical.insert(lower, (e.id, deg));
            }
        }

        let mut merged = 0usize;
        let mut absorbed: HashSet<i64> = HashSet::new();

        for e in &entities {
            if absorbed.contains(&e.id) || is_noise_type(&e.entity_type) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }
            let my_deg = degree.get(&e.id).copied().unwrap_or(0);

            // Try progressively shorter prefixes
            for take in (2..words.len()).rev() {
                let prefix = words[..take].join(" ").to_lowercase();
                if let Some(&(target_id, target_deg)) = canonical.get(&prefix) {
                    if target_id == e.id || absorbed.contains(&target_id) {
                        continue;
                    }
                    // Require target to have 3x+ degree (high confidence)
                    if target_deg < my_deg.max(1) * 3 {
                        continue;
                    }
                    self.brain.merge_entities(e.id, target_id)?;
                    absorbed.insert(e.id);
                    merged += 1;
                    break;
                }
            }
        }
        Ok(merged)
    }

    /// (adverbs ending in -ly, adjectives ending in -ous/-ive/-ful/-less, common nouns,
    /// past participles ending in -ed, gerunds ending in -ing, plurals of abstract concepts).
    /// We preserve single-word islands that are clearly proper nouns with known-entity signals.
    pub fn purge_generic_single_word_islands(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        let mut removed = 0usize;
        for e in &entities {
            if connected.contains(&e.id) {
                continue;
            }
            let facts = self.brain.get_facts_for(e.id)?;
            if !facts.is_empty() {
                continue;
            }
            let name = e.name.trim();
            let word_count = name.split_whitespace().count();
            if word_count != 1 {
                continue;
            }
            if should_purge_single_word(name, &e.entity_type) {
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                    Ok(())
                })?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Refine over-generic `associated_with` predicates using entity type pairs.
    /// E.g., person→place becomes "active_in", person→organization becomes "affiliated_with",
    /// concept→concept becomes "related_concept", person→person becomes "contemporary_of", etc.
    /// Returns count of relations updated.
    pub fn refine_associated_with(&self) -> Result<usize> {
        let relations = self.brain.all_relations()?;
        let entities = self.brain.all_entities()?;
        let id_to_type: HashMap<i64, &str> = entities
            .iter()
            .map(|e| (e.id, e.entity_type.as_str()))
            .collect();

        let mut updated = 0usize;
        for r in &relations {
            if r.predicate != "associated_with" {
                continue;
            }
            let s_type = id_to_type.get(&r.subject_id).copied().unwrap_or("unknown");
            let o_type = id_to_type.get(&r.object_id).copied().unwrap_or("unknown");
            let new_pred = match (s_type, o_type) {
                ("person", "place") | ("place", "person") => "active_in",
                ("person", "organization") | ("organization", "person") => "affiliated_with",
                ("person", "person") => "contemporary_of",
                ("person", "concept") => "contributed_to",
                ("concept", "person") => "pioneered_by",
                ("person", "event") | ("event", "person") => "participated_in",
                ("organization", "place") | ("place", "organization") => "based_in",
                ("organization", "organization") => "partner_of",
                ("organization", "concept") => "works_on",
                ("concept", "concept") => "related_concept",
                ("concept", "place") | ("place", "concept") => "relevant_to",
                ("event", "place") | ("place", "event") => "held_in",
                _ => continue,
            };
            self.brain.with_conn(|conn| {
                conn.execute(
                    "UPDATE relations SET predicate = ?1 WHERE id = ?2",
                    params![new_pred, r.id],
                )?;
                Ok(())
            })?;
            updated += 1;
        }
        Ok(updated)
    }

    /// Refine overly generic `contributed_to` predicates into more specific ones
    /// based on entity types. `contributed_to` makes up ~33% of all relations —
    /// specializing it improves semantic precision and discovery quality.
    pub fn refine_contributed_to(&self) -> Result<usize> {
        let relations = self.brain.all_relations()?;
        let entities = self.brain.all_entities()?;
        let id_to_type: HashMap<i64, &str> = entities
            .iter()
            .map(|e| (e.id, e.entity_type.as_str()))
            .collect();

        let mut updated = 0usize;
        for r in &relations {
            if r.predicate != "contributed_to" {
                continue;
            }
            let s_type = id_to_type.get(&r.subject_id).copied().unwrap_or("unknown");
            let o_type = id_to_type.get(&r.object_id).copied().unwrap_or("unknown");
            let new_pred = match (s_type, o_type) {
                ("person", "concept") => "pioneered",
                ("person", "organization") => "affiliated_with",
                ("person", "place") => "active_in",
                ("person", "event") => "participated_in",
                ("organization", "concept") => "works_on",
                ("organization", "event") => "organized",
                ("concept", "concept") => "influenced",
                ("place", "concept") | ("concept", "place") => "relevant_to",
                _ => continue,
            };
            self.brain.with_conn(|conn| {
                conn.execute(
                    "UPDATE relations SET predicate = ?1 WHERE id = ?2",
                    params![new_pred, r.id],
                )?;
                Ok(())
            })?;
            updated += 1;
        }
        Ok(updated)
    }

    /// Promote high-confidence testing hypotheses to confirmed if they've been
    /// testing for over `min_days` and confidence is above `min_conf`.
    /// This prevents hypothesis limbo — if nothing contradicts a plausible hypothesis
    /// after several discovery cycles, it's likely valid.
    pub fn promote_mature_hypotheses(&self, min_days: i64, min_conf: f64) -> Result<usize> {
        let cutoff = (Utc::now() - chrono::Duration::days(min_days))
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let count = self.brain.with_conn(|conn| {
            let updated = conn.execute(
                "UPDATE hypotheses SET status = 'confirmed' WHERE status = 'testing' AND confidence >= ?1 AND discovered_at < ?2",
                params![min_conf, cutoff],
            )?;
            Ok(updated)
        })?;
        Ok(count)
    }

    /// Decompose concatenated entity names: entities like "Caucasus Crimea Balkans"
    /// or "South-East Asia Africa" where multiple entity names got concatenated
    /// during NLP extraction. Splits them and creates relations to component entities.
    /// Returns count of relations created + entities cleaned up.
    pub fn split_concatenated_entities(&self) -> Result<(usize, usize)> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build name→id lookup for known entities (case-insensitive)
        let mut name_to_id: HashMap<String, i64> = HashMap::new();
        for e in &entities {
            if !is_noise_name(&e.name) && !is_noise_type(&e.entity_type) {
                name_to_id.entry(e.name.to_lowercase()).or_insert(e.id);
            }
        }

        let mut rels_created = 0usize;
        let mut entities_cleaned = 0usize;

        for e in &entities {
            if is_noise_name(&e.name) || is_noise_type(&e.entity_type) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }

            // Try all possible splits into 2+ known entity names
            // E.g., "Caucasus Crimea Balkans" → try "Caucasus" + "Crimea Balkans",
            //   "Caucasus Crimea" + "Balkans", etc.
            let mut found_components: Vec<(i64, String)> = Vec::new();
            for split_at in 1..words.len() {
                let left: String = words[..split_at].join(" ");
                let right: String = words[split_at..].join(" ");
                let left_lower = left.to_lowercase();
                let right_lower = right.to_lowercase();

                // Both parts must be known entities (different from this entity)
                if let (Some(&lid), Some(&rid)) =
                    (name_to_id.get(&left_lower), name_to_id.get(&right_lower))
                {
                    if lid != e.id && rid != e.id && lid != rid {
                        found_components.push((lid, left));
                        found_components.push((rid, right));
                        break;
                    }
                }
            }

            if found_components.len() >= 2 {
                // Create relations from this entity to its components
                for (comp_id, _comp_name) in &found_components {
                    self.brain
                        .upsert_relation(e.id, "references", *comp_id, "")?;
                    rels_created += 1;
                }
                // If this entity is an island, merge it away
                if !connected.contains(&e.id) {
                    let facts = self.brain.get_facts_for(e.id)?;
                    if facts.is_empty() {
                        self.brain.with_conn(|conn| {
                            conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                            Ok(())
                        })?;
                        entities_cleaned += 1;
                    }
                }
            }
        }
        Ok((rels_created, entities_cleaned))
    }

    /// Dissolve single-word name fragment hubs.
    ///
    /// Entities like "Charles" (concept, 35 relations) are NLP extraction artifacts
    /// that absorb connections meant for real entities ("Charles Babbage", "Charles Darwin").
    /// This method:
    /// 1. Finds single-word entities with high degree that look like name fragments
    /// 2. For each, checks if multi-word entities exist containing that name
    /// 3. If so, merges the fragment into the most connected matching entity
    /// 4. Or if the fragment is just a hub linking unrelated "Charles *" entities,
    ///    deletes it and its relations (they're noise)
    ///
    /// Returns count of fragments dissolved (merged or removed).
    pub fn dissolve_name_fragment_hubs(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Known real single-word entities that should NOT be dissolved
        let real_single_words: HashSet<&str> = [
            // Countries & major places
            "China",
            "France",
            "Germany",
            "Japan",
            "India",
            "Russia",
            "Italy",
            "Spain",
            "Brazil",
            "Canada",
            "Mexico",
            "Egypt",
            "Iran",
            "Iraq",
            "Turkey",
            "Greece",
            "Sweden",
            "Norway",
            "Finland",
            "Denmark",
            "Poland",
            "Austria",
            "Belgium",
            "Portugal",
            "Romania",
            "Hungary",
            "Ireland",
            "Scotland",
            "England",
            "Wales",
            "Africa",
            "Asia",
            "Europe",
            "America",
            "Antarctica",
            "Australia",
            "Paris",
            "London",
            "Berlin",
            "Rome",
            "Tokyo",
            "Moscow",
            "Vienna",
            "Prague",
            "Madrid",
            "Lisbon",
            "Athens",
            "Cairo",
            "Baghdad",
            "Tehran",
            "Delhi",
            "Zurich",
            "Geneva",
            "Bern",
            "Basel",
            "Lausanne",
            "Lucerne",
            "Svalbard",
            "Crimea",
            "Balkans",
            "Caucasus",
            "Anatolia",
            "Mesopotamia",
            // Major concepts/things
            "Internet",
            "Bitcoin",
            "Linux",
            "Wikipedia",
            "Amazon",
            "Google",
            "Apple",
            "Microsoft",
            "Tesla",
            "Netflix",
            "Facebook",
            "Twitter",
            "YouTube",
            "DNA",
            "RNA",
            "CRISPR",
            "NATO",
            "UNESCO",
            "UNICEF",
            // Historical entities
            "Renaissance",
            "Reformation",
            "Enlightenment",
        ]
        .iter()
        .copied()
        .collect();

        // Build lookup: lowercase_name → Vec<(id, degree, name)> for multi-word entities
        let mut multiword_by_word: HashMap<String, Vec<(i64, usize, String)>> = HashMap::new();
        for e in &entities {
            if is_noise_type(&e.entity_type) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() >= 2 {
                let deg = degree.get(&e.id).copied().unwrap_or(0);
                for w in &words {
                    multiword_by_word
                        .entry(w.to_lowercase())
                        .or_default()
                        .push((e.id, deg, e.name.clone()));
                }
            }
        }

        let mut dissolved = 0usize;

        // Debug: count how many single-word entities with deg>=5 we find
        let mut dbg_count = 0usize;
        for e in &entities {
            let name = e.name.trim();
            if name.split_whitespace().count() == 1 && !is_noise_type(&e.entity_type) {
                let deg = degree.get(&e.id).copied().unwrap_or(0);
                if deg >= 5 && !real_single_words.contains(name) {
                    dbg_count += 1;
                    if dbg_count <= 5 {
                        eprintln!(
                            "  [dissolve-debug] '{}' type={} deg={}",
                            name, e.entity_type, deg
                        );
                    }
                }
            }
        }
        // eprintln!("  [dissolve] total candidates with deg>=5: {}", dbg_count);

        for e in &entities {
            if is_noise_type(&e.entity_type) {
                continue;
            }
            let name = e.name.trim();
            let word_count = name.split_whitespace().count();
            if word_count != 1 {
                continue;
            }
            // Skip known real entities
            if real_single_words.contains(name) {
                continue;
            }
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            if deg < 5 {
                continue; // Only target high-degree fragments
            }

            // Check: does this look like a name fragment?
            let lower = name.to_lowercase();
            let matching = multiword_by_word.get(&lower).cloned().unwrap_or_default();

            if name == "Charles" || name == "Noether" || name == "Lovelace" {
                eprintln!(
                    "  [dissolve-trace] '{}' deg={} matching.len()={}",
                    name,
                    deg,
                    matching.len()
                );
            }

            if matching.len() < 2 {
                continue; // Not enough evidence it's a fragment
            }
            eprintln!(
                "  [dissolve] candidate: '{}' (deg={}, {} matching multi-word entities)",
                name,
                deg,
                matching.len()
            );

            // This is a fragment hub. Find the best target to merge into.
            // Best = most connected multi-word entity containing this name.
            let mut best_target: Option<(i64, usize, String)> = None;
            for (mid, mdeg, mname) in &matching {
                if *mid == e.id {
                    continue;
                }
                if best_target.is_none() || mdeg > &best_target.as_ref().unwrap().1 {
                    best_target = Some((*mid, *mdeg, mname.clone()));
                }
            }

            if let Some((target_id, _target_deg, _target_name)) = best_target {
                // Merge fragment into best matching multi-word entity
                self.brain.merge_entities(e.id, target_id)?;
                dissolved += 1;
            }
        }
        Ok(dissolved)
    }

    /// Strip leading adjectives/demonyms from entity names.
    /// E.g., "American Eli Whitney" → merge into "Eli Whitney",
    /// "French Marie Curie" → merge into "Marie Curie".
    /// Only acts when the stripped version exists as a higher-degree entity.
    pub fn strip_leading_adjectives(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Build name → (id, degree) lookup
        let mut name_lookup: HashMap<String, (i64, usize)> = HashMap::new();
        for e in &entities {
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            let lower = e.name.to_lowercase();
            // Keep highest-degree version
            let entry = name_lookup.entry(lower).or_insert((e.id, deg));
            if deg > entry.1 {
                *entry = (e.id, deg);
            }
        }

        // Leading words that are likely adjectives/demonyms, not part of proper names
        let leading_adjectives: HashSet<&str> = [
            "american",
            "british",
            "french",
            "german",
            "italian",
            "spanish",
            "russian",
            "chinese",
            "japanese",
            "indian",
            "canadian",
            "australian",
            "dutch",
            "swiss",
            "swedish",
            "norwegian",
            "danish",
            "finnish",
            "polish",
            "austrian",
            "belgian",
            "portuguese",
            "hungarian",
            "irish",
            "scottish",
            "english",
            "welsh",
            "greek",
            "turkish",
            "persian",
            "arab",
            "african",
            "asian",
            "european",
            "latin",
            "western",
            "eastern",
            "northern",
            "southern",
            "central",
            "ancient",
            "modern",
            "medieval",
            "classical",
            "early",
            "late",
            "young",
            "old",
            "great",
            "little",
            "big",
            "new",
            "former",
            "professor",
            "dr",
            "sir",
            "lord",
            "king",
            "queen",
            "emperor",
            "admiral",
            "general",
            "generals",
            "colonel",
            "captain",
            "marshal",
            "field-marshal",
            "sergeant",
            "lieutenant",
            "commander",
            "commodore",
            "minister",
            "governor",
            "archbishop",
            "bishop",
            "sultan",
            "kaiser",
            "tsar",
            "czar",
            "shah",
            "prince",
            "princess",
            "duke",
            "duchess",
            "count",
            "countess",
            "baron",
            "baroness",
            "cardinal",
            "pope",
            "saint",
            "san",
            "scientific",
            "royal",
            "imperial",
            "national",
            "international",
        ]
        .iter()
        .copied()
        .collect();

        // Country/region names that appear as prefixes in NLP extraction errors
        // e.g. "Netherlands Oskar Klein", "Switzerland CERN", "Japan Toyota"
        let country_prefixes: HashSet<&str> = [
            "netherlands",
            "switzerland",
            "germany",
            "france",
            "italy",
            "spain",
            "portugal",
            "belgium",
            "austria",
            "sweden",
            "norway",
            "denmark",
            "finland",
            "poland",
            "hungary",
            "romania",
            "bulgaria",
            "serbia",
            "croatia",
            "greece",
            "turkey",
            "russia",
            "ukraine",
            "japan",
            "korea",
            "taiwan",
            "thailand",
            "vietnam",
            "indonesia",
            "malaysia",
            "singapore",
            "philippines",
            "brazil",
            "argentina",
            "mexico",
            "colombia",
            "chile",
            "peru",
            "egypt",
            "israel",
            "iran",
            "iraq",
            "pakistan",
            "bangladesh",
            "nigeria",
            "kenya",
            "ethiopia",
            "tanzania",
            "australia",
            "zealand",
            "ireland",
            "scotland",
            "england",
            "wales",
            "canada",
            "cuba",
            "prussia",
            "saxony",
            "bavaria",
            "bohemia",
            "catalonia",
            "lombardy",
            "tuscany",
            "andalusia",
            "normandy",
        ]
        .iter()
        .copied()
        .collect();

        // Combine all strippable prefixes
        let all_prefixes: HashSet<&str> = leading_adjectives
            .union(&country_prefixes)
            .copied()
            .collect();

        let mut merged = 0usize;
        for e in &entities {
            let words: Vec<&str> = e.name.split_whitespace().collect();
            // For person entities, allow stripping to 2-word result (e.g. "Netherlands Oskar Klein" → "Oskar Klein")
            // For other types, still require 3+ words
            let min_words = if e.entity_type == "person" { 2 } else { 3 };
            if words.len() < min_words + 1 {
                continue;
            }
            let first_lower = words[0].to_lowercase();
            if !all_prefixes.contains(first_lower.as_str()) {
                continue;
            }
            // Try stripping the first word
            let stripped = words[1..].join(" ");
            let stripped_lower = stripped.to_lowercase();
            if let Some(&(target_id, _target_deg)) = name_lookup.get(&stripped_lower) {
                if target_id == e.id {
                    continue;
                }
                let _my_deg = degree.get(&e.id).copied().unwrap_or(0);
                // Merge the adjective-prefixed form into the canonical stripped form.
                // The stripped form is the correct name regardless of degree.
                eprintln!(
                    "  [adj-strip] merging '{}' (id={}) → '{}' (id={})",
                    e.name, e.id, stripped, target_id
                );
                self.brain.merge_entities(e.id, target_id)?;
                merged += 1;
            }
        }
        Ok(merged)
    }

    /// Information-theoretic entity scoring: rank entities by how much they contribute
    /// to the knowledge graph's information content. Uses a combination of:
    /// - Structural importance (betweenness centrality proxy via degree * clustering)
    /// - Uniqueness (inverse of how many similar entities exist)
    /// - Connectivity quality (ratio of diverse predicates to total degree)
    /// Returns (entity_name, entity_type, info_score) sorted descending.
    pub fn information_content_ranking(&self, limit: usize) -> Result<Vec<(String, String, f64)>> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        let mut predicates: HashMap<i64, HashSet<String>> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
            predicates
                .entry(r.subject_id)
                .or_default()
                .insert(r.predicate.clone());
            predicates
                .entry(r.object_id)
                .or_default()
                .insert(r.predicate.clone());
        }

        // Type frequency for uniqueness scoring
        let mut type_count: HashMap<String, usize> = HashMap::new();
        for e in &entities {
            *type_count.entry(e.entity_type.clone()).or_insert(0) += 1;
        }

        let total_entities = entities.len().max(1) as f64;
        let mut scores: Vec<(String, String, f64)> = Vec::new();

        for e in &entities {
            if is_noise_type(&e.entity_type) || is_noise_name(&e.name) {
                continue;
            }
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            if deg == 0 {
                continue;
            }

            let pred_diversity = predicates.get(&e.id).map(|s| s.len()).unwrap_or(0) as f64;
            let type_freq = type_count.get(&e.entity_type).copied().unwrap_or(1) as f64;

            // Information score components:
            // 1. Connectivity: log(degree + 1) — diminishing returns for hubs
            let connectivity = (deg as f64 + 1.0).ln();
            // 2. Predicate diversity: more diverse connections = more informative
            let diversity = pred_diversity / (deg as f64).max(1.0);
            // 3. Type uniqueness: rarer types are more informative
            let uniqueness = (total_entities / type_freq).ln().max(0.1);
            // 4. Fact richness
            let facts = self.brain.get_facts_for(e.id)?.len() as f64;
            let fact_bonus = (facts + 1.0).ln() * 0.5;

            let info_score = connectivity * diversity * uniqueness + fact_bonus;
            scores.push((e.name.clone(), e.entity_type.clone(), info_score));
        }

        scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        Ok(scores)
    }

    /// Token-based island reconnection: match island entities to connected entities
    /// via significant shared name tokens. E.g., island "Möngke Khan" connects to
    /// "Genghis Khan" via shared token "Khan". Uses TF-IDF-like weighting — rare
    /// tokens that appear in few entity names are more informative than common ones.
    /// Only connects when the shared tokens are significant (not stopwords, not too common).
    pub fn reconnect_islands_by_tokens(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Build direct-connection set
        let mut edges: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            edges.insert(key);
        }

        // Tokenize all entity names, compute document frequency per token
        let stopwords: HashSet<&str> = [
            "the",
            "of",
            "and",
            "in",
            "on",
            "for",
            "to",
            "by",
            "from",
            "with",
            "at",
            "an",
            "or",
            "a",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "has",
            "had",
            "have",
            "do",
            "does",
            "did",
            "not",
            "no",
            "but",
            "if",
            "as",
            "it",
            "its",
            "his",
            "her",
            "he",
            "she",
            "they",
            "their",
            "this",
            "that",
            "which",
            "who",
            "whom",
            "de",
            "la",
            "le",
            "les",
            "du",
            "des",
            "von",
            "van",
            "der",
            "den",
            "di",
            "el",
            // Geographic/descriptive terms that cause spurious token matches
            "sea",
            "basin",
            "island",
            "islands",
            "cape",
            "bay",
            "gulf",
            "strait",
            "river",
            "lake",
            "mountain",
            "mount",
            "valley",
            "desert",
            "forest",
            "park",
            "port",
            "north",
            "south",
            "east",
            "west",
            "northern",
            "southern",
            "eastern",
            "western",
            "new",
            "old",
            "great",
            "big",
            "little",
            "upper",
            "lower",
            "central",
            // Academic/publication terms
            "university",
            "college",
            "institute",
            "school",
            "academy",
            "society",
            "journal",
            "review",
            "press",
            "news",
            "radio",
            "time",
            "standard",
            // Common descriptors that link unrelated entities
            "national",
            "international",
            "royal",
            "imperial",
            "federal",
            "state",
            "general",
            "special",
            "grand",
            "holy",
            "sacred",
            "ancient",
            "modern",
            // Common surname-like words that cause false person matches
            "deep",
            "ridge",
            "dover",
            "london",
            "paris",
            "berlin",
            "prize",
            "award",
            "medal",
            "order",
            "courier",
            "herald",
            // Common English words that cause spurious token matches
            "you",
            "your",
            "all",
            "one",
            "two",
            "three",
            "first",
            "last",
            "next",
            "only",
            "just",
            "more",
            "most",
            "some",
            "many",
            "any",
            "each",
            "every",
            "other",
            "such",
            "same",
            "way",
            "day",
            "year",
            "man",
            "men",
            "out",
            "into",
            "over",
            "also",
            "after",
            "before",
            "how",
            "why",
            "what",
            "when",
            "where",
            "there",
            "here",
            "about",
            "between",
            "through",
            "during",
            "under",
            "along",
            "both",
            "well",
            "back",
            "even",
            "still",
            "then",
            "than",
            "very",
            "too",
            "much",
            "now",
            "long",
            "made",
            "make",
            "like",
            "will",
            "can",
            "may",
            "could",
            "would",
            "should",
            "use",
            "used",
            "core",
            "easy",
            "fast",
            "high",
            "low",
            "left",
            "right",
            "real",
            "true",
            "full",
            "good",
            "best",
            "need",
            "take",
            "give",
            "find",
            "know",
            "come",
            "part",
            "work",
            "world",
            "life",
            "being",
            "place",
            "thing",
            "point",
            "small",
            "large",
            "early",
            "late",
            "half",
            "end",
            "side",
            "line",
            "land",
            "head",
            "hand",
            "eye",
            "face",
            "book",
            "war",
            "bury",
            "compare",
            // Common nouns that cause spurious cross-domain matches
            "cube",
            "theory",
            "problem",
            "system",
            "model",
            "code",
            "map",
            "maps",
            "guide",
            "lost",
            "red",
            "blue",
            "green",
            "black",
            "white",
            "golden",
            "silver",
            "dark",
            "light",
            "star",
            "sun",
            "moon",
            "fire",
            "ice",
            "iron",
            "steel",
            "stone",
            "rock",
            "sand",
            "snow",
            "storm",
            "wind",
            "rain",
            "cloud",
            "wave",
            "twin",
            "double",
            "triple",
            "super",
            "mega",
            "ultra",
            "mini",
            "micro",
            "nano",
            "digital",
            "analog",
            "hybrid",
            "memory",
            "power",
            "energy",
            "force",
            "speed",
            "test",
            "trial",
            "game",
            "play",
            "race",
            "match",
            "fight",
            "rule",
            "rules",
            "act",
            "finally",
            "scientist",
            "tri",
            "bin",
            "hasan",
            "webb",
            "revenge",
            "encoder",
            "belt",
            "road",
        ]
        .into_iter()
        .collect();

        // Token → entity IDs (only for meaningful, non-noise entities)
        let mut token_entities: HashMap<String, Vec<i64>> = HashMap::new();
        let mut entity_tokens: HashMap<i64, Vec<String>> = HashMap::new();
        let entity_map: HashMap<i64, &crate::db::Entity> =
            entities.iter().map(|e| (e.id, e)).collect();

        for e in &entities {
            if !meaningful.contains(&e.id)
                || is_noise_name(&e.name)
                || is_noise_type(&e.entity_type)
            {
                continue;
            }
            let tokens: Vec<String> = e
                .name
                .split_whitespace()
                .filter(|w| w.len() >= 3 && !stopwords.contains(&w.to_lowercase().as_str()))
                .map(|w| w.to_lowercase())
                .collect();
            for t in &tokens {
                token_entities.entry(t.clone()).or_default().push(e.id);
            }
            entity_tokens.insert(e.id, tokens);
        }

        let total_entities = entities.len().max(1) as f64;

        // For each island entity, find best connected match via shared significant tokens
        let mut reconnected = 0usize;
        let islands: Vec<i64> = entities
            .iter()
            .filter(|e| {
                meaningful.contains(&e.id)
                    && !connected.contains(&e.id)
                    && !is_noise_name(&e.name)
                    && !is_noise_type(&e.entity_type)
                    && e.name.split_whitespace().count() >= 2 // At least 2-word names for token matching
            })
            .map(|e| e.id)
            .collect();

        for &island_id in &islands {
            let island_toks = match entity_tokens.get(&island_id) {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };

            // Score each connected entity by TF-IDF-like shared token weight
            let mut candidates: HashMap<i64, f64> = HashMap::new();
            for tok in island_toks {
                let doc_freq = token_entities.get(tok).map(|v| v.len()).unwrap_or(0) as f64;
                if doc_freq < 2.0 || doc_freq > total_entities * 0.1 {
                    continue; // Too rare (unique to island) or too common
                }
                let idf = (total_entities / doc_freq).ln();
                if let Some(matching_ids) = token_entities.get(tok) {
                    for &mid in matching_ids {
                        if mid == island_id || !connected.contains(&mid) {
                            continue;
                        }
                        let key = if island_id < mid {
                            (island_id, mid)
                        } else {
                            (mid, island_id)
                        };
                        if edges.contains(&key) {
                            continue;
                        }
                        *candidates.entry(mid).or_insert(0.0) += idf;
                    }
                }
            }

            // Find best candidate — require significant shared tokens
            if let Some((&best_id, &best_score)) = candidates
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                // Require minimum score (lower for person-type since surname sharing is strong signal)
                let min_score = if entity_map.get(&island_id).map(|e| e.entity_type.as_str())
                    == Some("person")
                {
                    4.0
                } else {
                    5.0
                };
                if best_score < min_score {
                    continue;
                }
                // Count actual shared tokens (not just score)
                let best_toks = entity_tokens.get(&best_id);
                let shared_count = island_toks
                    .iter()
                    .filter(|t| best_toks.map(|bt| bt.contains(t)).unwrap_or(false))
                    .count();
                // Require ≥2 shared tokens for non-person entities to avoid
                // single-word spurious matches like "Rubik Cube" ↔ "Hybrid Memory Cube"
                let is_person =
                    entity_map.get(&island_id).map(|e| e.entity_type.as_str()) == Some("person");
                if !is_person && shared_count < 2 && island_toks.len() >= 2 {
                    continue;
                }
                let island_name = entity_map
                    .get(&island_id)
                    .map(|e| e.name.as_str())
                    .unwrap_or("?");
                let target_name = entity_map
                    .get(&best_id)
                    .map(|e| e.name.as_str())
                    .unwrap_or("?");
                let island_type = entity_map
                    .get(&island_id)
                    .map(|e| e.entity_type.as_str())
                    .unwrap_or("?");
                let target_type = entity_map
                    .get(&best_id)
                    .map(|e| e.entity_type.as_str())
                    .unwrap_or("?");

                // For person-person matches, require shared LAST name (not just first name).
                // "Frank Sinatra" and "Denis Frank" share "frank" but are unrelated.
                // Require that the shared tokens include the last token of at least one name.
                if island_type == "person" && target_type == "person" {
                    let island_last = island_name
                        .split_whitespace()
                        .last()
                        .unwrap_or("")
                        .to_lowercase();
                    let target_last = target_name
                        .split_whitespace()
                        .last()
                        .unwrap_or("")
                        .to_lowercase();
                    let target_toks = entity_tokens.get(&best_id);
                    let shared_toks: Vec<&String> = island_toks
                        .iter()
                        .filter(|t| target_toks.map(|tt| tt.contains(t)).unwrap_or(false))
                        .collect();
                    // Must share a last name token, not just first names
                    let shares_surname = shared_toks
                        .iter()
                        .any(|t| **t == island_last || **t == target_last);
                    if !shares_surname {
                        continue;
                    }
                    // Require ≥2 shared tokens for person-person (avoid "John X" ↔ "John Y")
                    if shared_toks.len() < 2 && island_toks.len() >= 2 {
                        continue;
                    }
                }

                // For person ↔ non-person matches, require that shared tokens
                // include the person's surname (last word), not just first name.
                // Avoids "Don Backer" ↔ "Don River" (shares "don" = first name only).
                if (island_type == "person") != (target_type == "person") {
                    let (person_name, person_toks) = if island_type == "person" {
                        (island_name, island_toks.as_slice())
                    } else {
                        (target_name, best_toks.map(|t| t.as_slice()).unwrap_or(&[]))
                    };
                    if person_name.split_whitespace().count() >= 2 {
                        let person_last = person_name
                            .split_whitespace()
                            .last()
                            .unwrap_or("")
                            .to_lowercase();
                        let other_toks = if island_type == "person" {
                            best_toks.map(|t| t.as_slice()).unwrap_or(&[])
                        } else {
                            island_toks.as_slice()
                        };
                        let shares_surname = other_toks.iter().any(|t| *t == person_last);
                        if !shares_surname {
                            continue;
                        }
                    }
                }

                // Skip connections where one entity name contains noise words suggesting
                // it's a sentence fragment, not a real entity (e.g., "Stalin Carpet",
                // "Dead Christ", "Good-bye David Bowie")
                let noise_context_words = [
                    "carpet",
                    "dead",
                    "goodbye",
                    "good-bye",
                    "let",
                    "wealth",
                    "textbooks",
                    "traces",
                    "fever",
                    "goddamn",
                    "god",
                    "particle",
                    "wall",
                    "came",
                    "goes",
                    "down",
                    "preceded",
                    "succeeded",
                    "mint",
                    "re-explained",
                    "piled",
                    "higher",
                    "next-level",
                ];
                let island_lower = island_name.to_lowercase();
                let target_lower = target_name.to_lowercase();
                let has_noise_context = island_lower
                    .split_whitespace()
                    .any(|w| noise_context_words.contains(&w))
                    || target_lower
                        .split_whitespace()
                        .any(|w| noise_context_words.contains(&w));
                if has_noise_context {
                    continue;
                }

                // Determine predicate based on types
                let predicate = match (island_type, target_type) {
                    ("person", "person") => "contemporary_of",
                    ("place", "place") => "associated_with",
                    ("concept", "concept") => "related_concept",
                    ("person", "organization") | ("organization", "person") => "affiliated_with",
                    ("person", "place") | ("place", "person") => "associated_with",
                    _ => "associated_with",
                };

                eprintln!(
                    "  [token-reconnect] {} → {} (score: {:.2}, pred: {})",
                    island_name, target_name, best_score, predicate
                );
                self.brain.upsert_relation(
                    island_id,
                    predicate,
                    best_id,
                    "prometheus:token_reconnect",
                )?;
                reconnected += 1;
                if reconnected >= 200 {
                    break; // Cap per run
                }
            }
        }
        Ok(reconnected)
    }

    /// Connect island entities to connected entities of the same type where the
    /// island name is a substring of the connected entity name (or vice versa),
    /// weighted by name overlap ratio. E.g., island "Fourier Transform" → connected
    /// "Discrete Fourier Transform" with high confidence.
    /// Returns count of new relations created.
    pub fn reconnect_islands_by_name_containment(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        let mut connected: HashSet<i64> = HashSet::new();
        let mut edges: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            edges.insert(key);
        }

        // Build type → connected entities index
        let mut type_connected: HashMap<String, Vec<(i64, String)>> = HashMap::new();
        for e in &entities {
            if meaningful.contains(&e.id)
                && connected.contains(&e.id)
                && !is_noise_type(&e.entity_type)
                && !is_noise_name(&e.name)
                && e.name.split_whitespace().count() >= 2
            {
                type_connected
                    .entry(e.entity_type.clone())
                    .or_default()
                    .push((e.id, e.name.to_lowercase()));
            }
        }

        let mut reconnected = 0usize;
        for e in &entities {
            if connected.contains(&e.id)
                || !meaningful.contains(&e.id)
                || is_noise_type(&e.entity_type)
                || is_noise_name(&e.name)
                || e.name.split_whitespace().count() < 2
            {
                continue;
            }
            let island_lower = e.name.to_lowercase();
            let island_words: usize = island_lower.split_whitespace().count();

            // Look for same-type connected entities where one name contains the other
            if let Some(candidates) = type_connected.get(&e.entity_type) {
                let mut best: Option<(i64, f64)> = None;
                for (cid, cname) in candidates {
                    let key = if e.id < *cid {
                        (e.id, *cid)
                    } else {
                        (*cid, e.id)
                    };
                    if edges.contains(&key) {
                        continue;
                    }
                    let cand_words: usize = cname.split_whitespace().count();

                    // One must contain the other, and the shorter must be ≥2 words
                    let (shorter, longer): (&str, &str) = if island_words <= cand_words {
                        (island_lower.as_str(), cname.as_str())
                    } else {
                        (cname.as_str(), island_lower.as_str())
                    };
                    let shorter_words = shorter.split_whitespace().count();
                    if shorter_words < 2 {
                        continue;
                    }
                    if longer.contains(shorter) {
                        // Overlap ratio: how much of the longer name is covered
                        let ratio = shorter.len() as f64 / longer.len().max(1) as f64;
                        if ratio > 0.4 && (best.is_none() || ratio > best.unwrap().1) {
                            best = Some((*cid, ratio));
                        }
                    }
                }

                if let Some((target_id, _ratio)) = best {
                    let predicate = if island_words
                        < type_connected
                            .get(&e.entity_type)
                            .map(|v| {
                                v.iter()
                                    .find(|(id, _)| *id == target_id)
                                    .map(|(_, n)| n.split_whitespace().count())
                                    .unwrap_or(0)
                            })
                            .unwrap_or(0)
                    {
                        "broader_form_of"
                    } else {
                        "specific_form_of"
                    };
                    self.brain.upsert_relation(
                        e.id,
                        predicate,
                        target_id,
                        "prometheus:name_containment_reconnect",
                    )?;
                    reconnected += 1;
                    if reconnected >= 150 {
                        break;
                    }
                }
            }
        }
        Ok(reconnected)
    }

    /// Reconnect single-word island entities to connected entities whose name contains
    /// that word as a significant token. E.g., island "Entropy" → connected "Shannon Entropy".
    /// Uses type-aware scoring: same-type matches are preferred, and high-value types
    /// (person, place, concept, technology) get a boost.
    /// Only reconnects when the match is unambiguous (one clearly best candidate).
    pub fn reconnect_single_word_islands(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;

        let mut connected: HashSet<i64> = HashSet::new();
        let mut edges: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            edges.insert(key);
        }

        // Build inverted index: lowercase token → list of connected entity IDs
        let entity_map: HashMap<i64, &crate::db::Entity> =
            entities.iter().map(|e| (e.id, e)).collect();
        let mut token_to_connected: HashMap<String, Vec<i64>> = HashMap::new();
        for e in &entities {
            if !meaningful.contains(&e.id)
                || !connected.contains(&e.id)
                || is_noise_type(&e.entity_type)
                || is_noise_name(&e.name)
            {
                continue;
            }
            // Index each significant word token
            for word in e.name.split_whitespace() {
                let lower = word.to_lowercase();
                if lower.len() >= 3 {
                    token_to_connected.entry(lower).or_default().push(e.id);
                }
            }
        }

        // Collect single-word island entities that are meaningful
        let islands: Vec<&crate::db::Entity> = entities
            .iter()
            .filter(|e| {
                meaningful.contains(&e.id)
                    && !connected.contains(&e.id)
                    && !is_noise_type(&e.entity_type)
                    && !is_noise_name(&e.name)
                    && e.name.split_whitespace().count() == 1
                    && e.name.len() >= 3
            })
            .collect();

        let mut reconnected = 0usize;
        for island in &islands {
            let island_lower = island.name.to_lowercase();

            // Find connected entities containing this word
            let candidates = match token_to_connected.get(&island_lower) {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };

            // Too many matches = too generic, skip
            if candidates.len() > 20 {
                continue;
            }

            // Score candidates: prefer same type, prefer shorter names (more specific match)
            let mut scored: Vec<(i64, f64)> = Vec::new();
            for &cid in candidates {
                let key = if island.id < cid {
                    (island.id, cid)
                } else {
                    (cid, island.id)
                };
                if edges.contains(&key) {
                    continue;
                }
                let cand = match entity_map.get(&cid) {
                    Some(e) => e,
                    None => continue,
                };
                let mut score = 1.0_f64;
                // Same type bonus
                if cand.entity_type == island.entity_type {
                    score += 2.0;
                }
                // High-value type bonus
                if HIGH_VALUE_TYPES.contains(&cand.entity_type.as_str()) {
                    score += 0.5;
                }
                // Shorter name = more specific match (word is bigger fraction of name)
                let name_words = cand.name.split_whitespace().count().max(1) as f64;
                score += 1.0 / name_words;
                // Exact word-boundary match bonus (not just substring)
                let cand_tokens: Vec<String> = cand
                    .name
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .collect();
                if cand_tokens.contains(&island_lower) {
                    score += 1.0;
                }
                scored.push((cid, score));
            }

            if scored.is_empty() {
                continue;
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Only reconnect if best candidate is clearly better than runner-up
            let best_score = scored[0].1;
            let runner_up = scored.get(1).map(|s| s.1).unwrap_or(0.0);
            if scored.len() > 1 && best_score < runner_up * 1.3 {
                continue; // Ambiguous — skip
            }

            let target_id = scored[0].0;
            let target_name = entity_map
                .get(&target_id)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let predicate = match (
                island.entity_type.as_str(),
                entity_map
                    .get(&target_id)
                    .map(|e| e.entity_type.as_str())
                    .unwrap_or("?"),
            ) {
                ("person", "person") => "related_person",
                ("place", "place") => "associated_with",
                ("concept", "concept") => "related_concept",
                ("technology", "technology") => "related_technology",
                _ => "associated_with",
            };

            eprintln!(
                "  [single-word-reconnect] {} ({}) → {} (score: {:.2}, pred: {})",
                island.name, island.entity_type, target_name, best_score, predicate
            );
            self.brain.upsert_relation(
                island.id,
                predicate,
                target_id,
                "prometheus:single_word_reconnect",
            )?;
            reconnected += 1;
            if reconnected >= 300 {
                break;
            }
        }
        Ok(reconnected)
    }

    /// Purge island entities mistyped as "person" that are clearly not people.
    /// Catches patterns like "Gravitational Waves" (concept), "Modern Japan" (place/concept),
    /// "Collected Works" (concept), etc.
    pub fn purge_mistyped_person_islands(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;
        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Non-person indicators: words that strongly suggest the entity is NOT a person
        let concept_words: HashSet<&str> = [
            "waves",
            "theory",
            "theorem",
            "equation",
            "equations",
            "method",
            "methods",
            "algorithm",
            "process",
            "effect",
            "phenomenon",
            "principle",
            "law",
            "laws",
            "paradox",
            "conjecture",
            "hypothesis",
            "model",
            "formula",
            "series",
            "function",
            "transform",
            "distribution",
            "constant",
            "number",
            "numbers",
            "problem",
            "space",
            "group",
            "ring",
            "field",
            "operator",
            "integral",
            "differential",
            "system",
            "systems",
            "machine",
            "engine",
            "device",
            "network",
            "protocol",
            "code",
            "language",
            "architecture",
            "framework",
            "structure",
            "works",
            "collection",
            "collected",
            "correspondence",
            "letters",
            "papers",
            "manuscript",
            "text",
            "book",
            "edition",
            "volume",
            "treaty",
            "declaration",
            "manifesto",
            "charter",
            "revolution",
            "movement",
            "campaign",
            "battle",
            "war",
            "siege",
            "invasion",
            "expedition",
            "conquest",
            "migration",
            "diaspora",
            "genocide",
            "massacre",
            "crisis",
            "reform",
            "policy",
            "agreement",
            "alliance",
            "coalition",
            "prices",
            "trade",
            "market",
            "economy",
            "currency",
            "values",
            "money",
            "modern",
            "ancient",
            "medieval",
            "classical",
            "contemporary",
            "early",
            "late",
            "online",
            "digital",
            "virtual",
            "mobile",
            "wireless",
            "syndrome",
            "doctrine",
            "ideology",
            "heresy",
            "orthodoxy",
            "islam",
            "christianity",
            "buddhism",
            "hinduism",
            "judaism",
            "crown",
            "throne",
            "dynasty",
            "braid",
            "golden",
            "eternal",
            "sacred",
            "holy",
            "divine",
            "celestial",
            "infernal",
            "captains",
            "great",
            "grand",
            "royal",
            "imperial",
            "nuovo",
            "cimento",
            "prinzipien",
            "physikalische",
            "friedhöfen",
            "minería",
            "sublemma",
            "raspberry",
            "powerful",
            "dissenting",
            "academies",
            "academy",
            "penny",
            "post",
            "exchange",
            "steam",
            "power",
            "fire",
            "trials",
            "brothers",
            // German institutional suffixes
            "gesellschaft",
            "hochschule",
            "nachrichten",
            "jahrbuch",
            "akademie",
            "verein",
            "zeitschrift",
            "institut",
            "bibliothek",
            "archiv",
            "anzeiger",
            "berichte",
            "abhandlungen",
            "kommmission",
            "kommission",
            "cultusgemeinde",
            "gemeinde",
            "figuren",
            "handbuch",
            "wörterbuch",
            "lexikon",
            // Latin document/treaty terms
            "instrumentum",
            "pacis",
            "universalis",
            "cosmographia",
            "encyclopedia",
            "dictionary",
            // Past participles indicating sentence fragments
            "disappeared",
            "commissioned",
            "collected",
            "completed",
            "established",
            "organized",
            "published",
            "translated",
            "compiled",
            "ratified",
            "succeeded",
            // Cross-cultural / descriptive terms
            "cross-cultural",
            "influences",
            "philanthropy",
            "educational",
            "electronic",
            "historical",
        ]
        .into_iter()
        .collect();

        let place_words: HashSet<&str> = [
            "japan",
            "china",
            "india",
            "europe",
            "asia",
            "africa",
            "america",
            "russia",
            "france",
            "germany",
            "england",
            "spain",
            "italy",
            "greece",
            "egypt",
            "persia",
            "arabia",
            "ottoman",
            "byzantine",
            "roman",
            "british",
            "french",
            "german",
            "danish",
            "english",
            "swedish",
            "norwegian",
            "finnish",
            "dutch",
            "polish",
            "czech",
            "hungarian",
            "austrian",
            "swiss",
            "scottish",
            "irish",
            "welsh",
        ]
        .into_iter()
        .collect();

        let mut purged = 0usize;
        for e in &entities {
            if e.entity_type != "person" || connected.contains(&e.id) {
                continue;
            }
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 2 {
                continue; // Single-word handled elsewhere
            }
            let lower_words: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();

            // Check if any word is a strong non-person indicator
            let has_concept_word = lower_words
                .iter()
                .any(|w| concept_words.contains(w.as_str()));
            let has_place_word = lower_words.iter().any(|w| place_words.contains(w.as_str()));

            // Names with non-ASCII characters that look like non-English phrases (not person names)
            // Check original-case words (not lower_words!) so "André Weil" passes the uppercase check
            let has_non_english_phrase = e
                .name
                .chars()
                .any(|c| matches!(c, 'ü' | 'ö' | 'ä' | 'é' | 'è' | 'ñ' | 'í'))
                && words.len() >= 3  // 2-word names with diacritics are usually real people
                && !words.iter().all(|w| {
                    w.chars()
                        .next()
                        .is_some_and(|c| c.is_uppercase() || w.len() <= 3)
                });
            // Check for all-lowercase multi-word (sentence fragment, not a name)
            let all_lowercase_words = words
                .iter()
                .all(|w| w.chars().next().is_some_and(|c| c.is_lowercase()));

            if has_concept_word
                || has_place_word
                || (has_non_english_phrase && e.confidence < 0.8)
                || (all_lowercase_words && words.len() >= 2)
            {
                eprintln!(
                    "  [mistyped-person-purge] deleting island '{}' (id={})",
                    e.name, e.id
                );
                self.brain.with_conn(|conn| {
                    conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                    Ok(())
                })?;
                purged += 1;
            }
        }
        Ok(purged)
    }

    /// Fix entities that are concatenated country/place names mistyped as persons.
    /// E.g., "Switzerland Germany Italy" (person) → delete or retype to "place".
    /// Also merges connected country-prefixed person entities into their canonical form.
    /// E.g., "Netherlands Oskar Klein" (11 rels) → merge into "Oskar Klein".
    /// Returns count of entities fixed.
    pub fn fix_country_concatenation_entities(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        let country_names: HashSet<&str> = [
            "netherlands",
            "switzerland",
            "germany",
            "france",
            "italy",
            "spain",
            "portugal",
            "belgium",
            "austria",
            "sweden",
            "norway",
            "denmark",
            "finland",
            "poland",
            "hungary",
            "romania",
            "bulgaria",
            "serbia",
            "croatia",
            "greece",
            "turkey",
            "russia",
            "ukraine",
            "japan",
            "korea",
            "taiwan",
            "china",
            "india",
            "brazil",
            "mexico",
            "egypt",
            "israel",
            "iran",
            "iraq",
            "prussia",
            "saxony",
            "bavaria",
            "bohemia",
            "england",
            "scotland",
            "ireland",
            "wales",
            "canada",
            "australia",
            "europe",
            "asia",
            "africa",
            "america",
            "crimea",
            "balkans",
            "caucasus",
            "mongolia",
            "persia",
            "arabia",
            "ottoman",
        ]
        .iter()
        .copied()
        .collect();

        // Build name → (id, degree) lookup for merge targets
        let mut name_lookup: HashMap<String, (i64, usize)> = HashMap::new();
        for e in &entities {
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            let lower = e.name.to_lowercase();
            let entry = name_lookup.entry(lower).or_insert((e.id, deg));
            if deg > entry.1 {
                *entry = (e.id, deg);
            }
        }

        let mut fixed = 0usize;
        for e in &entities {
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 2 {
                continue;
            }
            let lower_words: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();

            // Case 1: ALL words are country/place names → this is not a valid entity
            // E.g., "Switzerland Germany Italy", "Japan India", "Netherlands Switzerland"
            let all_countries = lower_words
                .iter()
                .all(|w| country_names.contains(w.as_str()));
            if all_countries && e.entity_type == "person" {
                let deg = degree.get(&e.id).copied().unwrap_or(0);
                if deg <= 2 {
                    // Low connectivity — safe to delete
                    eprintln!(
                        "  [country-concat] deleting all-country person '{}' (id={}, deg={})",
                        e.name, e.id, deg
                    );
                    self.brain.with_conn(|conn| {
                        conn.execute(
                            "DELETE FROM relations WHERE subject_id = ?1 OR object_id = ?1",
                            params![e.id],
                        )?;
                        conn.execute("DELETE FROM entities WHERE id = ?1", params![e.id])?;
                        Ok(())
                    })?;
                    fixed += 1;
                } else {
                    // Has many connections — retype to "place" instead of deleting
                    eprintln!(
                        "  [country-concat] retyping all-country person '{}' → place (id={}, deg={})",
                        e.name, e.id, deg
                    );
                    self.brain.with_conn(|conn| {
                        conn.execute(
                            "UPDATE entities SET entity_type = 'place' WHERE id = ?1",
                            params![e.id],
                        )?;
                        Ok(())
                    })?;
                    fixed += 1;
                }
                continue;
            }

            // Case 2: First word is a country name, rest looks like a real person name
            // E.g., "Netherlands Oskar Klein" → merge into "Oskar Klein"
            if e.entity_type == "person"
                && words.len() >= 3
                && country_names.contains(lower_words[0].as_str())
            {
                // Check if the non-country remainder is a proper name (starts with uppercase)
                let remainder = words[1..].join(" ");
                let remainder_lower = remainder.to_lowercase();
                if let Some(&(target_id, _)) = name_lookup.get(&remainder_lower) {
                    if target_id != e.id {
                        eprintln!(
                            "  [country-prefix] merging '{}' (id={}) → '{}' (id={})",
                            e.name, e.id, remainder, target_id
                        );
                        self.brain.merge_entities(e.id, target_id)?;
                        fixed += 1;
                    }
                }
            }
        }
        Ok(fixed)
    }

    /// Merge prefix-noise entities: longer entity names that end with a known shorter
    /// entity name, where the prefix is noise (e.g., "Devastated Tim Berners-Lee" → "Tim Berners-Lee",
    /// "Cannes Dev Patel" → "Dev Patel", "Moscow East Berlin" → "East Berlin").
    /// Only merges when the shorter name exists as a separate entity of the same type
    /// and the longer entity has low connectivity (likely an NLP extraction error).
    pub fn merge_prefix_noise_entities(&self) -> Result<usize> {
        let entities = self.brain.all_entities()?;
        let relations = self.brain.all_relations()?;

        let mut degree: HashMap<i64, usize> = HashMap::new();
        for r in &relations {
            *degree.entry(r.subject_id).or_insert(0) += 1;
            *degree.entry(r.object_id).or_insert(0) += 1;
        }

        // Build name → (id, degree) lookup
        let mut name_lookup: HashMap<String, (i64, usize)> = HashMap::new();
        for e in &entities {
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            let lower = e.name.to_lowercase();
            let entry = name_lookup.entry(lower).or_insert((e.id, deg));
            if deg > entry.1 {
                *entry = (e.id, deg);
            }
        }

        let mut merged = 0usize;
        for e in &entities {
            let words: Vec<&str> = e.name.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }
            let deg = degree.get(&e.id).copied().unwrap_or(0);
            if deg > 10 {
                continue; // skip highly connected entities
            }

            // Try dropping 1 or 2 prefix words and check if remainder exists
            for skip in 1..=(words.len() - 2).min(2) {
                let remainder = words[skip..].join(" ");
                let remainder_lower = remainder.to_lowercase();
                let remainder_words = words.len() - skip;

                // Remainder must be at least 2 words (to avoid false positives)
                if remainder_words < 2 {
                    continue;
                }

                if let Some(&(target_id, target_deg)) = name_lookup.get(&remainder_lower) {
                    if target_id == e.id {
                        continue;
                    }
                    // Target must have same type or be more connected
                    let target_entity = entities.iter().find(|x| x.id == target_id);
                    if let Some(target) = target_entity {
                        if target.entity_type != e.entity_type {
                            continue;
                        }
                    }
                    // Merge if the prefix word(s) look like noise (place names, adjectives, verbs)
                    // but not if they're genuine name parts (e.g., "Charles" in "Charles James Fox")
                    let prefix_words: Vec<String> =
                        words[..skip].iter().map(|w| w.to_lowercase()).collect();
                    let prefix_looks_noisy = prefix_words.iter().all(|pw| {
                        // Verb forms are always noise prefixes
                        let is_verb_form = (pw.ends_with("ed") && pw.len() > 5)
                            || (pw.ends_with("ing") && pw.len() > 5);
                        let is_demonym = pw.ends_with("man") && pw.len() > 5;
                        // If the prefix word exists as its own entity, it's likely
                        // a place/person name that got concatenated (NLP error)
                        let is_known_entity = name_lookup.contains_key(pw.as_str());
                        is_verb_form || is_demonym || (is_known_entity && e.entity_type == "person")
                    });
                    if prefix_looks_noisy {
                        eprintln!(
                            "  [prefix-noise] merging '{}' (id={}, deg={}) → '{}' (id={}, deg={})",
                            e.name, e.id, deg, remainder, target_id, target_deg
                        );
                        self.brain.merge_entities(e.id, target_id)?;
                        merged += 1;
                        break; // don't try more skips for this entity
                    }
                }
            }
        }
        Ok(merged)
    }

    /// Cross-type token bridge: find pairs of entities of different types that share
    /// significant name tokens and are in different components, suggesting a bridge.
    /// E.g., "Bayesian Learning" (concept) and "Thomas Bayes" (person) share "Bayes/Bayesian".
    /// Returns (entity_a_name, entity_b_name, shared_tokens, bridge_value).
    pub fn find_cross_type_token_bridges(&self) -> Result<Vec<(String, String, Vec<String>, f64)>> {
        let entities = self.brain.all_entities()?;
        let meaningful = meaningful_ids(self.brain)?;
        let relations = self.brain.all_relations()?;

        let mut connected: HashSet<i64> = HashSet::new();
        for r in &relations {
            connected.insert(r.subject_id);
            connected.insert(r.object_id);
        }

        // Only consider connected entities with decent types
        let high_value: HashSet<&str> = [
            "person",
            "organization",
            "concept",
            "technology",
            "place",
            "company",
        ]
        .into_iter()
        .collect();

        // Build stem → entity mapping (simple: lowercase, strip trailing 's', 'ed', 'ing')
        fn stem(word: &str) -> String {
            let w = word.to_lowercase();
            if w.len() > 4 {
                if let Some(s) = w.strip_suffix("ing") {
                    return s.to_string();
                }
                if let Some(s) = w.strip_suffix("ed") {
                    return s.to_string();
                }
                if let Some(s) = w.strip_suffix("ian") {
                    return s.to_string();
                }
                if let Some(s) = w.strip_suffix("ean") {
                    return s.to_string();
                }
            }
            if w.len() > 3 {
                if let Some(s) = w.strip_suffix('s') {
                    return s.to_string();
                }
            }
            w
        }

        let stopwords: HashSet<&str> = [
            "the", "of", "and", "in", "on", "for", "to", "by", "from", "with",
        ]
        .into_iter()
        .collect();

        let mut stem_entities: HashMap<String, Vec<i64>> = HashMap::new();
        let entity_map: HashMap<i64, &crate::db::Entity> =
            entities.iter().map(|e| (e.id, e)).collect();

        for e in &entities {
            if !meaningful.contains(&e.id)
                || !high_value.contains(e.entity_type.as_str())
                || !connected.contains(&e.id)
            {
                continue;
            }
            for word in e.name.split_whitespace() {
                if word.len() < 3 || stopwords.contains(&word.to_lowercase().as_str()) {
                    continue;
                }
                let s = stem(word);
                if s.len() >= 3 {
                    stem_entities.entry(s).or_default().push(e.id);
                }
            }
        }

        // Find cross-type pairs sharing stems
        let mut edges: HashSet<(i64, i64)> = HashSet::new();
        for r in &relations {
            let key = if r.subject_id < r.object_id {
                (r.subject_id, r.object_id)
            } else {
                (r.object_id, r.subject_id)
            };
            edges.insert(key);
        }

        let mut bridges: Vec<(String, String, Vec<String>, f64)> = Vec::new();
        let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();

        for (stem_word, eids) in &stem_entities {
            if eids.len() < 2 || eids.len() > 50 {
                continue; // Too unique or too common
            }
            for i in 0..eids.len().min(20) {
                for j in (i + 1)..eids.len().min(20) {
                    let a = eids[i];
                    let b = eids[j];
                    let key = if a < b { (a, b) } else { (b, a) };
                    if edges.contains(&key) || !seen_pairs.insert(key) {
                        continue;
                    }
                    let type_a = entity_map
                        .get(&a)
                        .map(|e| e.entity_type.as_str())
                        .unwrap_or("");
                    let type_b = entity_map
                        .get(&b)
                        .map(|e| e.entity_type.as_str())
                        .unwrap_or("");
                    if type_a == type_b {
                        continue; // Same type — not a cross-type bridge
                    }
                    let name_a = entity_map.get(&a).map(|e| e.name.as_str()).unwrap_or("");
                    let name_b = entity_map.get(&b).map(|e| e.name.as_str()).unwrap_or("");
                    bridges.push((
                        name_a.to_string(),
                        name_b.to_string(),
                        vec![stem_word.clone()],
                        1.0,
                    ));
                }
            }
        }

        bridges.truncate(50);
        Ok(bridges)
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Check if an entity name looks like a citation fragment (common in Wikipedia extraction).
fn looks_like_citation(name: &str) -> bool {
    let lower = name.to_lowercase();
    // Contains volume/page indicators
    if lower.contains(" vol ") || lower.contains(" pp ") || lower.contains(" no ") {
        return true;
    }
    // Contains "journal" anywhere
    if lower.contains("journal") {
        return true;
    }
    // Ends with common citation noise
    let citation_suffixes = ["press", "publishers", "publishing", "edition", "eds"];
    let last_word = lower.split_whitespace().last().unwrap_or("");
    if citation_suffixes.contains(&last_word) && lower.split_whitespace().count() >= 3 {
        return true;
    }
    false
}

/// Determine if an isolated entity is likely extraction noise from Wikipedia/encyclopedias.
/// Aggressive filter — only apply to entities with ZERO relations.
fn is_extraction_noise(name: &str, entity_type: &str) -> bool {
    // Already caught by noise filters
    if is_noise_name(name) || is_noise_type(entity_type) {
        return true;
    }
    let lower = name.to_lowercase();
    let word_count = lower.split_whitespace().count();

    // Citation fragments
    if looks_like_citation(name) {
        return true;
    }

    // Names with mixed case patterns suggesting concatenated references
    if word_count >= 3 {
        let words: Vec<&str> = name.split_whitespace().collect();
        let has_trailing_noise = [
            "During",
            "Resting",
            "Commentary",
            "Surveys",
            "Proceedings",
            "Magazine",
            "Review",
            "Bulletin",
            "Like",
        ];
        if words.len() >= 2 && has_trailing_noise.contains(&words[words.len() - 1]) {
            return true;
        }
    }

    // Pure acronyms that are too short to be meaningful when isolated
    if word_count == 1 && name.len() <= 3 && name.chars().all(|c| c.is_uppercase()) {
        return true;
    }

    // Possessive/genitive forms alone (e.g. "Switzerland's", "Ramanujan's")
    if lower.ends_with("'s") || lower.ends_with("\u{2019}s") {
        return true;
    }

    // Starts with "The " followed by a single generic word
    if lower.starts_with("the ") && word_count == 2 {
        return true;
    }

    // Multi-word fragments containing prepositions/articles mid-name suggest sentence chunks
    let filler_words = [
        "of", "the", "and", "in", "on", "for", "to", "by", "from", "with", "at", "an", "or",
    ];
    if word_count >= 3 {
        let words: Vec<&str> = lower.split_whitespace().collect();
        let filler_count = words.iter().filter(|w| filler_words.contains(w)).count();
        // If >40% of words are fillers + total is long, it's a sentence fragment
        if filler_count >= 2 && filler_count as f64 / word_count as f64 > 0.35 {
            return true;
        }
    }

    // Names that look like "Topic1 Topic2 Topic3" — concatenated unrelated words
    // Heuristic: 4+ capitalized words with no filler words = keyword salad
    if word_count >= 4 {
        let words: Vec<&str> = name.split_whitespace().collect();
        let cap_count = words
            .iter()
            .filter(|w| w.starts_with(|c: char| c.is_uppercase()))
            .count();
        let filler = words
            .iter()
            .filter(|w| filler_words.contains(&w.to_lowercase().as_str()))
            .count();
        if cap_count >= 4 && filler == 0 {
            return true;
        }
    }

    // Entity names containing "MacTutor", "Archive", "ISBN" — reference noise
    let ref_noise = ["mactutor", "archive", "isbn", "doi:", "arxiv", "springer"];
    if ref_noise.iter().any(|r| lower.contains(r)) {
        return true;
    }

    // Single generic words that got capitalized by NLP extractors
    let generic_singles = [
        "actor",
        "areas",
        "unity",
        "proof",
        "lieutenant",
        "terminology",
        "calinger",
        "estreicher",
        "improvement",
        "superchip",
        "location",
        "outcome",
        "abolition",
        "pickwick",
        "ruffini",
        "newton",
        "historically",
        "politically",
        "coalition",
    ];
    if word_count == 1 && generic_singles.contains(&lower.as_str()) {
        return true;
    }

    // Type mismatches: places classified as persons, etc.
    // "Crystal Palace" as person, "Middle East" as person, etc.
    let place_indicators = [
        "palace",
        "east",
        "west",
        "north",
        "south",
        "island",
        "ocean",
        "mountain",
        "valley",
        "river",
        "lake",
        "strait",
        "peninsula",
        "gulf",
        "bay",
        "coast",
        "border",
        "frontier",
        "colony",
        "republic",
        "kingdom",
        "empire",
    ];
    if entity_type == "person" && word_count >= 2 {
        let has_place_word = lower
            .split_whitespace()
            .any(|w| place_indicators.contains(&w));
        if has_place_word {
            return true;
        }
    }

    // Names starting with "Source " — extraction artifact
    if lower.starts_with("source ") {
        return true;
    }

    // Names that are clearly descriptions, not entities: contain "Years", "Survey", "Overview"
    let desc_words = [
        "survey",
        "overview",
        "übersicht",
        "origins",
        "recognition",
        "processing",
        "diagnosing",
        "improvement",
        "dependability",
        "embedded",
        "quantum",
        "symposium",
    ];
    if word_count >= 2 && desc_words.iter().any(|d| lower.contains(d)) {
        return true;
    }

    // Entity name == entity type (e.g. entity "Actor" of type "concept")
    if lower == entity_type {
        return true;
    }

    // Non-English words that commonly appear as extracted entity fragments
    // (French, German academic/citation text)
    let non_english_noise = [
        "accompagnées",
        "pensées",
        "chaires",
        "mémoires",
        "études",
        "régime",
        "même",
        "après",
        "année",
        "années",
        "siècle",
        "über",
        "und",
        "eine",
        "eines",
        "junge",
        "neue",
        "neuen",
        "des",
        "der",
        "die",
        "das",
        "dem",
        "den",
        "für",
        "avec",
        "dans",
        "pour",
        "les",
        "aux",
        "sur",
        "une",
        "della",
        "degli",
        "delle",
        "nelle",
        "nella",
        "dello",
    ];
    if word_count >= 2 {
        let words: Vec<&str> = lower.split_whitespace().collect();
        let foreign_count = words
            .iter()
            .filter(|w| non_english_noise.contains(w))
            .count();
        // If >30% of words are non-English noise, it's a citation fragment
        if foreign_count >= 1 && foreign_count as f64 / word_count as f64 > 0.3 {
            return true;
        }
    }

    // Academic journal abbreviation patterns: entities containing "Monthly Notices",
    // "Astrophysical Journal", "Physical Review", etc.
    let journal_patterns = [
        "monthly notices",
        "physical review",
        "astrophysical",
        "letters to",
        "annals of",
        "proceedings of",
        "transactions of",
        "reviews of",
        "reports on",
        "advances in",
        "frontiers in",
        "studies in",
    ];
    if word_count >= 2 && journal_patterns.iter().any(|jp| lower.contains(jp)) {
        return true;
    }

    // Entities where known entity name is prefixed/suffixed with random context words
    // Pattern: "SomeContext KnownEntity MoreContext" where the middle is what matters
    // Detect by checking for ALL-CAPS abbreviations mixed with regular words
    if word_count >= 3 {
        let words: Vec<&str> = name.split_whitespace().collect();
        let has_acronym = words.iter().any(|w| {
            w.len() >= 3
                && w.len() <= 8
                && w.chars().all(|c| c.is_uppercase() || c.is_ascii_digit())
        });
        let has_normal = words.iter().any(|w| {
            w.len() >= 3
                && w.starts_with(|c: char| c.is_uppercase())
                && w.chars().skip(1).any(|c| c.is_lowercase())
        });
        // "CFHTLenS Monthly Notices" pattern: acronym + generic words
        if has_acronym && has_normal && word_count <= 4 {
            let generic_trail: HashSet<&str> = [
                "monthly", "notices", "letters", "review", "reports", "papers", "notes", "studies",
                "focus", "dead", "mass",
            ]
            .iter()
            .copied()
            .collect();
            let generic_count = words
                .iter()
                .filter(|w| generic_trail.contains(&w.to_lowercase().as_str()))
                .count();
            if generic_count >= 1 {
                return true;
            }
        }
    }

    // Mixed-script names (Latin + Cyrillic/Arabic/CJK) — usually translation artifacts
    let has_latin = lower.chars().any(|c| c.is_ascii_alphabetic());
    let has_non_latin = name.chars().any(|c| {
        c.is_alphabetic()
            && !c.is_ascii_alphabetic()
            && c != 'é'
            && c != 'è'
            && c != 'ê'
            && c != 'ë'
            && c != 'à'
            && c != 'â'
            && c != 'ä'
            && c != 'ö'
            && c != 'ü'
            && c != 'ß'
            && c != 'ñ'
            && c != 'ç'
            && c != 'î'
            && c != 'ô'
            && c != 'û'
            && c != 'æ'
            && c != 'ø'
            && c != 'å'
            && c != 'í'
            && c != 'ó'
            && c != 'ú'
    });
    if has_latin && has_non_latin && word_count >= 3 {
        return true;
    }

    // Names ending with "Post-Intelligencer", "Modelling", "Diploma" — org/concept, not entities
    let noise_endings = ["post-intelligencer", "modelling", "diploma", "semantics"];
    if word_count >= 2 {
        let last = lower.split_whitespace().last().unwrap_or("");
        if noise_endings.contains(&last) {
            return true;
        }
    }

    // Sentence fragments disguised as entities: "unknown" type with verb-laden text.
    // Real entities are noun phrases; sentence fragments contain verbs, pronouns, determiners.
    if entity_type == "unknown" && word_count >= 3 {
        let words: Vec<&str> = lower.split_whitespace().collect();
        let sentence_verbs = [
            "is",
            "was",
            "were",
            "are",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "has",
            "had",
            "have",
            "did",
            "does",
            "do",
            "been",
            "being",
            "became",
            "become",
            "came",
            "went",
            "said",
            "made",
            "found",
            "gave",
            "took",
            "got",
            "proved",
            "lost",
            "destroyed",
            "decided",
            "published",
            "continued",
            "controlled",
            "succeeded",
            "predicted",
            "conquered",
            "invaded",
            "defeated",
            "established",
            "discovered",
            "built",
            "wrote",
            "studied",
            "increased",
            "decreased",
            "produced",
            "created",
            "remained",
            "returned",
            "appeared",
            "contained",
            "included",
            "involved",
            "began",
            "started",
            "ended",
            "died",
            "born",
            "lived",
        ];
        let pronouns = [
            "he", "she", "it", "his", "her", "its", "their", "they", "them", "him", "who", "whom",
            "whose", "which", "what", "that", "this", "these", "those",
        ];
        let verb_count = words.iter().filter(|w| sentence_verbs.contains(w)).count();
        let pronoun_count = words.iter().filter(|w| pronouns.contains(w)).count();
        // 2+ verbs/pronouns in a 3+ word "entity" = sentence fragment
        if verb_count + pronoun_count >= 2 {
            return true;
        }
        // Even 1 verb + starting with lowercase = sentence fragment
        if verb_count >= 1 {
            if let Some(first_char) = name.chars().next() {
                if first_char.is_lowercase() {
                    return true;
                }
            }
        }
    }

    // "unknown" type entities over 40 chars are almost always extraction errors
    if entity_type == "unknown" && name.len() > 40 {
        return true;
    }

    false
}

/// Determine if a single-word isolated entity should be purged.
/// Returns true for generic English words, citation surnames, adverbs, adjectives, etc.
/// Preserves entities that look like well-known proper nouns or technical terms.
fn should_purge_single_word(name: &str, entity_type: &str) -> bool {
    let lower = name.to_lowercase();
    let len = lower.len();

    // Very short names are almost always noise
    if len <= 3 {
        return true;
    }

    // Already caught by noise filters
    if is_noise_name(name) || is_noise_type(entity_type) {
        return true;
    }

    // Starts with lowercase → not a proper noun → generic word
    if name.starts_with(|c: char| c.is_lowercase()) {
        return true;
    }

    // For "concept" type: aggressively purge common English word patterns
    if entity_type == "concept" {
        // Adverbs ending in -ly
        if lower.ends_with("ly") && len >= 5 {
            return true;
        }
        // Adjectives: -ous, -ive, -ful, -less, -able, -ible, -ical, -ary, -ory
        let adj_suffixes = [
            "ous", "ive", "ful", "less", "able", "ible", "ical", "ary", "ory", "ish", "ular",
            "inal", "ular", "etic", "atic",
        ];
        if adj_suffixes.iter().any(|s| lower.ends_with(s)) && len >= 6 {
            return true;
        }
        // Past participles (-ed), gerunds (-ing) — often sentence fragments
        if (lower.ends_with("ed") || lower.ends_with("ing")) && len >= 5 {
            return true;
        }
        // Plural abstract concepts (-ies, -isms, -ists, -ments, -tions, -nesses)
        let abstract_suffixes = [
            "isms", "ists", "ments", "tions", "nesses", "ities", "ences", "ances",
        ];
        if abstract_suffixes.iter().any(|s| lower.ends_with(s)) {
            return true;
        }
        // Common generic concept words
        let generic_concepts = [
            "unlike",
            "others",
            "battle",
            "timeline",
            "further",
            "please",
            "learn",
            "multiple",
            "lectures",
            "origins",
            "bulletin",
            "elements",
            "universe",
            "physicists",
            "aside",
            "together",
            "video",
            "consider",
            "attempts",
            "fellows",
            "chamber",
            "user",
            "contexts",
            "humans",
            "lowest",
            "bells",
            "comic",
            "chariot",
            "comet",
            "ceramic",
            "stellar",
            "temperature",
            "falcon",
            "piranha",
            "mitten",
            "treatise",
            "television",
            "particle",
            "stratosphere",
            "antimatter",
            "apply",
            "strongly",
            "axioms",
            "debates",
            "problems",
            "ideas",
            "systems",
            "methods",
            "models",
            "levels",
            "forces",
            "fields",
            "waves",
            "forms",
            "rules",
            "tools",
            "parts",
            "types",
            "modes",
            "roles",
            "units",
            "rates",
            "phases",
            "zones",
            "loops",
            "paths",
            "nodes",
            "links",
            "terms",
            "claims",
            "facts",
            "texts",
            "codes",
            "tests",
            "maps",
            "keys",
            "data",
            "sets",
            "rows",
            "logs",
            "tags",
            "runs",
            "gaps",
            "ends",
            "bits",
            "aims",
        ];
        if generic_concepts.contains(&lower.as_str()) {
            return true;
        }
        // Single-word concept entities that look like surnames (capitalized, 5-12 chars,
        // ending in common surname suffixes) are almost always citation artifacts
        let surname_suffixes = [
            "ier", "iere", "ski", "sky", "ley", "ley", "ner", "ger", "ler", "sen", "son", "man",
            "men", "kov", "ova", "enko", "elli", "ini", "otti", "ardi", "ardy", "burg", "dorf",
            "feld", "stein", "berg", "wald", "rff", "off", "eff", "ych", "vich", "wicz",
        ];
        let chars: Vec<char> = name.chars().collect();
        if chars.len() >= 5
            && chars.len() <= 14
            && chars[0].is_uppercase()
            && chars[1..]
                .iter()
                .all(|c| c.is_lowercase() || *c == '-' || *c == '\'')
            && surname_suffixes.iter().any(|s| lower.ends_with(s))
        {
            return true;
        }
        return false;
    }

    // For "person" type: purge if it looks like a citation surname
    if entity_type == "person" {
        // Single-word "person" entities are almost always citation last names
        // unless they're a very well-known mononymous person
        let known_mononymous = [
            "aristotle",
            "plato",
            "socrates",
            "euclid",
            "archimedes",
            "confucius",
            "avicenna",
            "averroes",
            "fibonacci",
            "michelangelo",
            "raphael",
            "caravaggio",
            "rembrandt",
            "voltaire",
            "napoleon",
            "galileo",
            "copernicus",
            "hypatia",
            "ptolemy",
            "hippocrates",
            "pythagoras",
            "herodotus",
            "homer",
            "thales",
            "democritus",
            "epicurus",
            "seneca",
            "virgil",
            "ovid",
            "tacitus",
            "livy",
            "cicero",
            "nero",
            "caesar",
            "cleopatra",
            "hannibal",
            "xerxes",
            "charlemagne",
            "saladin",
            "tamerlane",
            "maimonides",
            "rumi",
            "hafez",
            "omar",
            "drake",
            "magellan",
            "columbus",
            "vespucci",
            "pizarro",
            "cortez",
            "nostradamus",
            "paracelsus",
            "vesalius",
            "kepler",
            "descartes",
            "pascal",
            "leibniz",
            "euler",
            "gauss",
            "riemann",
            "hilbert",
            "poincaré",
            "noether",
            "ramanujan",
            "turing",
            "gödel",
            "shannon",
            "babbage",
            "lovelace",
            "tesla",
            "edison",
            "faraday",
            "maxwell",
            "boltzmann",
            "heisenberg",
            "schrödinger",
            "dirac",
            "feynman",
            "hawking",
            "einstein",
            "newton",
            "darwin",
            "mendel",
            "pasteur",
            "curie",
            "planck",
            "bohr",
            "rutherford",
            "fermi",
            "oppenheimer",
            "madonna",
            "beyoncé",
            "shakira",
            "adele",
            "rihanna",
            "drake",
            "eminem",
            "bono",
            "cher",
            "prince",
            "moby",
            "björk",
            "sia",
            "pelé",
            "ronaldinho",
            "neymar",
            "ronaldo",
            "messi",
            "madonna",
            "picasso",
            "banksy",
            "kandinsky",
            "monet",
            "renoir",
            "cézanne",
            "matisse",
            "warhol",
            "pollock",
            "dostoevsky",
            "tolstoy",
            "chekhov",
            "pushkin",
            "nabokov",
            "kafka",
            "goethe",
            "nietzsche",
            "hegel",
            "kant",
            "spinoza",
            "hume",
            "locke",
            "hobbes",
            "rousseau",
            "montesquieu",
            "machiavelli",
            "buddha",
            "confucius",
            "laozi",
            "zoroaster",
            "muhammad",
            "moses",
            "jesus",
        ];
        if known_mononymous.contains(&lower.as_str()) {
            return false; // Keep well-known mononymous persons
        }
        // Otherwise, single-word person is almost certainly a citation surname
        return true;
    }

    // For "organization" type: single-word orgs are usually abbreviations or generic
    if entity_type == "organization" && len <= 5 {
        return true;
    }

    false
}

/// Detect if an entity type is wrong based on name patterns.
/// Returns Some(correct_type) or None if current type seems fine.
fn detect_correct_type(lower: &str, words: &[&str], current_type: &str) -> Option<&'static str> {
    let word_count = words.len();

    // Place-like names classified as person/concept
    let place_words = [
        "east",
        "west",
        "north",
        "south",
        "island",
        "islands",
        "ocean",
        "sea",
        "mountain",
        "mountains",
        "valley",
        "river",
        "lake",
        "strait",
        "peninsula",
        "gulf",
        "bay",
        "colony",
        "palace",
        "castle",
        "tower",
        "bridge",
        "airport",
        "harbor",
        "harbour",
        "province",
        "canton",
        "county",
        "district",
        "territory",
        "springs",
        "city",
        "creek",
        "basin",
        "desert",
        "plateau",
        "cape",
        "coast",
        "reef",
        "archipelago",
        "fjord",
        "steppe",
        "tundra",
        "savanna",
        "mesa",
        "gorge",
        "canyon",
        "ridge",
        "hills",
        "plains",
        "marsh",
        "swamp",
        "oasis",
        "inlet",
    ];
    if (current_type == "person" || current_type == "concept")
        && word_count >= 2
        && words.iter().any(|w| place_words.contains(w))
    {
        return Some("place");
    }

    // Place → person: names that look like "Firstname Lastname" classified as place
    // Heuristic: 2-3 words, all capitalized, no place/org indicators, no numbers
    if current_type == "place" && (word_count == 2 || word_count == 3) {
        let has_place_word = words.iter().any(|w| place_words.contains(w));
        let has_org_word = words.iter().any(|w| {
            [
                "university",
                "college",
                "institute",
                "company",
                "corp",
                "inc",
                "street",
                "avenue",
                "road",
                "turnpike",
                "post-intelligencer",
                "semantics",
                "modelling",
                "diploma",
            ]
            .contains(w)
        });
        let has_tech_word = words.iter().any(|w| {
            [
                "spark",
                "hadoop",
                "kafka",
                "kubernetes",
                "docker",
                "tensorflow",
                "pytorch",
                "bayes",
                "naive",
                "artificial",
                "neural",
                "quantum",
            ]
            .contains(w)
        });
        let has_digit = lower.chars().any(|c| c.is_ascii_digit());
        if !has_place_word && !has_org_word && !has_tech_word && !has_digit {
            // Check if words look like proper name components (capitalized, alphabetic)
            let all_name_like = words.iter().all(|w| {
                w.len() >= 2 && w.chars().all(|c| c.is_alphabetic() || c == '-' || c == '.')
            });
            // Person names don't usually contain these
            let non_person_words = [
                "soviet-allied",
                "off",
                "columbia",
                "cuban",
                "brazilian",
                "carolingian",
                "imperial",
                "byzantine",
            ];
            let has_non_person = words.iter().any(|w| non_person_words.contains(w));
            if all_name_like && !has_non_person && word_count <= 3 {
                // Additional check: is the last word a common surname-like word?
                // Avoid reclassifying "West Berlin" etc.
                let first_word = words[0];
                let last_word = words[word_count - 1];
                // If first word is a cardinal direction or common adjective, keep as place
                let adj_first = [
                    "new", "old", "great", "upper", "lower", "central", "western", "eastern",
                    "northern", "southern", "south", "north", "east", "west",
                ];
                if !adj_first.contains(&first_word) {
                    // Likely a person name misclassified as place
                    // But be conservative: only if last word >= 4 chars (likely surname)
                    if last_word.len() >= 4 {
                        return Some("person");
                    }
                }
            }
        }
        // Place → technology: known software/tech names
        if has_tech_word {
            return Some("technology");
        }
    }

    // Compound entities misclassified as person: "X Building", "X Day", "X Award", etc.
    if current_type == "person" && word_count >= 2 {
        if let Some(last) = words.last() {
            // These suffixes mean it's a place, not a person
            let place_suffixes = [
                "building",
                "center",
                "centre",
                "house",
                "suite",
                "hall",
                "park",
                "square",
                "street",
                "avenue",
                "boulevard",
                "museum",
                "library",
                "hospital",
                "station",
                "airport",
            ];
            if place_suffixes.contains(last) {
                return Some("place");
            }
            // These suffixes mean it's an organization
            let org_suffixes = [
                "institute",
                "foundation",
                "association",
                "society",
                "academy",
                "council",
                "committee",
                "commission",
                "agency",
                "bureau",
            ];
            if org_suffixes.contains(last) {
                return Some("organization");
            }
            // These suffixes mean it's an event
            let event_suffixes = ["day", "festival", "ceremony", "conference", "symposium"];
            if event_suffixes.contains(last) {
                return Some("event");
            }
            // These suffixes mean it's a concept/thing
            let concept_suffixes = [
                "award",
                "prize",
                "medal",
                "biography",
                "original",
                "wired",
                "founder",
                "countess",
                "notes",
                "letters",
                "papers",
            ];
            if concept_suffixes.contains(last) {
                return Some("concept");
            }
        }
    }

    // Technology-like names classified as organization
    let tech_words = [
        "network",
        "networks",
        "algorithm",
        "protocol",
        "framework",
        "neural",
        "processor",
        "architecture",
        "compiler",
        "runtime",
        "kernel",
        "driver",
    ];
    if current_type == "organization"
        && word_count >= 2
        && words.iter().any(|w| tech_words.contains(w))
    {
        return Some("technology");
    }

    // Concept-like names classified as person (multi-word abstractions)
    let concept_indicators = [
        "theory",
        "theorem",
        "principle",
        "effect",
        "law",
        "paradox",
        "hypothesis",
        "equation",
        "conjecture",
        "inequality",
        "transform",
        "function",
        "distribution",
        "constant",
        "number",
        "formula",
    ];
    if current_type == "person"
        && word_count >= 2
        && words.iter().any(|w| concept_indicators.contains(w))
    {
        // But "Euler's theorem" should be concept, "Leonhard Euler" should stay person
        // Check: if the last word is a concept indicator, it's probably a concept
        if let Some(last) = words.last() {
            if concept_indicators.contains(last) {
                return Some("concept");
            }
        }
    }

    // GDP/statistics entities are concepts, not persons
    if current_type == "person"
        && (lower.contains("gdp") || lower.contains("population") || lower.contains("statistics"))
    {
        return Some("concept");
    }

    // Well-known misclassifications: geographic entities classified as person
    let known_places: &[&str] = &[
        "hong kong",
        "new york",
        "new zealand",
        "sri lanka",
        "el salvador",
        "middle east",
        "south america",
        "north america",
        "central asia",
        "south asia",
        "east asia",
        "southeast asia",
        "sub-saharan africa",
        "saharan africa",
        "latin america",
        "central europe",
        "western europe",
        "eastern europe",
        "northern europe",
        "southern europe",
    ];
    if current_type == "person" && known_places.contains(&lower) {
        return Some("place");
    }
    // Entities ending with geographic suffixes that are definitely places
    if current_type == "person" && word_count >= 2 {
        let geo_suffixes = [
            "springs",
            "city",
            "islands",
            "island",
            "mountains",
            "mountain",
            "valley",
            "river",
            "lake",
            "strait",
            "peninsula",
            "gulf",
            "bay",
            "creek",
            "basin",
            "desert",
            "plateau",
            "cape",
            "coast",
            "reef",
            "archipelago",
            "steppe",
            "canyon",
            "ridge",
            "hills",
            "plains",
            "falls",
            "pass",
            "harbor",
            "harbour",
            "port",
            "inlet",
            "fjord",
        ];
        if let Some(last) = words.last() {
            if geo_suffixes.contains(last) {
                return Some("place");
            }
        }
    }

    // Well-known concepts misclassified as person
    let known_concepts: &[&str] = &[
        "big bang",
        "dark matter",
        "dark energy",
        "black hole",
        "quantum mechanics",
        "general relativity",
        "special relativity",
        "string theory",
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "natural selection",
        "climate change",
        "global warming",
        "plate tectonics",
        "continental drift",
        "new scientist",
        "scientific american",
    ];
    if current_type == "person" && known_concepts.contains(&lower) {
        return Some("concept");
    }

    // Organizations misclassified as person
    let known_orgs: &[&str] = &["new scientist", "scientific american", "nature"];
    if current_type == "person" && known_orgs.contains(&lower) {
        return Some("organization");
    }

    // "State X" patterns misclassified as person (e.g. "State Anthem", "Crusader States")
    // But keep "State Henry Kissinger" etc. — those are "Secretary of State + name" fragments
    if current_type == "person" && word_count >= 2 {
        let state_concept_suffixes = [
            "anthem",
            "church",
            "churches",
            "estate",
            "estates",
            "states",
            "clause",
            "sicilies",
            "council",
            "militia",
            "department",
        ];
        if let Some(last) = words.last() {
            if state_concept_suffixes.contains(last) {
                return Some("concept");
            }
        }
        // "X Church" → organization
        let org_last = ["church", "churches", "orthodox", "catholic"];
        if let Some(last) = words.last() {
            if org_last.contains(last) {
                return Some("organization");
            }
        }
        // "Federated States", "Crusader States", "First/Second/Third Estate"
        if let Some(first) = words.first() {
            let ordinal_first = ["first", "second", "third", "fourth", "fifth"];
            if ordinal_first.contains(first) {
                let concept_second = ["estate", "estates", "republic", "empire", "crusade"];
                if word_count == 2 && concept_second.contains(&words[1]) {
                    return Some("concept");
                }
            }
        }
    }

    // Single-word entities that are obviously not persons
    if current_type == "person" && word_count == 1 {
        let concept_singles = [
            "philosophy",
            "matrix",
            "supersymmetries",
            "supermembranes",
            "microcontrollers",
            "fundamentals",
            "correspondents",
        ];
        if concept_singles.contains(&lower) {
            return Some("concept");
        }
    }

    // Well-known countries/regions misclassified as concept or person
    let known_countries: &[&str] = &[
        "netherlands",
        "germany",
        "france",
        "spain",
        "portugal",
        "italy",
        "greece",
        "turkey",
        "egypt",
        "india",
        "china",
        "japan",
        "korea",
        "russia",
        "brazil",
        "mexico",
        "canada",
        "australia",
        "argentina",
        "chile",
        "peru",
        "colombia",
        "venezuela",
        "cuba",
        "iran",
        "iraq",
        "syria",
        "afghanistan",
        "pakistan",
        "bangladesh",
        "myanmar",
        "thailand",
        "vietnam",
        "indonesia",
        "malaysia",
        "singapore",
        "taiwan",
        "mongolia",
        "nepal",
        "cambodia",
        "laos",
        "austria",
        "switzerland",
        "belgium",
        "poland",
        "czechia",
        "hungary",
        "romania",
        "bulgaria",
        "serbia",
        "croatia",
        "ukraine",
        "belarus",
        "lithuania",
        "latvia",
        "estonia",
        "finland",
        "sweden",
        "norway",
        "denmark",
        "iceland",
        "ireland",
        "scotland",
        "wales",
        "england",
        "morocco",
        "algeria",
        "tunisia",
        "libya",
        "sudan",
        "ethiopia",
        "kenya",
        "tanzania",
        "uganda",
        "nigeria",
        "ghana",
        "senegal",
        "cameroon",
        "madagascar",
        "mozambique",
        "zimbabwe",
        "zambia",
        "angola",
        "namibia",
        "botswana",
        "somalia",
        "eritrea",
        "djibouti",
        "gabon",
        "congo",
        "shropshire",
        "bradford",
        "clermont",
        "esztergom",
        "lombardy",
        "saxony",
        "bavaria",
        "bohemia",
        "moravia",
        "silesia",
        "alsace",
        "catalonia",
        "andalusia",
        "galicia",
        "brittany",
        "normandy",
        "flanders",
        "wallonia",
        "tyrol",
        "transylvania",
        "thrace",
        "mesopotamia",
        "anatolia",
        "persia",
        "judea",
        "canaan",
        "baltic",
        "siberia",
        "sahara",
        "patagonia",
        "scandinavia",
        "balkans",
        "caucasus",
        "crimea",
        "cyprus",
        "crete",
        "sardinia",
        "sicily",
        "corsica",
        "sumatra",
        "borneo",
        "java",
        "ceylon",
        "formosa",
        "kashmir",
        "tibet",
        "manchuria",
        "xinjiang",
    ];
    if (current_type == "concept" || current_type == "person")
        && word_count == 1
        && known_countries.contains(&lower)
    {
        return Some("place");
    }

    // Multi-word place names misclassified as concept (e.g. "Spanish Netherlands", "Ottoman Empire")
    if current_type == "concept" && word_count >= 2 {
        let place_compounds: &[&str] = &[
            "spanish netherlands",
            "austrian netherlands",
            "southern netherlands",
            "ottoman empire",
            "roman empire",
            "byzantine empire",
            "british empire",
            "russian empire",
            "persian empire",
            "mughal empire",
            "holy roman empire",
            "soviet union",
            "austro-hungarian empire",
            "habsburg empire",
            "west germany",
            "east germany",
            "west berlin",
            "east berlin",
            "north korea",
            "south korea",
            "north vietnam",
            "south vietnam",
            "saudi arabia",
            "south africa",
            "costa rica",
            "puerto rico",
            "new guinea",
            "new caledonia",
            "new hebrides",
            "rhine valley",
            "nile delta",
            "ganges plain",
            "fertile crescent",
        ];
        if place_compounds.contains(&lower) {
            return Some("place");
        }
    }

    None
}

/// Heuristic: is this likely a real, notable entity worth keeping even if isolated?
fn is_likely_real_entity(name: &str, entity_type: &str) -> bool {
    let lower = name.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let word_count = words.len();

    // Already filtered by noise checks
    if is_noise_name(name) {
        return false;
    }

    // Person names: 2-3 words, each capitalized, no noise words
    if entity_type == "person" && (word_count == 2 || word_count == 3) {
        let all_cap = name
            .split_whitespace()
            .all(|w| w.starts_with(|c: char| c.is_uppercase()));
        if all_cap {
            return true;
        }
    }

    // Well-known organization patterns
    if entity_type == "organization" && word_count <= 4 {
        let org_suffixes = [
            "inc",
            "corp",
            "ltd",
            "gmbh",
            "ag",
            "llc",
            "foundation",
            "institute",
            "university",
            "college",
            "party",
            "association",
        ];
        if let Some(last) = words.last() {
            if org_suffixes.iter().any(|s| last.ends_with(s)) {
                return true;
            }
        }
    }

    // Places: 1-3 words, capitalized
    if entity_type == "place" && word_count <= 3 {
        let all_cap = name
            .split_whitespace()
            .all(|w| w.starts_with(|c: char| c.is_uppercase()));
        if all_cap {
            return true;
        }
    }

    // Single well-capitalized words that are proper nouns
    if word_count == 1 && name.starts_with(|c: char| c.is_uppercase()) && name.len() >= 4 {
        // Filter out common English words that sneak through as entities
        // These are NOT proper nouns even when capitalized
        if is_common_english_word(&lower) {
            return false;
        }
        // Words ending in common verb/adjective suffixes are likely not proper nouns
        if lower.ends_with("ing")
            || lower.ends_with("tion")
            || lower.ends_with("ment")
            || lower.ends_with("ness")
            || lower.ends_with("ists")
            || lower.ends_with("isms")
            || lower.ends_with("ally")
            || lower.ends_with("edly")
            || lower.ends_with("ious")
            || lower.ends_with("eous")
            || lower.ends_with("ible")
            || lower.ends_with("able")
        {
            // But allow known proper nouns with these endings (e.g. "Beijing", "Turing")
            let known_exceptions = [
                "beijing",
                "turing",
                "reading",
                "stirling",
                "darjeeling",
                "nanjing",
                "chongqing",
                "washington",
                "wellington",
                "nottingham",
                "birmingham",
                "buckingham",
                "manning",
                "browning",
                "kipling",
                "lessing",
                "göttingen",
            ];
            if !known_exceptions.contains(&lower.as_str()) {
                return false;
            }
        }
        return true;
    }

    false
}

/// Common English words that are NOT proper nouns even when capitalized.
/// Used to filter island entities that are clearly generic terms.
fn is_common_english_word(lower: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        // Verbs / verb forms
        "defeat",
        "subtract",
        "encode",
        "assuming",
        "conclude",
        "declare",
        "emerge",
        "evolve",
        "expand",
        "explore",
        "impose",
        "improve",
        "indicate",
        "interpret",
        "introduce",
        "invoke",
        "lapse",
        "mandate",
        "negotiate",
        "observe",
        "oppose",
        "organize",
        "overcome",
        "persist",
        "pledge",
        "possess",
        "precede",
        "preserve",
        "prevail",
        "proceed",
        "produce",
        "propose",
        "pursue",
        "reckon",
        "reform",
        "reign",
        "resolve",
        "restore",
        "retain",
        "retrieve",
        "settle",
        "stimulate",
        "succeed",
        "suppress",
        "sustain",
        "transform",
        "undermine",
        "withdrew",
        "yield",
        "abolish",
        "accomplish",
        // Nouns (generic)
        "beings",
        "championships",
        "director",
        "formula",
        "integers",
        "passages",
        "defeats",
        "pledges",
        "marshalls",
        "theorists",
        "adventists",
        "caliphs",
        "streifzüge",
        "bilanz",
        // Adjectives / adverbs
        "postwar",
        "naively",
        "paleolithic",
        "prehistoric",
        "medieval",
        "contemporary",
        "predominantly",
        "approximately",
        "consequently",
        "furthermore",
        "nevertheless",
        "subsequently",
        "alternatively",
        // German/French noise
        "jahren",
        "während",
        "zwischen",
        "bereits",
        "allerdings",
        "bibliothèque",
        "musique",
        "conseil",
        // Common nouns that appear as capitalized island entities
        "structure",
        "formation",
        "electric",
        "graph",
        "problem",
        "changes",
        "names",
        "upper",
        "studies",
        "addresses",
        "accuracy",
        "acoustics",
        "adulthood",
        "advancements",
        "afterwards",
        "lab",
        "bay",
        "base",
        "stem",
        "pass",
        "quart",
        "manus",
        "court",
        "office",
        "sea",
        "matter",
        "engine",
        "climate",
        "difference",
        "federal",
        "church",
        "states",
        "catholic",
        "reformed",
        "american",
        "scientific",
        "monthly",
        "notices",
        "colloquium",
        "academic",
        "accepted",
        "adiabatic",
        "chapters",
        "principles",
        "applications",
        "communications",
        "mechanics",
        "dynamics",
        "thermodynamics",
        "optics",
        "physics",
        "chemistry",
        "biology",
        "geometry",
        "algebra",
        "calculus",
        "statistics",
        "probability",
        "topology",
        "engineering",
        "architecture",
        "philosophy",
        "psychology",
        "sociology",
        "anthropology",
        "economics",
        "politics",
        "diplomacy",
        "agriculture",
        "industry",
        "commerce",
        "infrastructure",
        "parliament",
        "congress",
        "senate",
        "democracy",
        "monarchy",
        "aristocracy",
        "bureaucracy",
        "independence",
        "sovereignty",
        "territory",
        "population",
        "immigration",
        "emigration",
        "colonization",
        "modernization",
        "industrialization",
        "urbanization",
        "globalization",
        "reformation",
        "enlightenment",
        "renaissance",
        "conquest",
        "invasion",
        "rebellion",
        "uprising",
        "coup",
        "siege",
        "battle",
        "campaign",
        "alliance",
        "treaty",
        "armistice",
        "ceasefire",
        "occupation",
        "liberation",
        "resistance",
        "propaganda",
        "censorship",
        "persecution",
        "genocide",
        "massacre",
        "famine",
        "plague",
        "epidemic",
        "pandemic",
        "drought",
        "flood",
        "earthquake",
        "volcano",
        "tsunami",
        "catastrophe",
        "disaster",
        "crisis",
        "recession",
        "depression",
        "inflation",
        "prosperity",
    ];
    COMMON_WORDS.contains(&lower)
}

fn now_str() -> String {
    Utc::now()
        .naive_utc()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

fn parse_hypothesis_row(row: &rusqlite::Row) -> Hypothesis {
    let ef: String = row.get::<_, String>(5).unwrap_or_default();
    let ea: String = row.get::<_, String>(6).unwrap_or_default();
    let rc: String = row.get::<_, String>(7).unwrap_or_default();
    Hypothesis {
        id: row.get(0).unwrap_or(0),
        subject: row.get(1).unwrap_or_default(),
        predicate: row.get(2).unwrap_or_default(),
        object: row.get(3).unwrap_or_default(),
        confidence: row.get(4).unwrap_or(0.5),
        evidence_for: serde_json::from_str(&ef).unwrap_or_default(),
        evidence_against: serde_json::from_str(&ea).unwrap_or_default(),
        reasoning_chain: serde_json::from_str(&rc).unwrap_or_default(),
        status: HypothesisStatus::from_str(&row.get::<_, String>(8).unwrap_or_default()),
        discovered_at: row.get(9).unwrap_or_default(),
        pattern_source: row.get(10).unwrap_or_default(),
    }
}

/// Extract domain from a URL (best-effort, no external crate needed).
fn extract_domain(url: &str) -> String {
    let stripped = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    stripped
        .split('/')
        .next()
        .unwrap_or(stripped)
        .to_lowercase()
}

fn is_contradicting_predicate(a: &str, b: &str) -> bool {
    let pairs = [
        ("is", "is_not"),
        ("has", "lacks"),
        ("contains", "excludes"),
        ("member_of", "not_member_of"),
        ("created_by", "not_created_by"),
    ];
    for (p, q) in &pairs {
        if (a == *p && b == *q) || (a == *q && b == *p) {
            return true;
        }
    }
    false
}

fn predicates_similar(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let synonyms = [
        &["created", "created_by", "made", "built", "developed"][..],
        &["is", "is_a", "type_of"],
        &["has", "contains", "includes"],
        &["part_of", "belongs_to", "member_of"],
        &["located_in", "based_in", "in"],
        &["related_to", "associated_with", "connected_to", "knows"],
    ];
    for group in &synonyms {
        if group.contains(&a) && group.contains(&b) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Brain;

    fn test_brain() -> Brain {
        Brain::open_in_memory().unwrap()
    }

    fn setup_graph() -> Brain {
        let brain = test_brain();
        let a = brain.upsert_entity("Alice", "person").unwrap();
        let b = brain.upsert_entity("Bob", "person").unwrap();
        let c = brain.upsert_entity("Charlie", "person").unwrap();
        let d = brain.upsert_entity("Diana", "person").unwrap();
        brain.upsert_relation(a, "knows", b, "test").unwrap();
        brain.upsert_relation(a, "knows", c, "test").unwrap();
        brain.upsert_relation(b, "knows", d, "test").unwrap();
        brain.upsert_relation(c, "knows", d, "test").unwrap();
        brain
    }

    #[test]
    fn test_init_schema() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        // Should be able to list hypotheses on empty DB
        let hyps = p.list_hypotheses(None).unwrap();
        assert!(hyps.is_empty());
    }

    #[test]
    fn test_structural_holes() {
        let brain = setup_graph();
        let p = Prometheus::new(&brain).unwrap();
        let holes = p.find_structural_holes().unwrap();
        // B and C both connect to A and D but not to each other
        assert!(
            holes
                .iter()
                .any(|(a, b)| { (a == "Bob" && b == "Charlie") || (a == "Charlie" && b == "Bob") }),
            "Expected B-C hole, got: {:?}",
            holes
        );
    }

    #[test]
    fn test_co_occurrences() {
        let brain = setup_graph();
        let p = Prometheus::new(&brain).unwrap();
        let patterns = p.find_co_occurrences(1).unwrap();
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_frequent_subgraphs() {
        let brain = setup_graph();
        let p = Prometheus::new(&brain).unwrap();
        let patterns = p.find_frequent_subgraphs(1).unwrap();
        // Should find "knows, knows" motif at least twice (Alice and Diana both have 2 knows edges... wait, only outgoing)
        // Actually Alice has outgoing: knows->Bob, knows->Charlie. That's 1 entity with 2 outgoing of same pred.
        assert!(
            patterns
                .iter()
                .any(|p| p.pattern_type == PatternType::FrequentSubgraph),
            "patterns: {:?}",
            patterns
        );
    }

    #[test]
    fn test_type_gaps() {
        let brain = test_brain();
        // Create 4 persons, 3 of which have "works_at"
        let a = brain.upsert_entity("P1", "person").unwrap();
        let b = brain.upsert_entity("P2", "person").unwrap();
        let c = brain.upsert_entity("P3", "person").unwrap();
        let _d = brain.upsert_entity("P4", "person").unwrap();
        let co = brain.upsert_entity("Corp", "company").unwrap();
        brain.upsert_relation(a, "works_at", co, "test").unwrap();
        brain.upsert_relation(b, "works_at", co, "test").unwrap();
        brain.upsert_relation(c, "works_at", co, "test").unwrap();
        let p = Prometheus::new(&brain).unwrap();
        let gaps = p.find_type_gaps().unwrap();
        assert!(
            gaps.iter()
                .any(|(e, pred, _)| e == "P4" && pred == "works_at"),
            "Expected P4 missing works_at, got: {:?}",
            gaps
        );
    }

    #[test]
    fn test_anomalies() {
        let brain = test_brain();
        let a = brain.upsert_entity("X1", "widget").unwrap();
        let b = brain.upsert_entity("X2", "widget").unwrap();
        let c = brain.upsert_entity("X3", "widget").unwrap();
        let t = brain.upsert_entity("Target", "thing").unwrap();
        brain.upsert_relation(a, "has_feature", t, "test").unwrap();
        brain.upsert_relation(b, "has_feature", t, "test").unwrap();
        // X3 does NOT have has_feature
        let p = Prometheus::new(&brain).unwrap();
        let anomalies = p.find_anomalies().unwrap();
        assert!(
            anomalies
                .iter()
                .any(|p| p.entities_involved.contains(&"X3".to_string())),
            "anomalies: {:?}",
            anomalies
        );
    }

    #[test]
    fn test_generate_hypotheses_from_holes() {
        let brain = setup_graph();
        let p = Prometheus::new(&brain).unwrap();
        let hyps = p.generate_hypotheses_from_holes().unwrap();
        assert!(!hyps.is_empty());
        assert!(hyps.iter().any(|h| h.pattern_source == "structural_hole"));
    }

    #[test]
    fn test_generate_hypotheses_from_type_gaps() {
        let brain = test_brain();
        let a = brain.upsert_entity("A", "animal").unwrap();
        let b = brain.upsert_entity("B", "animal").unwrap();
        let c = brain.upsert_entity("C", "animal").unwrap();
        let _d = brain.upsert_entity("D", "animal").unwrap();
        let f = brain.upsert_entity("Food", "thing").unwrap();
        brain.upsert_relation(a, "eats", f, "test").unwrap();
        brain.upsert_relation(b, "eats", f, "test").unwrap();
        brain.upsert_relation(c, "eats", f, "test").unwrap();
        let p = Prometheus::new(&brain).unwrap();
        let hyps = p.generate_hypotheses_from_type_gaps().unwrap();
        assert!(hyps
            .iter()
            .any(|h| h.subject == "D" && h.predicate == "eats"));
    }

    #[test]
    fn test_check_contradiction_none() {
        let brain = setup_graph();
        let p = Prometheus::new(&brain).unwrap();
        let h = Hypothesis {
            id: 0,
            subject: "Alice".into(),
            predicate: "knows".into(),
            object: "Diana".into(),
            confidence: 0.5,
            evidence_for: vec![],
            evidence_against: vec![],
            reasoning_chain: vec![],
            status: HypothesisStatus::Proposed,
            discovered_at: now_str(),
            pattern_source: "test".into(),
        };
        assert!(!p.check_contradiction(&h).unwrap());
    }

    #[test]
    fn test_score_hypothesis() {
        let brain = setup_graph();
        let p = Prometheus::new(&brain).unwrap();
        let mut h = Hypothesis {
            id: 0,
            subject: "Bob".into(),
            predicate: "related_to".into(),
            object: "Charlie".into(),
            confidence: 0.5,
            evidence_for: vec!["shared neighbours".into()],
            evidence_against: vec![],
            reasoning_chain: vec![],
            status: HypothesisStatus::Proposed,
            discovered_at: now_str(),
            pattern_source: "structural_hole".into(),
        };
        let score = p.score_hypothesis(&mut h).unwrap();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_save_and_get_hypothesis() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let h = Hypothesis {
            id: 0,
            subject: "X".into(),
            predicate: "is".into(),
            object: "Y".into(),
            confidence: 0.7,
            evidence_for: vec!["reason 1".into()],
            evidence_against: vec![],
            reasoning_chain: vec!["step 1".into(), "step 2".into()],
            status: HypothesisStatus::Proposed,
            discovered_at: now_str(),
            pattern_source: "test".into(),
        };
        let id = p.save_hypothesis(&h).unwrap();
        let loaded = p.get_hypothesis(id).unwrap().unwrap();
        assert_eq!(loaded.subject, "X");
        assert_eq!(loaded.predicate, "is");
        assert_eq!(loaded.object, "Y");
        assert_eq!(loaded.reasoning_chain.len(), 2);
    }

    #[test]
    fn test_list_hypotheses_filter() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let h1 = Hypothesis {
            id: 0,
            subject: "A".into(),
            predicate: "is".into(),
            object: "B".into(),
            confidence: 0.5,
            evidence_for: vec![],
            evidence_against: vec![],
            reasoning_chain: vec![],
            status: HypothesisStatus::Proposed,
            discovered_at: now_str(),
            pattern_source: "test".into(),
        };
        let h2 = Hypothesis {
            status: HypothesisStatus::Confirmed,
            subject: "C".into(),
            object: "D".into(),
            ..h1.clone()
        };
        p.save_hypothesis(&h1).unwrap();
        p.save_hypothesis(&h2).unwrap();
        let all = p.list_hypotheses(None).unwrap();
        assert_eq!(all.len(), 2);
        let confirmed = p
            .list_hypotheses(Some(HypothesisStatus::Confirmed))
            .unwrap();
        assert_eq!(confirmed.len(), 1);
        assert_eq!(confirmed[0].subject, "C");
    }

    #[test]
    fn test_update_hypothesis_status() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let h = Hypothesis {
            id: 0,
            subject: "A".into(),
            predicate: "is".into(),
            object: "B".into(),
            confidence: 0.5,
            evidence_for: vec![],
            evidence_against: vec![],
            reasoning_chain: vec![],
            status: HypothesisStatus::Proposed,
            discovered_at: now_str(),
            pattern_source: "test".into(),
        };
        let id = p.save_hypothesis(&h).unwrap();
        p.update_hypothesis_status(id, HypothesisStatus::Confirmed)
            .unwrap();
        let loaded = p.get_hypothesis(id).unwrap().unwrap();
        assert_eq!(loaded.status, HypothesisStatus::Confirmed);
    }

    #[test]
    fn test_save_pattern() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let pat = Pattern {
            id: 0,
            pattern_type: PatternType::CoOccurrence,
            entities_involved: vec!["A".into(), "B".into()],
            frequency: 5,
            last_seen: now_str(),
            description: "test pattern".into(),
        };
        let id = p.save_pattern(&pat).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_save_discovery() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let h = Hypothesis {
            id: 0,
            subject: "A".into(),
            predicate: "is".into(),
            object: "B".into(),
            confidence: 0.9,
            evidence_for: vec![],
            evidence_against: vec![],
            reasoning_chain: vec![],
            status: HypothesisStatus::Confirmed,
            discovered_at: now_str(),
            pattern_source: "test".into(),
        };
        let hid = p.save_hypothesis(&h).unwrap();
        let did = p
            .save_discovery(hid, &["source1".into(), "source2".into()])
            .unwrap();
        assert!(did > 0);
    }

    #[test]
    fn test_record_outcome_and_weight() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        p.record_outcome("structural_hole", true).unwrap();
        p.record_outcome("structural_hole", true).unwrap();
        p.record_outcome("structural_hole", false).unwrap();
        let w = p.get_pattern_weight("structural_hole").unwrap();
        // 2 confirmations, 1 rejection => 2/3 ≈ 0.667
        assert!((w - 2.0 / 3.0).abs() < 0.01, "weight: {}", w);
    }

    #[test]
    fn test_get_pattern_weights() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        p.record_outcome("type_gap", true).unwrap();
        p.record_outcome("co_occurrence", false).unwrap();
        let weights = p.get_pattern_weights().unwrap();
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_discovery_scores_empty() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let scores = p.discovery_scores().unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_discovery_scores() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let h = Hypothesis {
            id: 0,
            subject: "A".into(),
            predicate: "is".into(),
            object: "B".into(),
            confidence: 0.9,
            evidence_for: vec![],
            evidence_against: vec![],
            reasoning_chain: vec![],
            status: HypothesisStatus::Confirmed,
            discovered_at: now_str(),
            pattern_source: "test".into(),
        };
        p.save_hypothesis(&h).unwrap();
        let scores = p.discovery_scores().unwrap();
        assert_eq!(*scores.get("A").unwrap(), 1);
        assert_eq!(*scores.get("B").unwrap(), 1);
    }

    #[test]
    fn test_explain_hypothesis() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let h = Hypothesis {
            id: 0,
            subject: "X".into(),
            predicate: "related_to".into(),
            object: "Y".into(),
            confidence: 0.6,
            evidence_for: vec!["ev1".into()],
            evidence_against: vec![],
            reasoning_chain: vec!["step1".into(), "step2".into()],
            status: HypothesisStatus::Testing,
            discovered_at: now_str(),
            pattern_source: "structural_hole".into(),
        };
        let id = p.save_hypothesis(&h).unwrap();
        let explanation = p.explain(id).unwrap().unwrap();
        assert!(explanation.contains("X"));
        assert!(explanation.contains("step1"));
        assert!(explanation.contains("Evidence For"));
    }

    #[test]
    fn test_explain_nonexistent() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        assert!(p.explain(999).unwrap().is_none());
    }

    #[test]
    fn test_full_discover_pipeline() {
        let brain = setup_graph();
        let p = Prometheus::new(&brain).unwrap();
        let report = p.discover().unwrap();
        assert!(!report.summary.is_empty());
        // Should find at least the B-C structural hole hypothesis
        assert!(
            report.hypotheses_generated.len() > 0 || report.patterns_found.len() > 0,
            "report: {:?}",
            report.summary
        );
    }

    #[test]
    fn test_report_json() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let report = DiscoveryReport {
            patterns_found: vec![],
            hypotheses_generated: vec![],
            gaps_detected: 0,
            summary: "test".into(),
        };
        let json = p.report_json(&report);
        assert!(json.contains("test"));
    }

    #[test]
    fn test_report_markdown() {
        let brain = test_brain();
        let p = Prometheus::new(&brain).unwrap();
        let report = DiscoveryReport {
            patterns_found: vec![Pattern {
                id: 1,
                pattern_type: PatternType::CoOccurrence,
                entities_involved: vec!["A".into()],
                frequency: 3,
                last_seen: now_str(),
                description: "test pattern".into(),
            }],
            hypotheses_generated: vec![],
            gaps_detected: 0,
            summary: "test summary".into(),
        };
        let md = p.report_markdown(&report);
        assert!(md.contains("PROMETHEUS"));
        assert!(md.contains("test pattern"));
    }

    #[test]
    fn test_validate_hypothesis_with_evidence() {
        let brain = test_brain();
        let a = brain.upsert_entity("Rust", "language").unwrap();
        let b = brain.upsert_entity("Mozilla", "org").unwrap();
        brain.upsert_relation(a, "created_by", b, "test").unwrap();
        let p = Prometheus::new(&brain).unwrap();
        let mut h = Hypothesis {
            id: 0,
            subject: "Rust".into(),
            predicate: "created_by".into(),
            object: "Mozilla".into(),
            confidence: 0.5,
            evidence_for: vec![],
            evidence_against: vec![],
            reasoning_chain: vec![],
            status: HypothesisStatus::Proposed,
            discovered_at: now_str(),
            pattern_source: "test".into(),
        };
        p.validate_hypothesis(&mut h).unwrap();
        assert!(h.confidence > 0.5);
        assert!(!h.evidence_for.is_empty());
    }

    #[test]
    fn test_temporal_patterns() {
        let brain = test_brain();
        let a = brain.upsert_entity("E1", "thing").unwrap();
        let b = brain.upsert_entity("E2", "thing").unwrap();
        let c = brain.upsert_entity("E3", "thing").unwrap();
        let d = brain.upsert_entity("E4", "thing").unwrap();
        // Same subject, different predicates learned sequentially
        brain.upsert_relation(a, "creates", b, "test").unwrap();
        brain.upsert_relation(a, "publishes", c, "test").unwrap();
        brain.upsert_relation(d, "creates", b, "test").unwrap();
        brain.upsert_relation(d, "publishes", c, "test").unwrap();
        let p = Prometheus::new(&brain).unwrap();
        let temporal = p.find_temporal_patterns(2).unwrap();
        assert!(
            temporal
                .iter()
                .any(|p| p.pattern_type == PatternType::TemporalSequence),
            "temporal: {:?}",
            temporal
        );
    }

    #[test]
    fn test_hypothesis_status_roundtrip() {
        assert_eq!(
            HypothesisStatus::from_str(HypothesisStatus::Proposed.as_str()),
            HypothesisStatus::Proposed
        );
        assert_eq!(
            HypothesisStatus::from_str(HypothesisStatus::Testing.as_str()),
            HypothesisStatus::Testing
        );
        assert_eq!(
            HypothesisStatus::from_str(HypothesisStatus::Confirmed.as_str()),
            HypothesisStatus::Confirmed
        );
        assert_eq!(
            HypothesisStatus::from_str(HypothesisStatus::Rejected.as_str()),
            HypothesisStatus::Rejected
        );
    }

    #[test]
    fn test_pattern_type_roundtrip() {
        let types = [
            PatternType::CoOccurrence,
            PatternType::StructuralHole,
            PatternType::TypeGap,
            PatternType::Analogy,
            PatternType::TemporalSequence,
            PatternType::FrequentSubgraph,
        ];
        for t in &types {
            assert_eq!(PatternType::from_str(t.as_str()), *t);
        }
    }

    #[test]
    fn test_predicate_chains() {
        let brain = test_brain();
        let a = brain.upsert_entity("Einstein", "person").unwrap();
        let b = brain.upsert_entity("Germany", "place").unwrap();
        let c = brain.upsert_entity("Europe", "place").unwrap();
        let d = brain.upsert_entity("Curie", "person").unwrap();
        brain.upsert_relation(a, "born_in", b, "test").unwrap();
        brain.upsert_relation(b, "located_in", c, "test").unwrap();
        brain.upsert_relation(d, "born_in", b, "test").unwrap();
        let p = Prometheus::new(&brain).unwrap();
        let chains = p.find_predicate_chains(2).unwrap();
        assert!(
            chains
                .iter()
                .any(|p| p.description.contains("born_in") && p.description.contains("located_in")),
            "chains: {:?}",
            chains
        );
        let hyps = p.generate_hypotheses_from_chains().unwrap();
        assert!(
            hyps.iter().any(|h| h.pattern_source == "predicate_chain"),
            "hyps: {:?}",
            hyps
        );
    }

    #[test]
    fn test_predicates_similar() {
        assert!(predicates_similar("created", "built"));
        assert!(predicates_similar("is", "is_a"));
        assert!(!predicates_similar("created", "eats"));
    }

    #[test]
    fn test_is_contradicting_predicate() {
        assert!(is_contradicting_predicate("is", "is_not"));
        assert!(is_contradicting_predicate("has", "lacks"));
        assert!(!is_contradicting_predicate("is", "has"));
    }
}
