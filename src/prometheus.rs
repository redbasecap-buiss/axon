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
    {
        return true;
    }
    // Long multi-word names that look like sentence fragments (>6 words)
    let word_count = lower_trimmed.split_whitespace().count();
    if word_count > 6 {
        return true;
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
        for (_src, entities) in &source_entities {
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
        let ids: Vec<i64> = neighbours.keys().copied().collect();
        let mut patterns = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = ids[i];
                let b = ids[j];
                let na = &neighbours[&a];
                let nb = &neighbours[&b];
                let shared: usize = na.intersection(nb).count();
                if shared >= min_shared {
                    let union_size = na.union(nb).count();
                    let jaccard = if union_size > 0 {
                        shared as f64 / union_size as f64
                    } else {
                        0.0
                    };
                    let a_name = self.entity_name(a)?;
                    let b_name = self.entity_name(b)?;
                    patterns.push(Pattern {
                        id: 0,
                        pattern_type: PatternType::CoOccurrence,
                        entities_involved: vec![a_name.clone(), b_name.clone()],
                        frequency: shared as i64,
                        last_seen: now_str(),
                        description: format!(
                            "{} and {} share {} neighbours (Jaccard: {:.2})",
                            a_name, b_name, shared, jaccard
                        ),
                    });
                }
            }
        }
        // Sort by frequency descending for better results
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
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
    pub fn generate_hypotheses_from_shared_objects(&self) -> Result<Vec<Hypothesis>> {
        let relations = self.brain.all_relations()?;
        let meaningful = meaningful_ids(self.brain)?;
        // Group by (predicate, object_id) → list of subject_ids
        // Only require subjects to be meaningful (objects can be any type)
        let mut groups: HashMap<(String, i64), Vec<i64>> = HashMap::new();
        for r in &relations {
            if meaningful.contains(&r.subject_id) {
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

    /// Score a hypothesis based on available evidence.
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

        // 3. Island entities as gaps
        let islands = self.find_island_entities()?;

        let gaps_detected = all_hypotheses.len() + islands.len();

        // 4. Deduplicate hypotheses against existing DB, then score and validate
        all_hypotheses.retain(|h| {
            !self
                .hypothesis_exists(&h.subject, &h.predicate, &h.object)
                .unwrap_or(true)
        });
        for h in all_hypotheses.iter_mut().take(200) {
            self.score_hypothesis(h)?;
            self.validate_hypothesis(h)?;
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

        // Decay old hypotheses (meta-learning)
        let decayed = self.decay_old_hypotheses(7).unwrap_or(0);

        // Auto-resolve existing testing hypotheses
        let (auto_confirmed, auto_rejected) = self.auto_resolve_hypotheses().unwrap_or((0, 0));

        // Find connectable islands
        let connectable = self.find_connectable_islands().unwrap_or_default();

        // Prioritize gaps
        let gap_priorities = self.prioritize_gaps().unwrap_or_default();

        // Predicate diversity analysis
        let (total_rels, _generic_rels, diverse_rels, div_ratio) =
            self.predicate_diversity().unwrap_or((0, 0, 0, 0.0));

        // Fuzzy duplicate detection
        let fuzzy_dupes = self.find_fuzzy_duplicates().unwrap_or_default();
        let merge_candidates = fuzzy_dupes.iter().filter(|d| d.3 == "merge").count();

        // Topic coverage
        let topic_coverage = self.topic_coverage_analysis().unwrap_or_default();
        let sparse_topics = topic_coverage.iter().filter(|t| t.3 < 0.01).count();

        let summary = format!(
            "Discovered {} patterns, generated {} new hypotheses ({} confirmed, {} rejected), \
             auto-resolved {} existing ({}✓ {}✗), {} island entities, {} meaningful relations, \
             {} cross-domain gaps, {} knowledge frontiers, {} bridge entities, {} connectable islands, \
             {} prioritized gaps, {} decayed, {} island clusters, \
             predicate diversity: {}/{} ({:.0}% diverse), \
             {} fuzzy duplicates ({} auto-merge), {} sparse topic domains",
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
            island_clusters.len(),
            diverse_rels,
            total_rels,
            div_ratio * 100.0,
            fuzzy_dupes.len(),
            merge_candidates,
            sparse_topics,
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
            let mut types: HashMap<String, usize> = HashMap::new();
            let mut names: Vec<String> = Vec::new();
            for &id in comp {
                if let Some(e) = id_to_entity.get(&id) {
                    if !noise_types.contains(e.entity_type.as_str()) {
                        *types.entry(e.entity_type.clone()).or_insert(0) += 1;
                        if names.len() < 5 {
                            names.push(e.name.clone());
                        }
                    }
                }
            }
            if !types.is_empty() {
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
                let path = crate::graph::shortest_path(self.brain, &h.subject, &h.object)?;
                if let Some(p) = &path {
                    if p.len() <= 3 {
                        // Close in graph — likely related
                        let new_conf = (h.confidence + 0.2).min(1.0);
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

        for (_etype, indices) in &by_type {
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
                        let action = if jaccard >= 0.8 {
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

    fn entity_name(&self, id: i64) -> Result<String> {
        Ok(self
            .brain
            .get_entity_by_id(id)?
            .map(|e| e.name)
            .unwrap_or_else(|| format!("#{}", id)))
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

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
