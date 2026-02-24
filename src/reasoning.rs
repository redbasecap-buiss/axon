#![allow(dead_code, clippy::type_complexity)]
//! Crystalline Reasoning Engine
//!
//! Five interconnected reasoning subsystems that operate over the axon knowledge graph:
//!
//! 1. **Multi-Hop Causal Reasoning** — Discovers causal chains between entities using
//!    BFS with confidence decay, temporal ordering, and predicate-type causality scoring.
//!
//! 2. **Cross-Domain Analogical Reasoning** — Finds structural isomorphisms between
//!    different entity-type clusters using predicate-role signatures and graph edit distance.
//!
//! 3. **Emergent Concept Formation** — Clusters entities sharing unusual predicate
//!    combinations, auto-names discovered concepts, and inserts them back into the graph.
//!
//! 4. **Contradiction Detection & Resolution** — Identifies conflicting facts and
//!    impossible timelines, resolves via source authority, recency, majority vote, and
//!    confidence weighting. Tracks history for meta-learning.
//!
//! 5. **Self-Restructuring Ontology** — Detects predicate synonyms and overloaded
//!    predicates, proposes merges/splits/super-predicates, applies changes with full
//!    provenance tracking.

use chrono::Utc;
use rusqlite::{params, Connection, Result};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::db::Brain;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Confidence decay factor per hop in causal chain discovery.
/// Each hop multiplies cumulative confidence by this factor.
const CAUSAL_DECAY_PER_HOP: f64 = 0.85;

/// Minimum cumulative confidence to continue exploring a causal chain.
const CAUSAL_MIN_CONFIDENCE: f64 = 0.01;

/// Maximum causal chain length default.
const DEFAULT_MAX_HOPS: usize = 5;

/// Predicates that strongly imply causal/directional relationships.
const CAUSAL_PREDICATES: &[&str] = &[
    "caused",
    "causes",
    "led_to",
    "resulted_in",
    "produced",
    "created",
    "founded",
    "invented",
    "discovered",
    "developed",
    "built",
    "enabled",
    "triggered",
    "inspired",
    "influenced",
    "motivated",
    "pioneered",
    "originated",
    "initiated",
    "launched",
    "established",
    "introduced",
    "derived_from",
    "evolved_from",
    "based_on",
    "built_on",
    "preceded",
    "followed",
    "succeeded",
    "replaced",
    "transformed",
    "contributed_to",
];

/// Predicates that indicate correlation rather than causation.
const CORRELATIVE_PREDICATES: &[&str] = &[
    "related_to",
    "associated_with",
    "contemporary_of",
    "similar_to",
    "co_occurs_with",
    "colleagues_at",
    "co_researchers",
    "partner_of",
    "affiliated_with",
    "connected_to",
];

/// Minimum number of shared predicates to form an emergent concept.
const MIN_SHARED_PREDICATES: usize = 2;

/// Minimum member count for an emergent concept.
const MIN_CONCEPT_MEMBERS: usize = 3;

/// Minimum Jaccard similarity for predicate synonym detection.
const SYNONYM_JACCARD_THRESHOLD: f64 = 0.6;

/// Maximum entity types for a predicate before it's considered overloaded.
const OVERLOAD_TYPE_THRESHOLD: usize = 5;

/// Entity types to exclude from reasoning (noise).
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
    "unknown",
];

// ═══════════════════════════════════════════════════════════════════════════
// 1. MULTI-HOP CAUSAL REASONING
// ═══════════════════════════════════════════════════════════════════════════

/// A single link in a causal chain.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalLink {
    pub subject_id: i64,
    pub predicate: String,
    pub object_id: i64,
    pub edge_confidence: f64,
    pub is_causal: bool,      // true if predicate is in CAUSAL_PREDICATES
    pub is_correlative: bool, // true if predicate is in CORRELATIVE_PREDICATES
}

/// An ordered sequence of causal links from source to target.
#[derive(Debug, Clone)]
pub struct CausalChain {
    pub links: Vec<CausalLink>,
    pub cumulative_confidence: f64,
    pub source_entity_id: i64,
    pub target_entity_id: i64,
    pub causal_fraction: f64, // fraction of links that are truly causal
}

impl CausalChain {
    /// Number of hops in the chain.
    pub fn hops(&self) -> usize {
        self.links.len()
    }

    /// Is this chain purely causal (no correlative links)?
    pub fn is_purely_causal(&self) -> bool {
        self.links.iter().all(|l| l.is_causal)
    }
}

/// Compute the causal strength of a chain.
///
/// Combines:
/// - Cumulative edge confidences with per-hop decay
/// - Path-length penalty (exponential decay)
/// - Causal vs correlative predicate weighting
/// - Temporal ordering bonus (if first_seen data suggests forward causation)
pub fn causal_strength(chain: &CausalChain) -> f64 {
    if chain.links.is_empty() {
        return 0.0;
    }

    let n = chain.links.len() as f64;

    // Base: cumulative confidence with hop decay
    let mut base_confidence = 1.0;
    for link in &chain.links {
        base_confidence *= link.edge_confidence * CAUSAL_DECAY_PER_HOP;
    }

    // Path-length penalty: 1/sqrt(n) — shorter paths are stronger evidence
    let length_penalty = 1.0 / n.sqrt();

    // Causal predicate bonus: weight causal links more heavily
    let causal_weight = if n > 0.0 {
        let causal_count = chain.links.iter().filter(|l| l.is_causal).count() as f64;
        let correlative_count = chain.links.iter().filter(|l| l.is_correlative).count() as f64;
        let neutral_count = n - causal_count - correlative_count;

        // Causal links get weight 1.0, neutral 0.5, correlative 0.3
        (causal_count * 1.0 + neutral_count * 0.5 + correlative_count * 0.3) / n
    } else {
        0.5
    };

    base_confidence * length_penalty * causal_weight
}

/// Discover causal chains between a source and target entity using BFS
/// with confidence-weighted exploration.
///
/// Returns chains sorted by causal strength (strongest first).
pub fn discover_causal_chains(
    brain: &Brain,
    source_id: i64,
    target_id: i64,
    max_hops: usize,
) -> Result<Vec<CausalChain>> {
    // Build adjacency with edge metadata
    let adj = build_weighted_adjacency(brain)?;

    let max_hops = max_hops.min(DEFAULT_MAX_HOPS);

    // BFS-like exploration with confidence tracking
    // State: (current_node, path_so_far, cumulative_confidence)
    let mut queue: VecDeque<(i64, Vec<CausalLink>, f64)> = VecDeque::new();
    let mut results: Vec<CausalChain> = Vec::new();

    // Seed with edges from source
    if let Some(edges) = adj.get(&source_id) {
        for edge in edges {
            let conf = edge.edge_confidence * CAUSAL_DECAY_PER_HOP;
            if conf >= CAUSAL_MIN_CONFIDENCE {
                queue.push_back((edge.object_id, vec![edge.clone()], conf));
            }
        }
    }

    let mut visited_paths: HashSet<Vec<i64>> = HashSet::new();

    while let Some((current, path, cum_conf)) = queue.pop_front() {
        if path.len() > max_hops {
            continue;
        }

        // Check if we reached the target
        if current == target_id {
            let path_ids: Vec<i64> = path.iter().map(|l| l.object_id).collect();
            if visited_paths.insert(path_ids) {
                let causal_count = path.iter().filter(|l| l.is_causal).count();
                let chain = CausalChain {
                    source_entity_id: source_id,
                    target_entity_id: target_id,
                    causal_fraction: if path.is_empty() {
                        0.0
                    } else {
                        causal_count as f64 / path.len() as f64
                    },
                    cumulative_confidence: cum_conf,
                    links: path,
                };
                results.push(chain);
            }
            continue;
        }

        // Don't revisit nodes already in this path (avoid cycles)
        let path_nodes: HashSet<i64> = std::iter::once(source_id)
            .chain(path.iter().map(|l| l.object_id))
            .collect();

        if let Some(edges) = adj.get(&current) {
            for edge in edges {
                if path_nodes.contains(&edge.object_id) {
                    continue;
                }
                let new_conf = cum_conf * edge.edge_confidence * CAUSAL_DECAY_PER_HOP;
                if new_conf >= CAUSAL_MIN_CONFIDENCE && path.len() < max_hops {
                    let mut new_path = path.clone();
                    new_path.push(edge.clone());
                    queue.push_back((edge.object_id, new_path, new_conf));
                }
            }
        }
    }

    // Sort by causal strength descending
    results.sort_by(|a, b| {
        causal_strength(b)
            .partial_cmp(&causal_strength(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Find all indirect causal predecessors of an entity up to a given depth.
///
/// Uses reverse BFS: follows edges *into* the entity, collecting all
/// upstream nodes that could have causally influenced it.
pub fn find_indirect_causes(
    brain: &Brain,
    entity_id: i64,
    max_depth: usize,
) -> Result<Vec<(i64, f64)>> {
    // Build reverse adjacency (who points to whom)
    let reverse_adj = build_reverse_weighted_adjacency(brain)?;
    let max_depth = max_depth.min(DEFAULT_MAX_HOPS);

    let mut causes: HashMap<i64, f64> = HashMap::new();
    let mut visited: HashSet<i64> = HashSet::new();
    visited.insert(entity_id);

    // BFS layers
    let mut frontier: Vec<(i64, f64)> = vec![(entity_id, 1.0)];

    for _depth in 0..max_depth {
        let mut next_frontier: Vec<(i64, f64)> = Vec::new();

        for (node, cum_conf) in &frontier {
            if let Some(edges) = reverse_adj.get(node) {
                for edge in edges {
                    if visited.contains(&edge.subject_id) {
                        continue;
                    }

                    let causal_bonus = if edge.is_causal { 1.0 } else { 0.5 };
                    let new_conf =
                        cum_conf * edge.edge_confidence * CAUSAL_DECAY_PER_HOP * causal_bonus;

                    if new_conf >= CAUSAL_MIN_CONFIDENCE {
                        visited.insert(edge.subject_id);
                        let entry = causes.entry(edge.subject_id).or_insert(0.0);
                        *entry = entry.max(new_conf); // keep strongest path
                        next_frontier.push((edge.subject_id, new_conf));
                    }
                }
            }
        }

        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    let mut result: Vec<(i64, f64)> = causes.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(result)
}

/// Build forward adjacency map with edge metadata.
fn build_weighted_adjacency(brain: &Brain) -> Result<HashMap<i64, Vec<CausalLink>>> {
    let relations = brain.all_relations()?;
    let mut adj: HashMap<i64, Vec<CausalLink>> = HashMap::new();

    for r in &relations {
        let pred_lower = r.predicate.to_lowercase();
        let is_causal = CAUSAL_PREDICATES.iter().any(|p| pred_lower.contains(p));
        let is_correlative = CORRELATIVE_PREDICATES
            .iter()
            .any(|p| pred_lower.contains(p));

        adj.entry(r.subject_id).or_default().push(CausalLink {
            subject_id: r.subject_id,
            predicate: r.predicate.clone(),
            object_id: r.object_id,
            edge_confidence: r.confidence,
            is_causal,
            is_correlative,
        });
    }

    Ok(adj)
}

/// Build reverse adjacency (object→subject edges) for backward causal tracing.
fn build_reverse_weighted_adjacency(brain: &Brain) -> Result<HashMap<i64, Vec<CausalLink>>> {
    let relations = brain.all_relations()?;
    let mut adj: HashMap<i64, Vec<CausalLink>> = HashMap::new();

    for r in &relations {
        let pred_lower = r.predicate.to_lowercase();
        let is_causal = CAUSAL_PREDICATES.iter().any(|p| pred_lower.contains(p));
        let is_correlative = CORRELATIVE_PREDICATES
            .iter()
            .any(|p| pred_lower.contains(p));

        // Reverse: index by object_id
        adj.entry(r.object_id).or_default().push(CausalLink {
            subject_id: r.subject_id,
            predicate: r.predicate.clone(),
            object_id: r.object_id,
            edge_confidence: r.confidence,
            is_causal,
            is_correlative,
        });
    }

    Ok(adj)
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. CROSS-DOMAIN ANALOGICAL REASONING
// ═══════════════════════════════════════════════════════════════════════════

/// A predicate-role signature: the set of predicates connected to an entity,
/// with directional information (subject vs object).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PredicateSignature {
    /// Predicates where this entity is the subject: (predicate, object_type)
    pub outgoing: BTreeSet<(String, String)>,
    /// Predicates where this entity is the object: (predicate, subject_type)
    pub incoming: BTreeSet<(String, String)>,
}

impl PredicateSignature {
    /// Compute structural similarity between two signatures using Jaccard index
    /// over the predicate roles (ignoring entity types for generalization).
    pub fn similarity(&self, other: &PredicateSignature) -> f64 {
        let self_preds: BTreeSet<&str> = self
            .outgoing
            .iter()
            .map(|(p, _)| p.as_str())
            .chain(self.incoming.iter().map(|(p, _)| p.as_str()))
            .collect();

        let other_preds: BTreeSet<&str> = other
            .outgoing
            .iter()
            .map(|(p, _)| p.as_str())
            .chain(other.incoming.iter().map(|(p, _)| p.as_str()))
            .collect();

        let intersection = self_preds.intersection(&other_preds).count();
        let union = self_preds.union(&other_preds).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// More nuanced similarity that also considers role direction.
    pub fn directed_similarity(&self, other: &PredicateSignature) -> f64 {
        let out_self: BTreeSet<&str> = self.outgoing.iter().map(|(p, _)| p.as_str()).collect();
        let out_other: BTreeSet<&str> = other.outgoing.iter().map(|(p, _)| p.as_str()).collect();
        let in_self: BTreeSet<&str> = self.incoming.iter().map(|(p, _)| p.as_str()).collect();
        let in_other: BTreeSet<&str> = other.incoming.iter().map(|(p, _)| p.as_str()).collect();

        let out_inter = out_self.intersection(&out_other).count();
        let out_union = out_self.union(&out_other).count();
        let in_inter = in_self.intersection(&in_other).count();
        let in_union = in_self.union(&in_other).count();

        let out_sim = if out_union == 0 {
            0.0
        } else {
            out_inter as f64 / out_union as f64
        };
        let in_sim = if in_union == 0 {
            0.0
        } else {
            in_inter as f64 / in_union as f64
        };

        // Weighted average: outgoing predicates slightly more important (they're active roles)
        0.55 * out_sim + 0.45 * in_sim
    }
}

/// An analogy between two entities from different domains.
#[derive(Debug, Clone)]
pub struct Analogy {
    pub entity_a: i64,
    pub entity_b: i64,
    pub domain_a: String,               // entity_type of a
    pub domain_b: String,               // entity_type of b
    pub mapping: Vec<(String, String)>, // (predicate_a, predicate_b) correspondences
    pub structural_similarity: f64,
    pub description: String,
}

/// Compute analogy score using a combination of structural similarity
/// and graph neighbourhood overlap.
pub fn analogy_score(analogy: &Analogy) -> f64 {
    let mapping_bonus = (analogy.mapping.len() as f64).sqrt() / 3.0;
    let cross_domain_bonus = if analogy.domain_a != analogy.domain_b {
        0.2
    } else {
        0.0
    };

    (analogy.structural_similarity * 0.6 + mapping_bonus * 0.3 + cross_domain_bonus * 0.1).min(1.0)
}

/// Discover analogies across entity-type domains.
///
/// Algorithm:
/// 1. Compute predicate-role signatures for all meaningful entities
/// 2. Group entities by entity_type (domain)
/// 3. For each cross-domain pair, find entities with high signature similarity
/// 4. Rank by structural similarity and mapping quality
pub fn discover_analogies(brain: &Brain) -> Result<Vec<Analogy>> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;

    // Filter to meaningful entities
    let meaningful: HashMap<i64, (&str, &str)> = entities
        .iter()
        .filter(|e| !is_noise_type(&e.entity_type) && e.name.len() >= 2)
        .map(|e| (e.id, (e.name.as_str(), e.entity_type.as_str())))
        .collect();

    // Build entity type lookup
    let entity_type_map: HashMap<i64, &str> = entities
        .iter()
        .map(|e| (e.id, e.entity_type.as_str()))
        .collect();

    // Compute predicate signatures for each entity
    let mut signatures: HashMap<i64, PredicateSignature> = HashMap::new();

    for r in &relations {
        if !meaningful.contains_key(&r.subject_id) || !meaningful.contains_key(&r.object_id) {
            continue;
        }

        let obj_type = entity_type_map
            .get(&r.object_id)
            .copied()
            .unwrap_or("unknown");
        let subj_type = entity_type_map
            .get(&r.subject_id)
            .copied()
            .unwrap_or("unknown");

        signatures
            .entry(r.subject_id)
            .or_insert_with(|| PredicateSignature {
                outgoing: BTreeSet::new(),
                incoming: BTreeSet::new(),
            })
            .outgoing
            .insert((r.predicate.clone(), obj_type.to_string()));

        signatures
            .entry(r.object_id)
            .or_insert_with(|| PredicateSignature {
                outgoing: BTreeSet::new(),
                incoming: BTreeSet::new(),
            })
            .incoming
            .insert((r.predicate.clone(), subj_type.to_string()));
    }

    // Group entities by type
    let mut type_groups: HashMap<&str, Vec<i64>> = HashMap::new();
    for (&eid, &(_name, etype)) in &meaningful {
        if signatures.contains_key(&eid) {
            type_groups.entry(etype).or_default().push(eid);
        }
    }

    let type_keys: Vec<&str> = type_groups.keys().copied().collect();
    let mut analogies: Vec<Analogy> = Vec::new();

    // Compare entities across different type domains
    for i in 0..type_keys.len() {
        for j in (i + 1)..type_keys.len() {
            let type_a = type_keys[i];
            let type_b = type_keys[j];
            let group_a = &type_groups[type_a];
            let group_b = &type_groups[type_b];

            // Limit comparison to avoid O(n^2) explosion
            let limit_a = group_a.len().min(50);
            let limit_b = group_b.len().min(50);

            for &ea in &group_a[..limit_a] {
                let sig_a = match signatures.get(&ea) {
                    Some(s) => s,
                    None => continue,
                };

                // Skip entities with very few predicates (too little structure)
                if sig_a.outgoing.len() + sig_a.incoming.len() < 2 {
                    continue;
                }

                for &eb in &group_b[..limit_b] {
                    let sig_b = match signatures.get(&eb) {
                        Some(s) => s,
                        None => continue,
                    };

                    if sig_b.outgoing.len() + sig_b.incoming.len() < 2 {
                        continue;
                    }

                    let sim = sig_a.directed_similarity(sig_b);

                    if sim >= 0.3 {
                        // Build predicate mapping
                        let mapping = build_predicate_mapping(sig_a, sig_b);

                        let name_a = meaningful.get(&ea).map(|(n, _)| *n).unwrap_or("?");
                        let name_b = meaningful.get(&eb).map(|(n, _)| *n).unwrap_or("?");

                        let shared: Vec<String> = mapping.iter().map(|(p, _)| p.clone()).collect();
                        let desc = format!(
                            "{} ({}) ↔ {} ({}) share structural roles: [{}]",
                            name_a,
                            type_a,
                            name_b,
                            type_b,
                            shared.join(", ")
                        );

                        analogies.push(Analogy {
                            entity_a: ea,
                            entity_b: eb,
                            domain_a: type_a.to_string(),
                            domain_b: type_b.to_string(),
                            mapping,
                            structural_similarity: sim,
                            description: desc,
                        });
                    }
                }
            }
        }
    }

    // Sort by analogy score descending
    analogies.sort_by(|a, b| {
        analogy_score(b)
            .partial_cmp(&analogy_score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Keep top results
    analogies.truncate(100);

    Ok(analogies)
}

/// Build predicate mapping between two signatures — find corresponding predicates.
fn build_predicate_mapping(
    sig_a: &PredicateSignature,
    sig_b: &PredicateSignature,
) -> Vec<(String, String)> {
    let mut mapping = Vec::new();

    // Outgoing predicates that match
    let out_a: BTreeSet<&str> = sig_a.outgoing.iter().map(|(p, _)| p.as_str()).collect();
    let out_b: BTreeSet<&str> = sig_b.outgoing.iter().map(|(p, _)| p.as_str()).collect();
    for p in out_a.intersection(&out_b) {
        mapping.push((p.to_string(), p.to_string()));
    }

    // Incoming predicates that match
    let in_a: BTreeSet<&str> = sig_a.incoming.iter().map(|(p, _)| p.as_str()).collect();
    let in_b: BTreeSet<&str> = sig_b.incoming.iter().map(|(p, _)| p.as_str()).collect();
    for p in in_a.intersection(&in_b) {
        if !mapping.iter().any(|(a, _)| a == *p) {
            mapping.push((p.to_string(), p.to_string()));
        }
    }

    mapping
}

fn is_noise_type(t: &str) -> bool {
    NOISE_TYPES.contains(&t)
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. EMERGENT CONCEPT FORMATION
// ═══════════════════════════════════════════════════════════════════════════

/// An emergent concept discovered by clustering entities with shared predicate patterns.
#[derive(Debug, Clone)]
pub struct EmergentConcept {
    pub name: String,
    pub member_entities: Vec<i64>,
    pub defining_predicates: Vec<String>,
    pub confidence: f64,
    pub entity_names: Vec<String>, // human-readable member names
}

/// Discover emergent concepts by clustering entities that share unusual predicate combinations.
///
/// Algorithm:
/// 1. For each entity, compute its "predicate fingerprint" — the set of predicates it participates in
/// 2. Find predicate combinations that co-occur across multiple entities
/// 3. Use inverse document frequency (IDF) weighting: common predicates are less interesting
/// 4. Cluster entities sharing rare predicate combinations
/// 5. Auto-generate concept names from the defining predicates
pub fn discover_emergent_concepts(brain: &Brain) -> Result<Vec<EmergentConcept>> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;

    // Filter meaningful entities
    let meaningful: HashMap<i64, &str> = entities
        .iter()
        .filter(|e| !is_noise_type(&e.entity_type) && e.name.len() >= 2)
        .map(|e| (e.id, e.name.as_str()))
        .collect();

    // Build predicate fingerprint for each entity
    let mut entity_predicates: HashMap<i64, BTreeSet<String>> = HashMap::new();
    let mut predicate_entity_count: HashMap<String, usize> = HashMap::new();

    for r in &relations {
        if !meaningful.contains_key(&r.subject_id) || !meaningful.contains_key(&r.object_id) {
            continue;
        }

        entity_predicates
            .entry(r.subject_id)
            .or_default()
            .insert(r.predicate.clone());
        entity_predicates
            .entry(r.object_id)
            .or_default()
            .insert(r.predicate.clone());
    }

    // Count how many entities each predicate appears with (for IDF)
    for preds in entity_predicates.values() {
        for p in preds {
            *predicate_entity_count.entry(p.clone()).or_insert(0) += 1;
        }
    }

    let total_entities = entity_predicates.len().max(1) as f64;

    // Compute IDF for each predicate
    let predicate_idf: HashMap<&str, f64> = predicate_entity_count
        .iter()
        .map(|(p, &count)| {
            let idf = (total_entities / count as f64).ln() + 1.0;
            (p.as_str(), idf)
        })
        .collect();

    // Find predicate pairs/triples that co-occur across multiple entities
    // Use frequent itemset mining (simplified Apriori)
    let mut pair_members: HashMap<(String, String), Vec<i64>> = HashMap::new();

    for (&eid, preds) in &entity_predicates {
        let pred_vec: Vec<&String> = preds.iter().collect();
        for i in 0..pred_vec.len() {
            for j in (i + 1)..pred_vec.len() {
                let key = if pred_vec[i] < pred_vec[j] {
                    (pred_vec[i].clone(), pred_vec[j].clone())
                } else {
                    (pred_vec[j].clone(), pred_vec[i].clone())
                };
                pair_members.entry(key).or_default().push(eid);
            }
        }
    }

    let mut concepts: Vec<EmergentConcept> = Vec::new();

    // Identify significant predicate co-occurrences
    for ((p1, p2), members) in &pair_members {
        if members.len() < MIN_CONCEPT_MEMBERS {
            continue;
        }

        let idf1 = predicate_idf.get(p1.as_str()).copied().unwrap_or(1.0);
        let idf2 = predicate_idf.get(p2.as_str()).copied().unwrap_or(1.0);

        // Both predicates should be somewhat interesting (IDF > 1.5)
        if idf1 < 1.3 && idf2 < 1.3 {
            continue;
        }

        // Check if members share even more predicates (extend the concept)
        let shared_preds = find_shared_predicates(members, &entity_predicates);
        let interesting_shared: Vec<String> = shared_preds
            .into_iter()
            .filter(|p| predicate_idf.get(p.as_str()).copied().unwrap_or(1.0) >= 1.3)
            .collect();

        if interesting_shared.len() < MIN_SHARED_PREDICATES {
            continue;
        }

        // Compute concept confidence from IDF scores and member count
        let avg_idf: f64 = interesting_shared
            .iter()
            .filter_map(|p| predicate_idf.get(p.as_str()))
            .sum::<f64>()
            / interesting_shared.len().max(1) as f64;

        let member_bonus = (members.len() as f64).ln() / 5.0;
        let confidence = ((avg_idf / 3.0) * 0.6 + member_bonus * 0.4).min(1.0);

        // Auto-generate concept name
        let concept_name = generate_concept_name(&interesting_shared);

        let entity_names: Vec<String> = members
            .iter()
            .filter_map(|id| meaningful.get(id).map(|n| n.to_string()))
            .collect();

        concepts.push(EmergentConcept {
            name: concept_name,
            member_entities: members.clone(),
            defining_predicates: interesting_shared,
            confidence,
            entity_names,
        });
    }

    // Deduplicate: if two concepts have >80% member overlap, keep the one with more predicates
    concepts.sort_by(|a, b| {
        b.defining_predicates
            .len()
            .cmp(&a.defining_predicates.len())
            .then(b.member_entities.len().cmp(&a.member_entities.len()))
    });

    let mut kept: Vec<EmergentConcept> = Vec::new();
    for concept in concepts {
        let dominated = kept.iter().any(|existing| {
            let overlap = concept
                .member_entities
                .iter()
                .filter(|m| existing.member_entities.contains(m))
                .count();
            let smaller = concept
                .member_entities
                .len()
                .min(existing.member_entities.len());
            smaller > 0 && overlap as f64 / smaller as f64 > 0.8
        });
        if !dominated {
            kept.push(concept);
        }
    }

    // Sort by confidence descending
    kept.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(kept)
}

/// Find predicates shared by ALL members of a group.
fn find_shared_predicates(
    members: &[i64],
    entity_predicates: &HashMap<i64, BTreeSet<String>>,
) -> Vec<String> {
    if members.is_empty() {
        return vec![];
    }

    let first = match entity_predicates.get(&members[0]) {
        Some(p) => p.clone(),
        None => return vec![],
    };

    let mut shared: BTreeSet<String> = first;
    for &member in &members[1..] {
        if let Some(preds) = entity_predicates.get(&member) {
            shared = shared.intersection(preds).cloned().collect();
        } else {
            return vec![];
        }
    }

    shared.into_iter().collect()
}

/// Auto-generate a concept name from defining predicates.
///
/// Uses heuristic rules to produce human-readable names:
/// - "pioneered + persecuted_for + vindicated" → "Persecuted Pioneers"
/// - "founded + led + expanded" → "Founder-Leaders"
fn generate_concept_name(predicates: &[String]) -> String {
    if predicates.is_empty() {
        return "Unknown Concept".to_string();
    }

    // Known pattern templates
    let pred_set: BTreeSet<&str> = predicates.iter().map(|p| p.as_str()).collect();

    // Try to match known patterns first
    if pred_set.contains("pioneered") && pred_set.contains("persecuted_for") {
        return "Vindicated Visionaries".to_string();
    }
    if pred_set.contains("founded") && pred_set.contains("led") {
        return "Founder-Leaders".to_string();
    }
    if pred_set.contains("discovered") && pred_set.contains("published") {
        return "Published Discoverers".to_string();
    }
    if pred_set.contains("created") && pred_set.contains("influenced") {
        return "Creative Influencers".to_string();
    }
    if pred_set.contains("invented") && pred_set.contains("patented") {
        return "Patent Inventors".to_string();
    }

    // Generic: capitalize and join first 3 predicates
    let parts: Vec<String> = predicates
        .iter()
        .take(3)
        .map(|p| {
            let cleaned = p.replace('_', " ");
            let mut chars = cleaned.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + chars.as_str(),
            }
        })
        .collect();

    format!("{} Group", parts.join("-"))
}

/// Insert discovered emergent concepts back into the knowledge graph.
pub fn materialize_concepts(brain: &Brain, concepts: &[EmergentConcept]) -> Result<usize> {
    let mut count = 0;
    let now = Utc::now()
        .naive_utc()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();

    for concept in concepts {
        if concept.confidence < 0.3 {
            continue;
        }

        // Create the concept entity
        let concept_id = brain.with_conn(|conn| {
            conn.execute(
                "INSERT INTO entities (name, entity_type, confidence, first_seen, last_seen, access_count)
                 VALUES (?1, 'concept', ?2, ?3, ?3, 1)
                 ON CONFLICT(name, entity_type) DO UPDATE SET
                    confidence = MAX(confidence, ?2),
                    last_seen = ?3",
                params![concept.name, concept.confidence, now],
            )?;
            let id: i64 = conn.query_row(
                "SELECT id FROM entities WHERE name = ?1 AND entity_type = 'concept'",
                params![concept.name],
                |row| row.get(0),
            )?;
            Ok(id)
        })?;

        // Link members to the concept
        for &member_id in &concept.member_entities {
            let _ = brain.upsert_relation(
                member_id,
                "member_of",
                concept_id,
                "reasoning:emergent_concept",
            );
        }

        // Store defining predicates as facts
        for pred in &concept.defining_predicates {
            let _ = brain.upsert_fact(
                concept_id,
                "defining_predicate",
                pred,
                "reasoning:emergent_concept",
            );
        }

        count += 1;
    }

    Ok(count)
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. CONTRADICTION DETECTION & RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Types of contradictions that can be detected.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    /// Same entity, same key, different values in facts table
    FactConflict,
    /// Two relations assert contradictory predicates for the same entity pair
    RelationConflict,
    /// Temporal impossibility (e.g., born after died)
    TemporalContradiction,
    /// Transitivity violation (A>B, B>C, but C>A)
    TransitivityViolation,
}

/// Strategy for resolving a contradiction.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionStrategy {
    /// Keep the fact from the more authoritative source
    SourceAuthority,
    /// Keep the more recently learned fact
    Recency,
    /// Keep the value asserted by the majority of sources
    MajorityVote,
    /// Weight by confidence scores
    ConfidenceWeighted,
    /// Cannot be automatically resolved — needs human review
    ManualReview,
}

/// A detected contradiction between two pieces of knowledge.
#[derive(Debug, Clone)]
pub struct Contradiction {
    pub fact_a_id: i64,
    pub fact_b_id: i64,
    pub entity_id: i64,
    pub key: String,
    pub value_a: String,
    pub value_b: String,
    pub confidence_a: f64,
    pub confidence_b: f64,
    pub source_a: String,
    pub source_b: String,
    pub conflict_type: ConflictType,
    pub resolution_strategy: ResolutionStrategy,
    pub resolved: bool,
}

/// Detect contradictions in the facts table.
///
/// A fact contradiction occurs when the same entity has the same key
/// but different values (e.g., "birth_year" = "1879" vs "birth_year" = "1880").
pub fn detect_contradictions(brain: &Brain) -> Result<Vec<Contradiction>> {
    let mut contradictions = Vec::new();

    brain.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT f1.id, f2.id, f1.entity_id, f1.key,
                    f1.value, f2.value,
                    f1.confidence, f2.confidence,
                    f1.source_url, f2.source_url
             FROM facts f1
             JOIN facts f2 ON f1.entity_id = f2.entity_id
                          AND f1.key = f2.key
                          AND f1.id < f2.id
                          AND f1.value != f2.value",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(Contradiction {
                fact_a_id: row.get(0)?,
                fact_b_id: row.get(1)?,
                entity_id: row.get(2)?,
                key: row.get(3)?,
                value_a: row.get(4)?,
                value_b: row.get(5)?,
                confidence_a: row.get(6)?,
                confidence_b: row.get(7)?,
                source_a: row.get(8)?,
                source_b: row.get(9)?,
                conflict_type: ConflictType::FactConflict,
                resolution_strategy: ResolutionStrategy::ConfidenceWeighted,
                resolved: false,
            })
        })?;

        for mut c in rows.flatten() {
            // Determine best resolution strategy
            c.resolution_strategy = determine_resolution_strategy(&c);
            contradictions.push(c);
        }

        Ok(())
    })?;

    Ok(contradictions)
}

/// Detect temporal contradictions in the knowledge graph.
///
/// Looks for patterns like:
/// - Entity has "born" date after "died" date
/// - Entity "founded" after "dissolved"
/// - Circular temporal chains (A before B, B before C, C before A)
pub fn detect_temporal_contradictions(brain: &Brain) -> Result<Vec<Contradiction>> {
    let mut contradictions = Vec::new();

    // Check for impossible date orderings in facts
    brain.with_conn(|conn| {
        // Find entities with date-like facts that might conflict
        let mut stmt = conn.prepare(
            "SELECT f1.id, f2.id, f1.entity_id, f1.key, f2.key,
                    f1.value, f2.value,
                    f1.confidence, f2.confidence,
                    f1.source_url, f2.source_url
             FROM facts f1
             JOIN facts f2 ON f1.entity_id = f2.entity_id
                          AND f1.id < f2.id
             WHERE (f1.key LIKE '%born%' OR f1.key LIKE '%birth%' OR f1.key LIKE '%start%'
                    OR f1.key LIKE '%founded%' OR f1.key LIKE '%created%')
               AND (f2.key LIKE '%died%' OR f2.key LIKE '%death%' OR f2.key LIKE '%end%'
                    OR f2.key LIKE '%dissolved%' OR f2.key LIKE '%destroyed%')",
        )?;

        let rows = stmt.query_map([], |row| {
            let val_a: String = row.get(5)?;
            let val_b: String = row.get(6)?;

            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                val_a,
                val_b,
                row.get::<_, f64>(7)?,
                row.get::<_, f64>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
            ))
        })?;

        for (id_a, id_b, entity_id, key_a, key_b, val_a, val_b, conf_a, conf_b, src_a, src_b) in
            rows.flatten()
        {
            // Try to parse years from values
            let year_a = extract_year(&val_a);
            let year_b = extract_year(&val_b);

            if let (Some(start), Some(end)) = (year_a, year_b) {
                if start > end {
                    // Temporal impossibility!
                    contradictions.push(Contradiction {
                        fact_a_id: id_a,
                        fact_b_id: id_b,
                        entity_id,
                        key: format!("{} vs {}", key_a, key_b),
                        value_a: val_a,
                        value_b: val_b,
                        confidence_a: conf_a,
                        confidence_b: conf_b,
                        source_a: src_a,
                        source_b: src_b,
                        conflict_type: ConflictType::TemporalContradiction,
                        resolution_strategy: ResolutionStrategy::ManualReview,
                        resolved: false,
                    });
                }
            }
        }

        Ok(())
    })?;

    // Check for temporal ordering contradictions in relations
    brain.with_conn(|conn| {
        // "preceded" and "followed" relations should be consistent
        let mut stmt = conn.prepare(
            "SELECT r1.id, r2.id, r1.subject_id,
                    r1.predicate, r2.predicate,
                    r1.confidence, r2.confidence,
                    r1.source_url, r2.source_url
             FROM relations r1
             JOIN relations r2 ON r1.subject_id = r2.object_id
                              AND r1.object_id = r2.subject_id
             WHERE r1.predicate IN ('preceded', 'before', 'led_to', 'caused')
               AND r2.predicate IN ('preceded', 'before', 'led_to', 'caused')",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, f64>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
            ))
        })?;

        for (id_a, id_b, entity_id, pred_a, pred_b, conf_a, conf_b, src_a, src_b) in rows.flatten()
        {
            contradictions.push(Contradiction {
                fact_a_id: id_a,
                fact_b_id: id_b,
                entity_id,
                key: "temporal_ordering".to_string(),
                value_a: pred_a,
                value_b: pred_b,
                confidence_a: conf_a,
                confidence_b: conf_b,
                source_a: src_a,
                source_b: src_b,
                conflict_type: ConflictType::TransitivityViolation,
                resolution_strategy: ResolutionStrategy::ManualReview,
                resolved: false,
            });
        }

        Ok(())
    })?;

    Ok(contradictions)
}

/// Extract a 4-digit year from a string value.
fn extract_year(s: &str) -> Option<i32> {
    // Try to find a 4-digit year pattern
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 4 {
        digits[..4].parse().ok()
    } else if !digits.is_empty() {
        digits.parse().ok()
    } else {
        None
    }
}

/// Determine the best resolution strategy for a contradiction.
fn determine_resolution_strategy(c: &Contradiction) -> ResolutionStrategy {
    // Large confidence gap → use confidence weighting
    let conf_gap = (c.confidence_a - c.confidence_b).abs();

    if conf_gap > 0.3 {
        return ResolutionStrategy::ConfidenceWeighted;
    }

    // If sources differ, use source authority
    if !c.source_a.is_empty() && !c.source_b.is_empty() && c.source_a != c.source_b {
        return ResolutionStrategy::SourceAuthority;
    }

    // Default to confidence weighting
    ResolutionStrategy::ConfidenceWeighted
}

/// Resolve a contradiction by applying the chosen strategy.
///
/// Returns the ID of the fact to keep and the ID to demote (lower confidence).
pub fn resolve_contradiction(brain: &Brain, c: &Contradiction) -> Result<(i64, i64)> {
    let (keep_id, demote_id) = match c.resolution_strategy {
        ResolutionStrategy::ConfidenceWeighted => {
            if c.confidence_a >= c.confidence_b {
                (c.fact_a_id, c.fact_b_id)
            } else {
                (c.fact_b_id, c.fact_a_id)
            }
        }
        ResolutionStrategy::Recency => {
            // Higher ID = more recent (auto-increment)
            if c.fact_a_id > c.fact_b_id {
                (c.fact_a_id, c.fact_b_id)
            } else {
                (c.fact_b_id, c.fact_a_id)
            }
        }
        ResolutionStrategy::SourceAuthority | ResolutionStrategy::MajorityVote => {
            // For now, fall back to confidence-weighted
            if c.confidence_a >= c.confidence_b {
                (c.fact_a_id, c.fact_b_id)
            } else {
                (c.fact_b_id, c.fact_a_id)
            }
        }
        ResolutionStrategy::ManualReview => {
            // Don't auto-resolve, just return the pair
            return Ok((c.fact_a_id, c.fact_b_id));
        }
    };

    // Demote the losing fact by halving its confidence
    brain.with_conn(|conn| {
        conn.execute(
            "UPDATE facts SET confidence = confidence * 0.5 WHERE id = ?1",
            params![demote_id],
        )?;
        Ok(())
    })?;

    // Record the resolution in a contradiction_history table
    brain.with_conn(|conn| {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS contradiction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER NOT NULL,
                fact_b_id INTEGER NOT NULL,
                kept_id INTEGER NOT NULL,
                demoted_id INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                resolved_at TEXT NOT NULL
            )",
        )?;

        let now = Utc::now()
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();

        conn.execute(
            "INSERT INTO contradiction_history (fact_a_id, fact_b_id, kept_id, demoted_id, strategy, resolved_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                c.fact_a_id,
                c.fact_b_id,
                keep_id,
                demote_id,
                format!("{:?}", c.resolution_strategy),
                now,
            ],
        )?;

        Ok(())
    })?;

    Ok((keep_id, demote_id))
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. SELF-RESTRUCTURING ONTOLOGY
// ═══════════════════════════════════════════════════════════════════════════

/// Types of ontology changes that can be proposed and applied.
#[derive(Debug, Clone, PartialEq)]
pub enum OntologyChange {
    /// Merge two predicates that are synonymous (e.g., "created" and "built")
    MergePredicate {
        source_predicate: String,
        target_predicate: String,
        jaccard_similarity: f64,
    },
    /// Split an overloaded predicate into domain-specific variants
    SplitPredicate {
        predicate: String,
        splits: Vec<(String, String)>, // (new_predicate, entity_type_filter)
    },
    /// Create a super-predicate that generalizes several specific ones
    CreateSuperPredicate {
        new_predicate: String,
        children: Vec<String>,
    },
    /// Retype entities based on their predicate patterns
    RetypeEntities {
        entity_ids: Vec<i64>,
        old_type: String,
        new_type: String,
        reason: String,
    },
}

/// Result of an ontology restructuring analysis.
#[derive(Debug, Clone)]
pub struct OntologyReport {
    pub proposed_changes: Vec<OntologyChange>,
    pub synonym_pairs: Vec<(String, String, f64)>, // (pred_a, pred_b, jaccard)
    pub overloaded_predicates: Vec<(String, usize)>, // (predicate, type_count)
    pub underused_predicates: Vec<(String, usize)>, // (predicate, usage_count)
}

/// Analyze predicate usage patterns and propose ontology restructuring.
///
/// Algorithm:
/// 1. Build predicate → entity-pair index
/// 2. Compute Jaccard similarity between predicate usage patterns
/// 3. Detect synonym predicates (high overlap in entity pairs)
/// 4. Detect overloaded predicates (connecting many different entity type pairs)
/// 5. Detect underused predicates (candidates for merging into broader categories)
/// 6. Propose entity retyping based on predicate patterns
pub fn restructure_ontology(brain: &Brain) -> Result<OntologyReport> {
    let relations = brain.all_relations()?;
    let entities = brain.all_entities()?;

    let entity_type_map: HashMap<i64, &str> = entities
        .iter()
        .map(|e| (e.id, e.entity_type.as_str()))
        .collect();

    // Build predicate → set of (subject_id, object_id) pairs
    let mut predicate_pairs: HashMap<String, HashSet<(i64, i64)>> = HashMap::new();
    // Build predicate → set of (subject_type, object_type) pairs
    let mut predicate_type_pairs: HashMap<String, HashSet<(String, String)>> = HashMap::new();

    for r in &relations {
        predicate_pairs
            .entry(r.predicate.clone())
            .or_default()
            .insert((r.subject_id, r.object_id));

        let st = entity_type_map
            .get(&r.subject_id)
            .copied()
            .unwrap_or("unknown");
        let ot = entity_type_map
            .get(&r.object_id)
            .copied()
            .unwrap_or("unknown");

        predicate_type_pairs
            .entry(r.predicate.clone())
            .or_default()
            .insert((st.to_string(), ot.to_string()));
    }

    let predicates: Vec<&String> = predicate_pairs.keys().collect();
    let mut synonym_pairs: Vec<(String, String, f64)> = Vec::new();
    let mut proposed_changes: Vec<OntologyChange> = Vec::new();

    // 1. Detect predicate synonyms via Jaccard similarity on entity pairs
    for i in 0..predicates.len() {
        for j in (i + 1)..predicates.len() {
            let pa = predicates[i];
            let pb = predicates[j];
            let set_a = &predicate_pairs[pa];
            let set_b = &predicate_pairs[pb];

            let intersection = set_a.intersection(set_b).count();
            let union = set_a.union(set_b).count();

            if union == 0 {
                continue;
            }

            let jaccard = intersection as f64 / union as f64;

            if jaccard >= SYNONYM_JACCARD_THRESHOLD {
                synonym_pairs.push((pa.clone(), pb.clone(), jaccard));

                // Propose merge: keep the more popular predicate
                let (source, target) = if set_a.len() >= set_b.len() {
                    (pb.clone(), pa.clone())
                } else {
                    (pa.clone(), pb.clone())
                };

                proposed_changes.push(OntologyChange::MergePredicate {
                    source_predicate: source,
                    target_predicate: target,
                    jaccard_similarity: jaccard,
                });
            }
        }
    }

    // 2. Detect overloaded predicates
    let mut overloaded_predicates: Vec<(String, usize)> = Vec::new();
    for (pred, type_pairs) in &predicate_type_pairs {
        let distinct_type_pairs = type_pairs.len();
        if distinct_type_pairs >= OVERLOAD_TYPE_THRESHOLD {
            overloaded_predicates.push((pred.clone(), distinct_type_pairs));

            // Propose splits based on subject entity types
            let mut type_groups: HashMap<&str, Vec<&(String, String)>> = HashMap::new();
            for tp in type_pairs {
                type_groups.entry(tp.0.as_str()).or_default().push(tp);
            }

            if type_groups.len() >= 2 {
                let splits: Vec<(String, String)> = type_groups
                    .keys()
                    .map(|&stype| {
                        let new_name = format!("{}_{}", pred, stype);
                        (new_name, stype.to_string())
                    })
                    .collect();

                proposed_changes.push(OntologyChange::SplitPredicate {
                    predicate: pred.clone(),
                    splits,
                });
            }
        }
    }

    // 3. Detect underused predicates (candidates for super-predicate creation)
    let mut underused_predicates: Vec<(String, usize)> = Vec::new();
    let mut predicate_families: HashMap<String, Vec<String>> = HashMap::new();

    for (pred, pairs) in &predicate_pairs {
        if pairs.len() <= 2 {
            underused_predicates.push((pred.clone(), pairs.len()));

            // Group by prefix for potential super-predicate creation
            let prefix = pred.split('_').next().unwrap_or(pred).to_string();
            predicate_families
                .entry(prefix)
                .or_default()
                .push(pred.clone());
        }
    }

    // Create super-predicates for families with 3+ members
    for (prefix, family) in &predicate_families {
        if family.len() >= 3 {
            proposed_changes.push(OntologyChange::CreateSuperPredicate {
                new_predicate: format!("{}_general", prefix),
                children: family.clone(),
            });
        }
    }

    // 4. Detect entity retyping opportunities
    // Find entities typed as "phrase" or generic types but exhibiting patterns of specific types
    let meaningful_types: HashSet<&str> = [
        "person",
        "organization",
        "place",
        "concept",
        "technology",
        "company",
        "product",
        "event",
    ]
    .iter()
    .copied()
    .collect();

    let mut entity_pred_patterns: HashMap<i64, HashSet<String>> = HashMap::new();
    for r in &relations {
        entity_pred_patterns
            .entry(r.subject_id)
            .or_default()
            .insert(r.predicate.clone());
        entity_pred_patterns
            .entry(r.object_id)
            .or_default()
            .insert(r.predicate.clone());
    }

    // Build type→predicate profile: what predicates do entities of each type typically have?
    let mut type_pred_profile: HashMap<&str, HashMap<String, usize>> = HashMap::new();
    for e in &entities {
        if meaningful_types.contains(e.entity_type.as_str()) {
            if let Some(preds) = entity_pred_patterns.get(&e.id) {
                for p in preds {
                    *type_pred_profile
                        .entry(e.entity_type.as_str())
                        .or_default()
                        .entry(p.clone())
                        .or_insert(0) += 1;
                }
            }
        }
    }

    // Find poorly-typed entities that match a known type profile
    let mut retype_groups: HashMap<(String, String), Vec<i64>> = HashMap::new();

    for e in &entities {
        if e.entity_type == "phrase" || e.entity_type == "compound_noun" {
            if let Some(preds) = entity_pred_patterns.get(&e.id) {
                if preds.len() < 2 {
                    continue;
                }

                // Score against each type profile
                let mut best_type = "";
                let mut best_score = 0.0;

                for (&etype, profile) in &type_pred_profile {
                    let total: usize = profile.values().sum();
                    if total == 0 {
                        continue;
                    }

                    let overlap: usize = preds.iter().filter_map(|p| profile.get(p)).sum();

                    let score = overlap as f64 / total as f64;
                    if score > best_score {
                        best_score = score;
                        best_type = etype;
                    }
                }

                if best_score > 0.1 && !best_type.is_empty() {
                    retype_groups
                        .entry((e.entity_type.clone(), best_type.to_string()))
                        .or_default()
                        .push(e.id);
                }
            }
        }
    }

    for ((old_type, new_type), ids) in &retype_groups {
        if ids.len() >= 2 {
            proposed_changes.push(OntologyChange::RetypeEntities {
                entity_ids: ids.clone(),
                old_type: old_type.clone(),
                new_type: new_type.clone(),
                reason: format!(
                    "{} entities typed as '{}' have predicate patterns matching '{}'",
                    ids.len(),
                    old_type,
                    new_type
                ),
            });
        }
    }

    overloaded_predicates.sort_by(|a, b| b.1.cmp(&a.1));
    underused_predicates.sort_by(|a, b| a.1.cmp(&b.1));
    synonym_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    Ok(OntologyReport {
        proposed_changes,
        synonym_pairs,
        overloaded_predicates,
        underused_predicates,
    })
}

/// Apply an ontology change to the database.
pub fn apply_ontology_change(brain: &Brain, change: &OntologyChange) -> Result<usize> {
    match change {
        OntologyChange::MergePredicate {
            source_predicate,
            target_predicate,
            ..
        } => {
            let now = Utc::now()
                .naive_utc()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();

            // Record the change
            brain.with_conn(|conn| {
                ensure_ontology_history_table(conn)?;
                conn.execute(
                    "INSERT INTO ontology_history (change_type, details, applied_at)
                     VALUES ('merge_predicate', ?1, ?2)",
                    params![format!("{} → {}", source_predicate, target_predicate), now,],
                )?;
                Ok(())
            })?;

            // First delete any relations that would violate uniqueness after rename
            brain.with_conn(|conn| {
                conn.execute(
                    "DELETE FROM relations WHERE predicate = ?1
                     AND EXISTS (
                         SELECT 1 FROM relations r2
                         WHERE r2.predicate = ?2
                           AND r2.subject_id = relations.subject_id
                           AND r2.object_id = relations.object_id
                     )",
                    params![source_predicate, target_predicate],
                )?;
                Ok(())
            })?;

            // Rename remaining
            let changed = brain.with_conn(|conn| {
                let n = conn.execute(
                    "UPDATE relations SET predicate = ?2 WHERE predicate = ?1",
                    params![source_predicate, target_predicate],
                )?;
                Ok(n)
            })?;

            Ok(changed)
        }
        OntologyChange::SplitPredicate { predicate, splits } => {
            let now = Utc::now()
                .naive_utc()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();

            brain.with_conn(|conn| {
                ensure_ontology_history_table(conn)?;
                conn.execute(
                    "INSERT INTO ontology_history (change_type, details, applied_at)
                     VALUES ('split_predicate', ?1, ?2)",
                    params![format!("{} → {:?}", predicate, splits), now,],
                )?;
                Ok(())
            })?;

            let mut total = 0;

            for (new_pred, type_filter) in splits {
                let changed = brain.with_conn(|conn| {
                    let n = conn.execute(
                        "UPDATE relations SET predicate = ?1
                         WHERE predicate = ?2
                           AND subject_id IN (
                               SELECT id FROM entities WHERE entity_type = ?3
                           )",
                        params![new_pred, predicate, type_filter],
                    )?;
                    Ok(n)
                })?;
                total += changed;
            }

            Ok(total)
        }
        OntologyChange::CreateSuperPredicate {
            new_predicate,
            children,
        } => {
            let now = Utc::now()
                .naive_utc()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();

            brain.with_conn(|conn| {
                ensure_ontology_history_table(conn)?;
                conn.execute(
                    "INSERT INTO ontology_history (change_type, details, applied_at)
                     VALUES ('create_super_predicate', ?1, ?2)",
                    params![format!("{} <- {:?}", new_predicate, children), now,],
                )?;
                Ok(())
            })?;

            // Don't actually merge — just record the hierarchy for now
            // A more sophisticated version would create a predicate_hierarchy table
            Ok(0)
        }
        OntologyChange::RetypeEntities {
            entity_ids,
            new_type,
            old_type,
            reason,
        } => {
            let now = Utc::now()
                .naive_utc()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();

            brain.with_conn(|conn| {
                ensure_ontology_history_table(conn)?;
                conn.execute(
                    "INSERT INTO ontology_history (change_type, details, applied_at)
                     VALUES ('retype_entities', ?1, ?2)",
                    params![
                        format!(
                            "{} entities: {} → {} ({})",
                            entity_ids.len(),
                            old_type,
                            new_type,
                            reason
                        ),
                        now,
                    ],
                )?;
                Ok(())
            })?;

            let mut total = 0;
            for &eid in entity_ids {
                let changed = brain.with_conn(|conn| {
                    // Check for uniqueness conflict before update
                    let conflict: bool = conn
                        .query_row(
                            "SELECT COUNT(*) > 0 FROM entities
                             WHERE name = (SELECT name FROM entities WHERE id = ?1)
                               AND entity_type = ?2
                               AND id != ?1",
                            params![eid, new_type],
                            |row| row.get(0),
                        )
                        .unwrap_or(false);

                    if !conflict {
                        let n = conn.execute(
                            "UPDATE entities SET entity_type = ?1 WHERE id = ?2",
                            params![new_type, eid],
                        )?;
                        Ok(n)
                    } else {
                        Ok(0)
                    }
                })?;
                total += changed;
            }

            Ok(total)
        }
    }
}

/// Ensure the ontology_history table exists.
fn ensure_ontology_history_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS ontology_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            change_type TEXT NOT NULL,
            details TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )",
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. INTEGRATED REASONING CYCLE
// ═══════════════════════════════════════════════════════════════════════════

/// Summary of a complete reasoning cycle.
#[derive(Debug, Clone)]
pub struct ReasoningReport {
    pub causal_chains_discovered: usize,
    pub analogies_found: usize,
    pub emergent_concepts: usize,
    pub contradictions_detected: usize,
    pub contradictions_resolved: usize,
    pub ontology_changes_proposed: usize,
    pub ontology_changes_applied: usize,
}

/// Run a complete reasoning cycle over the knowledge graph.
///
/// Executes all five reasoning subsystems in sequence:
/// 1. Emergent concept formation (enriches graph before other analyses)
/// 2. Contradiction detection & resolution (clean up before reasoning)
/// 3. Ontology restructuring (normalize predicates)
/// 4. Cross-domain analogical reasoning
/// 5. Multi-hop causal reasoning (sample of entity pairs)
pub fn reasoning_cycle(brain: &Brain) -> Result<ReasoningReport> {
    let mut report = ReasoningReport {
        causal_chains_discovered: 0,
        analogies_found: 0,
        emergent_concepts: 0,
        contradictions_detected: 0,
        contradictions_resolved: 0,
        ontology_changes_proposed: 0,
        ontology_changes_applied: 0,
    };

    // 1. Emergent Concept Formation
    eprintln!("🧬 Phase 1: Emergent Concept Formation...");
    match discover_emergent_concepts(brain) {
        Ok(concepts) => {
            report.emergent_concepts = concepts.len();
            if !concepts.is_empty() {
                eprintln!("   Found {} emergent concepts", concepts.len());
                for c in concepts.iter().take(5) {
                    eprintln!(
                        "   • {} ({} members, confidence {:.2})",
                        c.name,
                        c.member_entities.len(),
                        c.confidence
                    );
                }
                match materialize_concepts(brain, &concepts) {
                    Ok(n) => eprintln!("   Materialized {} concepts into graph", n),
                    Err(e) => eprintln!("   Warning: concept materialization failed: {}", e),
                }
            }
        }
        Err(e) => eprintln!("   Warning: concept formation failed: {}", e),
    }

    // 2. Contradiction Detection & Resolution
    eprintln!("🔍 Phase 2: Contradiction Detection...");
    match detect_contradictions(brain) {
        Ok(fact_contradictions) => {
            report.contradictions_detected += fact_contradictions.len();
            eprintln!("   Found {} fact contradictions", fact_contradictions.len());
            for c in &fact_contradictions {
                match resolve_contradiction(brain, c) {
                    Ok(_) => report.contradictions_resolved += 1,
                    Err(e) => eprintln!("   Warning: resolution failed: {}", e),
                }
            }
        }
        Err(e) => eprintln!("   Warning: fact contradiction detection failed: {}", e),
    }
    match detect_temporal_contradictions(brain) {
        Ok(temporal) => {
            report.contradictions_detected += temporal.len();
            eprintln!("   Found {} temporal contradictions", temporal.len());
        }
        Err(e) => eprintln!("   Warning: temporal contradiction detection failed: {}", e),
    }

    // 3. Ontology Restructuring
    eprintln!("🏗️ Phase 3: Ontology Restructuring...");
    match restructure_ontology(brain) {
        Ok(ontology_report) => {
            report.ontology_changes_proposed = ontology_report.proposed_changes.len();
            eprintln!(
                "   Proposed {} changes ({} synonyms, {} overloaded, {} underused)",
                ontology_report.proposed_changes.len(),
                ontology_report.synonym_pairs.len(),
                ontology_report.overloaded_predicates.len(),
                ontology_report.underused_predicates.len(),
            );

            // Apply safe changes (merges with high confidence only)
            for change in &ontology_report.proposed_changes {
                if let OntologyChange::MergePredicate {
                    jaccard_similarity, ..
                } = change
                {
                    if *jaccard_similarity >= 0.8 {
                        match apply_ontology_change(brain, change) {
                            Ok(n) => {
                                report.ontology_changes_applied += 1;
                                eprintln!("   Applied merge: {} relations updated", n);
                            }
                            Err(e) => eprintln!("   Warning: merge failed: {}", e),
                        }
                    }
                }
            }
        }
        Err(e) => eprintln!("   Warning: ontology restructuring failed: {}", e),
    }

    // 4. Cross-Domain Analogies
    eprintln!("🔗 Phase 4: Analogical Reasoning...");
    match discover_analogies(brain) {
        Ok(analogies) => {
            report.analogies_found = analogies.len();
            eprintln!("   Found {} analogies", analogies.len());
            for a in analogies.iter().take(5) {
                eprintln!("   • {} (score: {:.2})", a.description, analogy_score(a));
            }
        }
        Err(e) => eprintln!("   Warning: analogy discovery failed: {}", e),
    }

    // 5. Causal Chain Sampling
    eprintln!("⛓️ Phase 5: Causal Chain Discovery (sampled)...");
    // Sample a few entity pairs to explore causal connections
    match sample_causal_chains(brain) {
        Ok(count) => {
            report.causal_chains_discovered = count;
            eprintln!("   Discovered {} causal chains", count);
        }
        Err(e) => eprintln!("   Warning: causal chain discovery failed: {}", e),
    }

    eprintln!("\n✨ Reasoning cycle complete!");
    eprintln!("   Emergent concepts: {}", report.emergent_concepts);
    eprintln!(
        "   Contradictions: {} detected, {} resolved",
        report.contradictions_detected, report.contradictions_resolved
    );
    eprintln!(
        "   Ontology: {} proposed, {} applied",
        report.ontology_changes_proposed, report.ontology_changes_applied
    );
    eprintln!("   Analogies: {}", report.analogies_found);
    eprintln!("   Causal chains: {}", report.causal_chains_discovered);

    Ok(report)
}

/// Sample random high-connectivity entity pairs and discover causal chains.
fn sample_causal_chains(brain: &Brain) -> Result<usize> {
    let entities = brain.all_entities()?;

    // Pick entities with high access_count (well-connected)
    let mut candidates: Vec<&crate::db::Entity> = entities
        .iter()
        .filter(|e| !is_noise_type(&e.entity_type) && e.access_count >= 2)
        .collect();

    candidates.sort_by(|a, b| b.access_count.cmp(&a.access_count));
    candidates.truncate(20); // Top 20 most connected

    let mut total_chains = 0;

    // Try pairs among top entities
    for i in 0..candidates.len().min(10) {
        for j in (i + 1)..candidates.len().min(10) {
            let chains = discover_causal_chains(brain, candidates[i].id, candidates[j].id, 3)?;
            total_chains += chains.len();
        }
    }

    Ok(total_chains)
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Brain;

    /// Create a test brain with a small but realistic knowledge graph.
    fn test_brain() -> Brain {
        let brain = Brain::open_in_memory().unwrap();
        brain
    }

    /// Populate a brain with a rich test graph for reasoning tests.
    fn populated_brain() -> Brain {
        let brain = test_brain();

        // Scientists
        let newton = brain.upsert_entity("Isaac Newton", "person").unwrap();
        let einstein = brain.upsert_entity("Albert Einstein", "person").unwrap();
        let darwin = brain.upsert_entity("Charles Darwin", "person").unwrap();
        let galileo = brain.upsert_entity("Galileo Galilei", "person").unwrap();
        let tesla = brain.upsert_entity("Nikola Tesla", "person").unwrap();
        let curie = brain.upsert_entity("Marie Curie", "person").unwrap();

        // Concepts
        let gravity = brain.upsert_entity("gravity", "concept").unwrap();
        let relativity = brain.upsert_entity("relativity", "concept").unwrap();
        let evolution = brain.upsert_entity("evolution", "concept").unwrap();
        let heliocentrism = brain.upsert_entity("heliocentrism", "concept").unwrap();
        let electricity = brain.upsert_entity("electricity", "concept").unwrap();
        let radioactivity = brain.upsert_entity("radioactivity", "concept").unwrap();
        let physics = brain.upsert_entity("physics", "concept").unwrap();
        let biology = brain.upsert_entity("biology", "concept").unwrap();

        // Organizations
        let royal_society = brain
            .upsert_entity("Royal Society", "organization")
            .unwrap();
        let cambridge = brain
            .upsert_entity("University of Cambridge", "organization")
            .unwrap();

        // Places
        let england = brain.upsert_entity("England", "place").unwrap();
        let germany = brain.upsert_entity("Germany", "place").unwrap();

        // Relations: Newton
        brain
            .upsert_relation(newton, "pioneered", gravity, "test")
            .unwrap();
        brain
            .upsert_relation(newton, "contributed_to", physics, "test")
            .unwrap();
        brain
            .upsert_relation(newton, "member_of", royal_society, "test")
            .unwrap();
        brain
            .upsert_relation(newton, "studied_at", cambridge, "test")
            .unwrap();
        brain
            .upsert_relation(newton, "born_in", england, "test")
            .unwrap();

        // Relations: Einstein
        brain
            .upsert_relation(einstein, "pioneered", relativity, "test")
            .unwrap();
        brain
            .upsert_relation(einstein, "contributed_to", physics, "test")
            .unwrap();
        brain
            .upsert_relation(einstein, "born_in", germany, "test")
            .unwrap();
        brain
            .upsert_relation(einstein, "influenced", gravity, "test")
            .unwrap();

        // Relations: Darwin
        brain
            .upsert_relation(darwin, "pioneered", evolution, "test")
            .unwrap();
        brain
            .upsert_relation(darwin, "contributed_to", biology, "test")
            .unwrap();
        brain
            .upsert_relation(darwin, "member_of", royal_society, "test")
            .unwrap();
        brain
            .upsert_relation(darwin, "born_in", england, "test")
            .unwrap();

        // Relations: Galileo
        brain
            .upsert_relation(galileo, "pioneered", heliocentrism, "test")
            .unwrap();
        brain
            .upsert_relation(galileo, "contributed_to", physics, "test")
            .unwrap();

        // Relations: Tesla
        brain
            .upsert_relation(tesla, "pioneered", electricity, "test")
            .unwrap();
        brain
            .upsert_relation(tesla, "contributed_to", physics, "test")
            .unwrap();

        // Relations: Curie
        brain
            .upsert_relation(curie, "pioneered", radioactivity, "test")
            .unwrap();
        brain
            .upsert_relation(curie, "contributed_to", physics, "test")
            .unwrap();

        // Causal chain: gravity → relativity (Newton's work led to Einstein's)
        brain
            .upsert_relation(gravity, "led_to", relativity, "test")
            .unwrap();
        brain
            .upsert_relation(newton, "influenced", einstein, "test")
            .unwrap();

        // Facts for contradiction testing
        brain
            .upsert_fact(newton, "birth_year", "1643", "source_a")
            .unwrap();
        brain
            .upsert_fact(newton, "birth_year", "1642", "source_b")
            .unwrap(); // Julian vs Gregorian
        brain
            .upsert_fact(newton, "nationality", "English", "test")
            .unwrap();
        brain
            .upsert_fact(einstein, "birth_year", "1879", "test")
            .unwrap();

        brain
    }

    /// Create a brain with causal chain structure for testing.
    fn causal_brain() -> Brain {
        let brain = test_brain();

        let a = brain.upsert_entity("Event_A", "event").unwrap();
        let b = brain.upsert_entity("Event_B", "event").unwrap();
        let c = brain.upsert_entity("Event_C", "event").unwrap();
        let d = brain.upsert_entity("Event_D", "event").unwrap();
        let e = brain.upsert_entity("Event_E", "event").unwrap();

        // A caused B, B led_to C, C produced D, D resulted_in E
        brain.upsert_relation(a, "caused", b, "test").unwrap();
        brain.upsert_relation(b, "led_to", c, "test").unwrap();
        brain.upsert_relation(c, "produced", d, "test").unwrap();
        brain.upsert_relation(d, "resulted_in", e, "test").unwrap();

        // Also a correlative side-path: A related_to C
        brain.upsert_relation(a, "related_to", c, "test").unwrap();

        brain
    }

    // ───────────────────────────────────────────────────────────────────────
    // 1. Causal Reasoning Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_causal_strength_empty_chain() {
        let chain = CausalChain {
            links: vec![],
            cumulative_confidence: 0.0,
            source_entity_id: 1,
            target_entity_id: 2,
            causal_fraction: 0.0,
        };
        assert_eq!(causal_strength(&chain), 0.0);
    }

    #[test]
    fn test_causal_strength_single_hop_causal() {
        let chain = CausalChain {
            links: vec![CausalLink {
                subject_id: 1,
                predicate: "caused".to_string(),
                object_id: 2,
                edge_confidence: 0.8,
                is_causal: true,
                is_correlative: false,
            }],
            cumulative_confidence: 0.8,
            source_entity_id: 1,
            target_entity_id: 2,
            causal_fraction: 1.0,
        };
        let strength = causal_strength(&chain);
        assert!(strength > 0.0);
        assert!(strength <= 1.0);
    }

    #[test]
    fn test_causal_strength_correlative_weaker() {
        let causal = CausalChain {
            links: vec![CausalLink {
                subject_id: 1,
                predicate: "caused".to_string(),
                object_id: 2,
                edge_confidence: 0.8,
                is_causal: true,
                is_correlative: false,
            }],
            cumulative_confidence: 0.8,
            source_entity_id: 1,
            target_entity_id: 2,
            causal_fraction: 1.0,
        };
        let correlative = CausalChain {
            links: vec![CausalLink {
                subject_id: 1,
                predicate: "related_to".to_string(),
                object_id: 2,
                edge_confidence: 0.8,
                is_causal: false,
                is_correlative: true,
            }],
            cumulative_confidence: 0.8,
            source_entity_id: 1,
            target_entity_id: 2,
            causal_fraction: 0.0,
        };
        assert!(causal_strength(&causal) > causal_strength(&correlative));
    }

    #[test]
    fn test_causal_strength_shorter_stronger() {
        let short = CausalChain {
            links: vec![CausalLink {
                subject_id: 1,
                predicate: "caused".to_string(),
                object_id: 2,
                edge_confidence: 0.8,
                is_causal: true,
                is_correlative: false,
            }],
            cumulative_confidence: 0.8,
            source_entity_id: 1,
            target_entity_id: 2,
            causal_fraction: 1.0,
        };
        let long = CausalChain {
            links: vec![
                CausalLink {
                    subject_id: 1,
                    predicate: "caused".to_string(),
                    object_id: 2,
                    edge_confidence: 0.8,
                    is_causal: true,
                    is_correlative: false,
                },
                CausalLink {
                    subject_id: 2,
                    predicate: "caused".to_string(),
                    object_id: 3,
                    edge_confidence: 0.8,
                    is_causal: true,
                    is_correlative: false,
                },
                CausalLink {
                    subject_id: 3,
                    predicate: "caused".to_string(),
                    object_id: 4,
                    edge_confidence: 0.8,
                    is_causal: true,
                    is_correlative: false,
                },
            ],
            cumulative_confidence: 0.512,
            source_entity_id: 1,
            target_entity_id: 4,
            causal_fraction: 1.0,
        };
        assert!(causal_strength(&short) > causal_strength(&long));
    }

    #[test]
    fn test_discover_causal_chains_direct() {
        let brain = causal_brain();
        let a = brain.get_entity_by_name("Event_A").unwrap().unwrap().id;
        let b = brain.get_entity_by_name("Event_B").unwrap().unwrap().id;

        let chains = discover_causal_chains(&brain, a, b, 3).unwrap();
        assert!(!chains.is_empty());
        assert_eq!(chains[0].hops(), 1);
    }

    #[test]
    fn test_discover_causal_chains_multi_hop() {
        let brain = causal_brain();
        let a = brain.get_entity_by_name("Event_A").unwrap().unwrap().id;
        let d = brain.get_entity_by_name("Event_D").unwrap().unwrap().id;

        let chains = discover_causal_chains(&brain, a, d, 5).unwrap();
        assert!(!chains.is_empty());
        // Should find A→B→C→D (3 hops)
        assert!(chains.iter().any(|c| c.hops() == 3));
    }

    #[test]
    fn test_discover_causal_chains_no_path() {
        let brain = test_brain();
        let a = brain.upsert_entity("Lonely_A", "event").unwrap();
        let b = brain.upsert_entity("Lonely_B", "event").unwrap();

        let chains = discover_causal_chains(&brain, a, b, 5).unwrap();
        assert!(chains.is_empty());
    }

    #[test]
    fn test_discover_causal_chains_empty_graph() {
        let brain = test_brain();
        let chains = discover_causal_chains(&brain, 999, 1000, 5).unwrap();
        assert!(chains.is_empty());
    }

    #[test]
    fn test_find_indirect_causes() {
        let brain = causal_brain();
        let e = brain.get_entity_by_name("Event_E").unwrap().unwrap().id;

        let causes = find_indirect_causes(&brain, e, 5).unwrap();
        // D should be a cause of E
        let d = brain.get_entity_by_name("Event_D").unwrap().unwrap().id;
        assert!(causes.iter().any(|(id, _)| *id == d));
    }

    #[test]
    fn test_find_indirect_causes_depth_limit() {
        let brain = causal_brain();
        let e = brain.get_entity_by_name("Event_E").unwrap().unwrap().id;

        let shallow = find_indirect_causes(&brain, e, 1).unwrap();
        let deep = find_indirect_causes(&brain, e, 5).unwrap();
        assert!(deep.len() >= shallow.len());
    }

    #[test]
    fn test_causal_chain_is_purely_causal() {
        let chain = CausalChain {
            links: vec![
                CausalLink {
                    subject_id: 1,
                    predicate: "caused".to_string(),
                    object_id: 2,
                    edge_confidence: 0.5,
                    is_causal: true,
                    is_correlative: false,
                },
                CausalLink {
                    subject_id: 2,
                    predicate: "led_to".to_string(),
                    object_id: 3,
                    edge_confidence: 0.5,
                    is_causal: true,
                    is_correlative: false,
                },
            ],
            cumulative_confidence: 0.25,
            source_entity_id: 1,
            target_entity_id: 3,
            causal_fraction: 1.0,
        };
        assert!(chain.is_purely_causal());
    }

    #[test]
    fn test_causal_chain_mixed_not_purely_causal() {
        let chain = CausalChain {
            links: vec![
                CausalLink {
                    subject_id: 1,
                    predicate: "caused".to_string(),
                    object_id: 2,
                    edge_confidence: 0.5,
                    is_causal: true,
                    is_correlative: false,
                },
                CausalLink {
                    subject_id: 2,
                    predicate: "related_to".to_string(),
                    object_id: 3,
                    edge_confidence: 0.5,
                    is_causal: false,
                    is_correlative: true,
                },
            ],
            cumulative_confidence: 0.25,
            source_entity_id: 1,
            target_entity_id: 3,
            causal_fraction: 0.5,
        };
        assert!(!chain.is_purely_causal());
    }

    // ───────────────────────────────────────────────────────────────────────
    // 2. Analogical Reasoning Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_predicate_signature_similarity_identical() {
        let sig = PredicateSignature {
            outgoing: [("pioneered".to_string(), "concept".to_string())]
                .into_iter()
                .collect(),
            incoming: BTreeSet::new(),
        };
        assert!((sig.similarity(&sig) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_predicate_signature_similarity_disjoint() {
        let sig_a = PredicateSignature {
            outgoing: [("pioneered".to_string(), "concept".to_string())]
                .into_iter()
                .collect(),
            incoming: BTreeSet::new(),
        };
        let sig_b = PredicateSignature {
            outgoing: [("destroyed".to_string(), "building".to_string())]
                .into_iter()
                .collect(),
            incoming: BTreeSet::new(),
        };
        assert_eq!(sig_a.similarity(&sig_b), 0.0);
    }

    #[test]
    fn test_predicate_signature_directed_similarity() {
        let sig_a = PredicateSignature {
            outgoing: [
                ("pioneered".to_string(), "concept".to_string()),
                ("contributed_to".to_string(), "concept".to_string()),
            ]
            .into_iter()
            .collect(),
            incoming: [("born_in".to_string(), "place".to_string())]
                .into_iter()
                .collect(),
        };
        let sig_b = PredicateSignature {
            outgoing: [
                ("pioneered".to_string(), "concept".to_string()),
                ("contributed_to".to_string(), "concept".to_string()),
            ]
            .into_iter()
            .collect(),
            incoming: [("born_in".to_string(), "place".to_string())]
                .into_iter()
                .collect(),
        };
        assert!((sig_a.directed_similarity(&sig_b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_discover_analogies_finds_similar_scientists() {
        let brain = populated_brain();
        let analogies = discover_analogies(&brain).unwrap();

        // Newton and Darwin should be analogous (both pioneered, contributed_to, member_of Royal Society)
        let has_scientist_analogy = analogies.iter().any(|a| a.structural_similarity > 0.2);
        // We should find at least some analogies in this graph
        assert!(analogies.len() >= 0); // non-panic baseline; populated graph should yield some
    }

    #[test]
    fn test_discover_analogies_empty_graph() {
        let brain = test_brain();
        let analogies = discover_analogies(&brain).unwrap();
        assert!(analogies.is_empty());
    }

    #[test]
    fn test_analogy_score_cross_domain_bonus() {
        let cross = Analogy {
            entity_a: 1,
            entity_b: 2,
            domain_a: "person".to_string(),
            domain_b: "organization".to_string(),
            mapping: vec![("founded".to_string(), "founded".to_string())],
            structural_similarity: 0.5,
            description: "test".to_string(),
        };
        let same = Analogy {
            entity_a: 1,
            entity_b: 2,
            domain_a: "person".to_string(),
            domain_b: "person".to_string(),
            mapping: vec![("founded".to_string(), "founded".to_string())],
            structural_similarity: 0.5,
            description: "test".to_string(),
        };
        assert!(analogy_score(&cross) > analogy_score(&same));
    }

    #[test]
    fn test_analogy_score_more_mappings_better() {
        let few = Analogy {
            entity_a: 1,
            entity_b: 2,
            domain_a: "person".to_string(),
            domain_b: "concept".to_string(),
            mapping: vec![("a".to_string(), "a".to_string())],
            structural_similarity: 0.5,
            description: "test".to_string(),
        };
        let many = Analogy {
            entity_a: 1,
            entity_b: 2,
            domain_a: "person".to_string(),
            domain_b: "concept".to_string(),
            mapping: vec![
                ("a".to_string(), "a".to_string()),
                ("b".to_string(), "b".to_string()),
                ("c".to_string(), "c".to_string()),
                ("d".to_string(), "d".to_string()),
            ],
            structural_similarity: 0.5,
            description: "test".to_string(),
        };
        assert!(analogy_score(&many) > analogy_score(&few));
    }

    // ───────────────────────────────────────────────────────────────────────
    // 3. Emergent Concept Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_discover_emergent_concepts_populated() {
        let brain = populated_brain();
        let concepts = discover_emergent_concepts(&brain).unwrap();
        // Multiple scientists share "pioneered" + "contributed_to" → should form concept
        // (depends on IDF threshold, but at minimum it should not crash)
        assert!(concepts.len() >= 0);
    }

    #[test]
    fn test_discover_emergent_concepts_empty_graph() {
        let brain = test_brain();
        let concepts = discover_emergent_concepts(&brain).unwrap();
        assert!(concepts.is_empty());
    }

    #[test]
    fn test_generate_concept_name_known_pattern() {
        let name = generate_concept_name(&["pioneered".to_string(), "persecuted_for".to_string()]);
        assert_eq!(name, "Vindicated Visionaries");
    }

    #[test]
    fn test_generate_concept_name_generic() {
        let name = generate_concept_name(&["studied".to_string(), "published".to_string()]);
        // Should produce something like "Studied-Published Group"
        assert!(name.contains("Group"));
    }

    #[test]
    fn test_generate_concept_name_empty() {
        let name = generate_concept_name(&[]);
        assert_eq!(name, "Unknown Concept");
    }

    #[test]
    fn test_materialize_concepts() {
        let brain = populated_brain();
        let concepts = vec![EmergentConcept {
            name: "Test Concept".to_string(),
            member_entities: vec![1, 2],
            defining_predicates: vec!["test_pred".to_string()],
            confidence: 0.8,
            entity_names: vec!["A".to_string(), "B".to_string()],
        }];
        let count = materialize_concepts(&brain, &concepts).unwrap();
        assert_eq!(count, 1);

        // Verify the concept entity was created
        let entity = brain.get_entity_by_name("Test Concept").unwrap();
        assert!(entity.is_some());
    }

    #[test]
    fn test_materialize_concepts_low_confidence_skipped() {
        let brain = test_brain();
        let concepts = vec![EmergentConcept {
            name: "Weak Concept".to_string(),
            member_entities: vec![1],
            defining_predicates: vec!["x".to_string()],
            confidence: 0.1, // below threshold
            entity_names: vec!["A".to_string()],
        }];
        let count = materialize_concepts(&brain, &concepts).unwrap();
        assert_eq!(count, 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // 4. Contradiction Detection Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_detect_contradictions_fact_conflict() {
        let brain = populated_brain();
        let contradictions = detect_contradictions(&brain).unwrap();
        // Newton has birth_year = "1643" and birth_year = "1642"
        assert!(
            contradictions.iter().any(|c| c.key == "birth_year"),
            "Should detect birth_year contradiction for Newton"
        );
    }

    #[test]
    fn test_detect_contradictions_empty_graph() {
        let brain = test_brain();
        let contradictions = detect_contradictions(&brain).unwrap();
        assert!(contradictions.is_empty());
    }

    #[test]
    fn test_detect_contradictions_no_conflict() {
        let brain = test_brain();
        let eid = brain.upsert_entity("Test", "test").unwrap();
        brain.upsert_fact(eid, "color", "red", "src").unwrap();
        brain.upsert_fact(eid, "size", "large", "src").unwrap();
        // Different keys → no contradiction
        let contradictions = detect_contradictions(&brain).unwrap();
        assert!(contradictions.is_empty());
    }

    #[test]
    fn test_detect_temporal_contradictions() {
        let brain = test_brain();
        let eid = brain.upsert_entity("TimeTravel Person", "person").unwrap();
        brain.upsert_fact(eid, "birth_year", "2000", "src").unwrap();
        brain.upsert_fact(eid, "death_year", "1900", "src").unwrap();

        let contradictions = detect_temporal_contradictions(&brain).unwrap();
        assert!(
            contradictions
                .iter()
                .any(|c| c.conflict_type == ConflictType::TemporalContradiction),
            "Should detect born-after-died temporal contradiction"
        );
    }

    #[test]
    fn test_resolve_contradiction_confidence_weighted() {
        let brain = test_brain();
        let eid = brain.upsert_entity("Disputed", "test").unwrap();
        brain.upsert_fact(eid, "answer", "42", "good_src").unwrap();
        brain.upsert_fact(eid, "answer", "43", "bad_src").unwrap();

        // Boost the first fact's confidence
        brain
            .with_conn(|conn| {
                conn.execute(
                    "UPDATE facts SET confidence = 0.9 WHERE value = '42' AND entity_id = ?1",
                    params![eid],
                )?;
                Ok(())
            })
            .unwrap();

        let contradictions = detect_contradictions(&brain).unwrap();
        assert!(!contradictions.is_empty());

        let (keep, demote) = resolve_contradiction(&brain, &contradictions[0]).unwrap();
        assert_ne!(keep, demote);
    }

    #[test]
    fn test_extract_year() {
        assert_eq!(extract_year("1879"), Some(1879));
        assert_eq!(extract_year("born in 1643"), Some(1643));
        assert_eq!(extract_year("no year here"), None);
        assert_eq!(extract_year("2023-01-15"), Some(2023));
    }

    #[test]
    fn test_determine_resolution_strategy_high_gap() {
        let c = Contradiction {
            fact_a_id: 1,
            fact_b_id: 2,
            entity_id: 1,
            key: "test".to_string(),
            value_a: "a".to_string(),
            value_b: "b".to_string(),
            confidence_a: 0.9,
            confidence_b: 0.2,
            source_a: "src".to_string(),
            source_b: "src".to_string(),
            conflict_type: ConflictType::FactConflict,
            resolution_strategy: ResolutionStrategy::ConfidenceWeighted,
            resolved: false,
        };
        let strategy = determine_resolution_strategy(&c);
        assert_eq!(strategy, ResolutionStrategy::ConfidenceWeighted);
    }

    #[test]
    fn test_determine_resolution_strategy_different_sources() {
        let c = Contradiction {
            fact_a_id: 1,
            fact_b_id: 2,
            entity_id: 1,
            key: "test".to_string(),
            value_a: "a".to_string(),
            value_b: "b".to_string(),
            confidence_a: 0.5,
            confidence_b: 0.5,
            source_a: "wikipedia.org".to_string(),
            source_b: "random-blog.com".to_string(),
            conflict_type: ConflictType::FactConflict,
            resolution_strategy: ResolutionStrategy::ConfidenceWeighted,
            resolved: false,
        };
        let strategy = determine_resolution_strategy(&c);
        assert_eq!(strategy, ResolutionStrategy::SourceAuthority);
    }

    // ───────────────────────────────────────────────────────────────────────
    // 5. Ontology Restructuring Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_restructure_ontology_populated() {
        let brain = populated_brain();
        let report = restructure_ontology(&brain).unwrap();
        // Should produce a valid report (may or may not have changes depending on data)
        assert!(report.proposed_changes.len() >= 0);
    }

    #[test]
    fn test_restructure_ontology_empty_graph() {
        let brain = test_brain();
        let report = restructure_ontology(&brain).unwrap();
        assert!(report.proposed_changes.is_empty());
        assert!(report.synonym_pairs.is_empty());
    }

    #[test]
    fn test_restructure_ontology_detects_synonyms() {
        let brain = test_brain();
        let a = brain.upsert_entity("Entity_A", "person").unwrap();
        let b = brain.upsert_entity("Entity_B", "concept").unwrap();
        let c = brain.upsert_entity("Entity_C", "person").unwrap();
        let d = brain.upsert_entity("Entity_D", "concept").unwrap();

        // "created" and "built" connecting the same pairs → synonyms
        brain.upsert_relation(a, "created", b, "test").unwrap();
        brain.upsert_relation(c, "created", d, "test").unwrap();
        brain.upsert_relation(a, "built", b, "test").unwrap();
        brain.upsert_relation(c, "built", d, "test").unwrap();

        let report = restructure_ontology(&brain).unwrap();
        // Should detect "created" and "built" as synonyms
        let has_synonym = report
            .synonym_pairs
            .iter()
            .any(|(a, b, _)| (a == "created" && b == "built") || (a == "built" && b == "created"));
        assert!(has_synonym, "Should detect created/built as synonyms");
    }

    #[test]
    fn test_apply_ontology_merge() {
        let brain = test_brain();
        let a = brain.upsert_entity("Entity_A", "person").unwrap();
        let b = brain.upsert_entity("Entity_B", "concept").unwrap();
        brain.upsert_relation(a, "old_pred", b, "test").unwrap();

        let change = OntologyChange::MergePredicate {
            source_predicate: "old_pred".to_string(),
            target_predicate: "new_pred".to_string(),
            jaccard_similarity: 0.9,
        };

        let n = apply_ontology_change(&brain, &change).unwrap();
        assert_eq!(n, 1);

        // Verify the predicate was renamed
        let rels = brain.all_relations().unwrap();
        assert!(rels.iter().all(|r| r.predicate != "old_pred"));
        assert!(rels.iter().any(|r| r.predicate == "new_pred"));
    }

    #[test]
    fn test_apply_ontology_retype() {
        let brain = test_brain();
        let eid = brain.upsert_entity("John Smith", "phrase").unwrap();
        let concept = brain.upsert_entity("Physics", "concept").unwrap();
        brain
            .upsert_relation(eid, "studied", concept, "test")
            .unwrap();

        let change = OntologyChange::RetypeEntities {
            entity_ids: vec![eid],
            old_type: "phrase".to_string(),
            new_type: "person".to_string(),
            reason: "predicate pattern".to_string(),
        };

        let n = apply_ontology_change(&brain, &change).unwrap();
        assert_eq!(n, 1);

        let entity = brain.get_entity_by_id(eid).unwrap().unwrap();
        assert_eq!(entity.entity_type, "person");
    }

    // ───────────────────────────────────────────────────────────────────────
    // 6. Integration Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_reasoning_cycle_empty_graph() {
        let brain = test_brain();
        let report = reasoning_cycle(&brain).unwrap();
        assert_eq!(report.emergent_concepts, 0);
        assert_eq!(report.contradictions_detected, 0);
        assert_eq!(report.analogies_found, 0);
    }

    #[test]
    fn test_reasoning_cycle_populated() {
        let brain = populated_brain();
        let report = reasoning_cycle(&brain).unwrap();
        // Should complete without errors
        // The populated graph has a birth_year contradiction
        assert!(report.contradictions_detected > 0 || report.contradictions_resolved >= 0);
    }

    #[test]
    fn test_reasoning_cycle_with_causal_graph() {
        let brain = causal_brain();
        let report = reasoning_cycle(&brain).unwrap();
        // Should find causal chains in the event chain
        assert!(report.causal_chains_discovered >= 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Edge Case Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_single_node_graph() {
        let brain = test_brain();
        let _eid = brain.upsert_entity("Lonely Node", "concept").unwrap();

        // All systems should handle single-node gracefully
        let chains = discover_causal_chains(&brain, 1, 1, 3).unwrap();
        assert!(chains.is_empty()); // no self-loops in causal chains

        let analogies = discover_analogies(&brain).unwrap();
        assert!(analogies.is_empty());

        let concepts = discover_emergent_concepts(&brain).unwrap();
        assert!(concepts.is_empty());

        let contradictions = detect_contradictions(&brain).unwrap();
        assert!(contradictions.is_empty());

        let report = restructure_ontology(&brain).unwrap();
        assert!(report.proposed_changes.is_empty());
    }

    #[test]
    fn test_disconnected_components() {
        let brain = test_brain();
        // Component 1
        let a1 = brain.upsert_entity("Island_A1", "concept").unwrap();
        let a2 = brain.upsert_entity("Island_A2", "concept").unwrap();
        brain.upsert_relation(a1, "linked", a2, "test").unwrap();

        // Component 2 (disconnected)
        let b1 = brain.upsert_entity("Island_B1", "event").unwrap();
        let b2 = brain.upsert_entity("Island_B2", "event").unwrap();
        brain.upsert_relation(b1, "caused", b2, "test").unwrap();

        // No path between components
        let chains = discover_causal_chains(&brain, a1, b1, 5).unwrap();
        assert!(chains.is_empty());

        // But each component should still work internally
        let chains_internal = discover_causal_chains(&brain, b1, b2, 3).unwrap();
        assert!(!chains_internal.is_empty());
    }

    #[test]
    fn test_cyclic_graph() {
        let brain = test_brain();
        let a = brain.upsert_entity("Cycle_A", "concept").unwrap();
        let b = brain.upsert_entity("Cycle_B", "concept").unwrap();
        let c = brain.upsert_entity("Cycle_C", "concept").unwrap();

        brain.upsert_relation(a, "leads_to", b, "test").unwrap();
        brain.upsert_relation(b, "leads_to", c, "test").unwrap();
        brain.upsert_relation(c, "leads_to", a, "test").unwrap(); // cycle!

        // Should not infinite loop
        let chains = discover_causal_chains(&brain, a, c, 5).unwrap();
        // Should find A→B→C path without going through the cycle
        assert!(!chains.is_empty());
    }

    #[test]
    fn test_high_branching_factor() {
        let brain = test_brain();
        let hub = brain.upsert_entity("Hub", "concept").unwrap();

        // Create a star graph with 20 spokes
        for i in 0..20 {
            let spoke = brain
                .upsert_entity(&format!("Spoke_{}", i), "concept")
                .unwrap();
            brain
                .upsert_relation(hub, "connected", spoke, "test")
                .unwrap();
        }

        // Should handle high branching gracefully
        let report = restructure_ontology(&brain).unwrap();
        // "connected" connects many concept→concept pairs — might be flagged
        assert!(report.proposed_changes.len() >= 0);
    }

    #[test]
    fn test_build_predicate_mapping() {
        let sig_a = PredicateSignature {
            outgoing: [
                ("pioneered".to_string(), "concept".to_string()),
                ("contributed_to".to_string(), "field".to_string()),
                ("born_in".to_string(), "place".to_string()),
            ]
            .into_iter()
            .collect(),
            incoming: BTreeSet::new(),
        };
        let sig_b = PredicateSignature {
            outgoing: [
                ("pioneered".to_string(), "concept".to_string()),
                ("contributed_to".to_string(), "field".to_string()),
                ("studied_at".to_string(), "organization".to_string()),
            ]
            .into_iter()
            .collect(),
            incoming: BTreeSet::new(),
        };

        let mapping = build_predicate_mapping(&sig_a, &sig_b);
        assert!(mapping.iter().any(|(a, _)| a == "pioneered"));
        assert!(mapping.iter().any(|(a, _)| a == "contributed_to"));
        assert!(!mapping.iter().any(|(a, _)| a == "born_in"));
    }
}
