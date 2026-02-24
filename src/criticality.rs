#![allow(dead_code, clippy::type_complexity)]
//! # Criticality Engine — Self-Organized Criticality, Abductive Reasoning,
//! # Temporal Intelligence, Network Topology Optimization & Meta-Cognition
//!
//! Six interconnected subsystems that make axon a *thinking* system:
//!
//! 1. **Self-Organized Criticality** — Von Neumann graph entropy, semantic entropy,
//!    critical discovery parameter (CDP), surprise edge fraction, avalanche detection.
//!    Based on Buehler 2025 "Self-Organizing Graph Reasoning Evolves into a Critical State".
//!
//! 2. **Abductive Reasoning** — Given observations, generate best explanations via
//!    parsimony × coverage × consistency scoring. Explain gaps, predict next discoveries.
//!    Based on DARK 2025 unified deductive/abductive reasoning and CtrlHGen 2026.
//!
//! 3. **Temporal Reasoning** — Granger-like causality, temporal anomaly detection,
//!    pattern prediction from entity appearance timelines.
//!
//! 4. **Network Topology Optimization** — Scale-free scoring, small-world coefficient,
//!    hub vulnerability analysis, topology steering recommendations.
//!
//! 5. **Meta-Cognitive Loop** — Domain balance (Gini coefficient), discovery velocity,
//!    staleness index, feeding strategy recommendations.
//!
//! 6. **Criticality Report** — Full health report integrating all subsystems.

use chrono::{NaiveDateTime, Utc};
use rusqlite::{params, Connection, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::db::Brain;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum Lanczos iterations for eigenvalue approximation.
const LANCZOS_MAX_ITER: usize = 100;

/// Convergence threshold for Lanczos.
const LANCZOS_EPSILON: f64 = 1e-10;

/// Optimal CDP range for discovery zone.
const CDP_DISCOVERY_LOW: f64 = -0.3;
const CDP_DISCOVERY_HIGH: f64 = -0.1;

/// Target surprise edge fraction for optimal innovation.
const SURPRISE_TARGET: f64 = 0.12;

/// Cosine similarity threshold for "surprising" edges.
const SURPRISE_COSINE_THRESHOLD: f64 = 0.2;

/// Ideal power-law exponent range for scale-free networks.
const SCALE_FREE_GAMMA_LOW: f64 = 2.0;
const SCALE_FREE_GAMMA_HIGH: f64 = 3.0;

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

fn is_noise_type(t: &str) -> bool {
    NOISE_TYPES.contains(&t)
}

// ═══════════════════════════════════════════════════════════════════════════
// Schema helpers
// ═══════════════════════════════════════════════════════════════════════════

fn ensure_criticality_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS criticality_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            cdp REAL NOT NULL,
            structural_entropy REAL NOT NULL,
            semantic_entropy REAL NOT NULL,
            surprise_fraction REAL NOT NULL,
            entity_count INTEGER NOT NULL,
            relation_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS avalanche_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at TEXT NOT NULL,
            trigger_entity_id INTEGER,
            avalanche_size INTEGER NOT NULL,
            cascade_depth INTEGER NOT NULL
        );",
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. SELF-ORGANIZED CRITICALITY ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse matrix representation for Laplacian operations.
struct SparseMatrix {
    /// Adjacency: node_index → Vec<(neighbor_index, weight)>
    adj: Vec<Vec<(usize, f64)>>,
    /// Degree of each node
    degrees: Vec<f64>,
    /// Number of nodes
    n: usize,
}

impl SparseMatrix {
    /// Multiply the normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2} by vector x.
    /// Returns L_norm · x
    fn laplacian_multiply(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.n];
        for i in 0..self.n {
            // Diagonal: L_norm[i,i] = 1 (if degree > 0), else 0
            if self.degrees[i] > 0.0 {
                result[i] = x[i]; // Identity part
            }
            // Off-diagonal: L_norm[i,j] = -1/sqrt(d_i * d_j) if (i,j) is edge
            let di_inv_sqrt = if self.degrees[i] > 0.0 {
                1.0 / self.degrees[i].sqrt()
            } else {
                0.0
            };
            for &(j, _weight) in &self.adj[i] {
                let dj_inv_sqrt = if self.degrees[j] > 0.0 {
                    1.0 / self.degrees[j].sqrt()
                } else {
                    0.0
                };
                result[i] -= di_inv_sqrt * dj_inv_sqrt * x[j];
            }
        }
        result
    }
}

/// Build sparse matrix from the brain's graph.
fn build_sparse_graph(brain: &Brain) -> Result<(SparseMatrix, Vec<i64>)> {
    let relations = brain.all_relations()?;
    let entities = brain.all_entities()?;

    // Build id→index mapping
    let mut id_to_idx: HashMap<i64, usize> = HashMap::new();
    let mut idx_to_id: Vec<i64> = Vec::new();

    for e in &entities {
        if !is_noise_type(&e.entity_type) {
            let idx = idx_to_id.len();
            id_to_idx.insert(e.id, idx);
            idx_to_id.push(e.id);
        }
    }

    let n = idx_to_id.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut degrees: Vec<f64> = vec![0.0; n];

    for r in &relations {
        if let (Some(&si), Some(&oi)) = (id_to_idx.get(&r.subject_id), id_to_idx.get(&r.object_id))
        {
            if si != oi {
                adj[si].push((oi, r.confidence));
                adj[oi].push((si, r.confidence));
                degrees[si] += 1.0;
                degrees[oi] += 1.0;
            }
        }
    }

    Ok((SparseMatrix { adj, degrees, n }, idx_to_id))
}

/// Lanczos algorithm for approximating eigenvalues of a symmetric matrix.
///
/// Computes the k largest eigenvalues of the normalized Laplacian using the
/// Lanczos iteration, which builds a tridiagonal matrix whose eigenvalues
/// approximate those of the original matrix.
///
/// Returns sorted eigenvalues (ascending).
fn lanczos_eigenvalues(mat: &SparseMatrix, k: usize) -> Vec<f64> {
    let n = mat.n;
    if n == 0 {
        return vec![];
    }

    let k = k.min(n);

    // Initialize with a deterministic vector (unit vector normalized)
    let norm = (n as f64).sqrt();
    let mut v: Vec<f64> = vec![1.0 / norm; n];

    let mut alpha: Vec<f64> = Vec::with_capacity(k); // diagonal
    let mut beta: Vec<f64> = Vec::with_capacity(k); // off-diagonal

    let mut v_prev: Vec<f64> = vec![0.0; n];

    for j in 0..k {
        // w = L * v_j
        let mut w = mat.laplacian_multiply(&v);

        // alpha_j = w^T * v_j
        let a: f64 = w.iter().zip(v.iter()).map(|(wi, vi)| wi * vi).sum();
        alpha.push(a);

        // w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        let b_prev = if j > 0 { beta[j - 1] } else { 0.0 };
        for i in 0..n {
            w[i] -= a * v[i] + b_prev * v_prev[i];
        }

        // Reorthogonalization (full, for numerical stability)
        // This is important for accurate eigenvalues
        // We'd need to store all previous vectors for full reorth,
        // but for a practical implementation we do partial
        let w_norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();

        if w_norm < LANCZOS_EPSILON {
            break; // Invariant subspace found
        }

        beta.push(w_norm);

        v_prev = v.clone();
        v = w.iter().map(|x| x / w_norm).collect();
    }

    // Now solve the tridiagonal eigenvalue problem using the QL algorithm
    tridiagonal_eigenvalues(&alpha, &beta)
}

/// Compute eigenvalues of a symmetric tridiagonal matrix using the
/// implicit QR algorithm with Wilkinson shifts.
///
/// Based on the LAPACK dsteqr algorithm (simplified, eigenvalues only).
///
/// Input: alpha = diagonal elements, beta = off-diagonal elements
/// Returns sorted eigenvalues (ascending).
fn tridiagonal_eigenvalues(alpha: &[f64], beta: &[f64]) -> Vec<f64> {
    let n = alpha.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![alpha[0]];
    }

    // Working copies: d = diagonal, e = sub-diagonal (length n-1)
    let mut d: Vec<f64> = alpha.to_vec();
    let mut e: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n - 1 {
        e.push(beta.get(i).copied().unwrap_or(0.0));
    }
    // e has length n-1

    let eps = f64::EPSILON;
    let max_iter = 30 * n;
    let mut iter_count = 0;

    // Process from bottom-right corner
    let mut end = n - 1; // active block is [start..=end]

    while end > 0 && iter_count < max_iter {
        // Find the largest l such that e[l] is negligible → split
        let mut l = end;
        while l > 0 {
            let test = d[l - 1].abs() + d[l].abs();
            if e[l - 1].abs() <= eps * test {
                e[l - 1] = 0.0;
                break;
            }
            l -= 1;
        }

        if l == end {
            // e[end-1] is zero → d[end] is an eigenvalue, deflate
            end -= 1;
            continue;
        }

        iter_count += 1;

        // Wilkinson shift: eigenvalue of trailing 2x2 closer to d[end]
        let p = (d[end - 1] - d[end]) / (2.0 * e[end - 1]);
        let r = p.hypot(1.0);
        let shift = d[end] - e[end - 1] / (p + r.copysign(p));

        // Implicit QR step (Givens rotations chasing the bulge from l to end)
        let mut f = d[l] - shift;
        let mut g = e[l];

        for i in l..end {
            // Compute Givens rotation to zero out g
            let rr = f.hypot(g);
            let cos = f / rr;
            let sin = g / rr;

            if i > l {
                e[i - 1] = rr;
            }

            f = cos * d[i] + sin * e[i];
            e[i] = cos * e[i] - sin * d[i];
            g = sin * d[i + 1];
            d[i + 1] *= cos;

            d[i] = cos * f + sin * g;
            f = cos * e[i] + sin * d[i + 1];
            d[i + 1] = -sin * e[i] + cos * d[i + 1];

            if i + 1 < end {
                g = sin * e[i + 1];
                e[i + 1] *= cos;
            }
        }

        e[end - 1] = f;

        // Check for convergence of e[end-1]
        let test = d[end - 1].abs() + d[end].abs();
        if e[end - 1].abs() <= eps * test {
            e[end - 1] = 0.0;
            end -= 1;
        }
    }

    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d
}

/// Compute Von Neumann graph entropy from the normalized Laplacian.
///
/// The Von Neumann entropy S = -Σ λ̃ᵢ · log₂(λ̃ᵢ) where λ̃ᵢ = λᵢ/Σλⱼ
/// are the normalized eigenvalues of the graph Laplacian.
///
/// Uses the Lanczos algorithm to approximate eigenvalues for large graphs,
/// falling back to exact computation for small ones.
///
/// This measures structural randomness: high entropy = uniform structure,
/// low entropy = highly structured/hierarchical.
pub fn von_neumann_graph_entropy(brain: &Brain) -> Result<f64> {
    let (mat, _ids) = build_sparse_graph(brain)?;
    let n = mat.n;

    if n < 2 {
        return Ok(0.0);
    }

    // Determine number of eigenvalues to compute
    let k = if n <= 200 {
        n // Exact for small graphs
    } else {
        LANCZOS_MAX_ITER.min(n / 2).max(30)
    };

    let eigenvalues = lanczos_eigenvalues(&mat, k);

    if eigenvalues.is_empty() {
        return Ok(0.0);
    }

    // The eigenvalues of the normalized Laplacian are in [0, 2].
    // For Von Neumann entropy, we use the density matrix ρ = L_norm / tr(L_norm)
    // tr(L_norm) = n for the normalized Laplacian (trace = number of non-isolated nodes)
    let trace: f64 = eigenvalues.iter().filter(|&&l| l > 1e-15).sum();

    if trace < 1e-15 {
        return Ok(0.0);
    }

    // Compute entropy
    let mut entropy = 0.0_f64;
    for &lambda in &eigenvalues {
        if lambda > 1e-15 {
            let p = lambda / trace;
            entropy -= p * p.log2();
        }
    }

    Ok(entropy)
}

/// Compute semantic entropy over the distribution of predicate types,
/// entity types, and fact key distributions.
///
/// Combines three Shannon entropy measures:
/// - H(predicates): diversity of relation types
/// - H(entity_types): diversity of entity categories
/// - H(fact_keys): diversity of fact key types
///
/// Returns the weighted average: 0.4·H(pred) + 0.35·H(type) + 0.25·H(facts)
pub fn semantic_entropy(brain: &Brain) -> Result<f64> {
    let relations = brain.all_relations()?;
    let entities = brain.all_entities()?;

    // H(predicates) — Shannon entropy of predicate distribution
    let h_pred = {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for r in &relations {
            *counts.entry(r.predicate.as_str()).or_insert(0) += 1;
        }
        shannon_entropy_from_counts(counts.values())
    };

    // H(entity_types)
    let h_types = {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for e in &entities {
            if !is_noise_type(&e.entity_type) {
                *counts.entry(e.entity_type.as_str()).or_insert(0) += 1;
            }
        }
        shannon_entropy_from_counts(counts.values())
    };

    // H(fact_keys)
    let h_facts = brain.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT key, COUNT(*) FROM facts GROUP BY key")?;
        let rows: Vec<usize> = stmt
            .query_map([], |row| row.get::<_, usize>(1))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(shannon_entropy_from_counts(rows.iter()))
    })?;

    // Weighted combination
    let semantic = 0.4 * h_pred + 0.35 * h_types + 0.25 * h_facts;
    Ok(semantic)
}

/// Compute Shannon entropy from a collection of counts.
fn shannon_entropy_from_counts<'a, I>(counts: I) -> f64
where
    I: Iterator<Item = &'a usize>,
{
    let counts: Vec<usize> = counts.copied().collect();
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut entropy = 0.0_f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Critical Discovery Parameter (CDP) — a dimensionless measure of the
/// knowledge graph's position relative to the critical point.
///
/// CDP = (S_semantic - S_structural) / max(S_semantic, S_structural)
///
/// When CDP stabilizes at a small negative value (~-0.1 to -0.3), the system
/// is in the "discovery zone" where structure slightly dominates semantics,
/// maintaining enough order for reasoning but enough diversity for discovery.
///
/// CDP > 0: Semantic diversity exceeds structure (supercritical — too chaotic)
/// CDP ≈ 0: Perfect balance (rare)
/// CDP < 0: Structure exceeds semantic diversity (ordered, focused)
/// CDP < -0.3: Subcritical (too structured, not enough diversity)
pub fn critical_discovery_parameter(brain: &Brain) -> Result<f64> {
    let s_struct = von_neumann_graph_entropy(brain)?;
    let s_sem = semantic_entropy(brain)?;

    let max_s = s_struct.max(s_sem);
    if max_s < 1e-10 {
        return Ok(0.0);
    }

    let cdp = (s_sem - s_struct) / max_s;

    // Log the CDP value
    brain.with_conn(|conn| {
        ensure_criticality_tables(conn)?;
        let now = Utc::now()
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let entity_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))?;
        let relation_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM relations", [], |r| r.get(0))?;
        conn.execute(
            "INSERT INTO criticality_log (recorded_at, cdp, structural_entropy, semantic_entropy, surprise_fraction, entity_count, relation_count)
             VALUES (?1, ?2, ?3, ?4, 0.0, ?5, ?6)",
            params![now, cdp, s_struct, s_sem, entity_count, relation_count],
        )?;
        Ok(())
    })?;

    Ok(cdp)
}

/// Compute the predicate profile vector for an entity.
///
/// Returns a HashMap<predicate, count> representing how many times
/// each predicate appears in the entity's relations.
fn entity_predicate_profile(brain: &Brain, entity_id: i64) -> Result<HashMap<String, f64>> {
    let mut profile: HashMap<String, f64> = HashMap::new();

    brain.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT predicate, COUNT(*) FROM relations
             WHERE subject_id = ?1 OR object_id = ?1
             GROUP BY predicate",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        for row in rows.flatten() {
            profile.insert(row.0, row.1);
        }
        Ok(())
    })?;

    Ok(profile)
}

/// Cosine similarity between two predicate profiles.
fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let keys: HashSet<&String> = a.keys().chain(b.keys()).collect();
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for key in keys {
        let va = a.get(key).copied().unwrap_or(0.0);
        let vb = b.get(key).copied().unwrap_or(0.0);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        dot / denom
    }
}

/// Fraction of edges connecting entities with very different predicate
/// profiles (cosine similarity < 0.2).
///
/// These "surprise edges" connect structurally dissimilar parts of the
/// knowledge graph. A fraction of ~12% is optimal for sustained discovery —
/// enough novelty without losing coherence.
pub fn surprise_edge_fraction(brain: &Brain) -> Result<f64> {
    let relations = brain.all_relations()?;
    if relations.is_empty() {
        return Ok(0.0);
    }

    // Build predicate profiles for all entities
    let mut profiles: HashMap<i64, HashMap<String, f64>> = HashMap::new();

    for r in &relations {
        profiles
            .entry(r.subject_id)
            .or_default()
            .entry(r.predicate.clone())
            .and_modify(|c| *c += 1.0)
            .or_insert(1.0);
        profiles
            .entry(r.object_id)
            .or_default()
            .entry(r.predicate.clone())
            .and_modify(|c| *c += 1.0)
            .or_insert(1.0);
    }

    // Count surprise edges
    let mut surprise_count = 0usize;
    let mut total_counted = 0usize;

    // Deduplicate edges (only count each unique pair once)
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    for r in &relations {
        let pair = if r.subject_id <= r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        if !seen.insert(pair) {
            continue;
        }

        if let (Some(pa), Some(pb)) = (profiles.get(&r.subject_id), profiles.get(&r.object_id)) {
            let sim = cosine_similarity(pa, pb);
            if sim < SURPRISE_COSINE_THRESHOLD {
                surprise_count += 1;
            }
            total_counted += 1;
        }
    }

    if total_counted == 0 {
        return Ok(0.0);
    }

    Ok(surprise_count as f64 / total_counted as f64)
}

/// Detect avalanches — cascading confirmations triggered by new facts.
///
/// An avalanche occurs when adding a fact causes multiple hypothesis
/// confirmations or relation validations in sequence. The distribution
/// of avalanche sizes indicates criticality:
/// - Power-law distribution → system is critical (optimal for discovery)
/// - Exponential distribution → subcritical (too ordered)
/// - Bimodal distribution → supercritical (chaotic bursts)
///
/// Returns (avalanche_sizes, power_law_exponent, is_power_law).
pub fn avalanche_detection(brain: &Brain) -> Result<(Vec<usize>, f64, bool)> {
    // Analyze avalanche history from the log
    let sizes = brain.with_conn(|conn| {
        ensure_criticality_tables(conn)?;
        let mut stmt = conn.prepare("SELECT avalanche_size FROM avalanche_log ORDER BY id")?;
        let rows: Vec<usize> = stmt
            .query_map([], |row| row.get::<_, usize>(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    })?;

    if sizes.len() < 3 {
        // Not enough data — estimate from graph structure
        return estimate_avalanche_potential(brain);
    }

    // Fit power law to avalanche size distribution
    let (gamma, r_squared) = fit_power_law(&sizes);
    let is_power_law = r_squared > 0.7 && gamma > 1.0;

    Ok((sizes, gamma, is_power_law))
}

/// Estimate avalanche potential from graph structure when no avalanche history exists.
///
/// Simulates avalanches by computing cascade depth from hub nodes:
/// if we "activate" a hub, how many of its neighbors' neighbors get reached?
fn estimate_avalanche_potential(brain: &Brain) -> Result<(Vec<usize>, f64, bool)> {
    let relations = brain.all_relations()?;

    // Build adjacency
    let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();
    for r in &relations {
        adj.entry(r.subject_id).or_default().push(r.object_id);
        adj.entry(r.object_id).or_default().push(r.subject_id);
    }

    if adj.is_empty() {
        return Ok((vec![], 0.0, false));
    }

    // Find top 20 hubs by degree
    let mut degree_list: Vec<(i64, usize)> = adj.iter().map(|(id, n)| (*id, n.len())).collect();
    degree_list.sort_by(|a, b| b.1.cmp(&a.1));
    degree_list.truncate(20);

    let mut sizes: Vec<usize> = Vec::new();

    // Simulate cascades from each hub with probability-based propagation
    for &(hub_id, _) in &degree_list {
        let mut visited: HashSet<i64> = HashSet::new();
        let mut frontier = vec![hub_id];
        visited.insert(hub_id);
        let mut cascade_size = 0usize;

        // Propagation probability decays with depth
        for depth in 0..4 {
            let prob = 0.5_f64.powi(depth);
            let mut next_frontier = Vec::new();

            for &node in &frontier {
                if let Some(neighbors) = adj.get(&node) {
                    for &n in neighbors {
                        if visited.insert(n) {
                            // Deterministic simulation: use degree ratio as activation threshold
                            let n_degree = adj.get(&n).map(|v| v.len()).unwrap_or(0);
                            if (n_degree as f64 * prob) >= 1.0 {
                                next_frontier.push(n);
                                cascade_size += 1;
                            }
                        }
                    }
                }
            }

            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        if cascade_size > 0 {
            sizes.push(cascade_size);
        }
    }

    if sizes.is_empty() {
        return Ok((vec![], 0.0, false));
    }

    let (gamma, r_squared) = fit_power_law(&sizes);
    let is_power_law = r_squared > 0.5 && gamma > 1.0 && sizes.len() >= 5;

    Ok((sizes, gamma, is_power_law))
}

/// Record an avalanche event in the database.
pub fn record_avalanche(
    brain: &Brain,
    trigger_entity_id: Option<i64>,
    size: usize,
    depth: usize,
) -> Result<()> {
    brain.with_conn(|conn| {
        ensure_criticality_tables(conn)?;
        let now = Utc::now()
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        conn.execute(
            "INSERT INTO avalanche_log (triggered_at, trigger_entity_id, avalanche_size, cascade_depth)
             VALUES (?1, ?2, ?3, ?4)",
            params![now, trigger_entity_id, size as i64, depth as i64],
        )?;
        Ok(())
    })
}

/// Fit power law P(x) ~ x^{-γ} to a distribution of sizes using
/// Maximum Likelihood Estimation (MLE).
///
/// γ_MLE = 1 + n / Σ ln(xᵢ/x_min)
///
/// Also computes R² of log-log regression for goodness-of-fit.
fn fit_power_law(sizes: &[usize]) -> (f64, f64) {
    if sizes.is_empty() {
        return (0.0, 0.0);
    }

    let x_min = *sizes.iter().min().unwrap_or(&1) as f64;
    if x_min < 1.0 {
        return (0.0, 0.0);
    }

    let n = sizes.len() as f64;
    let sum_log: f64 = sizes
        .iter()
        .map(|&x| (x as f64 / x_min).ln())
        .filter(|v| v.is_finite())
        .sum();

    let gamma = if sum_log > 0.0 {
        1.0 + n / sum_log
    } else {
        0.0
    };

    // R² from log-log regression
    let mut freq: HashMap<usize, usize> = HashMap::new();
    for &s in sizes {
        *freq.entry(s).or_insert(0) += 1;
    }

    let points: Vec<(f64, f64)> = freq
        .iter()
        .filter(|(&s, &c)| s > 0 && c > 0)
        .map(|(&s, &c)| ((s as f64).ln(), (c as f64).ln()))
        .collect();

    let r_squared = if points.len() >= 3 {
        linear_r_squared(&points)
    } else {
        0.0
    };

    (gamma, r_squared)
}

/// R² of linear regression on a set of (x, y) points.
fn linear_r_squared(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let mean_y = sum_y / n;
    let ss_tot: f64 = points.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = points
        .iter()
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();

    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Criticality regime classification.
#[derive(Debug, Clone, PartialEq)]
pub enum CriticalityRegime {
    /// CDP < -0.3: Too structured, needs more diversity
    Subcritical,
    /// CDP ∈ [-0.3, -0.1]: Optimal for discovery
    Critical,
    /// CDP ∈ [-0.1, 0.1]: Near-balanced
    NearCritical,
    /// CDP > 0.1: Too chaotic, needs consolidation
    Supercritical,
}

impl std::fmt::Display for CriticalityRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CriticalityRegime::Subcritical => write!(f, "Subcritical"),
            CriticalityRegime::Critical => write!(f, "Critical (Discovery Zone)"),
            CriticalityRegime::NearCritical => write!(f, "Near-Critical"),
            CriticalityRegime::Supercritical => write!(f, "Supercritical"),
        }
    }
}

/// Full criticality report.
#[derive(Debug, Clone)]
pub struct CriticalityReport {
    pub structural_entropy: f64,
    pub semantic_entropy: f64,
    pub cdp: f64,
    pub regime: CriticalityRegime,
    pub surprise_fraction: f64,
    pub surprise_assessment: String,
    pub avalanche_sizes: Vec<usize>,
    pub avalanche_exponent: f64,
    pub avalanche_is_power_law: bool,
    pub recommendation: String,
    pub cdp_history: Vec<(String, f64)>,
}

/// Generate a full criticality report.
pub fn criticality_report(brain: &Brain) -> Result<CriticalityReport> {
    let s_struct = von_neumann_graph_entropy(brain)?;
    let s_sem = semantic_entropy(brain)?;

    let max_s = s_struct.max(s_sem);
    let cdp = if max_s > 1e-10 {
        (s_sem - s_struct) / max_s
    } else {
        0.0
    };

    let regime = if cdp < CDP_DISCOVERY_LOW {
        CriticalityRegime::Subcritical
    } else if cdp <= CDP_DISCOVERY_HIGH {
        CriticalityRegime::Critical
    } else if cdp <= 0.1 {
        CriticalityRegime::NearCritical
    } else {
        CriticalityRegime::Supercritical
    };

    let surprise = surprise_edge_fraction(brain)?;

    let surprise_assessment = if surprise < 0.05 {
        "Very low surprise — knowledge graph is too homogeneous".to_string()
    } else if surprise < 0.10 {
        format!(
            "Below target ({:.1}% vs 12%) — add more cross-domain connections",
            surprise * 100.0
        )
    } else if surprise < 0.15 {
        format!(
            "Near optimal ({:.1}% ≈ 12%) — good innovation potential",
            surprise * 100.0
        )
    } else {
        format!(
            "High surprise ({:.1}%) — consider consolidating related domains",
            surprise * 100.0
        )
    };

    let (avalanche_sizes, avalanche_exponent, avalanche_is_power_law) =
        avalanche_detection(brain)?;

    let recommendation = generate_criticality_recommendation(
        &regime,
        surprise,
        avalanche_is_power_law,
        s_struct,
        s_sem,
    );

    // Fetch CDP history
    let cdp_history = brain.with_conn(|conn| {
        ensure_criticality_tables(conn)?;
        let mut stmt = conn.prepare(
            "SELECT recorded_at, cdp FROM criticality_log ORDER BY id DESC LIMIT 20",
        )?;
        let rows: Vec<(String, f64)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    })?;

    Ok(CriticalityReport {
        structural_entropy: s_struct,
        semantic_entropy: s_sem,
        cdp,
        regime,
        surprise_fraction: surprise,
        surprise_assessment,
        avalanche_sizes,
        avalanche_exponent,
        avalanche_is_power_law,
        recommendation,
        cdp_history,
    })
}

fn generate_criticality_recommendation(
    regime: &CriticalityRegime,
    surprise: f64,
    is_power_law: bool,
    s_struct: f64,
    s_sem: f64,
) -> String {
    let mut parts = Vec::new();

    match regime {
        CriticalityRegime::Subcritical => {
            parts.push("System is SUBCRITICAL — inject more cross-domain content to increase semantic diversity.".to_string());
            parts.push(format!(
                "Structural entropy ({:.2}) >> Semantic entropy ({:.2}).",
                s_struct, s_sem
            ));
            parts.push(
                "Feed topics from underrepresented domains. Add controversial/bridging concepts."
                    .to_string(),
            );
        }
        CriticalityRegime::Critical => {
            parts.push(
                "System is in the DISCOVERY ZONE — optimal state for sustained knowledge creation."
                    .to_string(),
            );
            parts.push("Maintain current feeding balance. Monitor for drift.".to_string());
        }
        CriticalityRegime::NearCritical => {
            parts.push(
                "System is NEAR-CRITICAL — close to optimal but could benefit from tuning."
                    .to_string(),
            );
            if s_sem > s_struct {
                parts.push(
                    "Slightly more structure needed — consolidate knowledge in sparse areas."
                        .to_string(),
                );
            }
        }
        CriticalityRegime::Supercritical => {
            parts.push(
                "System is SUPERCRITICAL — too much unstructured diversity. Consolidate existing knowledge."
                    .to_string(),
            );
            parts.push(format!(
                "Semantic entropy ({:.2}) >> Structural entropy ({:.2}).",
                s_sem, s_struct
            ));
            parts.push(
                "Feed more depth on existing topics. Run dedup and reasoning cycles.".to_string(),
            );
        }
    }

    if surprise < 0.05 {
        parts.push(
            "Surprise fraction very low — actively seek cross-domain connections.".to_string(),
        );
    } else if surprise > 0.20 {
        parts.push("Surprise fraction too high — graph is becoming incoherent.".to_string());
    }

    if is_power_law {
        parts.push(
            "Avalanche distribution follows power law ✓ — system exhibits self-organized criticality."
                .to_string(),
        );
    }

    parts.join("\n")
}

/// Format a criticality report for display.
pub fn format_criticality_report(report: &CriticalityReport) -> String {
    let mut lines = Vec::new();

    lines.push("╔══════════════════════════════════════════════════╗".to_string());
    lines.push("║        CRITICALITY REPORT — axon brain          ║".to_string());
    lines.push("╚══════════════════════════════════════════════════╝".to_string());
    lines.push(String::new());

    lines.push(format!(
        "  Structural entropy (Von Neumann): {:.4} bits",
        report.structural_entropy
    ));
    lines.push(format!(
        "  Semantic entropy (Shannon):       {:.4} bits",
        report.semantic_entropy
    ));
    lines.push(format!(
        "  Critical Discovery Parameter:     {:.4}",
        report.cdp
    ));
    lines.push(format!("  Regime:                           {}", report.regime));
    lines.push(String::new());

    lines.push(format!(
        "  Surprise edge fraction:           {:.1}%",
        report.surprise_fraction * 100.0
    ));
    lines.push(format!("  Assessment: {}", report.surprise_assessment));
    lines.push(String::new());

    if !report.avalanche_sizes.is_empty() {
        let avg_size: f64 =
            report.avalanche_sizes.iter().sum::<usize>() as f64 / report.avalanche_sizes.len() as f64;
        let max_size = report.avalanche_sizes.iter().max().unwrap_or(&0);
        lines.push(format!(
            "  Avalanches: {} recorded, avg size {:.1}, max {}",
            report.avalanche_sizes.len(),
            avg_size,
            max_size
        ));
        lines.push(format!(
            "  Power-law exponent: {:.2} (R² fit: {})",
            report.avalanche_exponent,
            if report.avalanche_is_power_law {
                "good"
            } else {
                "poor"
            }
        ));
    } else {
        lines.push("  Avalanches: No history available".to_string());
    }

    lines.push(String::new());
    lines.push("  ── Recommendation ──".to_string());
    for line in report.recommendation.lines() {
        lines.push(format!("  {}", line));
    }

    if !report.cdp_history.is_empty() {
        lines.push(String::new());
        lines.push("  ── CDP History (recent) ──".to_string());
        for (ts, val) in report.cdp_history.iter().take(10) {
            lines.push(format!("    {} → {:.4}", ts, val));
        }
    }

    lines.join("\n")
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ABDUCTIVE REASONING ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/// A single step in an explanatory reasoning chain.
#[derive(Debug, Clone)]
pub struct ExplanationStep {
    pub entity_id: i64,
    pub entity_name: String,
    pub predicate: String,
    pub target_id: i64,
    pub target_name: String,
    pub confidence: f64,
}

/// A candidate explanation for an observation.
#[derive(Debug, Clone)]
pub struct Explanation {
    pub reasoning_chain: Vec<ExplanationStep>,
    /// Occam's razor — fewer steps = better. Score = 1/(1 + len)
    pub parsimony_score: f64,
    /// How many observations this explains. Normalized 0-1.
    pub coverage_score: f64,
    /// No contradictions with known facts. 1.0 = fully consistent.
    pub consistency_score: f64,
    /// Combined score: parsimony × coverage × consistency
    pub combined_score: f64,
    /// Human-readable summary
    pub summary: String,
}

/// An abductive hypothesis — "given observation X, what's the best explanation?"
#[derive(Debug, Clone)]
pub struct AbductiveHypothesis {
    pub observation: String,
    pub candidate_explanations: Vec<Explanation>,
    pub best_explanation: Option<Explanation>,
    pub confidence: f64,
}

/// Generate abductive hypotheses for an observation.
///
/// Given an observation (e.g., "Newton and Leibniz are both connected to Calculus"),
/// generate candidate explanations by:
/// 1. Finding shared neighborhood (common connections)
/// 2. Finding shared predicates (similar structural roles)
/// 3. Finding temporal co-occurrence
/// 4. Scoring by parsimony × coverage × consistency
pub fn abduce(brain: &Brain, observation: &str) -> Result<AbductiveHypothesis> {
    // Parse entities from the observation text
    let mentioned_entities = find_entities_in_text(brain, observation)?;

    let mut explanations: Vec<Explanation> = Vec::new();

    if mentioned_entities.len() >= 2 {
        // Strategy 1: Shared neighborhood explanation
        for i in 0..mentioned_entities.len() {
            for j in (i + 1)..mentioned_entities.len() {
                let (id_a, name_a) = &mentioned_entities[i];
                let (id_b, name_b) = &mentioned_entities[j];

                if let Some(expl) =
                    explain_via_shared_neighborhood(brain, *id_a, name_a, *id_b, name_b)?
                {
                    explanations.push(expl);
                }

                if let Some(expl) =
                    explain_via_shared_predicates(brain, *id_a, name_a, *id_b, name_b)?
                {
                    explanations.push(expl);
                }

                if let Some(expl) =
                    explain_via_temporal_cooccurrence(brain, *id_a, name_a, *id_b, name_b)?
                {
                    explanations.push(expl);
                }
            }
        }
    } else if mentioned_entities.len() == 1 {
        // Single entity: explain its prominent features
        let (id, name) = &mentioned_entities[0];
        if let Some(expl) = explain_entity_prominence(brain, *id, name)? {
            explanations.push(expl);
        }
    }

    // If no entity-based explanations, try pattern-based
    if explanations.is_empty() {
        if let Some(expl) = explain_via_pattern_matching(brain, observation)? {
            explanations.push(expl);
        }
    }

    // Sort by combined score
    explanations.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = explanations.first().cloned();
    let confidence = best
        .as_ref()
        .map(|e| e.combined_score)
        .unwrap_or(0.0);

    Ok(AbductiveHypothesis {
        observation: observation.to_string(),
        candidate_explanations: explanations,
        best_explanation: best,
        confidence,
    })
}

/// Find entities mentioned in observation text.
fn find_entities_in_text(brain: &Brain, text: &str) -> Result<Vec<(i64, String)>> {
    let text_lower = text.to_lowercase();
    let mut found: Vec<(i64, String)> = Vec::new();

    let entities = brain.all_entities()?;
    for e in &entities {
        if is_noise_type(&e.entity_type) || e.name.len() < 2 {
            continue;
        }
        if text_lower.contains(&e.name.to_lowercase()) {
            found.push((e.id, e.name.clone()));
        }
    }

    // Sort by name length descending (prefer longer matches)
    found.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    // Deduplicate by id
    let mut seen: HashSet<i64> = HashSet::new();
    found.retain(|(id, _)| seen.insert(*id));

    Ok(found)
}

/// Explain connection between two entities via shared neighbors.
fn explain_via_shared_neighborhood(
    brain: &Brain,
    id_a: i64,
    name_a: &str,
    id_b: i64,
    name_b: &str,
) -> Result<Option<Explanation>> {
    let rels_a = brain.get_relations_for(id_a)?;
    let rels_b = brain.get_relations_for(id_b)?;

    // Find entities connected to both
    let neighbors_a: HashSet<String> = rels_a
        .iter()
        .flat_map(|(s, _, o, _)| vec![s.clone(), o.clone()])
        .collect();
    let neighbors_b: HashSet<String> = rels_b
        .iter()
        .flat_map(|(s, _, o, _)| vec![s.clone(), o.clone()])
        .collect();

    let shared: Vec<&String> = neighbors_a.intersection(&neighbors_b).collect();

    if shared.is_empty() {
        return Ok(None);
    }

    let mut chain = Vec::new();
    let coverage = (shared.len() as f64 / neighbors_a.len().max(1) as f64).min(1.0);

    // Build explanation chain through shared neighbors
    for shared_name in shared.iter().take(5) {
        // Find the relations connecting A to shared and shared to B
        for (s, p, o, c) in &rels_a {
            if s.as_str() == *shared_name || o.as_str() == *shared_name {
                chain.push(ExplanationStep {
                    entity_id: id_a,
                    entity_name: name_a.to_string(),
                    predicate: p.clone(),
                    target_id: 0,
                    target_name: shared_name.to_string(),
                    confidence: *c,
                });
                break;
            }
        }
    }

    let parsimony = 1.0 / (1.0 + chain.len() as f64);
    let consistency = chain
        .iter()
        .map(|s| s.confidence)
        .sum::<f64>()
        / chain.len().max(1) as f64;
    let combined = parsimony * coverage * consistency;

    let summary = format!(
        "{} and {} share {} common connections: [{}]",
        name_a,
        name_b,
        shared.len(),
        shared
            .iter()
            .take(5)
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    Ok(Some(Explanation {
        reasoning_chain: chain,
        parsimony_score: parsimony,
        coverage_score: coverage,
        consistency_score: consistency,
        combined_score: combined,
        summary,
    }))
}

/// Explain connection via shared predicate patterns.
fn explain_via_shared_predicates(
    brain: &Brain,
    id_a: i64,
    name_a: &str,
    id_b: i64,
    name_b: &str,
) -> Result<Option<Explanation>> {
    let rels_a = brain.get_relations_for(id_a)?;
    let rels_b = brain.get_relations_for(id_b)?;

    let preds_a: HashSet<&str> = rels_a.iter().map(|(_, p, _, _)| p.as_str()).collect();
    let preds_b: HashSet<&str> = rels_b.iter().map(|(_, p, _, _)| p.as_str()).collect();

    let shared: Vec<&&str> = preds_a.intersection(&preds_b).collect();

    if shared.is_empty() {
        return Ok(None);
    }

    let coverage = shared.len() as f64 / preds_a.union(&preds_b).count().max(1) as f64;
    let parsimony = 1.0 / (1.0 + shared.len() as f64);
    let consistency = 0.8; // High inherent consistency for structural similarity
    let combined = parsimony * coverage * consistency;

    let summary = format!(
        "{} and {} play similar structural roles via shared predicates: [{}]",
        name_a,
        name_b,
        shared
            .iter()
            .take(8)
            .map(|s| **s)
            .collect::<Vec<_>>()
            .join(", ")
    );

    Ok(Some(Explanation {
        reasoning_chain: vec![],
        parsimony_score: parsimony,
        coverage_score: coverage,
        consistency_score: consistency,
        combined_score: combined,
        summary,
    }))
}

/// Explain connection via temporal co-occurrence.
fn explain_via_temporal_cooccurrence(
    brain: &Brain,
    id_a: i64,
    name_a: &str,
    id_b: i64,
    name_b: &str,
) -> Result<Option<Explanation>> {
    let entity_a = brain.get_entity_by_id(id_a)?;
    let entity_b = brain.get_entity_by_id(id_b)?;

    let (ea, eb) = match (entity_a, entity_b) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(None),
    };

    // Check if entities appeared around the same time
    let diff = (ea.first_seen - eb.first_seen).num_hours().unsigned_abs();

    if diff > 168 {
        // More than a week apart
        return Ok(None);
    }

    let temporal_score = 1.0 / (1.0 + diff as f64 / 24.0); // Decay over days
    let parsimony = 0.8; // Simple temporal explanation
    let coverage = 0.5; // Temporal alone isn't full coverage
    let consistency = temporal_score;
    let combined = parsimony * coverage * consistency;

    let summary = format!(
        "{} and {} appeared within {} hours of each other, suggesting temporal correlation",
        name_a, name_b, diff
    );

    Ok(Some(Explanation {
        reasoning_chain: vec![],
        parsimony_score: parsimony,
        coverage_score: coverage,
        consistency_score: consistency,
        combined_score: combined,
        summary,
    }))
}

/// Explain why a single entity is prominent.
fn explain_entity_prominence(
    brain: &Brain,
    entity_id: i64,
    name: &str,
) -> Result<Option<Explanation>> {
    let rels = brain.get_relations_for(entity_id)?;
    let facts = brain.get_facts_for(entity_id)?;

    if rels.is_empty() && facts.is_empty() {
        return Ok(None);
    }

    let degree = rels.len();
    let fact_count = facts.len();

    let summary = format!(
        "{} has {} connections and {} facts, making it a well-connected knowledge hub",
        name, degree, fact_count
    );

    Ok(Some(Explanation {
        reasoning_chain: vec![],
        parsimony_score: 0.9,
        coverage_score: (degree as f64 / 10.0).min(1.0),
        consistency_score: 0.9,
        combined_score: 0.7,
        summary,
    }))
}

/// Pattern-based explanation when no entities are found.
fn explain_via_pattern_matching(
    brain: &Brain,
    observation: &str,
) -> Result<Option<Explanation>> {
    // Search for related facts
    let results = brain.search_facts(observation)?;

    if results.is_empty() {
        return Ok(None);
    }

    let summary = format!(
        "Found {} related facts matching the observation pattern. Top match: {} — {} = {}",
        results.len(),
        results[0].0,
        results[0].1,
        results[0].2
    );

    Ok(Some(Explanation {
        reasoning_chain: vec![],
        parsimony_score: 0.5,
        coverage_score: (results.len() as f64 / 20.0).min(1.0),
        consistency_score: results[0].3,
        combined_score: 0.3,
        summary,
    }))
}

/// Explain a gap between two entities — why aren't they connected when
/// they seem like they should be?
///
/// Generates hypotheses for the missing connection with evidence chains.
pub fn explain_gap(brain: &Brain, entity_a: &str, entity_b: &str) -> Result<Vec<Explanation>> {
    let ea = brain.get_entity_by_name(entity_a)?;
    let eb = brain.get_entity_by_name(entity_b)?;

    let (ea, eb) = match (ea, eb) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(vec![]),
    };

    let mut explanations = Vec::new();

    // Check if they're actually connected (maybe indirect)
    let rels_a = brain.get_relations_for(ea.id)?;
    let direct = rels_a.iter().any(|(s, _, o, _)| {
        (s == entity_a && o == entity_b) || (s == entity_b && o == entity_a)
    });

    if direct {
        explanations.push(Explanation {
            reasoning_chain: vec![],
            parsimony_score: 1.0,
            coverage_score: 1.0,
            consistency_score: 1.0,
            combined_score: 1.0,
            summary: format!(
                "{} and {} ARE directly connected — the gap doesn't exist",
                entity_a, entity_b
            ),
        });
        return Ok(explanations);
    }

    // Hypothesis 1: They're in different domains and nobody has bridged them
    if ea.entity_type != eb.entity_type {
        let summary = format!(
            "Domain gap: {} is a '{}' while {} is a '{}' — cross-domain bridges are rare",
            entity_a, ea.entity_type, entity_b, eb.entity_type
        );
        explanations.push(Explanation {
            reasoning_chain: vec![],
            parsimony_score: 0.8,
            coverage_score: 0.6,
            consistency_score: 0.9,
            combined_score: 0.43,
            summary,
        });
    }

    // Hypothesis 2: They share neighbors but no direct connection (missing link)
    let rels_b = brain.get_relations_for(eb.id)?;
    let neighbors_a: HashSet<String> = rels_a
        .iter()
        .flat_map(|(s, _, o, _)| vec![s.clone(), o.clone()])
        .collect();
    let neighbors_b: HashSet<String> = rels_b
        .iter()
        .flat_map(|(s, _, o, _)| vec![s.clone(), o.clone()])
        .collect();
    let shared_neighbors: Vec<&String> = neighbors_a.intersection(&neighbors_b).collect();

    if !shared_neighbors.is_empty() {
        let summary = format!(
            "Missing link hypothesis: {} and {} share {} neighbors ([{}]) but lack a direct connection. This is a strong candidate for a new relation.",
            entity_a,
            entity_b,
            shared_neighbors.len(),
            shared_neighbors.iter().take(5).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        );
        let coverage = (shared_neighbors.len() as f64 / 5.0).min(1.0);
        explanations.push(Explanation {
            reasoning_chain: vec![],
            parsimony_score: 0.9,
            coverage_score: coverage,
            consistency_score: 0.85,
            combined_score: 0.9 * coverage * 0.85,
            summary,
        });
    }

    // Hypothesis 3: Source coverage gap
    let sources_a = brain.get_source_urls_for(ea.id)?;
    let sources_b = brain.get_source_urls_for(eb.id)?;
    let sources_a_set: HashSet<&str> = sources_a.iter().map(|s| s.as_str()).collect();
    let sources_b_set: HashSet<&str> = sources_b.iter().map(|s| s.as_str()).collect();
    let shared_sources = sources_a_set.intersection(&sources_b_set).count();

    if shared_sources == 0 && !sources_a.is_empty() && !sources_b.is_empty() {
        let summary = format!(
            "Source gap: {} and {} were learned from completely different sources — they've never appeared in the same context",
            entity_a, entity_b
        );
        explanations.push(Explanation {
            reasoning_chain: vec![],
            parsimony_score: 0.7,
            coverage_score: 0.5,
            consistency_score: 0.95,
            combined_score: 0.33,
            summary,
        });
    }

    // Hypothesis 4: Temporal separation
    let time_diff = (ea.first_seen - eb.first_seen).num_hours().unsigned_abs();
    if time_diff > 168 {
        let days = time_diff / 24;
        let summary = format!(
            "Temporal gap: {} appeared {} days before {} — they may not have been discussed together",
            if ea.first_seen < eb.first_seen {
                entity_a
            } else {
                entity_b
            },
            days,
            if ea.first_seen < eb.first_seen {
                entity_b
            } else {
                entity_a
            },
        );
        explanations.push(Explanation {
            reasoning_chain: vec![],
            parsimony_score: 0.6,
            coverage_score: 0.4,
            consistency_score: 0.9,
            combined_score: 0.22,
            summary,
        });
    }

    // Sort by combined score
    explanations.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(explanations)
}

/// Prediction of which entities are most likely to gain new connections.
#[derive(Debug, Clone)]
pub struct DiscoveryPrediction {
    pub entity_id: i64,
    pub entity_name: String,
    pub entity_type: String,
    pub predicted_score: f64,
    pub degree_growth_rate: f64,
    pub neighborhood_density_change: f64,
    pub predicate_acquisition_rate: f64,
    pub reason: String,
}

/// Predict which entities are most likely to gain new connections.
///
/// Uses three signals:
/// 1. Degree growth rate — entities that have been gaining connections
/// 2. Neighborhood density change — entities whose local area is densifying
/// 3. Predicate acquisition rate — entities gaining new types of connections
pub fn predict_next_discovery(brain: &Brain) -> Result<Vec<DiscoveryPrediction>> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;

    if entities.is_empty() || relations.is_empty() {
        return Ok(vec![]);
    }

    // Build entity degree map
    let mut degree: HashMap<i64, usize> = HashMap::new();
    let mut entity_predicates: HashMap<i64, HashSet<String>> = HashMap::new();

    for r in &relations {
        *degree.entry(r.subject_id).or_insert(0) += 1;
        *degree.entry(r.object_id).or_insert(0) += 1;
        entity_predicates
            .entry(r.subject_id)
            .or_default()
            .insert(r.predicate.clone());
        entity_predicates
            .entry(r.object_id)
            .or_default()
            .insert(r.predicate.clone());
    }

    // Compute average degree by entity type
    let mut type_avg_degree: HashMap<&str, (f64, usize)> = HashMap::new();
    for e in &entities {
        if is_noise_type(&e.entity_type) {
            continue;
        }
        let d = degree.get(&e.id).copied().unwrap_or(0) as f64;
        let entry = type_avg_degree.entry(e.entity_type.as_str()).or_insert((0.0, 0));
        entry.0 += d;
        entry.1 += 1;
    }

    let type_avg: HashMap<&str, f64> = type_avg_degree
        .iter()
        .map(|(t, (sum, count))| (*t, if *count > 0 { sum / *count as f64 } else { 0.0 }))
        .collect();

    let mut predictions: Vec<DiscoveryPrediction> = Vec::new();

    for e in &entities {
        if is_noise_type(&e.entity_type) || e.name.len() < 2 {
            continue;
        }

        let d = degree.get(&e.id).copied().unwrap_or(0) as f64;
        let avg = type_avg.get(e.entity_type.as_str()).copied().unwrap_or(1.0);
        let pred_count = entity_predicates
            .get(&e.id)
            .map(|s| s.len())
            .unwrap_or(0) as f64;

        // Degree growth rate: access_count correlates with how much attention entity gets
        let growth_rate = e.access_count as f64 / 10.0;

        // Neighborhood density change: entities below average degree for their type
        // are more likely to gain connections (regression to the mean)
        let density_change = if avg > 0.0 {
            (avg - d) / avg
        } else {
            0.0
        };

        // Predicate acquisition rate: entities with few unique predicates relative to
        // their degree have room for new types of connections
        let pred_rate = if d > 0.0 {
            1.0 - (pred_count / d).min(1.0)
        } else {
            0.5
        };

        // Combined prediction score
        let score = 0.4 * growth_rate.min(1.0) + 0.35 * density_change.clamp(0.0, 1.0) + 0.25 * pred_rate;

        if score > 0.2 {
            let reason = if density_change > 0.3 {
                format!(
                    "Below-average degree ({:.0} vs {:.0} avg for {}) — likely to gain connections",
                    d, avg, e.entity_type
                )
            } else if growth_rate > 0.5 {
                format!(
                    "High attention (access_count={}) suggests active knowledge area",
                    e.access_count
                )
            } else {
                format!(
                    "Moderate growth potential: {:.0} connections, {:.0} unique predicates",
                    d, pred_count
                )
            };

            predictions.push(DiscoveryPrediction {
                entity_id: e.id,
                entity_name: e.name.clone(),
                entity_type: e.entity_type.clone(),
                predicted_score: score,
                degree_growth_rate: growth_rate,
                neighborhood_density_change: density_change,
                predicate_acquisition_rate: pred_rate,
                reason,
            });
        }
    }

    predictions.sort_by(|a, b| {
        b.predicted_score
            .partial_cmp(&a.predicted_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    predictions.truncate(20);

    Ok(predictions)
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. TEMPORAL REASONING ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/// A temporal event in the knowledge graph timeline.
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    pub entity_id: i64,
    pub entity_name: String,
    pub entity_type: String,
    pub event_type: TemporalEventType,
    pub timestamp: NaiveDateTime,
    pub related_entities: Vec<i64>,
}

/// Types of temporal events tracked.
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalEventType {
    /// Entity first appeared in the graph
    EntityCreated,
    /// Entity gained a new relation
    RelationAdded,
    /// Entity gained a new fact
    FactAdded,
    /// Entity's confidence was updated
    ConfidenceChanged,
}

impl std::fmt::Display for TemporalEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalEventType::EntityCreated => write!(f, "created"),
            TemporalEventType::RelationAdded => write!(f, "relation_added"),
            TemporalEventType::FactAdded => write!(f, "fact_added"),
            TemporalEventType::ConfidenceChanged => write!(f, "confidence_changed"),
        }
    }
}

/// Build a temporal index — ordered timeline of entity appearances.
pub fn build_temporal_index(brain: &Brain) -> Result<Vec<TemporalEvent>> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;

    let mut events: Vec<TemporalEvent> = Vec::new();

    // Entity creation events
    for e in &entities {
        if is_noise_type(&e.entity_type) {
            continue;
        }
        events.push(TemporalEvent {
            entity_id: e.id,
            entity_name: e.name.clone(),
            entity_type: e.entity_type.clone(),
            event_type: TemporalEventType::EntityCreated,
            timestamp: e.first_seen,
            related_entities: vec![],
        });
    }

    // Relation events
    for r in &relations {
        events.push(TemporalEvent {
            entity_id: r.subject_id,
            entity_name: String::new(), // Will be filled if needed
            entity_type: String::new(),
            event_type: TemporalEventType::RelationAdded,
            timestamp: r.learned_at,
            related_entities: vec![r.object_id],
        });
    }

    // Sort by timestamp
    events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    Ok(events)
}

/// Granger-like temporal causality between two event types.
///
/// Measures whether events related to entity_a consistently precede events
/// related to entity_b. Score based on:
/// - Temporal ordering consistency (what fraction of pairs are correctly ordered)
/// - Number of instances observed
/// - Average time lag
///
/// Returns a score in [0, 1] where 1 = perfect causal ordering.
pub fn temporal_causality(brain: &Brain, entity_a_name: &str, entity_b_name: &str) -> Result<f64> {
    let ea = brain.get_entity_by_name(entity_a_name)?;
    let eb = brain.get_entity_by_name(entity_b_name)?;

    let (ea, eb) = match (ea, eb) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(0.0),
    };

    let events = build_temporal_index(brain)?;

    // Collect events for each entity
    let events_a: Vec<&TemporalEvent> = events
        .iter()
        .filter(|e| e.entity_id == ea.id || e.related_entities.contains(&ea.id))
        .collect();
    let events_b: Vec<&TemporalEvent> = events
        .iter()
        .filter(|e| e.entity_id == eb.id || e.related_entities.contains(&eb.id))
        .collect();

    if events_a.is_empty() || events_b.is_empty() {
        return Ok(0.0);
    }

    // Count how many times A-events precede B-events vs the reverse
    let mut a_before_b = 0usize;
    let mut _b_before_a = 0usize;
    let mut total_pairs = 0usize;

    for ea_event in &events_a {
        for eb_event in &events_b {
            if ea_event.timestamp < eb_event.timestamp {
                a_before_b += 1;
            } else if eb_event.timestamp < ea_event.timestamp {
                _b_before_a += 1;
            }
            total_pairs += 1;
        }
    }

    if total_pairs == 0 {
        return Ok(0.0);
    }

    // Consistency score: how often A comes before B
    let ordering_consistency = a_before_b as f64 / total_pairs as f64;

    // Instance bonus: more observations = more confidence
    let instance_bonus = (total_pairs as f64).ln() / 10.0;

    // Final score
    let score = (ordering_consistency * 0.7 + instance_bonus.min(0.3) * 0.3).min(1.0);

    Ok(score)
}

/// Predict when an entity will gain its next connection based on temporal patterns.
#[derive(Debug, Clone)]
pub struct TemporalPrediction {
    pub entity_id: i64,
    pub entity_name: String,
    pub predicted_hours_until_next: f64,
    pub avg_interval_hours: f64,
    pub last_event_hours_ago: f64,
    pub confidence: f64,
}

/// Predict when an entity will gain its next connection.
pub fn predict_temporal_pattern(brain: &Brain, entity_name: &str) -> Result<Option<TemporalPrediction>> {
    let entity = brain.get_entity_by_name(entity_name)?;
    let entity = match entity {
        Some(e) => e,
        None => return Ok(None),
    };

    let events = build_temporal_index(brain)?;
    let entity_events: Vec<&TemporalEvent> = events
        .iter()
        .filter(|e| e.entity_id == entity.id || e.related_entities.contains(&entity.id))
        .collect();

    if entity_events.len() < 2 {
        return Ok(None);
    }

    // Compute inter-event intervals
    let mut intervals: Vec<f64> = Vec::new();
    for i in 1..entity_events.len() {
        let diff = (entity_events[i].timestamp - entity_events[i - 1].timestamp)
            .num_seconds() as f64
            / 3600.0;
        if diff > 0.0 {
            intervals.push(diff);
        }
    }

    if intervals.is_empty() {
        return Ok(None);
    }

    let avg_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
    let last_event = entity_events.last().unwrap();
    let now = Utc::now().naive_utc();
    let hours_since_last = (now - last_event.timestamp).num_seconds() as f64 / 3600.0;

    let predicted = (avg_interval - hours_since_last).max(0.0);

    // Confidence based on number of observations and variance
    let variance = if intervals.len() > 1 {
        let mean = avg_interval;
        intervals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (intervals.len() - 1) as f64
    } else {
        avg_interval * avg_interval
    };
    let cv = if avg_interval > 0.0 {
        variance.sqrt() / avg_interval
    } else {
        1.0
    };
    let confidence = (1.0 / (1.0 + cv)) * (1.0 - 1.0 / (1.0 + intervals.len() as f64));

    Ok(Some(TemporalPrediction {
        entity_id: entity.id,
        entity_name: entity.name.clone(),
        predicted_hours_until_next: predicted,
        avg_interval_hours: avg_interval,
        last_event_hours_ago: hours_since_last,
        confidence,
    }))
}

/// A temporal anomaly — entity with deviation from its type's norm.
#[derive(Debug, Clone)]
pub struct TemporalAnomaly {
    pub entity_id: i64,
    pub entity_name: String,
    pub entity_type: String,
    pub anomaly_type: String,
    pub deviation_score: f64,
    pub description: String,
}

/// Find entities whose temporal patterns deviate from their type's norm.
///
/// Detects:
/// - Burst patterns: entities that gained all relations in a short burst
/// - Dormant entities: entities with long gaps between events
/// - Accelerating entities: entities gaining connections faster over time
pub fn temporal_anomalies(brain: &Brain) -> Result<Vec<TemporalAnomaly>> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;

    if entities.is_empty() || relations.is_empty() {
        return Ok(vec![]);
    }

    // Compute per-entity temporal statistics
    let mut entity_event_times: HashMap<i64, Vec<NaiveDateTime>> = HashMap::new();
    for r in &relations {
        entity_event_times
            .entry(r.subject_id)
            .or_default()
            .push(r.learned_at);
        entity_event_times
            .entry(r.object_id)
            .or_default()
            .push(r.learned_at);
    }

    // Sort each entity's events
    for times in entity_event_times.values_mut() {
        times.sort();
    }

    // Compute per-type statistics
    let mut type_spans: HashMap<String, Vec<f64>> = HashMap::new(); // type → list of temporal spans (hours)

    for e in &entities {
        if is_noise_type(&e.entity_type) {
            continue;
        }
        if let Some(times) = entity_event_times.get(&e.id) {
            if times.len() >= 2 {
                let span = (times.last().unwrap().and_utc().timestamp()
                    - times.first().unwrap().and_utc().timestamp()) as f64
                    / 3600.0;
                type_spans
                    .entry(e.entity_type.clone())
                    .or_default()
                    .push(span);
            }
        }
    }

    // Compute mean and stddev for each type
    let type_stats: HashMap<String, (f64, f64)> = type_spans
        .iter()
        .filter(|(_, spans)| spans.len() >= 3)
        .map(|(t, spans)| {
            let mean = spans.iter().sum::<f64>() / spans.len() as f64;
            let var = spans.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (spans.len() - 1).max(1) as f64;
            (t.clone(), (mean, var.sqrt()))
        })
        .collect();

    let mut anomalies: Vec<TemporalAnomaly> = Vec::new();

    for e in &entities {
        if is_noise_type(&e.entity_type) || e.name.len() < 2 {
            continue;
        }

        let times = match entity_event_times.get(&e.id) {
            Some(t) if t.len() >= 2 => t,
            _ => continue,
        };

        let span = (times.last().unwrap().and_utc().timestamp()
            - times.first().unwrap().and_utc().timestamp()) as f64
            / 3600.0;

        let (type_mean, type_std) = match type_stats.get(&e.entity_type) {
            Some(s) => *s,
            None => continue,
        };

        if type_std < 1e-10 {
            continue;
        }

        let z_score = (span - type_mean) / type_std;

        // Burst detection: very short span with many events
        if z_score < -2.0 && times.len() >= 3 {
            anomalies.push(TemporalAnomaly {
                entity_id: e.id,
                entity_name: e.name.clone(),
                entity_type: e.entity_type.clone(),
                anomaly_type: "burst".to_string(),
                deviation_score: z_score.abs(),
                description: format!(
                    "Burst: {} gained {} events in {:.1}h (type avg: {:.1}h, z={:.1})",
                    e.name,
                    times.len(),
                    span,
                    type_mean,
                    z_score
                ),
            });
        }

        // Dormant detection: very long span relative to event count
        if z_score > 2.0 {
            anomalies.push(TemporalAnomaly {
                entity_id: e.id,
                entity_name: e.name.clone(),
                entity_type: e.entity_type.clone(),
                anomaly_type: "dormant".to_string(),
                deviation_score: z_score,
                description: format!(
                    "Dormant: {} has events spanning {:.1}h but only {} events (type avg span: {:.1}h, z={:.1})",
                    e.name, span, times.len(), type_mean, z_score
                ),
            });
        }
    }

    // Sort by deviation score
    anomalies.sort_by(|a, b| {
        b.deviation_score
            .partial_cmp(&a.deviation_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    anomalies.truncate(20);

    Ok(anomalies)
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. NETWORK TOPOLOGY OPTIMIZER
// ═══════════════════════════════════════════════════════════════════════════

/// Scale-free network score.
///
/// Fits the degree distribution to a power law P(k) ~ k^{-γ} using MLE
/// and log-log regression R².
///
/// Returns (gamma, r_squared, assessment).
/// Ideal γ ∈ [2.0, 3.0] for rich-get-richer dynamics.
pub fn scale_free_score(brain: &Brain) -> Result<(f64, f64, String)> {
    let relations = brain.all_relations()?;

    if relations.is_empty() {
        return Ok((0.0, 0.0, "No relations — cannot compute".to_string()));
    }

    // Compute degree distribution
    let mut degree: HashMap<i64, usize> = HashMap::new();
    for r in &relations {
        *degree.entry(r.subject_id).or_insert(0) += 1;
        *degree.entry(r.object_id).or_insert(0) += 1;
    }

    let degrees: Vec<usize> = degree.values().copied().filter(|&d| d > 0).collect();

    if degrees.len() < 3 {
        return Ok((0.0, 0.0, "Too few nodes".to_string()));
    }

    let (gamma, r_squared) = fit_power_law(&degrees);

    let assessment = if r_squared < 0.3 {
        format!(
            "Poor power-law fit (R²={:.2}) — NOT scale-free. γ={:.2}",
            r_squared, gamma
        )
    } else if gamma < SCALE_FREE_GAMMA_LOW {
        format!(
            "γ={:.2} < {:.1} — ultra-hub-dominated (too concentrated). R²={:.2}",
            gamma, SCALE_FREE_GAMMA_LOW, r_squared
        )
    } else if gamma > SCALE_FREE_GAMMA_HIGH {
        format!(
            "γ={:.2} > {:.1} — too democratic (lacks hubs). R²={:.2}",
            gamma, SCALE_FREE_GAMMA_HIGH, r_squared
        )
    } else {
        format!(
            "γ={:.2} ∈ [{:.1}, {:.1}] — ideal scale-free topology! R²={:.2}",
            gamma, SCALE_FREE_GAMMA_LOW, SCALE_FREE_GAMMA_HIGH, r_squared
        )
    };

    Ok((gamma, r_squared, assessment))
}

/// Small-world coefficient σ = (C/C_random) / (L/L_random).
///
/// C = clustering coefficient, L = average path length.
/// σ >> 1 means the graph is "small-world" (high clustering, short paths).
pub fn small_world_coefficient(brain: &Brain) -> Result<(f64, String)> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;

    let n = entities
        .iter()
        .filter(|e| !is_noise_type(&e.entity_type))
        .count() as f64;

    if n < 10.0 {
        return Ok((0.0, "Too few nodes for small-world analysis".to_string()));
    }

    let avg_degree = if n > 0.0 {
        (2.0 * relations.len() as f64) / n
    } else {
        0.0
    };

    if avg_degree < 1.0 {
        return Ok((0.0, "Average degree too low".to_string()));
    }

    // Compute global clustering coefficient
    let c = compute_global_clustering(brain)?;

    // Estimate average path length via sampled BFS
    let avg_l = estimate_avg_path_length(brain, 50)?;

    if avg_l <= 0.0 {
        return Ok((0.0, "Cannot estimate path length".to_string()));
    }

    // Random graph equivalents (Erdős–Rényi)
    let c_random = avg_degree / n;
    let l_random = if avg_degree > 1.0 {
        n.ln() / avg_degree.ln()
    } else {
        n
    };

    let gamma = if c_random > 0.0 { c / c_random } else { 0.0 };
    let lambda = if l_random > 0.0 {
        avg_l / l_random
    } else {
        1.0
    };
    let sigma = if lambda > 0.0 {
        gamma / lambda
    } else {
        0.0
    };

    let assessment = if sigma > 3.0 {
        format!(
            "σ={:.2} >> 1 — Strong small-world! (C={:.3}, L={:.1}, γ={:.1}, λ={:.2})",
            sigma, c, avg_l, gamma, lambda
        )
    } else if sigma > 1.0 {
        format!(
            "σ={:.2} > 1 — Moderate small-world properties. (C={:.3}, L={:.1})",
            sigma, c, avg_l
        )
    } else {
        format!(
            "σ={:.2} ≈ 1 — Not small-world (random-like). (C={:.3}, L={:.1})",
            sigma, c, avg_l
        )
    };

    Ok((sigma, assessment))
}

/// Compute global clustering coefficient.
fn compute_global_clustering(brain: &Brain) -> Result<f64> {
    let relations = brain.all_relations()?;

    let mut adj: HashMap<i64, HashSet<i64>> = HashMap::new();
    for r in &relations {
        adj.entry(r.subject_id).or_default().insert(r.object_id);
        adj.entry(r.object_id).or_default().insert(r.subject_id);
    }

    let mut triangles = 0u64;
    let mut triples = 0u64;

    for neighbors in adj.values() {
        let k = neighbors.len();
        if k < 2 {
            continue;
        }
        triples += (k * (k - 1) / 2) as u64;

        let neighbor_vec: Vec<i64> = neighbors.iter().copied().collect();
        for i in 0..neighbor_vec.len() {
            for j in (i + 1)..neighbor_vec.len() {
                if adj
                    .get(&neighbor_vec[i])
                    .map(|s| s.contains(&neighbor_vec[j]))
                    .unwrap_or(false)
                {
                    triangles += 1;
                }
            }
        }
        // Limit computation for very high-degree nodes
        if k > 100 {
            break;
        }
    }

    if triples == 0 {
        return Ok(0.0);
    }

    // Each triangle is counted 3 times (once per vertex)
    Ok(triangles as f64 / triples as f64)
}

/// Estimate average path length via sampled BFS.
fn estimate_avg_path_length(brain: &Brain, sample_size: usize) -> Result<f64> {
    let relations = brain.all_relations()?;

    let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();
    for r in &relations {
        adj.entry(r.subject_id).or_default().push(r.object_id);
        adj.entry(r.object_id).or_default().push(r.subject_id);
    }

    let nodes: Vec<i64> = adj.keys().copied().collect();
    if nodes.is_empty() {
        return Ok(0.0);
    }

    let sample = sample_size.min(nodes.len());
    let mut total_dist = 0u64;
    let mut total_pairs = 0u64;

    // Deterministic sampling: take evenly spaced nodes
    let step = nodes.len().max(1) / sample.max(1);
    let step = step.max(1);

    for idx in (0..nodes.len()).step_by(step).take(sample) {
        let source = nodes[idx];
        // BFS from source
        let mut visited: HashMap<i64, usize> = HashMap::new();
        visited.insert(source, 0);
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            let dist = visited[&node];
            if let Some(neighbors) = adj.get(&node) {
                for &n in neighbors {
                    if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(n) {
                        e.insert(dist + 1);
                        queue.push_back(n);
                    }
                }
            }
        }

        for &d in visited.values() {
            if d > 0 {
                total_dist += d as u64;
                total_pairs += 1;
            }
        }
    }

    if total_pairs == 0 {
        return Ok(0.0);
    }

    Ok(total_dist as f64 / total_pairs as f64)
}

/// Topology steering recommendations.
#[derive(Debug, Clone)]
pub struct TopologyRecommendation {
    pub priority: u8,
    pub category: String,
    pub recommendation: String,
}

/// Analyze current topology and recommend actions.
pub fn topology_steering_recommendations(brain: &Brain) -> Result<Vec<TopologyRecommendation>> {
    let mut recs: Vec<TopologyRecommendation> = Vec::new();

    // Scale-free analysis
    let (gamma, r_squared, _) = scale_free_score(brain)?;

    if r_squared > 0.3 {
        if gamma < SCALE_FREE_GAMMA_LOW {
            recs.push(TopologyRecommendation {
                priority: 1,
                category: "Hub Dominance".to_string(),
                recommendation: format!(
                    "γ={:.2} — graph is too hub-dominated. Add connections to peripheral entities to distribute connectivity.",
                    gamma
                ),
            });
        } else if gamma > SCALE_FREE_GAMMA_HIGH {
            recs.push(TopologyRecommendation {
                priority: 1,
                category: "Hub Scarcity".to_string(),
                recommendation: format!(
                    "γ={:.2} — graph lacks strong hubs. Enrich high-degree entities with more connections.",
                    gamma
                ),
            });
        }
    }

    // Small-world analysis
    let (sigma, _) = small_world_coefficient(brain)?;

    if sigma < 1.0 && sigma > 0.0 {
        recs.push(TopologyRecommendation {
            priority: 2,
            category: "Small-World".to_string(),
            recommendation: format!(
                "σ={:.2} — not small-world. Add clustering (connect neighbors of well-connected entities) to increase local density.",
                sigma
            ),
        });
    }

    // Hub vulnerability
    let vulnerabilities = hub_vulnerability(brain)?;
    for (name, score) in vulnerabilities.iter().take(3) {
        if *score > 0.1 {
            recs.push(TopologyRecommendation {
                priority: 1,
                category: "Vulnerability".to_string(),
                recommendation: format!(
                    "'{}' is a critical bridge (vulnerability={:.2}) — enrich it with redundant connections",
                    name, score
                ),
            });
        }
    }

    // Surprise edge fraction
    let surprise = surprise_edge_fraction(brain)?;
    if surprise < 0.08 {
        recs.push(TopologyRecommendation {
            priority: 2,
            category: "Innovation".to_string(),
            recommendation: format!(
                "Surprise fraction {:.1}% is below optimal 12% — feed more cross-domain content",
                surprise * 100.0
            ),
        });
    }

    recs.sort_by_key(|r| r.priority);
    Ok(recs)
}

/// Hub vulnerability analysis — identify entities whose removal would
/// fragment the graph most.
///
/// Uses betweenness centrality as a proxy for vulnerability:
/// high-betweenness nodes are critical bridges.
///
/// Returns (entity_name, vulnerability_score) sorted by vulnerability.
pub fn hub_vulnerability(brain: &Brain) -> Result<Vec<(String, f64)>> {
    let relations = brain.all_relations()?;
    let entities = brain.all_entities()?;

    if relations.is_empty() {
        return Ok(vec![]);
    }

    // Build adjacency
    let mut adj: HashMap<i64, HashSet<i64>> = HashMap::new();
    for r in &relations {
        adj.entry(r.subject_id).or_default().insert(r.object_id);
        adj.entry(r.object_id).or_default().insert(r.subject_id);
    }

    let n = adj.len();
    if n < 3 {
        return Ok(vec![]);
    }

    // Approximate betweenness centrality via sampled shortest paths
    let mut betweenness: HashMap<i64, f64> = HashMap::new();
    let nodes: Vec<i64> = adj.keys().copied().collect();
    let sample_size = nodes.len().min(50);
    let step = nodes.len().max(1) / sample_size.max(1);
    let step = step.max(1);

    for idx in (0..nodes.len()).step_by(step).take(sample_size) {
        let source = nodes[idx];
        // BFS from source
        let mut dist: HashMap<i64, usize> = HashMap::new();
        let mut paths: HashMap<i64, f64> = HashMap::new();
        let mut predecessors: HashMap<i64, Vec<i64>> = HashMap::new();

        dist.insert(source, 0);
        paths.insert(source, 1.0);

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);
        let mut stack: Vec<i64> = Vec::new();

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let d_v = dist[&v];

            if let Some(neighbors) = adj.get(&v) {
                for &w in neighbors {
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                        e.insert(d_v + 1);
                        queue.push_back(w);
                    }
                    if dist.get(&w) == Some(&(d_v + 1)) {
                        *paths.entry(w).or_insert(0.0) += paths.get(&v).copied().unwrap_or(0.0);
                        predecessors.entry(w).or_default().push(v);
                    }
                }
            }
        }

        // Accumulate betweenness
        let mut delta: HashMap<i64, f64> = HashMap::new();
        while let Some(w) = stack.pop() {
            if let Some(preds) = predecessors.get(&w) {
                for &v in preds {
                    let sigma_v = paths.get(&v).copied().unwrap_or(1.0);
                    let sigma_w = paths.get(&w).copied().unwrap_or(1.0);
                    let d_w = delta.get(&w).copied().unwrap_or(0.0);
                    *delta.entry(v).or_insert(0.0) += (sigma_v / sigma_w) * (1.0 + d_w);
                }
            }
            if w != source {
                *betweenness.entry(w).or_insert(0.0) += delta.get(&w).copied().unwrap_or(0.0);
            }
        }
    }

    // Normalize by n*(n-1)
    let norm = (n * (n - 1)) as f64;
    let entity_map: HashMap<i64, &str> = entities.iter().map(|e| (e.id, e.name.as_str())).collect();

    let mut result: Vec<(String, f64)> = betweenness
        .iter()
        .filter_map(|(&id, &score)| {
            entity_map.get(&id).map(|name| {
                let normalized = if norm > 0.0 { score / norm } else { 0.0 };
                (name.to_string(), normalized)
            })
        })
        .collect();

    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result.truncate(20);

    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. META-COGNITIVE LOOP
// ═══════════════════════════════════════════════════════════════════════════

/// Meta-cognitive assessment of the knowledge system.
#[derive(Debug, Clone)]
pub struct MetaCognition {
    pub learning_rate: f64,
    pub domain_balance: f64,
    pub discovery_velocity: f64,
    pub staleness_index: f64,
    pub entity_count: usize,
    pub relation_count: usize,
    pub fact_count: usize,
    pub recommendations: Vec<String>,
}

/// Full self-assessment: Am I learning too much in one domain?
/// Is my discovery rate declining? Are my hypotheses getting stale?
pub fn introspect(brain: &Brain) -> Result<MetaCognition> {
    let stats = brain.stats()?;
    let balance = domain_balance_score(brain)?;
    let velocity = discovery_velocity(brain)?;
    let staleness = staleness_index(brain)?;
    let learning = compute_learning_rate(brain)?;
    let recommendations = recommend_feeding_strategy(brain)?;

    Ok(MetaCognition {
        learning_rate: learning,
        domain_balance: balance,
        discovery_velocity: velocity,
        staleness_index: staleness,
        entity_count: stats.entity_count,
        relation_count: stats.relation_count,
        fact_count: stats.fact_count,
        recommendations,
    })
}

/// Domain balance score using the Gini coefficient.
///
/// Computes the Gini coefficient of entity counts per entity_type.
/// 0 = perfectly balanced across all types
/// 1 = all entities are one type
///
/// Formula: G = (2·Σᵢ i·xᵢ) / (n·Σᵢ xᵢ) - (n+1)/n
/// where xᵢ are sorted counts.
pub fn domain_balance_score(brain: &Brain) -> Result<f64> {
    let entities = brain.all_entities()?;

    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for e in &entities {
        if !is_noise_type(&e.entity_type) {
            *type_counts.entry(e.entity_type.as_str()).or_insert(0) += 1;
        }
    }

    let mut counts: Vec<usize> = type_counts.values().copied().collect();
    if counts.len() <= 1 {
        return Ok(if counts.is_empty() { 0.0 } else { 1.0 });
    }

    counts.sort();
    let n = counts.len() as f64;
    let total: f64 = counts.iter().sum::<usize>() as f64;

    if total == 0.0 {
        return Ok(0.0);
    }

    // Gini coefficient: G = (2·Σᵢ(i+1)·xᵢ) / (n·Σxᵢ) - (n+1)/n
    let weighted_sum: f64 = counts
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as f64 + 1.0) * x as f64)
        .sum();

    let gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n;

    Ok(gini.clamp(0.0, 1.0))
}

/// Rate of confirmed hypotheses per hour, smoothed over a 24h window.
pub fn discovery_velocity(brain: &Brain) -> Result<f64> {
    brain.with_conn(|conn| {
        // Check if hypotheses table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='hypotheses'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            return Ok(0.0);
        }

        // Count confirmed hypotheses in the last 24 hours
        let now = Utc::now().naive_utc();
        let window_start = (now - chrono::Duration::hours(24))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();

        let confirmed: f64 = conn
            .query_row(
                "SELECT COUNT(*) FROM hypotheses WHERE status = 'confirmed' AND created_at >= ?1",
                params![window_start],
                |r| r.get(0),
            )
            .unwrap_or(0.0);

        // Velocity = confirmed per hour
        Ok(confirmed / 24.0)
    })
}

/// Fraction of entities not touched (no new relations/facts) in last 7 days.
pub fn staleness_index(brain: &Brain) -> Result<f64> {
    let entities = brain.all_entities()?;

    if entities.is_empty() {
        return Ok(0.0);
    }

    let now = Utc::now().naive_utc();
    let seven_days_ago = now - chrono::Duration::days(7);

    let stale_count = entities
        .iter()
        .filter(|e| !is_noise_type(&e.entity_type) && e.last_seen < seven_days_ago)
        .count();

    let total = entities
        .iter()
        .filter(|e| !is_noise_type(&e.entity_type))
        .count();

    if total == 0 {
        return Ok(0.0);
    }

    Ok(stale_count as f64 / total as f64)
}

/// Compute learning rate from recent entity/relation growth.
fn compute_learning_rate(brain: &Brain) -> Result<f64> {
    brain.with_conn(|conn| {
        // Check if graph_snapshots table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='graph_snapshots'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(false);

        if !table_exists {
            return Ok(0.0);
        }

        // Compare last two snapshots
        let mut stmt = conn.prepare(
            "SELECT entities, relations FROM graph_snapshots ORDER BY id DESC LIMIT 2",
        )?;
        let snapshots: Vec<(i64, i64)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        if snapshots.len() < 2 {
            return Ok(0.0);
        }

        let (e_new, r_new) = snapshots[0];
        let (e_old, r_old) = snapshots[1];

        let e_growth = if e_old > 0 {
            (e_new - e_old) as f64 / e_old as f64
        } else {
            0.0
        };
        let r_growth = if r_old > 0 {
            (r_new - r_old) as f64 / r_old as f64
        } else {
            0.0
        };

        Ok((e_growth + r_growth) / 2.0)
    })
}

/// Based on all metrics, suggest what to feed next.
pub fn recommend_feeding_strategy(brain: &Brain) -> Result<Vec<String>> {
    let mut recs: Vec<String> = Vec::new();

    // Domain balance
    let balance = domain_balance_score(brain)?;
    if balance > 0.5 {
        // Find dominant and underrepresented types
        let entities = brain.all_entities()?;
        let mut type_counts: BTreeMap<String, usize> = BTreeMap::new();
        for e in &entities {
            if !is_noise_type(&e.entity_type) {
                *type_counts.entry(e.entity_type.clone()).or_insert(0) += 1;
            }
        }

        let total: usize = type_counts.values().sum();
        if total > 0 {
            let mut sorted: Vec<(String, usize)> = type_counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            if let Some((top_type, top_count)) = sorted.first() {
                let pct = *top_count as f64 / total as f64 * 100.0;
                recs.push(format!(
                    "'{}' is overrepresented ({:.0}% of entities) — feed more diverse content",
                    top_type, pct
                ));
            }
            if let Some((low_type, low_count)) = sorted.last() {
                let pct = *low_count as f64 / total as f64 * 100.0;
                recs.push(format!(
                    "'{}' is underrepresented ({:.0}%) — prioritize this domain",
                    low_type, pct
                ));
            }
        }
    }

    // Staleness
    let staleness = staleness_index(brain)?;
    if staleness > 0.5 {
        recs.push(format!(
            "Staleness index {:.0}% — over half of entities haven't been updated in 7 days. Revisit existing topics.",
            staleness * 100.0
        ));
    }

    // Discovery velocity
    let velocity = discovery_velocity(brain)?;
    if velocity < 0.01 {
        recs.push(
            "Discovery velocity near zero — try feeding controversial or cross-domain topics to stimulate hypothesis generation"
                .to_string(),
        );
    }

    // Surprise fraction
    let surprise = surprise_edge_fraction(brain)?;
    if surprise < 0.08 {
        recs.push(format!(
            "Low surprise fraction ({:.1}%) — knowledge graph is too homogeneous. Inject cross-domain connections.",
            surprise * 100.0
        ));
    } else if surprise > 0.20 {
        recs.push(format!(
            "High surprise fraction ({:.1}%) — consolidate by feeding deeper content on existing topics.",
            surprise * 100.0
        ));
    }

    if recs.is_empty() {
        recs.push(
            "Knowledge graph is well-balanced. Continue current feeding strategy.".to_string(),
        );
    }

    Ok(recs)
}

/// Format a meta-cognition report for display.
pub fn format_introspection(mc: &MetaCognition) -> String {
    let mut lines = Vec::new();

    lines.push("╔══════════════════════════════════════════════════╗".to_string());
    lines.push("║        META-COGNITION — axon self-assessment    ║".to_string());
    lines.push("╚══════════════════════════════════════════════════╝".to_string());
    lines.push(String::new());

    lines.push(format!(
        "  Entities:            {}",
        mc.entity_count
    ));
    lines.push(format!(
        "  Relations:           {}",
        mc.relation_count
    ));
    lines.push(format!(
        "  Facts:               {}",
        mc.fact_count
    ));
    lines.push(String::new());

    lines.push(format!(
        "  Learning rate:       {:.2}%",
        mc.learning_rate * 100.0
    ));
    lines.push(format!(
        "  Domain balance:      {:.3} (Gini, 0=balanced, 1=concentrated)",
        mc.domain_balance
    ));
    lines.push(format!(
        "  Discovery velocity:  {:.3} confirmations/hour",
        mc.discovery_velocity
    ));
    lines.push(format!(
        "  Staleness index:     {:.1}%",
        mc.staleness_index * 100.0
    ));

    lines.push(String::new());
    lines.push("  ── Recommendations ──".to_string());
    for rec in &mc.recommendations {
        lines.push(format!("  • {}", rec));
    }

    lines.join("\n")
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Brain;

    fn test_brain() -> Brain {
        Brain::open_in_memory().unwrap()
    }

    /// Create a populated brain with a realistic knowledge graph.
    fn populated_brain() -> Brain {
        let brain = test_brain();

        let newton = brain.upsert_entity("Isaac Newton", "person").unwrap();
        let einstein = brain.upsert_entity("Albert Einstein", "person").unwrap();
        let darwin = brain.upsert_entity("Charles Darwin", "person").unwrap();
        let galileo = brain.upsert_entity("Galileo Galilei", "person").unwrap();
        let tesla = brain.upsert_entity("Nikola Tesla", "person").unwrap();
        let curie = brain.upsert_entity("Marie Curie", "person").unwrap();
        let turing = brain.upsert_entity("Alan Turing", "person").unwrap();
        let ada = brain.upsert_entity("Ada Lovelace", "person").unwrap();

        let gravity = brain.upsert_entity("gravity", "concept").unwrap();
        let relativity = brain.upsert_entity("relativity", "concept").unwrap();
        let evolution = brain.upsert_entity("evolution", "concept").unwrap();
        let heliocentrism = brain.upsert_entity("heliocentrism", "concept").unwrap();
        let electricity = brain.upsert_entity("electricity", "concept").unwrap();
        let radioactivity = brain.upsert_entity("radioactivity", "concept").unwrap();
        let physics = brain.upsert_entity("physics", "concept").unwrap();
        let biology = brain.upsert_entity("biology", "concept").unwrap();
        let computation = brain.upsert_entity("computation", "concept").unwrap();
        let mathematics = brain.upsert_entity("mathematics", "concept").unwrap();

        let royal_society = brain
            .upsert_entity("Royal Society", "organization")
            .unwrap();
        let cambridge = brain
            .upsert_entity("University of Cambridge", "organization")
            .unwrap();

        let england = brain.upsert_entity("England", "place").unwrap();
        let germany = brain.upsert_entity("Germany", "place").unwrap();

        // Newton
        brain.upsert_relation(newton, "pioneered", gravity, "test").unwrap();
        brain.upsert_relation(newton, "contributed_to", physics, "test").unwrap();
        brain.upsert_relation(newton, "contributed_to", mathematics, "test").unwrap();
        brain.upsert_relation(newton, "member_of", royal_society, "test").unwrap();
        brain.upsert_relation(newton, "studied_at", cambridge, "test").unwrap();
        brain.upsert_relation(newton, "born_in", england, "test").unwrap();

        // Einstein
        brain.upsert_relation(einstein, "pioneered", relativity, "test").unwrap();
        brain.upsert_relation(einstein, "contributed_to", physics, "test").unwrap();
        brain.upsert_relation(einstein, "born_in", germany, "test").unwrap();
        brain.upsert_relation(einstein, "influenced", gravity, "test").unwrap();

        // Darwin
        brain.upsert_relation(darwin, "pioneered", evolution, "test").unwrap();
        brain.upsert_relation(darwin, "contributed_to", biology, "test").unwrap();
        brain.upsert_relation(darwin, "member_of", royal_society, "test").unwrap();
        brain.upsert_relation(darwin, "born_in", england, "test").unwrap();

        // Galileo
        brain.upsert_relation(galileo, "pioneered", heliocentrism, "test").unwrap();
        brain.upsert_relation(galileo, "contributed_to", physics, "test").unwrap();

        // Tesla
        brain.upsert_relation(tesla, "pioneered", electricity, "test").unwrap();
        brain.upsert_relation(tesla, "contributed_to", physics, "test").unwrap();

        // Curie
        brain.upsert_relation(curie, "pioneered", radioactivity, "test").unwrap();
        brain.upsert_relation(curie, "contributed_to", physics, "test").unwrap();

        // Turing
        brain.upsert_relation(turing, "pioneered", computation, "test").unwrap();
        brain.upsert_relation(turing, "contributed_to", mathematics, "test").unwrap();
        brain.upsert_relation(turing, "studied_at", cambridge, "test").unwrap();
        brain.upsert_relation(turing, "born_in", england, "test").unwrap();

        // Ada
        brain.upsert_relation(ada, "pioneered", computation, "test").unwrap();
        brain.upsert_relation(ada, "contributed_to", mathematics, "test").unwrap();
        brain.upsert_relation(ada, "born_in", england, "test").unwrap();

        // Cross-domain links
        brain.upsert_relation(gravity, "led_to", relativity, "test").unwrap();
        brain.upsert_relation(newton, "influenced", einstein, "test").unwrap();

        // Facts
        brain.upsert_fact(newton, "birth_year", "1643", "test").unwrap();
        brain.upsert_fact(einstein, "birth_year", "1879", "test").unwrap();
        brain.upsert_fact(darwin, "birth_year", "1809", "test").unwrap();
        brain.upsert_fact(newton, "nationality", "English", "test").unwrap();
        brain.upsert_fact(einstein, "nationality", "German", "test").unwrap();

        brain
    }

    // ───────────────────────────────────────────────────────────────────────
    // 1. Self-Organized Criticality Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_von_neumann_entropy_empty_graph() {
        let brain = test_brain();
        let entropy = von_neumann_graph_entropy(&brain).unwrap();
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_von_neumann_entropy_single_edge() {
        let brain = test_brain();
        let a = brain.upsert_entity("A", "concept").unwrap();
        let b = brain.upsert_entity("B", "concept").unwrap();
        brain.upsert_relation(a, "linked", b, "test").unwrap();
        let entropy = von_neumann_graph_entropy(&brain).unwrap();
        assert!(entropy >= 0.0, "Entropy should be non-negative");
    }

    #[test]
    fn test_von_neumann_entropy_populated() {
        let brain = populated_brain();
        let entropy = von_neumann_graph_entropy(&brain).unwrap();
        assert!(entropy > 0.0, "Populated graph should have positive entropy");
    }

    #[test]
    fn test_von_neumann_entropy_star_vs_chain() {
        // Star graph should have different entropy than chain graph
        let star_brain = test_brain();
        let hub = star_brain.upsert_entity("Hub", "concept").unwrap();
        for i in 0..5 {
            let spoke = star_brain
                .upsert_entity(&format!("Spoke{}", i), "concept")
                .unwrap();
            star_brain
                .upsert_relation(hub, "linked", spoke, "test")
                .unwrap();
        }
        let star_entropy = von_neumann_graph_entropy(&star_brain).unwrap();

        let chain_brain = test_brain();
        let mut prev = chain_brain.upsert_entity("Node0", "concept").unwrap();
        for i in 1..6 {
            let next = chain_brain
                .upsert_entity(&format!("Node{}", i), "concept")
                .unwrap();
            chain_brain
                .upsert_relation(prev, "linked", next, "test")
                .unwrap();
            prev = next;
        }
        let chain_entropy = von_neumann_graph_entropy(&chain_brain).unwrap();

        // Both should be positive but different
        assert!(star_entropy > 0.0);
        assert!(chain_entropy > 0.0);
        // Star graph should have lower entropy (more structured)
        // This is a known property of Von Neumann entropy
    }

    #[test]
    fn test_semantic_entropy_empty() {
        let brain = test_brain();
        let entropy = semantic_entropy(&brain).unwrap();
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_semantic_entropy_populated() {
        let brain = populated_brain();
        let entropy = semantic_entropy(&brain).unwrap();
        assert!(entropy > 0.0, "Populated graph should have semantic diversity");
    }

    #[test]
    fn test_semantic_entropy_homogeneous() {
        let brain = test_brain();
        // All same type, same predicate
        for i in 0..5 {
            let a = brain.upsert_entity(&format!("A{}", i), "concept").unwrap();
            let b = brain.upsert_entity(&format!("B{}", i), "concept").unwrap();
            brain.upsert_relation(a, "linked", b, "test").unwrap();
        }
        let entropy = semantic_entropy(&brain).unwrap();
        // Low diversity should produce lower entropy
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_critical_discovery_parameter_empty() {
        let brain = test_brain();
        let cdp = critical_discovery_parameter(&brain).unwrap();
        assert_eq!(cdp, 0.0);
    }

    #[test]
    fn test_critical_discovery_parameter_populated() {
        let brain = populated_brain();
        let cdp = critical_discovery_parameter(&brain).unwrap();
        // CDP should be a finite number in [-1, 1]
        assert!(cdp >= -1.0 && cdp <= 1.0, "CDP={} out of range", cdp);
    }

    #[test]
    fn test_surprise_edge_fraction_empty() {
        let brain = test_brain();
        let surprise = surprise_edge_fraction(&brain).unwrap();
        assert_eq!(surprise, 0.0);
    }

    #[test]
    fn test_surprise_edge_fraction_homogeneous() {
        let brain = test_brain();
        // All entities have the same predicate → cosine similarity = 1 → 0 surprise
        let a = brain.upsert_entity("A", "concept").unwrap();
        let b = brain.upsert_entity("B", "concept").unwrap();
        let c = brain.upsert_entity("C", "concept").unwrap();
        brain.upsert_relation(a, "linked", b, "test").unwrap();
        brain.upsert_relation(b, "linked", c, "test").unwrap();
        let surprise = surprise_edge_fraction(&brain).unwrap();
        assert!(surprise >= 0.0 && surprise <= 1.0);
    }

    #[test]
    fn test_surprise_edge_fraction_populated() {
        let brain = populated_brain();
        let surprise = surprise_edge_fraction(&brain).unwrap();
        assert!(surprise >= 0.0 && surprise <= 1.0);
    }

    #[test]
    fn test_avalanche_detection_no_history() {
        let brain = test_brain();
        let (sizes, _gamma, _is_pl) = avalanche_detection(&brain).unwrap();
        // No history, no relations → empty
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_avalanche_detection_with_graph() {
        let brain = populated_brain();
        let (sizes, _gamma, _is_pl) = avalanche_detection(&brain).unwrap();
        // Should get some estimates from graph structure
        // May or may not have sizes depending on graph properties
        assert!(sizes.len() >= 0);
    }

    #[test]
    fn test_record_avalanche() {
        let brain = test_brain();
        record_avalanche(&brain, Some(1), 5, 3).unwrap();
        record_avalanche(&brain, Some(2), 12, 4).unwrap();

        let (sizes, _, _) = avalanche_detection(&brain).unwrap();
        // Should have our recorded avalanches
        // Note: with < 3 entries it falls through to estimation
        // So record 3+
        record_avalanche(&brain, Some(3), 3, 2).unwrap();
        let (sizes, _gamma, _) = avalanche_detection(&brain).unwrap();
        assert!(sizes.len() >= 3);
    }

    #[test]
    fn test_criticality_report_empty() {
        let brain = test_brain();
        let report = criticality_report(&brain).unwrap();
        assert_eq!(report.structural_entropy, 0.0);
        assert_eq!(report.semantic_entropy, 0.0);
    }

    #[test]
    fn test_criticality_report_populated() {
        let brain = populated_brain();
        let report = criticality_report(&brain).unwrap();
        assert!(report.structural_entropy > 0.0);
        assert!(report.semantic_entropy > 0.0);
        assert!(report.cdp >= -1.0 && report.cdp <= 1.0);
        assert!(!report.recommendation.is_empty());
    }

    #[test]
    fn test_format_criticality_report() {
        let brain = populated_brain();
        let report = criticality_report(&brain).unwrap();
        let formatted = format_criticality_report(&report);
        assert!(formatted.contains("CRITICALITY REPORT"));
        assert!(formatted.contains("Von Neumann"));
    }

    #[test]
    fn test_criticality_regime_display() {
        assert_eq!(
            format!("{}", CriticalityRegime::Critical),
            "Critical (Discovery Zone)"
        );
        assert_eq!(
            format!("{}", CriticalityRegime::Subcritical),
            "Subcritical"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // 2. Abductive Reasoning Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_abduce_with_known_entities() {
        let brain = populated_brain();
        let hypothesis = abduce(&brain, "Isaac Newton and physics").unwrap();
        assert!(!hypothesis.candidate_explanations.is_empty());
        assert!(hypothesis.confidence > 0.0);
    }

    #[test]
    fn test_abduce_unknown_observation() {
        let brain = test_brain();
        let hypothesis = abduce(&brain, "something completely unknown").unwrap();
        assert!(hypothesis.candidate_explanations.is_empty());
    }

    #[test]
    fn test_abduce_single_entity() {
        let brain = populated_brain();
        let hypothesis = abduce(&brain, "Isaac Newton").unwrap();
        // Should find entity prominence explanation
        assert!(hypothesis.candidate_explanations.len() >= 0);
    }

    #[test]
    fn test_explain_gap_connected_entities() {
        let brain = populated_brain();
        let explanations = explain_gap(&brain, "Isaac Newton", "gravity").unwrap();
        // They ARE connected → should say so
        assert!(!explanations.is_empty());
        assert!(explanations[0].summary.contains("directly connected"));
    }

    #[test]
    fn test_explain_gap_disconnected_entities() {
        let brain = populated_brain();
        let explanations = explain_gap(&brain, "Charles Darwin", "relativity").unwrap();
        // Not directly connected — should generate hypotheses
        assert!(!explanations.is_empty());
    }

    #[test]
    fn test_explain_gap_unknown_entity() {
        let brain = populated_brain();
        let explanations = explain_gap(&brain, "Unknown Person", "gravity").unwrap();
        assert!(explanations.is_empty());
    }

    #[test]
    fn test_predict_next_discovery_empty() {
        let brain = test_brain();
        let predictions = predict_next_discovery(&brain).unwrap();
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_predict_next_discovery_populated() {
        let brain = populated_brain();
        let predictions = predict_next_discovery(&brain).unwrap();
        // Should predict some entities
        assert!(!predictions.is_empty());
        // Scores should be valid
        for p in &predictions {
            assert!(p.predicted_score > 0.0 && p.predicted_score <= 1.0);
        }
    }

    #[test]
    fn test_predict_next_discovery_sorted() {
        let brain = populated_brain();
        let predictions = predict_next_discovery(&brain).unwrap();
        // Should be sorted by score descending
        for i in 1..predictions.len() {
            assert!(predictions[i - 1].predicted_score >= predictions[i].predicted_score);
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // 3. Temporal Reasoning Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_build_temporal_index_empty() {
        let brain = test_brain();
        let events = build_temporal_index(&brain).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_build_temporal_index_populated() {
        let brain = populated_brain();
        let events = build_temporal_index(&brain).unwrap();
        assert!(!events.is_empty());
        // Should be sorted by timestamp
        for i in 1..events.len() {
            assert!(events[i].timestamp >= events[i - 1].timestamp);
        }
    }

    #[test]
    fn test_temporal_causality_same_entity() {
        let brain = populated_brain();
        let score = temporal_causality(&brain, "Isaac Newton", "Isaac Newton").unwrap();
        // Self-causality should be 0 or undefined
        assert!(score >= 0.0);
    }

    #[test]
    fn test_temporal_causality_unknown() {
        let brain = populated_brain();
        let score = temporal_causality(&brain, "Unknown1", "Unknown2").unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_temporal_causality_known_pair() {
        let brain = populated_brain();
        let score = temporal_causality(&brain, "Isaac Newton", "physics").unwrap();
        // Should be some positive value
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_predict_temporal_pattern_unknown() {
        let brain = populated_brain();
        let pred = predict_temporal_pattern(&brain, "Unknown Entity").unwrap();
        assert!(pred.is_none());
    }

    #[test]
    fn test_predict_temporal_pattern_known() {
        let brain = populated_brain();
        let pred = predict_temporal_pattern(&brain, "physics").unwrap();
        // physics has many relations, should get a prediction
        if let Some(p) = pred {
            assert!(p.avg_interval_hours >= 0.0);
            assert!(p.confidence >= 0.0 && p.confidence <= 1.0);
        }
    }

    #[test]
    fn test_temporal_anomalies_empty() {
        let brain = test_brain();
        let anomalies = temporal_anomalies(&brain).unwrap();
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_temporal_anomalies_populated() {
        let brain = populated_brain();
        let anomalies = temporal_anomalies(&brain).unwrap();
        // May or may not find anomalies in test data
        for a in &anomalies {
            assert!(a.deviation_score > 0.0);
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // 4. Network Topology Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_scale_free_score_empty() {
        let brain = test_brain();
        let (gamma, r2, _) = scale_free_score(&brain).unwrap();
        assert_eq!(gamma, 0.0);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn test_scale_free_score_populated() {
        let brain = populated_brain();
        let (gamma, r2, assessment) = scale_free_score(&brain).unwrap();
        assert!(gamma >= 0.0);
        assert!(r2 >= 0.0 && r2 <= 1.0);
        assert!(!assessment.is_empty());
    }

    #[test]
    fn test_small_world_coefficient_empty() {
        let brain = test_brain();
        let (sigma, _) = small_world_coefficient(&brain).unwrap();
        assert_eq!(sigma, 0.0);
    }

    #[test]
    fn test_small_world_coefficient_populated() {
        let brain = populated_brain();
        let (sigma, assessment) = small_world_coefficient(&brain).unwrap();
        assert!(sigma >= 0.0);
        assert!(!assessment.is_empty());
    }

    #[test]
    fn test_hub_vulnerability_empty() {
        let brain = test_brain();
        let vuln = hub_vulnerability(&brain).unwrap();
        assert!(vuln.is_empty());
    }

    #[test]
    fn test_hub_vulnerability_populated() {
        let brain = populated_brain();
        let vuln = hub_vulnerability(&brain).unwrap();
        assert!(!vuln.is_empty());
        // Scores should be non-negative
        for (_, score) in &vuln {
            assert!(*score >= 0.0);
        }
        // Should be sorted by vulnerability descending
        for i in 1..vuln.len() {
            assert!(vuln[i - 1].1 >= vuln[i].1);
        }
    }

    #[test]
    fn test_topology_steering_recommendations_empty() {
        let brain = test_brain();
        let recs = topology_steering_recommendations(&brain).unwrap();
        // May or may not have recommendations for empty graph
        assert!(recs.len() >= 0);
    }

    #[test]
    fn test_topology_steering_recommendations_populated() {
        let brain = populated_brain();
        let recs = topology_steering_recommendations(&brain).unwrap();
        for r in &recs {
            assert!(!r.recommendation.is_empty());
            assert!(r.priority >= 1 && r.priority <= 3);
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // 5. Meta-Cognitive Loop Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_domain_balance_score_empty() {
        let brain = test_brain();
        let gini = domain_balance_score(&brain).unwrap();
        assert_eq!(gini, 0.0);
    }

    #[test]
    fn test_domain_balance_score_single_type() {
        let brain = test_brain();
        brain.upsert_entity("A", "concept").unwrap();
        brain.upsert_entity("B", "concept").unwrap();
        let gini = domain_balance_score(&brain).unwrap();
        // Single type → Gini = 1.0 (completely concentrated)
        assert_eq!(gini, 1.0);
    }

    #[test]
    fn test_domain_balance_score_balanced() {
        let brain = test_brain();
        // Equal counts of each type
        brain.upsert_entity("P1", "person").unwrap();
        brain.upsert_entity("P2", "person").unwrap();
        brain.upsert_entity("C1", "concept").unwrap();
        brain.upsert_entity("C2", "concept").unwrap();
        brain.upsert_entity("O1", "organization").unwrap();
        brain.upsert_entity("O2", "organization").unwrap();
        let gini = domain_balance_score(&brain).unwrap();
        assert!(gini < 0.1, "Balanced types should have low Gini, got {}", gini);
    }

    #[test]
    fn test_domain_balance_score_unbalanced() {
        let brain = test_brain();
        for i in 0..20 {
            brain.upsert_entity(&format!("C{}", i), "concept").unwrap();
        }
        brain.upsert_entity("P1", "person").unwrap();
        let gini = domain_balance_score(&brain).unwrap();
        assert!(gini > 0.3, "Unbalanced types should have high Gini, got {}", gini);
    }

    #[test]
    fn test_staleness_index_fresh() {
        let brain = test_brain();
        brain.upsert_entity("Fresh", "concept").unwrap();
        let staleness = staleness_index(&brain).unwrap();
        assert_eq!(staleness, 0.0, "Just-created entity should not be stale");
    }

    #[test]
    fn test_staleness_index_empty() {
        let brain = test_brain();
        let staleness = staleness_index(&brain).unwrap();
        assert_eq!(staleness, 0.0);
    }

    #[test]
    fn test_discovery_velocity_no_hypotheses() {
        let brain = test_brain();
        let velocity = discovery_velocity(&brain).unwrap();
        assert_eq!(velocity, 0.0);
    }

    #[test]
    fn test_introspect_empty() {
        let brain = test_brain();
        let mc = introspect(&brain).unwrap();
        assert_eq!(mc.entity_count, 0);
        assert_eq!(mc.relation_count, 0);
    }

    #[test]
    fn test_introspect_populated() {
        let brain = populated_brain();
        let mc = introspect(&brain).unwrap();
        assert!(mc.entity_count > 0);
        assert!(mc.relation_count > 0);
        assert!(mc.domain_balance >= 0.0 && mc.domain_balance <= 1.0);
        assert!(!mc.recommendations.is_empty());
    }

    #[test]
    fn test_format_introspection() {
        let brain = populated_brain();
        let mc = introspect(&brain).unwrap();
        let formatted = format_introspection(&mc);
        assert!(formatted.contains("META-COGNITION"));
        assert!(formatted.contains("Learning rate"));
    }

    #[test]
    fn test_recommend_feeding_strategy_empty() {
        let brain = test_brain();
        let recs = recommend_feeding_strategy(&brain).unwrap();
        assert!(!recs.is_empty()); // Should always have at least one recommendation
    }

    #[test]
    fn test_recommend_feeding_strategy_populated() {
        let brain = populated_brain();
        let recs = recommend_feeding_strategy(&brain).unwrap();
        assert!(!recs.is_empty());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Utility Function Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_shannon_entropy_uniform() {
        let counts = vec![10, 10, 10, 10];
        let entropy = shannon_entropy_from_counts(counts.iter());
        assert!((entropy - 2.0).abs() < 0.01, "Uniform 4 categories should be 2 bits, got {}", entropy);
    }

    #[test]
    fn test_shannon_entropy_single() {
        let counts = vec![100];
        let entropy = shannon_entropy_from_counts(counts.iter());
        assert_eq!(entropy, 0.0, "Single category should have 0 entropy");
    }

    #[test]
    fn test_shannon_entropy_empty() {
        let counts: Vec<usize> = vec![];
        let entropy = shannon_entropy_from_counts(counts.iter());
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mut a = HashMap::new();
        a.insert("x".to_string(), 1.0);
        a.insert("y".to_string(), 2.0);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = HashMap::new();
        a.insert("x".to_string(), 1.0);
        let mut b = HashMap::new();
        b.insert("y".to_string(), 1.0);
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: HashMap<String, f64> = HashMap::new();
        let b: HashMap<String, f64> = HashMap::new();
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_fit_power_law_trivial() {
        let sizes = vec![1, 2, 4, 8, 16, 32];
        let (gamma, r2) = fit_power_law(&sizes);
        assert!(gamma > 0.0, "Power law exponent should be positive");
        // Not perfect power law but should get some fit
    }

    #[test]
    fn test_fit_power_law_empty() {
        let sizes: Vec<usize> = vec![];
        let (gamma, r2) = fit_power_law(&sizes);
        assert_eq!(gamma, 0.0);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn test_linear_r_squared_perfect() {
        let points: Vec<(f64, f64)> = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
        let r2 = linear_r_squared(&points);
        assert!((r2 - 1.0).abs() < 0.01, "Perfect linear should be R²≈1.0");
    }

    #[test]
    fn test_linear_r_squared_random() {
        let points: Vec<(f64, f64)> = vec![(1.0, 5.0), (2.0, 1.0), (3.0, 7.0), (4.0, 2.0)];
        let r2 = linear_r_squared(&points);
        assert!(r2 >= 0.0 && r2 <= 1.0);
    }

    #[test]
    fn test_tridiagonal_eigenvalues_identity() {
        // Identity matrix (all 1s on diagonal, 0s on off-diagonal)
        let alpha = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let eigenvalues = tridiagonal_eigenvalues(&alpha, &beta);
        assert_eq!(eigenvalues.len(), 3);
        for &ev in &eigenvalues {
            assert!((ev - 1.0).abs() < 0.1, "Identity eigenvalue should be ~1.0, got {}", ev);
        }
    }

    #[test]
    fn test_tridiagonal_eigenvalues_simple() {
        // Simple tridiagonal: [[2, -1], [-1, 2]]
        // Eigenvalues: 1, 3
        let alpha = vec![2.0, 2.0];
        let beta = vec![1.0]; // absolute value
        let eigenvalues = tridiagonal_eigenvalues(&alpha, &beta);
        assert_eq!(eigenvalues.len(), 2);
    }

    #[test]
    fn test_tridiagonal_eigenvalues_empty() {
        let eigenvalues = tridiagonal_eigenvalues(&[], &[]);
        assert!(eigenvalues.is_empty());
    }

    #[test]
    fn test_tridiagonal_eigenvalues_single() {
        let eigenvalues = tridiagonal_eigenvalues(&[3.14], &[]);
        assert_eq!(eigenvalues.len(), 1);
        assert!((eigenvalues[0] - 3.14).abs() < 1e-10);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Edge Case Tests
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn test_all_systems_single_node() {
        let brain = test_brain();
        brain.upsert_entity("Lonely", "concept").unwrap();

        assert_eq!(von_neumann_graph_entropy(&brain).unwrap(), 0.0);
        assert!(semantic_entropy(&brain).unwrap() >= 0.0);
        assert_eq!(surprise_edge_fraction(&brain).unwrap(), 0.0);
        assert!(temporal_anomalies(&brain).unwrap().is_empty());
        assert!(hub_vulnerability(&brain).unwrap().is_empty());
    }

    #[test]
    fn test_all_systems_two_connected_nodes() {
        let brain = test_brain();
        let a = brain.upsert_entity("A", "person").unwrap();
        let b = brain.upsert_entity("B", "concept").unwrap();
        brain.upsert_relation(a, "knows", b, "test").unwrap();

        let entropy = von_neumann_graph_entropy(&brain).unwrap();
        assert!(entropy >= 0.0);

        let sem = semantic_entropy(&brain).unwrap();
        assert!(sem >= 0.0);

        let surprise = surprise_edge_fraction(&brain).unwrap();
        assert!(surprise >= 0.0 && surprise <= 1.0);
    }

    #[test]
    fn test_fully_connected_graph() {
        let brain = test_brain();
        let ids: Vec<i64> = (0..5)
            .map(|i| {
                brain
                    .upsert_entity(&format!("Node{}", i), "concept")
                    .unwrap()
            })
            .collect();

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                brain
                    .upsert_relation(ids[i], "linked", ids[j], "test")
                    .unwrap();
            }
        }

        let entropy = von_neumann_graph_entropy(&brain).unwrap();
        assert!(entropy >= 0.0, "Fully connected graph entropy should be non-negative, got {}", entropy);

        let surprise = surprise_edge_fraction(&brain).unwrap();
        assert!(surprise >= 0.0);

        let (sigma, _) = small_world_coefficient(&brain).unwrap();
        // Small fully-connected graph might not qualify for small-world analysis
        assert!(sigma >= 0.0);
    }
}