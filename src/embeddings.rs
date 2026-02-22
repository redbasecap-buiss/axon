//! TF-IDF embeddings, cosine similarity, HNSW index, and k-means clustering.
//!
//! No external ML models — pure Rust, sparse vectors, navigable small world graphs.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read as _, Write as _};
use std::path::Path;

use crate::db::Brain;

// ---------------------------------------------------------------------------
// Sparse vector
// ---------------------------------------------------------------------------

/// A sparse vector stored as sorted (dimension, value) pairs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseVec {
    /// Sorted by dimension index.
    pub entries: Vec<(u32, f64)>,
}

impl SparseVec {
    pub fn new(entries: Vec<(u32, f64)>) -> Self {
        let mut e = entries;
        e.sort_by_key(|(d, _)| *d);
        Self { entries: e }
    }

    pub fn norm(&self) -> f64 {
        self.entries.iter().map(|(_, v)| v * v).sum::<f64>().sqrt()
    }

    pub fn dot(&self, other: &SparseVec) -> f64 {
        let (a, b) = (&self.entries, &other.entries);
        let (mut i, mut j) = (0, 0);
        let mut sum = 0.0;
        while i < a.len() && j < b.len() {
            match a[i].0.cmp(&b[j].0) {
                std::cmp::Ordering::Equal => {
                    sum += a[i].1 * b[j].1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        sum
    }
}

/// Cosine similarity in \[-1, 1\].  Returns 0.0 when either vector is zero.
pub fn cosine_similarity(a: &SparseVec, b: &SparseVec) -> f64 {
    let denom = a.norm() * b.norm();
    if denom == 0.0 {
        return 0.0;
    }
    a.dot(b) / denom
}

// ---------------------------------------------------------------------------
// TF-IDF
// ---------------------------------------------------------------------------

/// Tokenise text into lowercase alphanumeric words.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(String::from)
        .collect()
}

/// Corpus statistics for IDF computation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vocabulary {
    /// word → dimension index
    pub word_to_dim: BTreeMap<String, u32>,
    /// dimension → document frequency
    pub doc_freq: Vec<u32>,
    /// total number of documents used to build the vocab
    pub num_docs: u32,
}

impl Vocabulary {
    /// Build vocabulary from a set of documents (each is a bag of words).
    pub fn build(docs: &[Vec<String>]) -> Self {
        let mut word_to_dim: BTreeMap<String, u32> = BTreeMap::new();
        let mut doc_freq_map: HashMap<String, u32> = HashMap::new();

        for doc in docs {
            let unique: HashSet<&String> = doc.iter().collect();
            for w in unique {
                *doc_freq_map.entry(w.clone()).or_insert(0) += 1;
                if !word_to_dim.contains_key(w.as_str()) {
                    let idx = word_to_dim.len() as u32;
                    word_to_dim.insert(w.clone(), idx);
                }
            }
        }

        let mut doc_freq = vec![0u32; word_to_dim.len()];
        for (w, count) in &doc_freq_map {
            if let Some(&dim) = word_to_dim.get(w) {
                doc_freq[dim as usize] = *count;
            }
        }

        Self {
            word_to_dim,
            doc_freq,
            num_docs: docs.len() as u32,
        }
    }

    /// Compute TF-IDF sparse vector for a single document.
    pub fn tfidf(&self, tokens: &[String]) -> SparseVec {
        if tokens.is_empty() {
            return SparseVec::new(vec![]);
        }
        let mut tf: HashMap<u32, u32> = HashMap::new();
        for t in tokens {
            if let Some(&dim) = self.word_to_dim.get(t) {
                *tf.entry(dim).or_insert(0) += 1;
            }
        }
        let max_tf = *tf.values().max().unwrap_or(&1) as f64;
        let n = (self.num_docs as f64).max(1.0);
        let entries: Vec<(u32, f64)> = tf
            .into_iter()
            .map(|(dim, count)| {
                let tf_val = 0.5 + 0.5 * (count as f64 / max_tf);
                let df = self.doc_freq[dim as usize].max(1) as f64;
                let idf = (n / df).ln() + 1.0;
                (dim, tf_val * idf)
            })
            .collect();
        SparseVec::new(entries)
    }
}

// ---------------------------------------------------------------------------
// HNSW Index
// ---------------------------------------------------------------------------

/// Configuration for the HNSW index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswConfig {
    /// Max connections per node per layer.
    pub m: usize,
    /// Max connections for layer 0 (typically 2*M).
    pub m_max0: usize,
    /// Size of dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search.
    pub ef_search: usize,
    /// Normalisation factor for level generation.
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (16_f64).ln(),
        }
    }
}

/// A single node in the HNSW graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct HnswNode {
    /// External id (entity name or other identifier).
    id: String,
    /// The vector.
    vector: SparseVec,
    /// Neighbours per layer. neighbours[layer] = vec of internal indices.
    neighbours: Vec<Vec<usize>>,
    /// Max layer this node lives in.
    max_layer: usize,
}

/// Hierarchical Navigable Small World index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    /// Map from external id to internal index.
    id_to_idx: HashMap<String, usize>,
    /// Current max layer in the graph.
    max_layer: usize,
    /// Entry point (internal index). None if empty.
    entry_point: Option<usize>,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            max_layer: 0,
            entry_point: None,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Assign a random layer for a new node.
    fn random_level(&self) -> usize {
        let r: f64 = rand_f64();
        (-r.ln() * self.config.ml).floor() as usize
    }

    /// Insert a vector with a given external id.
    pub fn insert(&mut self, id: String, vector: SparseVec) {
        if let Some(&existing) = self.id_to_idx.get(&id) {
            // Update vector in place.
            self.nodes[existing].vector = vector;
            return;
        }

        let level = self.random_level();
        let idx = self.nodes.len();

        let node = HnswNode {
            id: id.clone(),
            vector,
            neighbours: vec![Vec::new(); level + 1],
            max_layer: level,
        };
        self.nodes.push(node);
        self.id_to_idx.insert(id, idx);

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_layer = level;
            return;
        }

        let ep = self.entry_point.unwrap();
        let mut current = ep;

        // Traverse layers above the new node's level — greedy search.
        let top = self.max_layer;
        for layer in (level + 1..=top).rev() {
            current = self.greedy_closest(current, idx, layer);
        }

        // For each layer the new node participates in, find and connect neighbours.
        let insert_layers = level.min(top);
        for layer in (0..=insert_layers).rev() {
            let m_max = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };
            let candidates = self.search_layer(current, idx, self.config.ef_construction, layer);
            let neighbours: Vec<usize> = candidates.iter().take(m_max).map(|&(n, _)| n).collect();

            self.nodes[idx].neighbours[layer] = neighbours.clone();

            // Add back-connections, pruning if needed.
            for &nb in &neighbours {
                self.nodes[nb].neighbours[layer].push(idx);
                if self.nodes[nb].neighbours[layer].len() > m_max {
                    self.prune_neighbours(nb, layer, m_max);
                }
            }

            if !candidates.is_empty() {
                current = candidates[0].0;
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(idx);
        }
    }

    /// Greedy walk to the closest node to `target_idx` starting from `ep` at `layer`.
    fn greedy_closest(&self, ep: usize, target_idx: usize, layer: usize) -> usize {
        let target_vec = &self.nodes[target_idx].vector;
        let mut current = ep;
        let mut best_dist = distance(target_vec, &self.nodes[current].vector);
        loop {
            let mut changed = false;
            let nbs = &self.nodes[current].neighbours;
            if layer < nbs.len() {
                for &nb in &nbs[layer] {
                    let d = distance(target_vec, &self.nodes[nb].vector);
                    if d < best_dist {
                        best_dist = d;
                        current = nb;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Search a single layer, returning up to `ef` closest nodes as (idx, distance).
    fn search_layer(
        &self,
        ep: usize,
        target_idx: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f64)> {
        let target_vec = &self.nodes[target_idx].vector;
        self.search_layer_by_vec(ep, target_vec, ef, layer)
    }

    /// Search a single layer by vector.
    fn search_layer_by_vec(
        &self,
        ep: usize,
        target_vec: &SparseVec,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f64)> {
        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(ep);

        let ep_dist = distance(target_vec, &self.nodes[ep].vector);
        // candidates: min-heap by distance (closest first)
        let mut candidates: Vec<(usize, f64)> = vec![(ep, ep_dist)];
        // results: kept sorted, worst last
        let mut results: Vec<(usize, f64)> = vec![(ep, ep_dist)];

        while let Some(pos) = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
            .map(|(i, _)| i)
        {
            let (c_idx, c_dist) = candidates.remove(pos);
            let worst_result = results.last().map(|r| r.1).unwrap_or(f64::MAX);
            if c_dist > worst_result && results.len() >= ef {
                break;
            }

            let nbs = &self.nodes[c_idx].neighbours;
            if layer < nbs.len() {
                for &nb in &nbs[layer] {
                    if visited.insert(nb) {
                        let d = distance(target_vec, &self.nodes[nb].vector);
                        let worst = results.last().map(|r| r.1).unwrap_or(f64::MAX);
                        if d < worst || results.len() < ef {
                            candidates.push((nb, d));
                            results.push((nb, d));
                            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Prune neighbours of a node at a given layer to at most `m_max`.
    fn prune_neighbours(&mut self, node: usize, layer: usize, m_max: usize) {
        let node_vec = self.nodes[node].vector.clone();
        let mut scored: Vec<(usize, f64)> = self.nodes[node].neighbours[layer]
            .iter()
            .map(|&nb| (nb, distance(&node_vec, &self.nodes[nb].vector)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(m_max);
        self.nodes[node].neighbours[layer] = scored.into_iter().map(|(n, _)| n).collect();
    }

    /// Search for k nearest neighbours by external id.
    pub fn search_by_id(&self, id: &str, k: usize) -> Vec<(String, f64)> {
        let Some(&idx) = self.id_to_idx.get(id) else {
            return vec![];
        };
        let vec = &self.nodes[idx].vector;
        self.search(vec, k)
            .into_iter()
            .filter(|(name, _)| name != id)
            .collect()
    }

    /// Search for k nearest neighbours of a query vector.
    pub fn search(&self, query: &SparseVec, k: usize) -> Vec<(String, f64)> {
        if self.nodes.is_empty() {
            return vec![];
        }
        let ep = self.entry_point.unwrap();
        let mut current = ep;

        // Traverse top layers greedily.
        for layer in (1..=self.max_layer).rev() {
            current = self.greedy_closest_vec(current, query, layer);
        }

        // Search layer 0.
        let ef = self.config.ef_search.max(k);
        let results = self.search_layer_by_vec(current, query, ef, 0);

        results
            .into_iter()
            .take(k)
            .map(|(idx, dist)| {
                let sim = 1.0 - dist; // convert distance back to similarity
                (self.nodes[idx].id.clone(), sim)
            })
            .collect()
    }

    fn greedy_closest_vec(&self, ep: usize, target_vec: &SparseVec, layer: usize) -> usize {
        let mut current = ep;
        let mut best_dist = distance(target_vec, &self.nodes[current].vector);
        loop {
            let mut changed = false;
            let nbs = &self.nodes[current].neighbours;
            if layer < nbs.len() {
                for &nb in &nbs[layer] {
                    let d = distance(target_vec, &self.nodes[nb].vector);
                    if d < best_dist {
                        best_dist = d;
                        current = nb;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Remove an element by external id.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(&idx) = self.id_to_idx.get(id) else {
            return false;
        };

        // Remove all back-connections to this node.
        for layer in 0..=self.nodes[idx].max_layer {
            let nbs: Vec<usize> = self.nodes[idx].neighbours[layer].clone();
            for nb in nbs {
                self.nodes[nb].neighbours[layer].retain(|&x| x != idx);
            }
        }

        // We don't compact the vec — just mark as removed.
        self.id_to_idx.remove(id);
        self.nodes[idx].id = String::new();
        self.nodes[idx].neighbours = vec![];
        self.nodes[idx].vector = SparseVec::new(vec![]);

        // If this was the entry point, pick another.
        if self.entry_point == Some(idx) {
            self.entry_point = self.id_to_idx.values().copied().next();
            if let Some(new_ep) = self.entry_point {
                self.max_layer = self.nodes[new_ep].max_layer;
            }
        }

        true
    }

    /// Serialize to a file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_vec(self).map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&data)?;
        Ok(())
    }

    /// Deserialize from a file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut data = Vec::new();
        f.read_to_end(&mut data)?;
        serde_json::from_slice(&data).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// Distance metric: 1 - cosine_similarity (so 0 = identical).
fn distance(a: &SparseVec, b: &SparseVec) -> f64 {
    1.0 - cosine_similarity(a, b)
}

/// Simple pseudo-random f64 in (0, 1) using thread-local state.
fn rand_f64() -> f64 {
    use std::cell::Cell;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let mut h = DefaultHasher::new();
            std::thread::current().id().hash(&mut h);
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .hash(&mut h);
            h.finish()
        });
    }
    STATE.with(|s| {
        // xorshift64
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f64) / (u64::MAX as f64)
    })
}

// ---------------------------------------------------------------------------
// K-Means clustering
// ---------------------------------------------------------------------------

/// Cluster assignment result.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// cluster_id for each input vector (parallel to input).
    pub assignments: Vec<usize>,
    /// Number of clusters.
    pub k: usize,
}

/// Run k-means on sparse vectors.
pub fn kmeans(vectors: &[SparseVec], k: usize, max_iter: usize) -> ClusterResult {
    let n = vectors.len();
    if n == 0 || k == 0 {
        return ClusterResult {
            assignments: vec![],
            k: 0,
        };
    }
    let k = k.min(n);

    // Initialise centroids by picking k evenly-spaced items.
    let mut centroids: Vec<SparseVec> = (0..k).map(|i| vectors[i * n / k].clone()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for (i, v) in vectors.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, cosine_similarity(v, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Recompute centroids (mean of assigned vectors).
        centroids = (0..k)
            .map(|ci| {
                let members: Vec<&SparseVec> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| assignments[*j] == ci)
                    .map(|(_, v)| v)
                    .collect();
                if members.is_empty() {
                    return SparseVec::new(vec![]);
                }
                mean_sparse(&members)
            })
            .collect();
    }

    ClusterResult { assignments, k }
}

/// Compute the mean of a set of sparse vectors.
fn mean_sparse(vecs: &[&SparseVec]) -> SparseVec {
    let n = vecs.len() as f64;
    let mut acc: BTreeMap<u32, f64> = BTreeMap::new();
    for v in vecs {
        for &(dim, val) in &v.entries {
            *acc.entry(dim).or_insert(0.0) += val;
        }
    }
    SparseVec::new(acc.into_iter().map(|(d, v)| (d, v / n)).collect())
}

// ---------------------------------------------------------------------------
// Brain integration
// ---------------------------------------------------------------------------

/// Build TF-IDF vectors for all entities in the brain.
/// Returns (entity_names, vocabulary, vectors) — all parallel.
pub fn build_entity_vectors(
    brain: &Brain,
) -> anyhow::Result<(Vec<String>, Vocabulary, Vec<SparseVec>)> {
    let entities = brain.all_entities()?;
    let mut docs: Vec<Vec<String>> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    for entity in &entities {
        let mut tokens = tokenize(&entity.name);
        tokens.extend(tokenize(&entity.entity_type));
        // Include facts and relations as context.
        if let Ok(facts) = brain.get_facts_for(entity.id) {
            for f in facts {
                tokens.extend(tokenize(&f.key));
                tokens.extend(tokenize(&f.value));
            }
        }
        if let Ok(rels) = brain.get_relations_for(entity.id) {
            for (subj, pred, obj, _) in rels {
                tokens.extend(tokenize(&subj));
                tokens.extend(tokenize(&pred));
                tokens.extend(tokenize(&obj));
            }
        }
        names.push(entity.name.clone());
        docs.push(tokens);
    }

    let vocab = Vocabulary::build(&docs);
    let vectors: Vec<SparseVec> = docs.iter().map(|d| vocab.tfidf(d)).collect();

    Ok((names, vocab, vectors))
}

/// Build an HNSW index from the brain's entities.
pub fn build_index(brain: &Brain) -> anyhow::Result<(HnswIndex, Vec<String>)> {
    let (names, _vocab, vectors) = build_entity_vectors(brain)?;
    let mut index = HnswIndex::new(HnswConfig::default());
    for (name, vec) in names.iter().zip(vectors.into_iter()) {
        index.insert(name.clone(), vec);
    }
    Ok((index, names))
}

/// Find similar entities to the given entity name.
pub fn find_similar(brain: &Brain, entity: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
    let (index, _names) = build_index(brain)?;
    Ok(index.search_by_id(entity, k))
}

/// Auto-cluster entities, guessing k as sqrt(n).
pub fn cluster_entities(brain: &Brain) -> anyhow::Result<Vec<(usize, Vec<String>)>> {
    let (names, _vocab, vectors) = build_entity_vectors(brain)?;
    if names.is_empty() {
        return Ok(vec![]);
    }
    let k = ((names.len() as f64).sqrt().ceil() as usize)
        .max(2)
        .min(names.len());
    let result = kmeans(&vectors, k, 50);

    let mut clusters: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (i, &cluster_id) in result.assignments.iter().enumerate() {
        clusters
            .entry(cluster_id)
            .or_default()
            .push(names[i].clone());
    }

    Ok(clusters.into_iter().collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_from(entries: &[(u32, f64)]) -> SparseVec {
        SparseVec::new(entries.to_vec())
    }

    #[test]
    fn test_sparse_vec_norm() {
        let v = vec_from(&[(0, 3.0), (1, 4.0)]);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_vec_dot() {
        let a = vec_from(&[(0, 1.0), (1, 2.0), (2, 3.0)]);
        let b = vec_from(&[(0, 4.0), (2, 6.0)]);
        assert!((a.dot(&b) - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec_from(&[(0, 1.0), (1, 2.0)]);
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec_from(&[(0, 1.0)]);
        let b = vec_from(&[(1, 1.0)]);
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec_from(&[]);
        let b = vec_from(&[(0, 1.0)]);
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! Rust-is-great.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
    }

    #[test]
    fn test_vocabulary_build() {
        let docs = vec![
            tokenize("the cat sat"),
            tokenize("the dog sat"),
            tokenize("the cat ran"),
        ];
        let vocab = Vocabulary::build(&docs);
        assert!(vocab.word_to_dim.contains_key("cat"));
        assert!(vocab.word_to_dim.contains_key("the"));
        assert_eq!(vocab.num_docs, 3);
    }

    #[test]
    fn test_tfidf_nonzero() {
        let docs = vec![tokenize("hello world"), tokenize("goodbye world")];
        let vocab = Vocabulary::build(&docs);
        let v = vocab.tfidf(&tokenize("hello world"));
        assert!(!v.entries.is_empty());
        assert!(v.norm() > 0.0);
    }

    #[test]
    fn test_tfidf_unique_word_higher() {
        // "hello" appears in 1 doc, "world" in 2 — hello should have higher TF-IDF
        let docs = vec![tokenize("hello world"), tokenize("goodbye world")];
        let vocab = Vocabulary::build(&docs);
        let v = vocab.tfidf(&tokenize("hello world"));
        let hello_dim = vocab.word_to_dim["hello"];
        let world_dim = vocab.word_to_dim["world"];
        let hello_val = v.entries.iter().find(|(d, _)| *d == hello_dim).unwrap().1;
        let world_val = v.entries.iter().find(|(d, _)| *d == world_dim).unwrap().1;
        assert!(hello_val > world_val);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a".into(), vec_from(&[(0, 1.0), (1, 0.0)]));
        index.insert("b".into(), vec_from(&[(0, 0.9), (1, 0.1)]));
        index.insert("c".into(), vec_from(&[(0, 0.0), (1, 1.0)]));

        let results = index.search_by_id("a", 2);
        assert!(!results.is_empty());
        // "b" should be more similar to "a" than "c"
        assert_eq!(results[0].0, "b");
    }

    #[test]
    fn test_hnsw_empty_search() {
        let index = HnswIndex::new(HnswConfig::default());
        let results = index.search(&vec_from(&[(0, 1.0)]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_remove() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("a".into(), vec_from(&[(0, 1.0)]));
        index.insert("b".into(), vec_from(&[(1, 1.0)]));
        assert_eq!(index.len(), 2);
        assert!(index.remove("a"));
        assert!(!index.remove("a")); // already removed
        let results = index.search_by_id("b", 5);
        assert!(results.iter().all(|(name, _)| name != "a"));
    }

    #[test]
    fn test_hnsw_serialize_roundtrip() {
        let mut index = HnswIndex::new(HnswConfig::default());
        index.insert("x".into(), vec_from(&[(0, 1.0), (1, 2.0)]));
        index.insert("y".into(), vec_from(&[(0, 3.0), (1, 4.0)]));

        let dir = std::env::temp_dir().join("axon_hnsw_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_index.json");

        index.save(&path).unwrap();
        let loaded = HnswIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);

        let r1 = index.search_by_id("x", 1);
        let r2 = loaded.search_by_id("x", 1);
        assert_eq!(r1.len(), r2.len());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_hnsw_many_inserts() {
        let mut index = HnswIndex::new(HnswConfig {
            m: 4,
            m_max0: 8,
            ef_construction: 20,
            ef_search: 10,
            ml: 1.0 / 4_f64.ln(),
        });
        for i in 0..50 {
            let v = vec_from(&[(i as u32, 1.0), ((i + 1) as u32, 0.5)]);
            index.insert(format!("n{i}"), v);
        }
        assert_eq!(index.len(), 50);
        let results = index.search_by_id("n25", 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_kmeans_basic() {
        let vecs = vec![
            vec_from(&[(0, 1.0)]),
            vec_from(&[(0, 0.9)]),
            vec_from(&[(1, 1.0)]),
            vec_from(&[(1, 0.9)]),
        ];
        let result = kmeans(&vecs, 2, 20);
        assert_eq!(result.k, 2);
        assert_eq!(result.assignments.len(), 4);
        // First two should be in the same cluster, last two in another.
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[2], result.assignments[3]);
        assert_ne!(result.assignments[0], result.assignments[2]);
    }

    #[test]
    fn test_kmeans_empty() {
        let result = kmeans(&[], 3, 10);
        assert_eq!(result.k, 0);
    }

    #[test]
    fn test_distance_metric() {
        let a = vec_from(&[(0, 1.0)]);
        let b = vec_from(&[(0, 1.0)]);
        assert!(distance(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_mean_sparse() {
        let a = vec_from(&[(0, 2.0), (1, 4.0)]);
        let b = vec_from(&[(0, 4.0), (1, 6.0)]);
        let m = mean_sparse(&[&a, &b]);
        let v0 = m.entries.iter().find(|(d, _)| *d == 0).unwrap().1;
        let v1 = m.entries.iter().find(|(d, _)| *d == 1).unwrap().1;
        assert!((v0 - 3.0).abs() < 1e-10);
        assert!((v1 - 5.0).abs() < 1e-10);
    }
}
