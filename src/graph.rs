#![allow(dead_code)]
use std::collections::{HashMap, HashSet, VecDeque};

use crate::db::Brain;
use crate::nlp::levenshtein;

pub fn shortest_path(
    brain: &Brain,
    from: &str,
    to: &str,
) -> Result<Option<Vec<i64>>, rusqlite::Error> {
    let from_entity = brain.get_entity_by_name(from)?;
    let to_entity = brain.get_entity_by_name(to)?;
    let (from_id, to_id) = match (from_entity, to_entity) {
        (Some(f), Some(t)) => (f.id, t.id),
        _ => return Ok(None),
    };
    let adj = build_adjacency(brain)?;
    bfs_path(&adj, from_id, to_id)
}

pub fn all_paths(
    brain: &Brain,
    from: &str,
    to: &str,
    max_depth: usize,
) -> Result<Vec<Vec<i64>>, rusqlite::Error> {
    let from_entity = brain.get_entity_by_name(from)?;
    let to_entity = brain.get_entity_by_name(to)?;
    let (from_id, to_id) = match (from_entity, to_entity) {
        (Some(f), Some(t)) => (f.id, t.id),
        _ => return Ok(vec![]),
    };
    let adj = build_adjacency(brain)?;
    let mut results = Vec::new();
    let mut path = vec![from_id];
    let mut visited = HashSet::new();
    visited.insert(from_id);
    dfs_all_paths(
        &adj,
        from_id,
        to_id,
        max_depth,
        &mut visited,
        &mut path,
        &mut results,
    );
    Ok(results)
}

fn dfs_all_paths(
    adj: &HashMap<i64, Vec<i64>>,
    current: i64,
    target: i64,
    max_depth: usize,
    visited: &mut HashSet<i64>,
    path: &mut Vec<i64>,
    results: &mut Vec<Vec<i64>>,
) {
    if current == target {
        results.push(path.clone());
        return;
    }
    if path.len() > max_depth {
        return;
    }
    if let Some(neighbors) = adj.get(&current) {
        for &next in neighbors {
            if !visited.contains(&next) {
                visited.insert(next);
                path.push(next);
                dfs_all_paths(adj, next, target, max_depth, visited, path, results);
                path.pop();
                visited.remove(&next);
            }
        }
    }
}

pub fn detect_communities(brain: &Brain) -> Result<HashMap<i64, Vec<i64>>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let adj = build_adjacency(brain)?;
    let mut labels: HashMap<i64, i64> = entities.iter().map(|e| (e.id, e.id)).collect();
    for _ in 0..50 {
        let mut changed = false;
        for e in &entities {
            if let Some(neighbors) = adj.get(&e.id) {
                if neighbors.is_empty() {
                    continue;
                }
                let mut freq: HashMap<i64, usize> = HashMap::new();
                for &n in neighbors {
                    if let Some(&lbl) = labels.get(&n) {
                        *freq.entry(lbl).or_insert(0) += 1;
                    }
                }
                if let Some((&best_label, _)) = freq.iter().max_by_key(|(_, &c)| c) {
                    if labels.get(&e.id) != Some(&best_label) {
                        labels.insert(e.id, best_label);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut communities: HashMap<i64, Vec<i64>> = HashMap::new();
    for (eid, lbl) in &labels {
        communities.entry(*lbl).or_default().push(*eid);
    }
    Ok(communities)
}

pub fn pagerank(
    brain: &Brain,
    damping: f64,
    iterations: usize,
) -> Result<HashMap<i64, f64>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let n = entities.len();
    if n == 0 {
        return Ok(HashMap::new());
    }
    let ids: Vec<i64> = entities.iter().map(|e| e.id).collect();
    let id_set: HashSet<i64> = ids.iter().copied().collect();
    let mut out_links: HashMap<i64, Vec<i64>> = HashMap::new();
    for r in &relations {
        if id_set.contains(&r.subject_id) && id_set.contains(&r.object_id) {
            out_links.entry(r.subject_id).or_default().push(r.object_id);
        }
    }
    let mut scores: HashMap<i64, f64> = ids.iter().map(|&id| (id, 1.0 / n as f64)).collect();
    for _ in 0..iterations {
        let mut new_scores: HashMap<i64, f64> = ids
            .iter()
            .map(|&id| (id, (1.0 - damping) / n as f64))
            .collect();
        for &id in &ids {
            let out = out_links.get(&id);
            let out_count = out.map_or(0, |v| v.len());
            if out_count > 0 {
                let share = scores[&id] / out_count as f64;
                for &target in out.unwrap() {
                    *new_scores.entry(target).or_insert(0.0) += damping * share;
                }
            } else {
                let share = scores[&id] / n as f64;
                for &other in &ids {
                    *new_scores.entry(other).or_insert(0.0) += damping * share;
                }
            }
        }
        scores = new_scores;
    }
    Ok(scores)
}

pub fn infer_transitive(brain: &Brain) -> Result<Vec<(String, String, String)>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let transitive_preds: HashSet<&str> = [
        "is",
        "contains",
        "part_of",
        "located_in",
        "member_of",
        "subclass_of",
        "belongs_to",
        "created_by",
        "owned_by",
    ]
    .iter()
    .copied()
    .collect();
    let mut by_pred: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
    for r in &relations {
        by_pred
            .entry(r.predicate.clone())
            .or_default()
            .push((r.subject_id, r.object_id));
    }
    let mut inferred = Vec::new();
    let existing: HashSet<(i64, String, i64)> = relations
        .iter()
        .map(|r| (r.subject_id, r.predicate.clone(), r.object_id))
        .collect();
    for (pred, edges) in &by_pred {
        if !transitive_preds.contains(pred.as_str()) {
            continue;
        }
        let mut fwd: HashMap<i64, Vec<i64>> = HashMap::new();
        for &(s, o) in edges {
            fwd.entry(s).or_default().push(o);
        }
        for &(a, b) in edges {
            if let Some(cs) = fwd.get(&b) {
                for &c in cs {
                    if a != c && !existing.contains(&(a, pred.clone(), c)) {
                        let a_name = brain
                            .get_entity_by_id(a)?
                            .map(|e| e.name)
                            .unwrap_or_default();
                        let c_name = brain
                            .get_entity_by_id(c)?
                            .map(|e| e.name)
                            .unwrap_or_default();
                        if !a_name.is_empty() && !c_name.is_empty() {
                            inferred.push((a_name, pred.clone(), c_name));
                        }
                    }
                }
            }
        }
    }
    Ok(inferred)
}

pub fn detect_contradictions(
    brain: &Brain,
) -> Result<Vec<(String, String, Vec<String>)>, rusqlite::Error> {
    let singleton_keys: HashSet<&str> = [
        "capital",
        "population",
        "founded",
        "ceo",
        "president",
        "born",
        "died",
        "headquarters",
        "currency",
        "language",
    ]
    .iter()
    .copied()
    .collect();
    let entities = brain.all_entities()?;
    let mut contradictions = Vec::new();
    for entity in &entities {
        let facts = brain.get_facts_for(entity.id)?;
        let mut by_key: HashMap<String, Vec<String>> = HashMap::new();
        for f in &facts {
            by_key
                .entry(f.key.clone())
                .or_default()
                .push(f.value.clone());
        }
        for (key, values) in &by_key {
            if singleton_keys.contains(key.as_str()) && values.len() > 1 {
                contradictions.push((entity.name.clone(), key.clone(), values.clone()));
            }
        }
    }
    Ok(contradictions)
}

pub fn merge_near_duplicates(brain: &Brain) -> Result<Vec<(String, String)>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let mut merged = Vec::new();
    let mut absorbed: HashSet<i64> = HashSet::new();
    for i in 0..entities.len() {
        if absorbed.contains(&entities[i].id) {
            continue;
        }
        for j in (i + 1)..entities.len() {
            if absorbed.contains(&entities[j].id) {
                continue;
            }
            if entities[i].entity_type != entities[j].entity_type {
                continue;
            }
            let dist = levenshtein(
                &entities[i].name.to_lowercase(),
                &entities[j].name.to_lowercase(),
            );
            if dist > 0 && dist < 3 {
                let (keep, remove) = if entities[i].confidence >= entities[j].confidence {
                    (&entities[i], &entities[j])
                } else {
                    (&entities[j], &entities[i])
                };
                brain.merge_entities(remove.id, keep.id)?;
                absorbed.insert(remove.id);
                merged.push((keep.name.clone(), remove.name.clone()));
            }
        }
    }
    Ok(merged)
}

fn build_adjacency(brain: &Brain) -> Result<HashMap<i64, Vec<i64>>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();
    for r in &relations {
        adj.entry(r.subject_id).or_default().push(r.object_id);
        adj.entry(r.object_id).or_default().push(r.subject_id);
    }
    Ok(adj)
}

fn bfs_path(
    adj: &HashMap<i64, Vec<i64>>,
    from: i64,
    to: i64,
) -> Result<Option<Vec<i64>>, rusqlite::Error> {
    if from == to {
        return Ok(Some(vec![from]));
    }
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut parent: HashMap<i64, i64> = HashMap::new();
    visited.insert(from);
    queue.push_back(from);
    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = adj.get(&current) {
            for &next in neighbors {
                if !visited.contains(&next) {
                    visited.insert(next);
                    parent.insert(next, current);
                    if next == to {
                        let mut path = vec![to];
                        let mut cur = to;
                        while let Some(&p) = parent.get(&cur) {
                            path.push(p);
                            cur = p;
                        }
                        path.reverse();
                        return Ok(Some(path));
                    }
                    queue.push_back(next);
                }
            }
        }
    }
    Ok(None)
}

/// Betweenness centrality — approximate via sampled BFS from up to `sample` nodes.
pub fn betweenness_centrality(
    brain: &Brain,
    sample: usize,
) -> Result<HashMap<i64, f64>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let ids: Vec<i64> = adj.keys().copied().collect();
    let n = ids.len();
    if n == 0 {
        return Ok(HashMap::new());
    }
    let mut centrality: HashMap<i64, f64> = ids.iter().map(|&id| (id, 0.0)).collect();

    // Sample source nodes (deterministic: evenly spaced)
    let step = if n <= sample { 1 } else { n / sample };
    let sources: Vec<i64> = ids.iter().step_by(step).copied().take(sample).collect();

    for &s in &sources {
        // Brandes-style single-source shortest paths
        let mut stack = Vec::new();
        let mut pred: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut sigma: HashMap<i64, f64> = ids.iter().map(|&id| (id, 0.0)).collect();
        let mut dist: HashMap<i64, i64> = ids.iter().map(|&id| (id, -1)).collect();
        let mut delta: HashMap<i64, f64> = ids.iter().map(|&id| (id, 0.0)).collect();

        *sigma.get_mut(&s).unwrap() = 1.0;
        *dist.get_mut(&s).unwrap() = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let d_v = dist[&v];
            if let Some(neighbors) = adj.get(&v) {
                for &w in neighbors {
                    if dist[&w] < 0 {
                        *dist.get_mut(&w).unwrap() = d_v + 1;
                        queue.push_back(w);
                    }
                    if dist[&w] == d_v + 1 {
                        *sigma.get_mut(&w).unwrap() += sigma[&v];
                        pred.entry(w).or_default().push(v);
                    }
                }
            }
        }

        while let Some(w) = stack.pop() {
            if let Some(preds) = pred.get(&w) {
                for &v in preds {
                    let d = sigma[&v] / sigma[&w] * (1.0 + delta[&w]);
                    *delta.get_mut(&v).unwrap() += d;
                }
            }
            if w != s {
                *centrality.get_mut(&w).unwrap() += delta[&w];
            }
        }
    }

    // Normalize
    let scale = if sources.len() < n {
        n as f64 / sources.len() as f64
    } else {
        1.0
    };
    for v in centrality.values_mut() {
        *v *= scale;
    }
    Ok(centrality)
}

/// Find bridge edges — edges whose removal disconnects components.
pub fn find_bridges(brain: &Brain) -> Result<Vec<(i64, i64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let ids: Vec<i64> = adj.keys().copied().collect();
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let mut disc: HashMap<i64, i64> = HashMap::new();
    let mut low: HashMap<i64, i64> = HashMap::new();
    let mut bridges = Vec::new();
    let mut timer: i64 = 0;

    fn dfs_bridge(
        u: i64,
        parent: i64,
        adj: &HashMap<i64, Vec<i64>>,
        disc: &mut HashMap<i64, i64>,
        low: &mut HashMap<i64, i64>,
        timer: &mut i64,
        bridges: &mut Vec<(i64, i64)>,
    ) {
        disc.insert(u, *timer);
        low.insert(u, *timer);
        *timer += 1;
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                if !disc.contains_key(&v) {
                    dfs_bridge(v, u, adj, disc, low, timer, bridges);
                    let lv = low[&v];
                    let lu = low[&u];
                    if lv < lu {
                        low.insert(u, lv);
                    }
                    if low[&v] > disc[&u] {
                        bridges.push((u, v));
                    }
                } else if v != parent {
                    let dv = disc[&v];
                    let lu = low[&u];
                    if dv < lu {
                        low.insert(u, dv);
                    }
                }
            }
        }
    }

    for &id in &ids {
        if !disc.contains_key(&id) {
            dfs_bridge(id, -1, &adj, &mut disc, &mut low, &mut timer, &mut bridges);
        }
    }
    Ok(bridges)
}

/// Knowledge density: average relations per entity, grouped by entity_type.
pub fn knowledge_density(brain: &Brain) -> Result<HashMap<String, (usize, f64)>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut rel_count: HashMap<i64, usize> = HashMap::new();
    for r in &relations {
        *rel_count.entry(r.subject_id).or_insert(0) += 1;
        *rel_count.entry(r.object_id).or_insert(0) += 1;
    }
    let mut type_stats: HashMap<String, (usize, usize)> = HashMap::new(); // (entity_count, total_rels)
    for e in &entities {
        let entry = type_stats.entry(e.entity_type.clone()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += rel_count.get(&e.id).copied().unwrap_or(0);
    }
    Ok(type_stats
        .into_iter()
        .map(|(t, (count, rels))| {
            let density = if count > 0 {
                rels as f64 / count as f64
            } else {
                0.0
            };
            (t, (count, density))
        })
        .collect())
}

/// Find connected components and their sizes.
pub fn connected_components(brain: &Brain) -> Result<Vec<Vec<i64>>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let mut visited = HashSet::new();
    let mut components = Vec::new();
    for &start in adj.keys() {
        if visited.contains(&start) {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(node) = queue.pop_front() {
            component.push(node);
            if let Some(neighbors) = adj.get(&node) {
                for &n in neighbors {
                    if visited.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
        }
        components.push(component);
    }
    components.sort_by(|a, b| b.len().cmp(&a.len()));
    Ok(components)
}

pub fn format_path(brain: &Brain, path: &[i64]) -> Result<String, rusqlite::Error> {
    let mut names = Vec::new();
    for &id in path {
        let name = brain
            .get_entity_by_id(id)?
            .map(|e| e.name)
            .unwrap_or_else(|| format!("#{id}"));
        names.push(name);
    }
    Ok(names.join(" → "))
}

/// Graph health summary: component count, largest component %, avg degree, isolated nodes.
pub fn graph_health(brain: &Brain) -> Result<HashMap<String, f64>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let n = entities.len();
    let mut stats: HashMap<String, f64> = HashMap::new();
    stats.insert("entities".into(), n as f64);
    stats.insert("relations".into(), relations.len() as f64);

    // Degree distribution
    let mut degree: HashMap<i64, usize> = HashMap::new();
    for r in &relations {
        *degree.entry(r.subject_id).or_insert(0) += 1;
        *degree.entry(r.object_id).or_insert(0) += 1;
    }
    let connected = degree.len();
    let isolated = n.saturating_sub(connected);
    stats.insert("isolated_entities".into(), isolated as f64);

    let avg_degree = if connected > 0 {
        degree.values().sum::<usize>() as f64 / connected as f64
    } else {
        0.0
    };
    stats.insert("avg_degree".into(), avg_degree);

    // Max degree (hub)
    let max_deg = degree.values().copied().max().unwrap_or(0);
    stats.insert("max_degree".into(), max_deg as f64);

    // Components
    let components = connected_components(brain)?;
    stats.insert("components".into(), components.len() as f64);
    if let Some(largest) = components.first() {
        stats.insert("largest_component".into(), largest.len() as f64);
        if n > 0 {
            stats.insert(
                "largest_component_pct".into(),
                100.0 * largest.len() as f64 / n as f64,
            );
        }
    }

    Ok(stats)
}

/// Clustering coefficient per entity: fraction of neighbour pairs that are connected.
pub fn clustering_coefficients(brain: &Brain) -> Result<HashMap<i64, f64>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let mut coeffs = HashMap::new();
    for (&node, neighbors) in &adj {
        let k = neighbors.len();
        if k < 2 {
            coeffs.insert(node, 0.0);
            continue;
        }
        let nb_set: HashSet<i64> = neighbors.iter().copied().collect();
        let mut triangles = 0usize;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if nb_set.contains(&neighbors[j])
                    && adj
                        .get(&neighbors[i])
                        .is_some_and(|v| v.contains(&neighbors[j]))
                {
                    triangles += 1;
                }
            }
        }
        let possible = k * (k - 1) / 2;
        coeffs.insert(node, triangles as f64 / possible as f64);
    }
    Ok(coeffs)
}

/// Weighted PageRank: uses relation confidence as edge weight.
/// Falls back to uniform weights for edges without confidence.
pub fn weighted_pagerank(
    brain: &Brain,
    damping: f64,
    iterations: usize,
) -> Result<HashMap<i64, f64>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let n = entities.len();
    if n == 0 {
        return Ok(HashMap::new());
    }
    let ids: Vec<i64> = entities.iter().map(|e| e.id).collect();
    let id_set: HashSet<i64> = ids.iter().copied().collect();

    // Build weighted adjacency: source → [(target, weight)]
    let mut out_links: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
    for r in &relations {
        if id_set.contains(&r.subject_id) && id_set.contains(&r.object_id) {
            out_links
                .entry(r.subject_id)
                .or_default()
                .push((r.object_id, r.confidence.max(0.01)));
        }
    }

    let mut scores: HashMap<i64, f64> = ids.iter().map(|&id| (id, 1.0 / n as f64)).collect();
    for _ in 0..iterations {
        let mut new_scores: HashMap<i64, f64> = ids
            .iter()
            .map(|&id| (id, (1.0 - damping) / n as f64))
            .collect();
        for &id in &ids {
            if let Some(edges) = out_links.get(&id) {
                let total_weight: f64 = edges.iter().map(|(_, w)| w).sum();
                if total_weight > 0.0 {
                    for &(target, weight) in edges {
                        let share = scores[&id] * weight / total_weight;
                        *new_scores.entry(target).or_insert(0.0) += damping * share;
                    }
                }
            } else {
                // Dangling node: distribute evenly
                let share = scores[&id] / n as f64;
                for &other in &ids {
                    *new_scores.entry(other).or_insert(0.0) += damping * share;
                }
            }
        }
        scores = new_scores;
    }
    Ok(scores)
}

/// Louvain community detection — modularity-based, better quality than label propagation.
pub fn louvain_communities(brain: &Brain) -> Result<HashMap<i64, usize>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let mut adj: HashMap<i64, HashMap<i64, f64>> = HashMap::new();
    let mut total_weight = 0.0_f64;

    for r in &relations {
        let w = r.confidence.max(0.01);
        *adj.entry(r.subject_id)
            .or_default()
            .entry(r.object_id)
            .or_insert(0.0) += w;
        *adj.entry(r.object_id)
            .or_default()
            .entry(r.subject_id)
            .or_insert(0.0) += w;
        total_weight += 2.0 * w; // undirected
    }

    if total_weight == 0.0 {
        return Ok(HashMap::new());
    }

    let nodes: Vec<i64> = adj.keys().copied().collect();
    let mut community: HashMap<i64, usize> = HashMap::new();
    for (i, &n) in nodes.iter().enumerate() {
        community.insert(n, i);
    }

    // Weighted degree
    let mut k: HashMap<i64, f64> = HashMap::new();
    for (&node, neighbors) in &adj {
        k.insert(node, neighbors.values().sum());
    }

    // Iterative modularity optimization
    let m2 = total_weight; // 2m
    for _ in 0..20 {
        let mut moved = false;
        for &node in &nodes {
            let node_comm = community[&node];
            let ki = k[&node];

            // Sum of weights to each neighboring community
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            if let Some(neighbors) = adj.get(&node) {
                for (&nb, &w) in neighbors {
                    let c = community[&nb];
                    *comm_weights.entry(c).or_insert(0.0) += w;
                }
            }

            // Sum of k for each community
            let mut sigma_tot: HashMap<usize, f64> = HashMap::new();
            for (&n, &c) in &community {
                *sigma_tot.entry(c).or_insert(0.0) += k.get(&n).copied().unwrap_or(0.0);
            }

            // Find best community
            let mut best_comm = node_comm;
            let mut best_delta = 0.0_f64;

            // Remove node from its community for calculation
            let sigma_in_old = comm_weights.get(&node_comm).copied().unwrap_or(0.0);
            let sigma_tot_old = sigma_tot.get(&node_comm).copied().unwrap_or(0.0) - ki;

            for (&c, &w_in) in &comm_weights {
                if c == node_comm {
                    continue;
                }
                let sigma_tot_c = sigma_tot.get(&c).copied().unwrap_or(0.0);
                // Delta Q = [w_in_new/m - sigma_tot_new*ki/m²] - [w_in_old/m - sigma_tot_old*ki/m²]
                let delta =
                    (w_in - sigma_in_old) / m2 - ki * (sigma_tot_c - sigma_tot_old) / (m2 * m2);
                if delta > best_delta {
                    best_delta = delta;
                    best_comm = c;
                }
            }

            if best_comm != node_comm {
                community.insert(node, best_comm);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    Ok(community)
}

/// Global clustering coefficient (average of local coefficients).
pub fn global_clustering(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let coeffs = clustering_coefficients(brain)?;
    if coeffs.is_empty() {
        return Ok(0.0);
    }
    let sum: f64 = coeffs.values().sum();
    Ok(sum / coeffs.len() as f64)
}

/// Find hub entities (highest degree) — returns (entity_id, degree) sorted descending.
pub fn find_hubs(brain: &Brain, limit: usize) -> Result<Vec<(i64, usize)>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let mut degree: HashMap<i64, usize> = HashMap::new();
    for r in &relations {
        *degree.entry(r.subject_id).or_insert(0) += 1;
        *degree.entry(r.object_id).or_insert(0) += 1;
    }
    let mut ranked: Vec<(i64, usize)> = degree.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    ranked.truncate(limit);
    Ok(ranked)
}

/// Degree distribution: returns (degree → count) histogram.
pub fn degree_distribution(brain: &Brain) -> Result<Vec<(usize, usize)>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let mut degree: HashMap<i64, usize> = HashMap::new();
    for r in &relations {
        *degree.entry(r.subject_id).or_insert(0) += 1;
        *degree.entry(r.object_id).or_insert(0) += 1;
    }
    let mut dist: HashMap<usize, usize> = HashMap::new();
    for &d in degree.values() {
        *dist.entry(d).or_insert(0) += 1;
    }
    let mut result: Vec<(usize, usize)> = dist.into_iter().collect();
    result.sort_by_key(|&(d, _)| d);
    Ok(result)
}

/// Graph density: ratio of actual edges to possible edges.
pub fn graph_density(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let n = brain.all_entities()?.len() as f64;
    let m = brain.all_relations()?.len() as f64;
    if n <= 1.0 {
        return Ok(0.0);
    }
    Ok(2.0 * m / (n * (n - 1.0)))
}

/// Power-law exponent estimate via log-log regression on degree distribution.
/// Returns (exponent, r_squared) — r² > 0.8 suggests scale-free network.
pub fn power_law_estimate(brain: &Brain) -> Result<(f64, f64), rusqlite::Error> {
    let dist = degree_distribution(brain)?;
    let points: Vec<(f64, f64)> = dist
        .iter()
        .filter(|&&(d, c)| d > 0 && c > 0)
        .map(|&(d, c)| ((d as f64).ln(), (c as f64).ln()))
        .collect();
    if points.len() < 3 {
        return Ok((0.0, 0.0));
    }
    // Simple linear regression on log-log
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return Ok((0.0, 0.0));
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    // R²
    let mean_y = sum_y / n;
    let ss_tot: f64 = points.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = points
        .iter()
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };
    Ok((-slope, r2)) // Negate slope since power law has negative exponent
}

/// Entity similarity: Jaccard similarity based on shared predicates.
/// Returns top similar pairs above threshold.
pub fn entity_similarity(
    brain: &Brain,
    min_similarity: f64,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    // Build predicate-set per entity (both as subject and object)
    let mut entity_preds: HashMap<i64, HashSet<String>> = HashMap::new();
    for r in &relations {
        entity_preds
            .entry(r.subject_id)
            .or_default()
            .insert(format!("out:{}", r.predicate));
        entity_preds
            .entry(r.object_id)
            .or_default()
            .insert(format!("in:{}", r.predicate));
    }
    // Only consider entities with >= 2 predicates
    let candidates: Vec<(i64, &HashSet<String>)> = entity_preds
        .iter()
        .filter(|(_, preds)| preds.len() >= 2)
        .map(|(&id, preds)| (id, preds))
        .collect();

    let mut results = Vec::new();
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let (id_a, preds_a) = candidates[i];
            let (id_b, preds_b) = candidates[j];
            let intersection = preds_a.intersection(preds_b).count();
            if intersection == 0 {
                continue;
            }
            let union = preds_a.union(preds_b).count();
            let jaccard = intersection as f64 / union as f64;
            if jaccard >= min_similarity {
                results.push((id_a, id_b, jaccard));
            }
        }
    }
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Graph summary: compact human-readable overview.
pub fn graph_summary(brain: &Brain) -> Result<String, rusqlite::Error> {
    let health = graph_health(brain)?;
    let density = graph_density(brain)?;
    let (exponent, r2) = power_law_estimate(brain)?;

    let entities = health.get("entities").copied().unwrap_or(0.0) as usize;
    let rels = health.get("relations").copied().unwrap_or(0.0) as usize;
    let isolated = health.get("isolated_entities").copied().unwrap_or(0.0) as usize;
    let components = health.get("components").copied().unwrap_or(0.0) as usize;
    let largest_pct = health.get("largest_component_pct").copied().unwrap_or(0.0);
    let avg_deg = health.get("avg_degree").copied().unwrap_or(0.0);

    Ok(format!(
        "Graph: {} entities, {} relations, density {:.6}\n\
         Connected: {} ({:.1}% isolated), {} components (largest {:.1}%)\n\
         Avg degree: {:.2}, Power law: α={:.2} (R²={:.2} {})\n",
        entities,
        rels,
        density,
        entities - isolated,
        if entities > 0 {
            100.0 * isolated as f64 / entities as f64
        } else {
            0.0
        },
        components,
        largest_pct,
        avg_deg,
        exponent,
        r2,
        if r2 > 0.8 {
            "scale-free"
        } else {
            "not scale-free"
        }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Brain;

    fn setup() -> Brain {
        let brain = Brain::open_in_memory().unwrap();
        let a = brain.upsert_entity("Alice", "person").unwrap();
        let b = brain.upsert_entity("Bob", "person").unwrap();
        let c = brain.upsert_entity("Charlie", "person").unwrap();
        let d = brain.upsert_entity("Diana", "person").unwrap();
        brain.upsert_relation(a, "knows", b, "test").unwrap();
        brain.upsert_relation(b, "knows", c, "test").unwrap();
        brain.upsert_relation(c, "knows", d, "test").unwrap();
        brain
    }

    #[test]
    fn test_shortest_path_direct() {
        let brain = setup();
        let path = shortest_path(&brain, "Alice", "Bob").unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 2);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let brain = setup();
        let path = shortest_path(&brain, "Alice", "Diana").unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 4);
    }

    #[test]
    fn test_shortest_path_not_found() {
        let brain = setup();
        brain.upsert_entity("Isolated", "person").unwrap();
        let path = shortest_path(&brain, "Alice", "Isolated").unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_unknown() {
        let brain = setup();
        let path = shortest_path(&brain, "Alice", "Nobody").unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_all_paths() {
        let brain = setup();
        let a = brain.get_entity_by_name("Alice").unwrap().unwrap().id;
        let c = brain.get_entity_by_name("Charlie").unwrap().unwrap().id;
        brain.upsert_relation(a, "friend_of", c, "test").unwrap();
        let paths = all_paths(&brain, "Alice", "Charlie", 5).unwrap();
        assert!(paths.len() >= 2);
    }

    #[test]
    fn test_all_paths_depth_limit() {
        let brain = setup();
        let paths = all_paths(&brain, "Alice", "Diana", 2).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn test_communities() {
        let brain = Brain::open_in_memory().unwrap();
        let a = brain.upsert_entity("A1", "node").unwrap();
        let b = brain.upsert_entity("A2", "node").unwrap();
        let c = brain.upsert_entity("B1", "node").unwrap();
        let d = brain.upsert_entity("B2", "node").unwrap();
        brain.upsert_relation(a, "link", b, "test").unwrap();
        brain.upsert_relation(c, "link", d, "test").unwrap();
        let communities = detect_communities(&brain).unwrap();
        assert!(communities.len() >= 2);
    }

    #[test]
    fn test_pagerank() {
        let brain = setup();
        let scores = pagerank(&brain, 0.85, 20).unwrap();
        assert_eq!(scores.len(), 4);
        let total: f64 = scores.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_empty() {
        let brain = Brain::open_in_memory().unwrap();
        let scores = pagerank(&brain, 0.85, 20).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_infer_transitive() {
        let brain = Brain::open_in_memory().unwrap();
        let a = brain.upsert_entity("Paris", "city").unwrap();
        let b = brain.upsert_entity("France", "country").unwrap();
        let c = brain.upsert_entity("Europe", "continent").unwrap();
        brain.upsert_relation(a, "located_in", b, "test").unwrap();
        brain.upsert_relation(b, "located_in", c, "test").unwrap();
        let inferred = infer_transitive(&brain).unwrap();
        assert!(!inferred.is_empty());
        assert!(inferred
            .iter()
            .any(|(s, _, o)| s == "Paris" && o == "Europe"));
    }

    #[test]
    fn test_detect_contradictions() {
        let brain = Brain::open_in_memory().unwrap();
        let e = brain.upsert_entity("France", "country").unwrap();
        brain.upsert_fact(e, "capital", "Paris", "src1").unwrap();
        brain.upsert_fact(e, "capital", "Lyon", "src2").unwrap();
        let contradictions = detect_contradictions(&brain).unwrap();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0].2.len(), 2);
    }

    #[test]
    fn test_no_contradictions() {
        let brain = Brain::open_in_memory().unwrap();
        let e = brain.upsert_entity("France", "country").unwrap();
        brain.upsert_fact(e, "capital", "Paris", "src1").unwrap();
        assert!(detect_contradictions(&brain).unwrap().is_empty());
    }

    #[test]
    fn test_merge_near_duplicates() {
        let brain = Brain::open_in_memory().unwrap();
        brain.upsert_entity("Google", "company").unwrap();
        brain.upsert_entity("Gogle", "company").unwrap();
        brain.upsert_entity("Microsoft", "company").unwrap();
        let merged = merge_near_duplicates(&brain).unwrap();
        assert_eq!(merged.len(), 1);
        let entities = brain.all_entities().unwrap();
        assert_eq!(
            entities
                .iter()
                .filter(|e| e.entity_type == "company")
                .count(),
            2
        );
    }

    #[test]
    fn test_format_path() {
        let brain = setup();
        let path = shortest_path(&brain, "Alice", "Charlie").unwrap().unwrap();
        let formatted = format_path(&brain, &path).unwrap();
        assert!(formatted.contains("Alice"));
        assert!(formatted.contains("Charlie"));
    }

    #[test]
    fn test_bfs_self_path() {
        let brain = setup();
        let path = shortest_path(&brain, "Alice", "Alice").unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 1);
    }
}
