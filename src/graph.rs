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
        // Collect dangling node mass in one pass (avoid O(n*d) inner loop)
        let mut dangling_sum = 0.0_f64;
        let mut new_scores: HashMap<i64, f64> = ids
            .iter()
            .map(|&id| (id, (1.0 - damping) / n as f64))
            .collect();
        for &id in &ids {
            if let Some(out) = out_links.get(&id) {
                let share = scores[&id] / out.len() as f64;
                for &target in out {
                    *new_scores.entry(target).or_insert(0.0) += damping * share;
                }
            } else {
                dangling_sum += scores[&id];
            }
        }
        // Distribute dangling mass uniformly
        let dangling_share = damping * dangling_sum / n as f64;
        for v in new_scores.values_mut() {
            *v += dangling_share;
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
    components.sort_by_key(|b| std::cmp::Reverse(b.len()));
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
        let mut dangling_sum = 0.0_f64;
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
                } else {
                    dangling_sum += scores[&id];
                }
            } else {
                dangling_sum += scores[&id];
            }
        }
        let dangling_share = damping * dangling_sum / n as f64;
        for v in new_scores.values_mut() {
            *v += dangling_share;
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

    // Iterative modularity optimization with incremental sigma_tot
    let m2 = total_weight; // 2m

    // Maintain sigma_tot incrementally instead of recomputing O(n) per node
    let mut sigma_tot: HashMap<usize, f64> = HashMap::new();
    for (&n, &c) in &community {
        *sigma_tot.entry(c).or_insert(0.0) += k.get(&n).copied().unwrap_or(0.0);
    }

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
                // Update sigma_tot incrementally
                *sigma_tot.entry(node_comm).or_insert(0.0) -= ki;
                *sigma_tot.entry(best_comm).or_insert(0.0) += ki;
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

/// Compute modularity Q for a given community assignment.
/// Q = (1/2m) Σ [A_ij - k_i*k_j/(2m)] δ(c_i, c_j)
pub fn modularity(
    brain: &Brain,
    communities: &HashMap<i64, usize>,
) -> Result<f64, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let m = relations.len() as f64;
    if m == 0.0 {
        return Ok(0.0);
    }
    let mut degree: HashMap<i64, f64> = HashMap::new();
    for r in &relations {
        *degree.entry(r.subject_id).or_insert(0.0) += 1.0;
        *degree.entry(r.object_id).or_insert(0.0) += 1.0;
    }
    let m2 = 2.0 * m;
    let mut q = 0.0_f64;
    for r in &relations {
        let ci = communities
            .get(&r.subject_id)
            .copied()
            .unwrap_or(usize::MAX);
        let cj = communities.get(&r.object_id).copied().unwrap_or(usize::MAX);
        if ci == cj {
            let ki = degree.get(&r.subject_id).copied().unwrap_or(0.0);
            let kj = degree.get(&r.object_id).copied().unwrap_or(0.0);
            q += 1.0 - ki * kj / m2;
        }
    }
    Ok(q / m2)
}

/// Graph compaction ratio: what fraction of entities are in the largest connected component.
/// Higher = more connected graph. Useful for tracking improvement over time.
pub fn compaction_ratio(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let n = entities.len();
    if n == 0 {
        return Ok(0.0);
    }
    let components = connected_components(brain)?;
    let largest = components.first().map(|c| c.len()).unwrap_or(0);
    Ok(largest as f64 / n as f64)
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

/// Estimated graph diameter and average shortest path length via sampled BFS.
/// Returns (estimated_diameter, avg_path_length, sample_size).
/// Samples up to `sample` source nodes for efficiency.
pub fn estimated_diameter(
    brain: &Brain,
    sample: usize,
) -> Result<(usize, f64, usize), rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let ids: Vec<i64> = adj.keys().copied().collect();
    let n = ids.len();
    if n == 0 {
        return Ok((0, 0.0, 0));
    }

    let step = if n <= sample { 1 } else { n / sample };
    let sources: Vec<i64> = ids.iter().step_by(step).copied().take(sample).collect();

    let mut max_dist = 0usize;
    let mut total_dist = 0u64;
    let mut pair_count = 0u64;

    for &s in &sources {
        let mut visited: HashMap<i64, usize> = HashMap::new();
        visited.insert(s, 0);
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            let d = visited[&v];
            if let Some(neighbors) = adj.get(&v) {
                for &w in neighbors {
                    if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(w) {
                        let nd = d + 1;
                        e.insert(nd);
                        if nd > max_dist {
                            max_dist = nd;
                        }
                        total_dist += nd as u64;
                        pair_count += 1;
                        queue.push_back(w);
                    }
                }
            }
        }
    }

    let avg = if pair_count > 0 {
        total_dist as f64 / pair_count as f64
    } else {
        0.0
    };
    Ok((max_dist, avg, sources.len()))
}

/// Edge reciprocity: fraction of directed edges that have a reverse edge.
/// High reciprocity suggests symmetric relationships (knows, related_to).
/// Low reciprocity suggests hierarchical relationships (part_of, located_in).
pub fn edge_reciprocity(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let mut directed: HashSet<(i64, i64)> = HashSet::new();
    for r in &relations {
        directed.insert((r.subject_id, r.object_id));
    }
    if directed.is_empty() {
        return Ok(0.0);
    }
    let reciprocal = directed
        .iter()
        .filter(|(s, o)| directed.contains(&(*o, *s)))
        .count();
    Ok(reciprocal as f64 / directed.len() as f64)
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

/// Find articulation points — entities whose removal disconnects the graph.
pub fn find_articulation_points(brain: &Brain) -> Result<Vec<i64>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let ids: Vec<i64> = adj.keys().copied().collect();
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let mut disc: HashMap<i64, i64> = HashMap::new();
    let mut low: HashMap<i64, i64> = HashMap::new();
    let mut parent: HashMap<i64, i64> = HashMap::new();
    let mut ap: HashSet<i64> = HashSet::new();
    let mut timer: i64 = 0;

    fn dfs_ap(
        u: i64,
        adj: &HashMap<i64, Vec<i64>>,
        disc: &mut HashMap<i64, i64>,
        low: &mut HashMap<i64, i64>,
        parent: &mut HashMap<i64, i64>,
        ap: &mut HashSet<i64>,
        timer: &mut i64,
    ) {
        disc.insert(u, *timer);
        low.insert(u, *timer);
        *timer += 1;
        let mut children = 0i64;
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                if !disc.contains_key(&v) {
                    children += 1;
                    parent.insert(v, u);
                    dfs_ap(v, adj, disc, low, parent, ap, timer);
                    let lv = low[&v];
                    let lu = low[&u];
                    if lv < lu {
                        low.insert(u, lv);
                    }
                    // u is AP if: (1) root with 2+ children, or (2) non-root with low[v] >= disc[u]
                    let is_root = !parent.contains_key(&u);
                    if is_root && children > 1 {
                        ap.insert(u);
                    }
                    if !is_root && low[&v] >= disc[&u] {
                        ap.insert(u);
                    }
                } else if parent.get(&u) != Some(&v) {
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
            dfs_ap(
                id,
                &adj,
                &mut disc,
                &mut low,
                &mut parent,
                &mut ap,
                &mut timer,
            );
        }
    }
    Ok(ap.into_iter().collect())
}

/// K-core decomposition: find the maximal subgraph where every node has degree ≥ k.
/// Returns a map of entity_id → core_number (the highest k for which the entity is in the k-core).
/// Higher core number = more deeply embedded in the dense part of the graph.
pub fn k_core_decomposition(brain: &Brain) -> Result<HashMap<i64, usize>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let mut degree: HashMap<i64, usize> = adj
        .iter()
        .map(|(&id, neighbors)| (id, neighbors.len()))
        .collect();
    let mut core: HashMap<i64, usize> = HashMap::new();
    let mut remaining: HashSet<i64> = degree.keys().copied().collect();

    // Iteratively peel nodes with lowest degree
    while !remaining.is_empty() {
        // Find current minimum degree among remaining nodes
        let min_deg = remaining
            .iter()
            .map(|id| degree.get(id).copied().unwrap_or(0))
            .min()
            .unwrap_or(0);

        // Collect all nodes with this minimum degree
        let to_remove: Vec<i64> = remaining
            .iter()
            .filter(|id| degree.get(id).copied().unwrap_or(0) <= min_deg)
            .copied()
            .collect();

        for &node in &to_remove {
            core.insert(node, min_deg);
            remaining.remove(&node);
            // Reduce degree of remaining neighbors
            if let Some(neighbors) = adj.get(&node) {
                for &nb in neighbors {
                    if remaining.contains(&nb) {
                        if let Some(d) = degree.get_mut(&nb) {
                            *d = d.saturating_sub(1);
                        }
                    }
                }
            }
        }
    }
    Ok(core)
}

/// Find the densest k-core (highest k with ≥ min_size nodes).
/// Returns (k, node_ids) for the densest core meeting the size threshold.
pub fn densest_core(brain: &Brain, min_size: usize) -> Result<(usize, Vec<i64>), rusqlite::Error> {
    let cores = k_core_decomposition(brain)?;
    let max_k = cores.values().copied().max().unwrap_or(0);

    for k in (1..=max_k).rev() {
        let members: Vec<i64> = cores
            .iter()
            .filter(|(_, &v)| v >= k)
            .map(|(&id, _)| id)
            .collect();
        if members.len() >= min_size {
            return Ok((k, members));
        }
    }
    Ok((0, vec![]))
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

    let (diameter, avg_path, _) = estimated_diameter(brain, 50)?;

    Ok(format!(
        "Graph: {} entities, {} relations, density {:.6}\n\
         Connected: {} ({:.1}% isolated), {} components (largest {:.1}%)\n\
         Avg degree: {:.2}, Diameter: ~{}, Avg path: {:.2}\n\
         Power law: α={:.2} (R²={:.2} {})\n",
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
        diameter,
        avg_path,
        exponent,
        r2,
        if r2 > 0.8 {
            "scale-free"
        } else {
            "not scale-free"
        }
    ))
}

/// Small-world coefficient: σ = (C/C_rand) / (L/L_rand)
/// where C = clustering coefficient, L = avg path length,
/// C_rand = k/n, L_rand = ln(n)/ln(k) for random graph with same n,k.
/// σ >> 1 indicates small-world network. Returns (sigma, C, L).
pub fn small_world_coefficient(brain: &Brain) -> Result<(f64, f64, f64), rusqlite::Error> {
    let health = graph_health(brain)?;
    let n = health.get("entities").copied().unwrap_or(0.0);
    let avg_k = health.get("avg_degree").copied().unwrap_or(0.0);

    if n < 10.0 || avg_k < 1.0 {
        return Ok((0.0, 0.0, 0.0));
    }

    let c = global_clustering(brain)?;
    let (_, avg_l, _) = estimated_diameter(brain, 50)?;

    if avg_l <= 0.0 {
        return Ok((0.0, c, avg_l));
    }

    // Random graph equivalents
    let c_rand = avg_k / n;
    let l_rand = if avg_k > 1.0 {
        n.ln() / avg_k.ln()
    } else {
        n // degenerate
    };

    let gamma = if c_rand > 0.0 { c / c_rand } else { 0.0 };
    let lambda = if l_rand > 0.0 { avg_l / l_rand } else { 1.0 };
    let sigma = if lambda > 0.0 { gamma / lambda } else { 0.0 };

    Ok((sigma, c, avg_l))
}

/// Inter-community edge density: ratio of edges between communities vs within.
/// Higher ratio suggests the graph needs more bridging connections.
/// Returns (intra_edges, inter_edges, ratio).
pub fn inter_community_density(brain: &Brain) -> Result<(usize, usize, f64), rusqlite::Error> {
    let communities = louvain_communities(brain)?;
    let relations = brain.all_relations()?;
    let mut intra = 0usize;
    let mut inter = 0usize;
    for r in &relations {
        let cs = communities.get(&r.subject_id).copied();
        let co = communities.get(&r.object_id).copied();
        match (cs, co) {
            (Some(a), Some(b)) if a == b => intra += 1,
            (Some(_), Some(_)) => inter += 1,
            _ => {}
        }
    }
    let ratio = if intra > 0 {
        inter as f64 / intra as f64
    } else {
        0.0
    };
    Ok((intra, inter, ratio))
}

/// Save a snapshot of graph health metrics for historical trend tracking.
/// Creates the snapshots table if needed, then inserts current metrics.
/// Returns the snapshot id.
pub fn save_graph_snapshot(brain: &Brain) -> Result<i64, rusqlite::Error> {
    brain.with_conn(|conn| {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS graph_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                taken_at TEXT NOT NULL,
                entities INTEGER NOT NULL,
                relations INTEGER NOT NULL,
                components INTEGER NOT NULL,
                largest_component_pct REAL NOT NULL,
                avg_degree REAL NOT NULL,
                isolated INTEGER NOT NULL,
                density REAL NOT NULL,
                fragmentation REAL NOT NULL DEFAULT 0.0,
                modularity REAL NOT NULL DEFAULT 0.0
            );",
        )?;
        Ok(())
    })?;

    let health = graph_health(brain)?;
    let density_val = graph_density(brain)?;
    let frag = fragmentation_score(brain).unwrap_or(1.0);
    let communities = louvain_communities(brain)?;
    let mod_val = modularity(brain, &communities).unwrap_or(0.0);

    let entities = health.get("entities").copied().unwrap_or(0.0) as i64;
    let relations = health.get("relations").copied().unwrap_or(0.0) as i64;
    let components = health.get("components").copied().unwrap_or(0.0) as i64;
    let largest_pct = health.get("largest_component_pct").copied().unwrap_or(0.0);
    let avg_deg = health.get("avg_degree").copied().unwrap_or(0.0);
    let isolated = health.get("isolated_entities").copied().unwrap_or(0.0) as i64;

    brain.with_conn(|conn| {
        conn.execute(
            "INSERT INTO graph_snapshots (taken_at, entities, relations, components, largest_component_pct, avg_degree, isolated, density, fragmentation, modularity)
             VALUES (datetime('now'), ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![entities, relations, components, largest_pct, avg_deg, isolated, density_val, frag, mod_val],
        )?;
        Ok(conn.last_insert_rowid())
    })
}

/// Get recent graph snapshots for trend analysis.
/// Returns Vec of (taken_at, entities, relations, components, largest_pct, avg_degree, isolated, density, fragmentation, modularity).
#[allow(clippy::type_complexity)]
pub fn get_graph_snapshots(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(String, i64, i64, i64, f64, f64, i64, f64, f64, f64)>, rusqlite::Error> {
    brain.with_conn(|conn| {
        // Table might not exist yet
        let exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='graph_snapshots'",
            [],
            |row| row.get(0),
        )?;
        if !exists {
            return Ok(vec![]);
        }
        let mut stmt = conn.prepare(
            "SELECT taken_at, entities, relations, components, largest_component_pct, avg_degree, isolated, density, fragmentation, modularity
             FROM graph_snapshots ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, f64>(7)?,
                row.get::<_, f64>(8)?,
                row.get::<_, f64>(9)?,
            ))
        })?;
        rows.collect()
    })
}

/// Format a trend comparison between two snapshots.
pub fn format_trend(
    current: &(String, i64, i64, i64, f64, f64, i64, f64, f64, f64),
    previous: &(String, i64, i64, i64, f64, f64, i64, f64, f64, f64),
) -> String {
    let delta = |curr: f64, prev: f64| -> String {
        let d = curr - prev;
        if d.abs() < 0.001 {
            "→".to_string()
        } else if d > 0.0 {
            format!("↑{:.1}", d)
        } else {
            format!("↓{:.1}", d.abs())
        }
    };
    let delta_i = |curr: i64, prev: i64| -> String {
        let d = curr - prev;
        if d == 0 {
            "→".to_string()
        } else if d > 0 {
            format!("↑{}", d)
        } else {
            format!("↓{}", d.abs())
        }
    };
    format!(
        "Since {}: entities {} ({}), relations {} ({}), components {} ({}), \
         largest {:.1}% ({}), avg_degree {:.2} ({}), isolated {} ({}), \
         density {:.6} ({}), fragmentation {:.4} ({})",
        previous.0,
        current.1,
        delta_i(current.1, previous.1),
        current.2,
        delta_i(current.2, previous.2),
        current.3,
        delta_i(current.3, previous.3),
        current.4,
        delta(current.4, previous.4),
        current.5,
        delta(current.5, previous.5),
        current.6,
        delta_i(current.6, previous.6),
        current.7,
        delta(current.7, previous.7),
        current.8,
        delta(current.8, previous.8),
    )
}

/// Actionable graph health recommendations based on current metrics.
/// Returns a list of (priority, recommendation) pairs sorted by importance.
pub fn graph_recommendations(brain: &Brain) -> Result<Vec<(u8, String)>, rusqlite::Error> {
    let health = graph_health(brain)?;
    let mut recs: Vec<(u8, String)> = Vec::new();

    let entities = health.get("entities").copied().unwrap_or(0.0) as usize;
    let isolated = health.get("isolated_entities").copied().unwrap_or(0.0) as usize;
    let components = health.get("components").copied().unwrap_or(0.0) as usize;
    let largest_pct = health.get("largest_component_pct").copied().unwrap_or(0.0);
    let avg_deg = health.get("avg_degree").copied().unwrap_or(0.0);

    if entities > 0 {
        let isolation_pct = 100.0 * isolated as f64 / entities as f64;
        if isolation_pct > 50.0 {
            recs.push((1, format!(
                "CRITICAL: {:.0}% of entities are isolated ({}/{}). Run aggressive entity merging and crawl enrichment.",
                isolation_pct, isolated, entities
            )));
        } else if isolation_pct > 20.0 {
            recs.push((2, format!(
                "HIGH: {:.0}% isolated entities ({}/{}). Consider entity deduplication and targeted crawling.",
                isolation_pct, isolated, entities
            )));
        }
    }

    if components > 50 && largest_pct < 50.0 {
        recs.push((
            2,
            format!(
            "Graph is fragmented: {} components, largest only {:.1}%. Need cross-topic bridging.",
            components, largest_pct
        ),
        ));
    }

    if avg_deg < 2.0 && entities > 100 {
        recs.push((2, format!(
            "Sparse graph: avg degree {:.2}. Most entities have ≤1 connection. Crawl more content for existing topics.",
            avg_deg
        )));
    }

    let density_val = graph_density(brain)?;
    if density_val < 0.0001 && entities > 500 {
        recs.push((3, format!(
            "Very low density ({:.6}). Graph has {} entities but few connections. Focus on depth over breadth.",
            density_val, entities
        )));
    }

    if largest_pct > 90.0 && components < 5 {
        recs.push((
            4,
            "Graph is well-connected! Consider exploring new topic domains.".to_string(),
        ));
    }

    recs.sort_by_key(|r| r.0);
    Ok(recs)
}

/// Adamic-Adar link prediction: scores non-connected entity pairs by summing
/// 1/ln(degree) for each shared neighbor. Higher score = more likely missing link.
/// This is a proven link prediction algorithm from network science.
/// Returns top `limit` predicted links as (entity_a, entity_b, score).
pub fn adamic_adar_predict(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    // Precompute degrees
    let degree: HashMap<i64, usize> = adj
        .iter()
        .map(|(&id, neighbors)| (id, neighbors.len()))
        .collect();

    // Build direct-connection set
    let relations = brain.all_relations()?;
    let mut connected: HashSet<(i64, i64)> = HashSet::new();
    for r in &relations {
        let key = if r.subject_id < r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        connected.insert(key);
    }

    // Only score pairs where both nodes have degree >= 2 (otherwise no shared neighbors)
    let candidates: Vec<i64> = degree
        .iter()
        .filter(|(_, &d)| d >= 2)
        .map(|(&id, _)| id)
        .collect();

    let mut scores: Vec<(i64, i64, f64)> = Vec::new();
    let nb_sets: HashMap<i64, HashSet<i64>> = candidates
        .iter()
        .map(|&id| {
            let set: HashSet<i64> = adj
                .get(&id)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();
            (id, set)
        })
        .collect();

    // Sort candidates by degree descending for better coverage of high-value nodes
    let mut sorted_candidates = candidates.clone();
    sorted_candidates.sort_by(|a, b| degree.get(b).unwrap_or(&0).cmp(degree.get(a).unwrap_or(&0)));
    let scan = sorted_candidates.len().min(500);
    for i in 0..scan {
        let a = sorted_candidates[i];
        let na = &nb_sets[&a];
        for b in &sorted_candidates[(i + 1)..scan] {
            let b = *b;
            let key = if a < b { (a, b) } else { (b, a) };
            if connected.contains(&key) {
                continue;
            }
            let nb = &nb_sets[&b];
            let score: f64 = na
                .intersection(nb)
                .map(|&z| {
                    let dz = degree.get(&z).copied().unwrap_or(1).max(2);
                    1.0 / (dz as f64).ln()
                })
                .sum();
            if score > 0.0 {
                scores.push((a, b, score));
            }
        }
    }
    scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    Ok(scores)
}

/// Resource Allocation Index for link prediction — better than Adamic-Adar for sparse graphs.
/// RA(x,y) = Σ_{z ∈ Γ(x) ∩ Γ(y)} 1/|Γ(z)| (uses inverse degree, not inverse log-degree).
/// In sparse graphs where most nodes have degree 1-3, RA discriminates better because
/// 1/k decays faster than 1/ln(k) for small k, giving more weight to exclusive shared neighbors.
/// Returns top `limit` predicted links as (entity_a, entity_b, score).
pub fn resource_allocation_predict(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let degree: HashMap<i64, usize> = adj
        .iter()
        .map(|(&id, neighbors)| (id, neighbors.len()))
        .collect();

    let relations = brain.all_relations()?;
    let mut connected: HashSet<(i64, i64)> = HashSet::new();
    for r in &relations {
        let key = if r.subject_id < r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        connected.insert(key);
    }

    let candidates: Vec<i64> = degree
        .iter()
        .filter(|(_, &d)| d >= 2)
        .map(|(&id, _)| id)
        .collect();

    let nb_sets: HashMap<i64, HashSet<i64>> = candidates
        .iter()
        .map(|&id| {
            let set: HashSet<i64> = adj
                .get(&id)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();
            (id, set)
        })
        .collect();

    let mut scores: Vec<(i64, i64, f64)> = Vec::new();
    let scan = candidates.len().min(500);
    for i in 0..scan {
        let a = candidates[i];
        let na = &nb_sets[&a];
        for b in &candidates[(i + 1)..scan] {
            let b = *b;
            let key = if a < b { (a, b) } else { (b, a) };
            if connected.contains(&key) {
                continue;
            }
            let nb = &nb_sets[&b];
            let score: f64 = na
                .intersection(nb)
                .map(|&z| {
                    let dz = degree.get(&z).copied().unwrap_or(1).max(1);
                    1.0 / dz as f64
                })
                .sum();
            if score > 0.0 {
                scores.push((a, b, score));
            }
        }
    }
    scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    Ok(scores)
}

/// Common Neighbors with Type Affinity (CNTA) link prediction.
/// Extends common-neighbor scoring by boosting pairs that share the same entity type.
/// In knowledge graphs, same-type entities that share neighbors are more likely to be related
/// (e.g., two "person" entities who share "organization" neighbors likely collaborate).
/// Returns top `limit` predicted links as (entity_a, entity_b, score).
pub fn type_aware_link_predict(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let entities = brain.all_entities()?;
    let id_to_type: HashMap<i64, String> = entities
        .iter()
        .map(|e| (e.id, e.entity_type.clone()))
        .collect();
    let degree: HashMap<i64, usize> = adj
        .iter()
        .map(|(&id, neighbors)| (id, neighbors.len()))
        .collect();

    let relations = brain.all_relations()?;
    let mut connected: HashSet<(i64, i64)> = HashSet::new();
    for r in &relations {
        let key = if r.subject_id < r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        connected.insert(key);
    }

    let candidates: Vec<i64> = degree
        .iter()
        .filter(|(_, &d)| d >= 2)
        .map(|(&id, _)| id)
        .collect();

    let nb_sets: HashMap<i64, HashSet<i64>> = candidates
        .iter()
        .map(|&id| {
            let set: HashSet<i64> = adj
                .get(&id)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();
            (id, set)
        })
        .collect();

    let mut scores: Vec<(i64, i64, f64)> = Vec::new();
    let scan = candidates.len().min(500);
    for i in 0..scan {
        let a = candidates[i];
        let na = &nb_sets[&a];
        let type_a = id_to_type.get(&a).cloned().unwrap_or_default();
        for b in &candidates[(i + 1)..scan] {
            let b = *b;
            let key = if a < b { (a, b) } else { (b, a) };
            if connected.contains(&key) {
                continue;
            }
            let nb = &nb_sets[&b];
            let cn = na.intersection(nb).count() as f64;
            if cn == 0.0 {
                continue;
            }
            let type_b = id_to_type.get(&b).cloned().unwrap_or_default();
            // Type affinity boost: same high-value type gets 1.5x, same type gets 1.2x
            let type_boost = if type_a == type_b {
                let high_value = ["person", "organization", "concept", "technology", "place"];
                if high_value.contains(&type_a.as_str()) {
                    1.5
                } else {
                    1.2
                }
            } else {
                1.0
            };
            let score = cn * type_boost;
            scores.push((a, b, score));
        }
    }
    scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    Ok(scores)
}

/// Predicate entropy: measures the information content / diversity of predicate usage.
/// Higher entropy = more diverse predicate vocabulary = richer semantic graph.
/// Low entropy = over-reliance on a few generic predicates.
/// Returns (entropy_bits, top_predicate, top_predicate_fraction).
pub fn predicate_entropy(brain: &Brain) -> Result<(f64, String, f64), rusqlite::Error> {
    let relations = brain.all_relations()?;
    if relations.is_empty() {
        return Ok((0.0, String::new(), 0.0));
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for r in &relations {
        *counts.entry(r.predicate.clone()).or_insert(0) += 1;
    }
    let n = relations.len() as f64;
    let mut entropy = 0.0_f64;
    let mut top_pred = String::new();
    let mut top_count = 0usize;
    for (pred, &count) in &counts {
        if count > top_count {
            top_count = count;
            top_pred = pred.clone();
        }
        let p = count as f64 / n;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    let top_frac = top_count as f64 / n;
    Ok((entropy, top_pred, top_frac))
}

/// Entity type distribution entropy — measures how diverse the entity types are.
/// Returns (entropy_bits, num_types, dominant_type, dominant_fraction).
pub fn type_entropy(brain: &Brain) -> Result<(f64, usize, String, f64), rusqlite::Error> {
    let entities = brain.all_entities()?;
    if entities.is_empty() {
        return Ok((0.0, 0, String::new(), 0.0));
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for e in &entities {
        *counts.entry(e.entity_type.clone()).or_insert(0) += 1;
    }
    let n = entities.len() as f64;
    let mut entropy = 0.0_f64;
    let mut top_type = String::new();
    let mut top_count = 0usize;
    for (etype, &count) in &counts {
        if count > top_count {
            top_count = count;
            top_type = etype.clone();
        }
        let p = count as f64 / n;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    Ok((entropy, counts.len(), top_type, top_count as f64 / n))
}

/// Score how much adding an edge between two entities would reduce fragmentation.
/// Returns the increase in reachable pairs normalized by total possible pairs.
/// Higher score = more valuable connection. Useful for prioritizing hypothesis validation.
pub fn edge_connectivity_value(
    brain: &Brain,
    entity_a: i64,
    entity_b: i64,
) -> Result<f64, rusqlite::Error> {
    let components = connected_components(brain)?;
    let n: usize = components.iter().map(|c| c.len()).sum();
    if n <= 1 {
        return Ok(0.0);
    }

    // Find which components a and b belong to
    let mut comp_a = None;
    let mut comp_b = None;
    for (idx, comp) in components.iter().enumerate() {
        if comp.contains(&entity_a) {
            comp_a = Some(idx);
        }
        if comp.contains(&entity_b) {
            comp_b = Some(idx);
        }
    }

    match (comp_a, comp_b) {
        (Some(a), Some(b)) if a != b => {
            // Connecting two different components: new reachable pairs = size_a * size_b
            let sa = components[a].len() as f64;
            let sb = components[b].len() as f64;
            let max_pairs = n as f64 * (n as f64 - 1.0) / 2.0;
            if max_pairs > 0.0 {
                Ok(sa * sb / max_pairs)
            } else {
                Ok(0.0)
            }
        }
        _ => Ok(0.0), // Same component or not found
    }
}

/// Graph fragmentation score: 0.0 = perfectly connected, 1.0 = completely fragmented.
/// Based on fraction of node pairs that are unreachable from each other.
pub fn fragmentation_score(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let components = connected_components(brain)?;
    let n = components.iter().map(|c| c.len()).sum::<usize>() as f64;
    if n <= 1.0 {
        return Ok(0.0);
    }
    // Sum of s_i * (n - s_i) for each component = total unreachable pairs * 2
    let mut unreachable_pairs = 0.0_f64;
    for c in &components {
        let s = c.len() as f64;
        unreachable_pairs += s * (n - s);
    }
    // Normalize: max unreachable = n*(n-1) when all singletons
    Ok(unreachable_pairs / (n * (n - 1.0)))
}

/// HITS (Hyperlink-Induced Topic Search) algorithm.
/// Computes hub and authority scores for each entity.
/// Authorities = entities pointed to by many good hubs.
/// Hubs = entities pointing to many good authorities.
/// Returns (hub_scores, authority_scores).
#[allow(clippy::type_complexity)]
pub fn hits(
    brain: &Brain,
    iterations: usize,
) -> Result<(HashMap<i64, f64>, HashMap<i64, f64>), rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let n = entities.len();
    if n == 0 {
        return Ok((HashMap::new(), HashMap::new()));
    }
    let ids: Vec<i64> = entities.iter().map(|e| e.id).collect();
    let id_set: HashSet<i64> = ids.iter().copied().collect();

    let mut out_links: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut in_links: HashMap<i64, Vec<i64>> = HashMap::new();
    for r in &relations {
        if id_set.contains(&r.subject_id) && id_set.contains(&r.object_id) {
            out_links.entry(r.subject_id).or_default().push(r.object_id);
            in_links.entry(r.object_id).or_default().push(r.subject_id);
        }
    }

    let mut hub: HashMap<i64, f64> = ids.iter().map(|&id| (id, 1.0)).collect();
    let mut auth: HashMap<i64, f64> = ids.iter().map(|&id| (id, 1.0)).collect();

    for _ in 0..iterations {
        let mut new_auth: HashMap<i64, f64> = ids.iter().map(|&id| (id, 0.0)).collect();
        for &v in &ids {
            if let Some(sources) = in_links.get(&v) {
                for &u in sources {
                    *new_auth.get_mut(&v).unwrap() += hub[&u];
                }
            }
        }
        let mut new_hub: HashMap<i64, f64> = ids.iter().map(|&id| (id, 0.0)).collect();
        for &v in &ids {
            if let Some(targets) = out_links.get(&v) {
                for &u in targets {
                    *new_hub.get_mut(&v).unwrap() += new_auth[&u];
                }
            }
        }
        let auth_norm: f64 = new_auth
            .values()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-10);
        let hub_norm: f64 = new_hub
            .values()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-10);
        for v in new_auth.values_mut() {
            *v /= auth_norm;
        }
        for v in new_hub.values_mut() {
            *v /= hub_norm;
        }
        auth = new_auth;
        hub = new_hub;
    }

    Ok((hub, auth))
}

/// Graph cohesion score: combines clustering coefficient, compaction ratio,
/// and inverse fragmentation into a single 0-1 metric.
/// Higher = more cohesive knowledge graph.
pub fn cohesion_score(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let compaction = compaction_ratio(brain)?;
    let clustering = global_clustering(brain)?;
    let frag = fragmentation_score(brain)?;
    let score = 0.5 * compaction + 0.3 * (1.0 - frag) + 0.2 * clustering;
    Ok(score.clamp(0.0, 1.0))
}

/// Topic-bridging score: identifies entities that connect different entity type domains.
/// An entity linking persons to organizations to places is more "bridging" than one
/// connected only to other persons. Uses Shannon entropy of neighbor types.
/// Returns (entity_id, bridging_score, neighbor_type_count) sorted by score.
pub fn topic_bridging_scores(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, f64, usize)>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let adj = build_adjacency(brain)?;
    let id_to_type: HashMap<i64, String> = entities
        .iter()
        .map(|e| (e.id, e.entity_type.clone()))
        .collect();

    let mut scores: Vec<(i64, f64, usize)> = Vec::new();
    for (&node, neighbors) in &adj {
        if neighbors.len() < 2 {
            continue;
        }
        // Count neighbor types
        let mut type_counts: HashMap<&str, usize> = HashMap::new();
        for &nb in neighbors {
            let t = id_to_type.get(&nb).map(|s| s.as_str()).unwrap_or("unknown");
            *type_counts.entry(t).or_insert(0) += 1;
        }
        let n = neighbors.len() as f64;
        let mut entropy = 0.0_f64;
        for &count in type_counts.values() {
            let p = count as f64 / n;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        let num_types = type_counts.len();
        if num_types >= 2 {
            scores.push((node, entropy, num_types));
        }
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    Ok(scores)
}

/// Find the best entity pairs to bridge disconnected communities.
/// Returns (component_a_rep, component_b_rep, entity_a_name, entity_b_name, combined_size)
/// sorted by combined component size (connecting larger components is more impactful).
pub fn suggest_community_bridges(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(String, String, usize)>, rusqlite::Error> {
    let components = connected_components(brain)?;
    if components.len() < 2 {
        return Ok(vec![]);
    }
    let entities = brain.all_entities()?;
    let id_to_name: HashMap<i64, &str> = entities.iter().map(|e| (e.id, e.name.as_str())).collect();
    let id_to_type: HashMap<i64, &str> = entities
        .iter()
        .map(|e| (e.id, e.entity_type.as_str()))
        .collect();

    // For each component, find its best representative (highest-quality entity name)
    let noise_types: HashSet<&str> = ["phrase", "source", "url", "relative_date", "number_unit"]
        .into_iter()
        .collect();

    let mut comp_reps: Vec<(String, usize)> = Vec::new();
    for comp in &components {
        if comp.len() < 2 {
            continue; // skip singletons
        }
        // Find best representative: prefer multi-word names of high-value types
        let mut best_name = String::new();
        let mut best_score = 0i32;
        for &id in comp {
            let etype = id_to_type.get(&id).copied().unwrap_or("unknown");
            if noise_types.contains(etype) {
                continue;
            }
            let name = id_to_name.get(&id).copied().unwrap_or("");
            let score = match etype {
                "person" | "organization" | "company" => 10,
                "concept" | "technology" | "place" => 8,
                "event" | "product" => 6,
                _ => 3,
            } + name.split_whitespace().count() as i32;
            if score > best_score {
                best_score = score;
                best_name = name.to_string();
            }
        }
        if !best_name.is_empty() {
            comp_reps.push((best_name, comp.len()));
        }
    }

    // Sort by size descending, suggest bridging pairs of large components
    comp_reps.sort_by(|a, b| b.1.cmp(&a.1));
    let mut bridges = Vec::new();
    for i in 0..comp_reps.len().min(limit * 2) {
        for j in (i + 1)..comp_reps.len().min(limit * 2) {
            bridges.push((
                comp_reps[i].0.clone(),
                comp_reps[j].0.clone(),
                comp_reps[i].1 + comp_reps[j].1,
            ));
            if bridges.len() >= limit {
                return Ok(bridges);
            }
        }
    }
    Ok(bridges)
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
    fn test_k_core_decomposition() {
        let brain = Brain::open_in_memory().unwrap();
        // Create a triangle (3-clique) + one pendant node
        let a = brain.upsert_entity("A", "node").unwrap();
        let b = brain.upsert_entity("B", "node").unwrap();
        let c = brain.upsert_entity("C", "node").unwrap();
        let d = brain.upsert_entity("D", "node").unwrap();
        brain.upsert_relation(a, "link", b, "test").unwrap();
        brain.upsert_relation(b, "link", c, "test").unwrap();
        brain.upsert_relation(a, "link", c, "test").unwrap();
        brain.upsert_relation(a, "link", d, "test").unwrap(); // D is pendant (degree 1)
        let cores = k_core_decomposition(&brain).unwrap();
        // A, B, C form a 2-core (triangle), D is in 1-core only
        assert_eq!(*cores.get(&d).unwrap(), 1);
        assert!(*cores.get(&a).unwrap() >= 2);
        assert!(*cores.get(&b).unwrap() >= 2);
        assert!(*cores.get(&c).unwrap() >= 2);
    }

    #[test]
    fn test_densest_core() {
        let brain = Brain::open_in_memory().unwrap();
        let a = brain.upsert_entity("A", "node").unwrap();
        let b = brain.upsert_entity("B", "node").unwrap();
        let c = brain.upsert_entity("C", "node").unwrap();
        brain.upsert_relation(a, "link", b, "test").unwrap();
        brain.upsert_relation(b, "link", c, "test").unwrap();
        brain.upsert_relation(a, "link", c, "test").unwrap();
        let (k, members) = densest_core(&brain, 3).unwrap();
        assert_eq!(k, 2);
        assert_eq!(members.len(), 3);
    }

    #[test]
    fn test_estimated_diameter() {
        let brain = setup();
        let (diam, avg, samples) = estimated_diameter(&brain, 10).unwrap();
        assert!(diam >= 3, "diameter should be at least 3 for A-B-C-D chain");
        assert!(avg > 0.0);
        assert!(samples > 0);
    }

    #[test]
    fn test_small_world() {
        let brain = setup();
        let (sigma, c, l) = small_world_coefficient(&brain).unwrap();
        // Small graph, just check it doesn't panic and returns valid values
        assert!(sigma >= 0.0);
        assert!(c >= 0.0);
        assert!(l >= 0.0);
    }

    #[test]
    fn test_resource_allocation_predict() {
        let brain = Brain::open_in_memory().unwrap();
        let a = brain.upsert_entity("A", "person").unwrap();
        let b = brain.upsert_entity("B", "person").unwrap();
        let c = brain.upsert_entity("C", "person").unwrap();
        let d = brain.upsert_entity("D", "person").unwrap();
        brain.upsert_relation(a, "knows", c, "test").unwrap();
        brain.upsert_relation(b, "knows", c, "test").unwrap();
        brain.upsert_relation(a, "knows", d, "test").unwrap();
        brain.upsert_relation(b, "knows", d, "test").unwrap();
        let preds = resource_allocation_predict(&brain, 10).unwrap();
        // A and B share neighbors C and D but aren't directly connected
        assert!(!preds.is_empty());
        // The unconnected pair (A,B) should appear somewhere in predictions with score > 0
        let ab_pred = preds
            .iter()
            .find(|(x, y, _)| (*x == a && *y == b) || (*x == b && *y == a));
        assert!(ab_pred.is_some(), "A-B pair should be predicted");
        assert!(ab_pred.unwrap().2 > 0.0);
    }

    #[test]
    fn test_type_aware_link_predict() {
        let brain = Brain::open_in_memory().unwrap();
        let a = brain.upsert_entity("Alice", "person").unwrap();
        let b = brain.upsert_entity("Bob", "person").unwrap();
        let c = brain.upsert_entity("Org1", "organization").unwrap();
        let d = brain.upsert_entity("Org2", "organization").unwrap();
        brain.upsert_relation(a, "works_at", c, "test").unwrap();
        brain.upsert_relation(b, "works_at", c, "test").unwrap();
        brain.upsert_relation(a, "member_of", d, "test").unwrap();
        brain.upsert_relation(b, "member_of", d, "test").unwrap();
        let preds = type_aware_link_predict(&brain, 10).unwrap();
        assert!(!preds.is_empty());
        // Should predict a link (score > 0)
        assert!(preds[0].2 > 0.0);
    }

    #[test]
    fn test_bfs_self_path() {
        let brain = setup();
        let path = shortest_path(&brain, "Alice", "Alice").unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 1);
    }

    #[test]
    fn test_neighborhood_overlap() {
        let brain = setup();
        // The test graph may be too sparse for overlaps; just ensure no crash
        let overlaps = neighborhood_overlap(&brain, 0.1, 20).unwrap();
        // Result may be empty for sparse test data — that's OK
        assert!(overlaps.len() <= 20);
    }
}

/// Neighborhood overlap coefficient for link prediction.
/// For each pair of unconnected nodes with shared neighbors, compute:
///   overlap = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
/// This is better than Jaccard for hub-heavy graphs because it normalizes
/// by the smaller neighborhood, making it robust to degree imbalance.
/// Returns (entity_a_id, entity_b_id, overlap_score) sorted by score desc.
/// Degree assortativity coefficient — measures whether high-degree nodes connect
/// to other high-degree nodes (positive) or low-degree nodes (negative).
/// Returns r ∈ [-1, 1]. Positive = assortative, negative = disassortative.
pub fn degree_assortativity(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let relations = brain.all_relations()?;

    if relations.is_empty() {
        return Ok(0.0);
    }

    // Each undirected edge contributes once: (deg(u), deg(v))
    let mut seen: HashSet<(i64, i64)> = HashSet::new();
    let mut sum_xy = 0.0_f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;
    let mut m = 0.0_f64;

    for r in &relations {
        let key = if r.subject_id < r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        if !seen.insert(key) {
            continue;
        }
        let dx = adj.get(&r.subject_id).map(|v| v.len()).unwrap_or(0) as f64;
        let dy = adj.get(&r.object_id).map(|v| v.len()).unwrap_or(0) as f64;
        sum_xy += dx * dy;
        sum_x += dx;
        sum_y += dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
        m += 1.0;
    }

    if m == 0.0 {
        return Ok(0.0);
    }

    let numerator = sum_xy / m - (sum_x / m) * (sum_y / m);
    let denom = ((sum_x2 / m - (sum_x / m).powi(2)) * (sum_y2 / m - (sum_y / m).powi(2))).sqrt();

    if denom < 1e-12 {
        Ok(0.0)
    } else {
        Ok(numerator / denom)
    }
}

/// Type assortativity — fraction of edges connecting same-type entities.
/// Higher values mean entities tend to connect within their type.
pub fn type_assortativity(brain: &Brain) -> Result<(f64, HashMap<String, f64>), rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let id_type: HashMap<i64, String> = entities
        .iter()
        .map(|e| (e.id, e.entity_type.clone()))
        .collect();

    let mut total = 0usize;
    let mut same_type = 0usize;
    let mut type_same: HashMap<String, usize> = HashMap::new();
    let mut type_total: HashMap<String, usize> = HashMap::new();

    for r in &relations {
        let st = id_type
            .get(&r.subject_id)
            .map(|s| s.as_str())
            .unwrap_or("?");
        let ot = id_type.get(&r.object_id).map(|s| s.as_str()).unwrap_or("?");
        total += 1;
        *type_total.entry(st.to_string()).or_insert(0) += 1;
        *type_total.entry(ot.to_string()).or_insert(0) += 1;
        if st == ot {
            same_type += 1;
            *type_same.entry(st.to_string()).or_insert(0) += 1;
        }
    }

    let global = if total > 0 {
        same_type as f64 / total as f64
    } else {
        0.0
    };
    let per_type: HashMap<String, f64> = type_total
        .iter()
        .map(|(t, &tot)| {
            let s = type_same.get(t).copied().unwrap_or(0);
            (
                t.clone(),
                if tot > 0 {
                    s as f64 * 2.0 / tot as f64
                } else {
                    0.0
                },
            )
        })
        .collect();

    Ok((global, per_type))
}

/// Personalized PageRank (PPR): measures relevance of all entities to a seed entity.
/// Unlike global PageRank, PPR teleports back to the seed node, revealing which
/// entities are most contextually relevant to it. Useful for "what else should I
/// know about X?" queries and targeted crawl suggestions.
/// Returns entity scores sorted descending (excludes the seed itself).
pub fn personalized_pagerank(
    brain: &Brain,
    seed_id: i64,
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

    if !id_set.contains(&seed_id) {
        return Ok(HashMap::new());
    }

    let mut out_links: HashMap<i64, Vec<i64>> = HashMap::new();
    for r in &relations {
        if id_set.contains(&r.subject_id) && id_set.contains(&r.object_id) {
            out_links.entry(r.subject_id).or_default().push(r.object_id);
        }
    }

    let mut scores: HashMap<i64, f64> = ids
        .iter()
        .map(|&id| (id, if id == seed_id { 1.0 } else { 0.0 }))
        .collect();

    for _ in 0..iterations {
        let mut dangling_sum = 0.0_f64;
        // Teleport goes to seed only (not uniform)
        let mut new_scores: HashMap<i64, f64> = ids
            .iter()
            .map(|&id| (id, if id == seed_id { 1.0 - damping } else { 0.0 }))
            .collect();
        for &id in &ids {
            if let Some(out) = out_links.get(&id) {
                let share = scores[&id] / out.len() as f64;
                for &target in out {
                    *new_scores.entry(target).or_insert(0.0) += damping * share;
                }
            } else {
                dangling_sum += scores[&id];
            }
        }
        // Dangling mass goes to seed (personalized)
        *new_scores.get_mut(&seed_id).unwrap() += damping * dangling_sum;
        scores = new_scores;
    }
    scores.remove(&seed_id);
    Ok(scores)
}

/// Find the top-N most relevant entities to a given seed entity using PPR.
/// Returns (entity_id, score) pairs sorted by relevance.
pub fn top_relevant(
    brain: &Brain,
    seed_name: &str,
    limit: usize,
) -> Result<Vec<(i64, f64)>, rusqlite::Error> {
    let seed = brain.get_entity_by_name(seed_name)?;
    let seed_id = match seed {
        Some(e) => e.id,
        None => return Ok(vec![]),
    };
    let scores = personalized_pagerank(brain, seed_id, 0.85, 30)?;
    let mut ranked: Vec<(i64, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);
    Ok(ranked)
}

pub fn neighborhood_overlap(
    brain: &Brain,
    min_score: f64,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;

    // Only consider nodes with degree >= 2 (need at least 2 neighbors to share)
    let candidates: Vec<(i64, &Vec<i64>)> = adj
        .iter()
        .filter(|(_, neighbors)| neighbors.len() >= 2)
        .map(|(&id, neighbors)| (id, neighbors))
        .collect();

    // Build neighbor sets
    let neighbor_sets: HashMap<i64, HashSet<i64>> = candidates
        .iter()
        .map(|&(id, neighbors)| (id, neighbors.iter().copied().collect()))
        .collect();

    // Direct edge set for filtering
    let mut direct: HashSet<(i64, i64)> = HashSet::new();
    for (&id, neighbors) in &adj {
        for &n in neighbors {
            let key = if id < n { (id, n) } else { (n, id) };
            direct.insert(key);
        }
    }

    // For efficiency with large graphs, use inverted index: neighbor → set of nodes
    let mut inv_index: HashMap<i64, Vec<i64>> = HashMap::new();
    for &(id, neighbors) in &candidates {
        for &n in neighbors {
            inv_index.entry(n).or_default().push(id);
        }
    }

    let mut results: Vec<(i64, i64, f64)> = Vec::new();
    let mut seen: HashSet<(i64, i64)> = HashSet::new();

    // For each shared neighbor, consider all pairs of its "parents"
    for parents in inv_index.values() {
        if parents.len() < 2 || parents.len() > 200 {
            continue; // Skip super-hubs to avoid O(n²) blowup
        }
        for i in 0..parents.len() {
            for j in (i + 1)..parents.len() {
                let a = parents[i].min(parents[j]);
                let b = parents[i].max(parents[j]);
                let key = (a, b);
                if direct.contains(&key) || !seen.insert(key) {
                    continue;
                }
                if let (Some(sa), Some(sb)) = (neighbor_sets.get(&a), neighbor_sets.get(&b)) {
                    let shared_count = sa.intersection(sb).count();
                    let min_size = sa.len().min(sb.len());
                    if min_size > 0 {
                        let score = shared_count as f64 / min_size as f64;
                        if score >= min_score {
                            results.push((a, b, score));
                        }
                    }
                }
            }
        }
    }

    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Semantic similarity via shared predicate-object fingerprints.
/// Two entities are semantically similar if they share the same (predicate, object) pairs.
/// Returns (entity_a, entity_b, shared_count, jaccard) sorted by jaccard desc.
pub fn semantic_fingerprint_similarity(
    brain: &Brain,
    min_shared: usize,
    limit: usize,
) -> Result<Vec<(i64, i64, usize, f64)>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let mut fingerprints: HashMap<i64, HashSet<(String, i64)>> = HashMap::new();
    for r in &relations {
        fingerprints
            .entry(r.subject_id)
            .or_default()
            .insert((r.predicate.clone(), r.object_id));
    }
    let candidates: Vec<(i64, &HashSet<(String, i64)>)> = fingerprints
        .iter()
        .filter(|(_, fp)| fp.len() >= min_shared)
        .map(|(&id, fp)| (id, fp))
        .collect();
    let mut inv: HashMap<(String, i64), Vec<usize>> = HashMap::new();
    for (idx, &(_, fp)) in candidates.iter().enumerate() {
        for key in fp {
            inv.entry(key.clone()).or_default().push(idx);
        }
    }
    let mut pair_shared: HashMap<(usize, usize), usize> = HashMap::new();
    for posting in inv.values() {
        if posting.len() < 2 || posting.len() > 100 {
            continue;
        }
        for i in 0..posting.len() {
            for j in (i + 1)..posting.len() {
                let key = (posting[i].min(posting[j]), posting[i].max(posting[j]));
                *pair_shared.entry(key).or_insert(0) += 1;
            }
        }
    }
    let mut results = Vec::new();
    for ((i, j), shared) in &pair_shared {
        if *shared >= min_shared {
            let (a, fp_a) = candidates[*i];
            let (b, fp_b) = candidates[*j];
            let union_size = fp_a.union(fp_b).count();
            let jaccard = if union_size > 0 {
                *shared as f64 / union_size as f64
            } else {
                0.0
            };
            results.push((a, b, *shared, jaccard));
        }
    }
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Predicate co-occurrence matrix: which predicates tend to appear together on the same entity.
/// Returns (pred_a, pred_b, co_occurrence_count, pmi) sorted by PMI desc.
pub fn predicate_co_occurrence(
    brain: &Brain,
    min_count: usize,
) -> Result<Vec<(String, String, usize, f64)>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let total_entities = brain.all_entities()?.len() as f64;
    if total_entities < 2.0 {
        return Ok(vec![]);
    }
    let mut entity_preds: HashMap<i64, HashSet<String>> = HashMap::new();
    for r in &relations {
        entity_preds
            .entry(r.subject_id)
            .or_default()
            .insert(r.predicate.clone());
    }
    let mut pred_freq: HashMap<String, usize> = HashMap::new();
    for preds in entity_preds.values() {
        for p in preds {
            *pred_freq.entry(p.clone()).or_insert(0) += 1;
        }
    }
    let mut pair_count: HashMap<(String, String), usize> = HashMap::new();
    for preds in entity_preds.values() {
        let preds_vec: Vec<&String> = preds.iter().collect();
        for i in 0..preds_vec.len() {
            for j in (i + 1)..preds_vec.len() {
                let key = if preds_vec[i] <= preds_vec[j] {
                    (preds_vec[i].clone(), preds_vec[j].clone())
                } else {
                    (preds_vec[j].clone(), preds_vec[i].clone())
                };
                *pair_count.entry(key).or_insert(0) += 1;
            }
        }
    }
    let n = total_entities;
    let mut results = Vec::new();
    for ((p1, p2), count) in &pair_count {
        if *count >= min_count {
            let f1 = pred_freq.get(p1).copied().unwrap_or(1) as f64;
            let f2 = pred_freq.get(p2).copied().unwrap_or(1) as f64;
            let p_ab = *count as f64 / n;
            let p_a = f1 / n;
            let p_b = f2 / n;
            let pmi = if p_a > 0.0 && p_b > 0.0 {
                (p_ab / (p_a * p_b)).log2()
            } else {
                0.0
            };
            results.push((p1.clone(), p2.clone(), *count, pmi));
        }
    }
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

/// Graph structural diversity score: measures how varied the graph structure is
/// by combining component count entropy, degree distribution entropy, and predicate entropy.
/// Higher score = more structurally diverse graph. Range roughly 0-10.
pub fn structural_diversity(brain: &Brain) -> Result<f64, rusqlite::Error> {
    // Degree distribution entropy
    let dist = degree_distribution(brain)?;
    let total: usize = dist.iter().map(|(_, c)| c).sum();
    let deg_entropy: f64 = if total > 0 {
        dist.iter()
            .map(|(_, c)| {
                let p = *c as f64 / total as f64;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

    // Component size entropy
    let components = connected_components(brain)?;
    let total_nodes: usize = components.iter().map(|c| c.len()).sum();
    let comp_entropy: f64 = if total_nodes > 0 {
        components
            .iter()
            .map(|c| {
                let p = c.len() as f64 / total_nodes as f64;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

    Ok(deg_entropy + comp_entropy)
}

/// Analyze discovery velocity trend: detect diminishing returns and suggest course corrections.
/// Returns (trend_direction, avg_recent_patterns, avg_recent_confirmed, recommendation).
/// trend_direction: "accelerating", "steady", "decelerating", "stalled"
pub fn analyze_discovery_trend(
    brain: &Brain,
    window: usize,
) -> Result<(String, f64, f64, String), rusqlite::Error> {
    let snapshots = get_graph_snapshots(brain, window.max(4))?;
    if snapshots.len() < 2 {
        return Ok((
            "insufficient_data".into(),
            0.0,
            0.0,
            "Need at least 2 discovery runs for trend analysis.".into(),
        ));
    }

    // Look at relation growth rate across snapshots (most recent first)
    let recent = &snapshots[..snapshots.len().min(window)];
    let mut deltas: Vec<i64> = Vec::new();
    for w in recent.windows(2) {
        // w[0] is newer, w[1] is older
        deltas.push(w[0].2 - w[1].2); // relation delta
    }

    let avg_delta = if !deltas.is_empty() {
        deltas.iter().sum::<i64>() as f64 / deltas.len() as f64
    } else {
        0.0
    };

    // Check if deltas are decreasing (diminishing returns)
    let (trend, rec): (String, String) = if deltas.len() >= 3 {
        let first_half_avg = deltas[deltas.len() / 2..].iter().sum::<i64>() as f64
            / (deltas.len() - deltas.len() / 2) as f64;
        let second_half_avg =
            deltas[..deltas.len() / 2].iter().sum::<i64>() as f64 / (deltas.len() / 2) as f64;

        if second_half_avg > first_half_avg * 1.2 {
            (
                "accelerating".into(),
                "Discovery is accelerating — current strategies are effective. Keep going.".into(),
            )
        } else if second_half_avg < first_half_avg * 0.3 && second_half_avg < 10.0 {
            ("stalled".into(), "Discovery has stalled — consider crawling new topic domains or adjusting NLP extraction.".into())
        } else if second_half_avg < first_half_avg * 0.7 {
            ("decelerating".into(), "Diminishing returns detected — existing topics are saturating. Explore new domains.".into())
        } else {
            (
                "steady".into(),
                "Discovery rate is steady. Continue current approach.".into(),
            )
        }
    } else if avg_delta < 5.0 {
        (
            "slow".into(),
            "Low discovery rate — need more diverse source material.".into(),
        )
    } else {
        ("steady".into(), "Discovery rate looks healthy.".into())
    };

    // Fragmentation trend
    let frag_current = snapshots[0].8;
    let frag_oldest = snapshots.last().map(|s| s.8).unwrap_or(1.0);
    let frag_delta = frag_current - frag_oldest;
    let frag_note = if frag_delta < -0.01 {
        " Graph cohesion improving."
    } else if frag_delta > 0.01 {
        " Warning: graph becoming more fragmented — focus on connecting existing clusters."
    } else {
        ""
    };

    Ok((
        trend,
        avg_delta,
        0.0, // confirmed rate not tracked in snapshots
        format!("{}{}", rec, frag_note),
    ))
}

/// Topic-aware gap detection: for each community, find entities that have
/// high betweenness centrality within the community but low connections to other communities.
/// These are "local hubs" that could serve as bridges if connected cross-community.
/// Returns (entity_id, community_id, internal_degree, external_degree).
pub fn community_boundary_nodes(
    brain: &Brain,
    min_internal_degree: usize,
) -> Result<Vec<(i64, usize, usize, usize)>, rusqlite::Error> {
    let communities = louvain_communities(brain)?;
    let adj = build_adjacency(brain)?;

    let mut results = Vec::new();
    for (&node, &comm) in &communities {
        if let Some(neighbors) = adj.get(&node) {
            let internal = neighbors
                .iter()
                .filter(|&&nb| communities.get(&nb) == Some(&comm))
                .count();
            let external = neighbors.len() - internal;
            if internal >= min_internal_degree && external == 0 {
                results.push((node, comm, internal, external));
            }
        }
    }
    results.sort_by(|a, b| b.2.cmp(&a.2));
    Ok(results)
}

/// Jaccard-based link prediction: for each pair of entities sharing at least
/// `min_common` neighbors but NOT directly connected, compute Jaccard similarity.
/// Returns top `limit` predictions sorted by score descending.
pub fn jaccard_link_prediction(
    brain: &Brain,
    min_common: usize,
    limit: usize,
) -> Result<Vec<(i64, i64, f64, usize)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let entities = brain.all_entities()?;

    // Only consider entities with at least 2 neighbors (otherwise no meaningful overlap)
    let candidates: Vec<i64> = entities
        .iter()
        .filter(|e| adj.get(&e.id).is_some_and(|n| n.len() >= 2))
        .map(|e| e.id)
        .collect();

    let edge_set: HashSet<(i64, i64)> = {
        let rels = brain.all_relations()?;
        let mut s = HashSet::new();
        for r in &rels {
            s.insert((r.subject_id, r.object_id));
            s.insert((r.object_id, r.subject_id));
        }
        s
    };

    let neighbor_sets: HashMap<i64, HashSet<i64>> = candidates
        .iter()
        .filter_map(|&id| {
            adj.get(&id)
                .map(|ns| (id, ns.iter().copied().collect::<HashSet<i64>>()))
        })
        .collect();

    let mut predictions: Vec<(i64, i64, f64, usize)> = Vec::new();

    // For efficiency, iterate through shared-neighbor inverted index
    let mut neighbor_to_entities: HashMap<i64, Vec<i64>> = HashMap::new();
    for &eid in &candidates {
        if let Some(ns) = neighbor_sets.get(&eid) {
            for &n in ns {
                neighbor_to_entities.entry(n).or_default().push(eid);
            }
        }
    }

    let mut seen: HashSet<(i64, i64)> = HashSet::new();
    for entities_sharing in neighbor_to_entities.values() {
        if entities_sharing.len() > 200 {
            continue; // Skip overly-connected hubs
        }
        for i in 0..entities_sharing.len() {
            for j in (i + 1)..entities_sharing.len() {
                let a = entities_sharing[i];
                let b = entities_sharing[j];
                let key = if a < b { (a, b) } else { (b, a) };
                if edge_set.contains(&(a, b)) || !seen.insert(key) {
                    continue;
                }
                if let (Some(na), Some(nb)) = (neighbor_sets.get(&a), neighbor_sets.get(&b)) {
                    let common = na.intersection(nb).count();
                    if common >= min_common {
                        let union = na.union(nb).count();
                        let jaccard = if union > 0 {
                            common as f64 / union as f64
                        } else {
                            0.0
                        };
                        predictions.push((a, b, jaccard, common));
                    }
                }
            }
        }
    }

    predictions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    predictions.truncate(limit);
    Ok(predictions)
}

/// Compute graph evolution metrics by comparing current state to the most recent snapshot.
/// Returns a map of metric_name → (current_value, delta_from_last_snapshot).
/// Snapshot tuple: (timestamp, entities, relations, components, largest_pct, avg_degree, isolated, density, fragmentation, health_score)
pub fn graph_evolution(brain: &Brain) -> Result<HashMap<String, (f64, f64)>, rusqlite::Error> {
    let current_health = graph_health(brain)?;
    let snapshots = get_graph_snapshots(brain, 2)?;

    let mut result: HashMap<String, (f64, f64)> = HashMap::new();
    if snapshots.len() >= 2 {
        let curr = &snapshots[0];
        let prev = &snapshots[1];
        // Map snapshot fields to named metrics
        result.insert("entities".into(), (curr.1 as f64, (curr.1 - prev.1) as f64));
        result.insert(
            "relations".into(),
            (curr.2 as f64, (curr.2 - prev.2) as f64),
        );
        result.insert(
            "components".into(),
            (curr.3 as f64, (curr.3 - prev.3) as f64),
        );
        result.insert("largest_pct".into(), (curr.4, curr.4 - prev.4));
        result.insert("avg_degree".into(), (curr.5, curr.5 - prev.5));
        result.insert("isolated".into(), (curr.6 as f64, (curr.6 - prev.6) as f64));
        result.insert("density".into(), (curr.7, curr.7 - prev.7));
        result.insert("fragmentation".into(), (curr.8, curr.8 - prev.8));
    }
    for (key, &current) in &current_health {
        result.entry(key.clone()).or_insert((current, 0.0));
    }
    Ok(result)
}

/// Identify "knowledge deserts" — entity types with low average degree compared to the graph mean.
/// Returns (type, count, avg_degree, graph_avg_degree) sorted by avg_degree ascending.
pub fn knowledge_deserts(brain: &Brain) -> Result<Vec<(String, usize, f64, f64)>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let adj = build_adjacency(brain)?;

    let mut type_degrees: HashMap<String, Vec<f64>> = HashMap::new();
    let mut total_degree = 0.0_f64;
    let mut total_count = 0usize;

    for e in &entities {
        let deg = adj.get(&e.id).map_or(0, |n| n.len()) as f64;
        type_degrees
            .entry(e.entity_type.clone())
            .or_default()
            .push(deg);
        total_degree += deg;
        total_count += 1;
    }

    let graph_avg = if total_count > 0 {
        total_degree / total_count as f64
    } else {
        0.0
    };

    let mut deserts: Vec<(String, usize, f64, f64)> = type_degrees
        .iter()
        .map(|(t, degs)| {
            let avg = degs.iter().sum::<f64>() / degs.len() as f64;
            (t.clone(), degs.len(), avg, graph_avg)
        })
        .filter(|(_, count, avg, _)| *count >= 5 && *avg < graph_avg * 0.5)
        .collect();

    deserts.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok(deserts)
}

/// Neighborhood cosine similarity for link prediction.
/// cos(u,v) = |Γ(u) ∩ Γ(v)| / sqrt(|Γ(u)| * |Γ(v)|)
/// More robust than Jaccard for heterogeneous degree distributions — doesn't penalize
/// high-degree nodes as heavily, which matters in scale-free knowledge graphs.
/// Returns top `limit` pairs above `min_sim` threshold.
pub fn cosine_similarity_predict(
    brain: &Brain,
    min_sim: f64,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let relations = brain.all_relations()?;
    let mut connected: HashSet<(i64, i64)> = HashSet::new();
    for r in &relations {
        let key = if r.subject_id < r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        connected.insert(key);
    }

    // Only consider nodes with degree >= 3 (need overlap to be meaningful)
    let mut candidates: Vec<(i64, usize)> = adj
        .iter()
        .filter(|(_, nb)| nb.len() >= 3)
        .map(|(&id, nb)| (id, nb.len()))
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.truncate(800);

    let nb_sets: HashMap<i64, HashSet<i64>> = candidates
        .iter()
        .map(|&(id, _)| {
            let set: HashSet<i64> = adj[&id].iter().copied().collect();
            (id, set)
        })
        .collect();

    let mut results: Vec<(i64, i64, f64)> = Vec::new();
    for (i, &(a, da)) in candidates.iter().enumerate() {
        let na = &nb_sets[&a];
        for &(b, db) in &candidates[(i + 1)..] {
            let key = if a < b { (a, b) } else { (b, a) };
            if connected.contains(&key) {
                continue;
            }
            let nb = &nb_sets[&b];
            let intersection = na.intersection(nb).count() as f64;
            if intersection == 0.0 {
                continue;
            }
            let denom = (da as f64 * db as f64).sqrt();
            let sim = intersection / denom;
            if sim >= min_sim {
                results.push((a, b, sim));
            }
        }
    }
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Hub authority analysis: identify entities that are critical connectors between
/// different entity types. Returns (entity_id, type_diversity, degree, authority_score).
/// Entities connecting many different types are more valuable bridge nodes.
pub fn type_bridge_authority(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, usize, usize, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let entities = brain.all_entities()?;
    let id_to_type: HashMap<i64, String> = entities
        .iter()
        .map(|e| (e.id, e.entity_type.clone()))
        .collect();

    let mut results: Vec<(i64, usize, usize, f64)> = Vec::new();
    for (&node, neighbors) in &adj {
        if neighbors.len() < 3 {
            continue;
        }
        let neighbor_types: HashSet<&String> =
            neighbors.iter().filter_map(|n| id_to_type.get(n)).collect();
        let type_div = neighbor_types.len();
        if type_div < 2 {
            continue;
        }
        // Authority = degree * type_diversity_bonus
        let authority = neighbors.len() as f64 * (type_div as f64).ln();
        results.push((node, type_div, neighbors.len(), authority));
    }
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Structural similarity between two entities based on Adamic-Adar index,
/// normalized by geometric mean of degrees. Returns [0, 1].
pub fn structural_similarity(
    brain: &Brain,
    entity_a: i64,
    entity_b: i64,
) -> Result<f64, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let na = adj.get(&entity_a).cloned().unwrap_or_default();
    let nb: HashSet<i64> = adj
        .get(&entity_b)
        .map(|v| v.iter().copied().collect())
        .unwrap_or_default();

    if na.is_empty() || nb.is_empty() {
        return Ok(0.0);
    }

    let mut aa_sum = 0.0_f64;
    for &n in &na {
        if nb.contains(&n) {
            let deg = adj.get(&n).map(|v| v.len()).unwrap_or(1);
            aa_sum += 1.0 / (deg as f64).ln().max(1.0);
        }
    }

    let norm = (na.len() as f64 * nb.len() as f64).sqrt();
    Ok(if norm > 0.0 { aa_sum / norm } else { 0.0 })
}

/// Graph fragmentation index: 1 - (largest_component / total_entities).
/// 0 = fully connected, close to 1 = highly fragmented.
pub fn fragmentation_index(brain: &Brain) -> Result<f64, rusqlite::Error> {
    let n = brain.all_entities()?.len();
    if n == 0 {
        return Ok(0.0);
    }
    let components = connected_components(brain)?;
    let largest = components.first().map(|c| c.len()).unwrap_or(0);
    Ok(1.0 - (largest as f64 / n as f64))
}

/// Identify the most central entity per community (Louvain + PageRank).
/// Returns (community_id, leader_entity_id, leader_name, community_size, leader_pagerank).
#[allow(clippy::type_complexity)]
pub fn community_leaders(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(usize, i64, String, usize, f64)>, rusqlite::Error> {
    let communities = louvain_communities(brain)?;
    let pr = pagerank(brain, 0.85, 30)?;
    let entities = brain.all_entities()?;
    let id_to_name: HashMap<i64, &str> = entities.iter().map(|e| (e.id, e.name.as_str())).collect();

    let mut comm_members: HashMap<usize, Vec<(i64, f64)>> = HashMap::new();
    for (&eid, &cid) in &communities {
        let score = pr.get(&eid).copied().unwrap_or(0.0);
        comm_members.entry(cid).or_default().push((eid, score));
    }

    let mut results: Vec<(usize, i64, String, usize, f64)> = Vec::new();
    for (cid, members) in &comm_members {
        let size = members.len();
        if size < 3 {
            continue;
        }
        if let Some(&(leader_id, leader_pr)) = members
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let name = id_to_name
                .get(&leader_id)
                .map(|s| s.to_string())
                .unwrap_or_default();
            results.push((*cid, leader_id, name, size, leader_pr));
        }
    }
    results.sort_by(|a, b| b.3.cmp(&a.3));
    results.truncate(limit);
    Ok(results)
}

/// Jaccard coefficient link prediction: J(x,y) = |Γ(x) ∩ Γ(y)| / |Γ(x) ∪ Γ(y)|.
/// Normalizes for degree, so high-degree nodes don't dominate.
/// Returns top `limit` predicted links as (entity_a, entity_b, score).
pub fn jaccard_predict(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let degree: HashMap<i64, usize> = adj
        .iter()
        .map(|(&id, neighbors)| (id, neighbors.len()))
        .collect();

    let relations = brain.all_relations()?;
    let mut connected: HashSet<(i64, i64)> = HashSet::new();
    for r in &relations {
        let key = if r.subject_id < r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        connected.insert(key);
    }

    let candidates: Vec<i64> = degree
        .iter()
        .filter(|(_, &d)| d >= 2)
        .map(|(&id, _)| id)
        .collect();

    let nb_sets: HashMap<i64, HashSet<i64>> = candidates
        .iter()
        .map(|&id| {
            let set: HashSet<i64> = adj
                .get(&id)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();
            (id, set)
        })
        .collect();

    let mut sorted_candidates = candidates.clone();
    sorted_candidates.sort_by(|a, b| degree.get(b).unwrap_or(&0).cmp(degree.get(a).unwrap_or(&0)));

    let mut scores: Vec<(i64, i64, f64)> = Vec::new();
    let top: Vec<i64> = sorted_candidates.into_iter().take(500).collect();
    for (i, &a) in top.iter().enumerate() {
        let na = &nb_sets[&a];
        for &b in &top[i + 1..] {
            let key = if a < b { (a, b) } else { (b, a) };
            if connected.contains(&key) {
                continue;
            }
            let nb = &nb_sets[&b];
            let intersection = na.intersection(nb).count();
            if intersection == 0 {
                continue;
            }
            let union = na.union(nb).count();
            let score = intersection as f64 / union as f64;
            scores.push((a, b, score));
        }
    }
    scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    Ok(scores)
}

/// Preferential attachment link prediction: PA(x,y) = |Γ(x)| * |Γ(y)|.
/// Models "rich get richer" — high-degree nodes are more likely to form links.
/// Returns top `limit` predicted links among unconnected pairs.
pub fn preferential_attachment_predict(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let degree: HashMap<i64, usize> = adj
        .iter()
        .map(|(&id, neighbors)| (id, neighbors.len()))
        .collect();

    let relations = brain.all_relations()?;
    let mut connected: HashSet<(i64, i64)> = HashSet::new();
    for r in &relations {
        let key = if r.subject_id < r.object_id {
            (r.subject_id, r.object_id)
        } else {
            (r.object_id, r.subject_id)
        };
        connected.insert(key);
    }

    // Only consider top-degree nodes (high-degree * high-degree = strongest signal)
    let mut sorted: Vec<(i64, usize)> = degree.iter().map(|(&id, &d)| (id, d)).collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let top: Vec<i64> = sorted.into_iter().take(300).map(|(id, _)| id).collect();

    let mut scores: Vec<(i64, i64, f64)> = Vec::new();
    for (i, &a) in top.iter().enumerate() {
        let da = degree.get(&a).copied().unwrap_or(0);
        for &b in &top[i + 1..] {
            let key = if a < b { (a, b) } else { (b, a) };
            if connected.contains(&key) {
                continue;
            }
            let db = degree.get(&b).copied().unwrap_or(0);
            scores.push((a, b, (da * db) as f64));
        }
    }
    scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    Ok(scores)
}

/// Katz centrality: sums paths of all lengths with exponential decay β^k.
/// More robust than PageRank for small/sparse graphs.
/// Returns (entity_id, katz_score) for top `limit` entities.
pub fn katz_centrality(
    brain: &Brain,
    beta: f64,
    max_depth: usize,
    limit: usize,
) -> Result<Vec<(i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let all_nodes: Vec<i64> = adj.keys().copied().collect();
    let mut scores: HashMap<i64, f64> = HashMap::new();

    // For each source, BFS up to max_depth, accumulating β^k for each reachable node
    for &source in all_nodes.iter().take(500) {
        let mut frontier: Vec<i64> = vec![source];
        let mut visited: HashSet<i64> = HashSet::new();
        visited.insert(source);
        let mut depth = 0;
        while depth < max_depth && !frontier.is_empty() {
            depth += 1;
            let weight = beta.powi(depth as i32);
            let mut next_frontier = Vec::new();
            for &node in &frontier {
                if let Some(neighbors) = adj.get(&node) {
                    for &nb in neighbors {
                        *scores.entry(nb).or_insert(0.0) += weight;
                        if visited.insert(nb) {
                            next_frontier.push(nb);
                        }
                    }
                }
            }
            frontier = next_frontier;
        }
    }

    let mut result: Vec<(i64, f64)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result.truncate(limit);
    Ok(result)
}

/// Random Walk with Restart (RWR) similarity between two entities.
/// Simulates a random walker starting from `source` that at each step either
/// follows a random edge (prob `1-alpha`) or teleports back to `source` (prob `alpha`).
/// The stationary probability at `target` measures structural relatedness.
/// More robust than shortest-path for dense subgraphs because it considers ALL paths.
/// Returns a score in [0, 1]; higher = more structurally related.
pub fn rwr_similarity(
    brain: &Brain,
    source_id: i64,
    target_id: i64,
    alpha: f64,
    iterations: usize,
) -> Result<f64, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    if !adj.contains_key(&source_id) {
        return Ok(0.0);
    }

    // Power-iteration RWR
    let all_nodes: Vec<i64> = adj.keys().copied().collect();
    let mut prob: HashMap<i64, f64> = HashMap::new();
    prob.insert(source_id, 1.0);

    for _ in 0..iterations {
        let mut new_prob: HashMap<i64, f64> = HashMap::new();
        // Teleport component
        *new_prob.entry(source_id).or_insert(0.0) += alpha;

        // Walk component
        for &node in &all_nodes {
            let p = prob.get(&node).copied().unwrap_or(0.0);
            if p < 1e-12 {
                continue;
            }
            if let Some(neighbors) = adj.get(&node) {
                if neighbors.is_empty() {
                    // Dangling node: teleport to source
                    *new_prob.entry(source_id).or_insert(0.0) += (1.0 - alpha) * p;
                } else {
                    let share = (1.0 - alpha) * p / neighbors.len() as f64;
                    for &nb in neighbors {
                        *new_prob.entry(nb).or_insert(0.0) += share;
                    }
                }
            } else {
                // Isolated in adjacency: teleport
                *new_prob.entry(source_id).or_insert(0.0) += (1.0 - alpha) * p;
            }
        }
        prob = new_prob;
    }

    Ok(prob.get(&target_id).copied().unwrap_or(0.0))
}

/// Batch RWR: find top-N entities most similar to a seed via Random Walk with Restart.
/// More informative than PPR for discovery because alpha controls locality vs globality.
/// Lower alpha = more global exploration, higher alpha = more local.
/// Returns (entity_id, rwr_score) sorted descending, excluding the seed.
pub fn rwr_top_similar(
    brain: &Brain,
    seed_id: i64,
    alpha: f64,
    iterations: usize,
    limit: usize,
) -> Result<Vec<(i64, f64)>, rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    if !adj.contains_key(&seed_id) {
        return Ok(vec![]);
    }

    let all_nodes: Vec<i64> = adj.keys().copied().collect();
    let mut prob: HashMap<i64, f64> = HashMap::new();
    prob.insert(seed_id, 1.0);

    for _ in 0..iterations {
        let mut new_prob: HashMap<i64, f64> = HashMap::new();
        *new_prob.entry(seed_id).or_insert(0.0) += alpha;

        for &node in &all_nodes {
            let p = prob.get(&node).copied().unwrap_or(0.0);
            if p < 1e-12 {
                continue;
            }
            if let Some(neighbors) = adj.get(&node) {
                if neighbors.is_empty() {
                    *new_prob.entry(seed_id).or_insert(0.0) += (1.0 - alpha) * p;
                } else {
                    let share = (1.0 - alpha) * p / neighbors.len() as f64;
                    for &nb in neighbors {
                        *new_prob.entry(nb).or_insert(0.0) += share;
                    }
                }
            } else {
                *new_prob.entry(seed_id).or_insert(0.0) += (1.0 - alpha) * p;
            }
        }
        prob = new_prob;
    }

    prob.remove(&seed_id);
    let mut result: Vec<(i64, f64)> = prob.into_iter().filter(|(_, s)| *s > 1e-10).collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result.truncate(limit);
    Ok(result)
}

/// Algebraic connectivity estimate (Fiedler value / spectral gap).
/// Approximates the second-smallest eigenvalue of the graph Laplacian using
/// power iteration on the inverse-shifted Laplacian. This value measures how
/// well-connected the graph is: 0 = disconnected, higher = more robust connectivity.
/// For knowledge graphs, this indicates whether removing a few bridge entities
/// would fragment the graph. Returns (algebraic_connectivity, interpretation).
pub fn algebraic_connectivity_estimate(
    brain: &Brain,
    iterations: usize,
) -> Result<(f64, String), rusqlite::Error> {
    let adj = build_adjacency(brain)?;
    let nodes: Vec<i64> = adj.keys().copied().collect();
    let n = nodes.len();
    if n < 3 {
        return Ok((0.0, "Graph too small for spectral analysis".into()));
    }
    let node_idx: HashMap<i64, usize> = nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Power iteration to find largest eigenvalue of adjacency matrix,
    // then λ₂(L) ≈ max_degree - λ₁(A) for connected graphs.
    // More practical approach: inverse iteration on L to find smallest non-trivial eigenvalue.
    // We use Rayleigh quotient iteration with deflation of the constant eigenvector.

    let degrees: Vec<f64> = nodes
        .iter()
        .map(|id| adj.get(id).map_or(0, |n| n.len()) as f64)
        .collect();

    // Random initial vector, orthogonalized against the constant vector (1/√n)
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7 + 0.3).sin()).collect();
    // Remove projection onto constant vector
    let proj: f64 = v.iter().sum::<f64>() * inv_sqrt_n;
    for vi in v.iter_mut() {
        *vi -= proj * inv_sqrt_n;
    }
    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return Ok((0.0, "Degenerate vector".into()));
    }
    for vi in v.iter_mut() {
        *vi /= norm;
    }

    let mut lambda = 0.0_f64;
    for _ in 0..iterations {
        // Compute L*v where L = D - A (Laplacian)
        let mut lv = vec![0.0_f64; n];
        for (i, &node) in nodes.iter().enumerate() {
            lv[i] = degrees[i] * v[i]; // D*v component
            if let Some(neighbors) = adj.get(&node) {
                for &nb in neighbors {
                    if let Some(&j) = node_idx.get(&nb) {
                        lv[i] -= v[j]; // -A*v component
                    }
                }
            }
        }

        // Deflate: remove projection onto constant eigenvector
        let proj: f64 = lv.iter().sum::<f64>() * inv_sqrt_n;
        for lvi in lv.iter_mut() {
            *lvi -= proj * inv_sqrt_n;
        }

        // Rayleigh quotient: λ = v^T L v / v^T v
        lambda = v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum::<f64>();
        let vtv: f64 = v.iter().map(|x| x * x).sum();
        if vtv > 0.0 {
            lambda /= vtv;
        }

        // Update v = Lv (power iteration finds LARGEST eigenvalue of L, but
        // with deflation of constant vector, this gives us λ₂)
        v = lv;
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for vi in v.iter_mut() {
            *vi /= norm;
        }
    }

    // Normalize by max possible (max_degree for d-regular graph)
    let max_deg = degrees.iter().cloned().fold(0.0_f64, f64::max);
    let normalized = if max_deg > 0.0 { lambda / max_deg } else { 0.0 };

    let interpretation = if lambda < 0.01 {
        "Near-disconnected: graph has weak bridges that could easily fragment".into()
    } else if lambda < 0.1 {
        "Low connectivity: a few key entities hold the graph together".into()
    } else if lambda < 0.5 {
        "Moderate connectivity: reasonably robust structure".into()
    } else {
        "High connectivity: well-interconnected graph".into()
    };

    Ok((normalized, interpretation))
}

/// Temporal momentum scoring: entities gaining connections in recent facts score higher.
/// Uses fact timestamps to weight edges by recency (exponential decay).
/// Returns entity_id → momentum score, sorted descending.
pub fn temporal_momentum(
    brain: &Brain,
    half_life_days: f64,
    limit: usize,
) -> Result<Vec<(i64, f64)>, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let id_set: HashSet<i64> = entities.iter().map(|e| e.id).collect();

    // Parse the most recent timestamp as "now" reference
    let now = chrono::Utc::now().timestamp() as f64;
    let decay_rate = (2.0_f64).ln() / (half_life_days * 86400.0);

    // Accumulate recency-weighted degree for each entity
    let mut momentum: HashMap<i64, f64> = HashMap::new();
    for r in &relations {
        if !id_set.contains(&r.subject_id) || !id_set.contains(&r.object_id) {
            continue;
        }
        // Use relation's created_at as timestamp proxy
        let age_secs = (now - r.learned_at.and_utc().timestamp() as f64).max(0.0);
        let weight = (-decay_rate * age_secs).exp() * r.confidence.max(0.01);
        *momentum.entry(r.subject_id).or_insert(0.0) += weight;
        *momentum.entry(r.object_id).or_insert(0.0) += weight;
    }

    let mut scores: Vec<(i64, f64)> = momentum.into_iter().collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    Ok(scores)
}

/// Network motif census: count occurrences of 3-node motifs (triangles, feed-forward loops,
/// chains, mutual dyads). Returns a map from motif name to count.
/// Useful for understanding the structural character of the knowledge graph.
pub fn motif_census(brain: &Brain) -> Result<HashMap<String, usize>, rusqlite::Error> {
    let relations = brain.all_relations()?;
    let entities = brain.all_entities()?;
    let id_set: HashSet<i64> = entities.iter().map(|e| e.id).collect();

    // Build directed adjacency
    let mut out: HashMap<i64, HashSet<i64>> = HashMap::new();
    for r in &relations {
        if id_set.contains(&r.subject_id) && id_set.contains(&r.object_id) {
            out.entry(r.subject_id).or_default().insert(r.object_id);
        }
    }

    // Also build undirected neighbors for sampling
    let mut neighbors: HashMap<i64, HashSet<i64>> = HashMap::new();
    for r in &relations {
        if id_set.contains(&r.subject_id) && id_set.contains(&r.object_id) {
            neighbors
                .entry(r.subject_id)
                .or_default()
                .insert(r.object_id);
            neighbors
                .entry(r.object_id)
                .or_default()
                .insert(r.subject_id);
        }
    }

    let mut triangles = 0usize;
    let mut feed_forward = 0usize; // A→B, A→C, B→C
    let mut chains = 0usize; // A→B→C (no A→C)
    let mut mutual_dyads = 0usize; // A↔B

    // Count mutual dyads
    for (&a, targets) in &out {
        for &b in targets {
            if let Some(b_targets) = out.get(&b) {
                if b_targets.contains(&a) && a < b {
                    mutual_dyads += 1;
                }
            }
        }
    }

    // Sample 3-node motifs: for each edge A→B, check common neighbors
    // Limit sampling to prevent O(n³) blowup
    let mut sampled = 0usize;
    let max_samples = 100_000usize;
    'outer: for (&a, a_targets) in &out {
        for &b in a_targets {
            if sampled >= max_samples {
                break 'outer;
            }
            // Check all neighbors of B
            let b_targets = out.get(&b).cloned().unwrap_or_default();
            for &c in b_targets.iter() {
                if c == a {
                    continue;
                }
                sampled += 1;
                if sampled >= max_samples {
                    break 'outer;
                }
                let a_to_c = out.get(&a).is_some_and(|t| t.contains(&c));
                let c_to_a = out.get(&c).is_some_and(|t| t.contains(&a));
                if a_to_c && c_to_a {
                    triangles += 1;
                } else if a_to_c {
                    feed_forward += 1;
                } else {
                    chains += 1;
                }
            }
        }
    }

    let mut census = HashMap::new();
    census.insert("triangles".into(), triangles);
    census.insert("feed_forward_loops".into(), feed_forward);
    census.insert("chains".into(), chains);
    census.insert("mutual_dyads".into(), mutual_dyads);
    census.insert("samples".into(), sampled);
    Ok(census)
}

/// Identify "rising star" entities: high temporal momentum but low current degree.
/// These are entities rapidly gaining connections — potential discovery hotspots.
pub fn rising_stars(
    brain: &Brain,
    limit: usize,
) -> Result<Vec<(i64, String, f64, usize)>, rusqlite::Error> {
    let momentum = temporal_momentum(brain, 30.0, 500)?;
    let entities = brain.all_entities()?;
    let id_to_name: HashMap<i64, &str> = entities.iter().map(|e| (e.id, e.name.as_str())).collect();

    // Count degree
    let relations = brain.all_relations()?;
    let mut degree: HashMap<i64, usize> = HashMap::new();
    for r in &relations {
        *degree.entry(r.subject_id).or_insert(0) += 1;
        *degree.entry(r.object_id).or_insert(0) += 1;
    }

    // Rising star score = momentum / (degree + 1) — high momentum relative to current size
    let mut stars: Vec<(i64, String, f64, usize)> = momentum
        .iter()
        .filter_map(|&(id, m)| {
            let d = degree.get(&id).copied().unwrap_or(0);
            let name = id_to_name.get(&id).copied().unwrap_or("?");
            let score = m / (d as f64 + 1.0);
            if score > 0.01 {
                Some((id, name.to_string(), score, d))
            } else {
                None
            }
        })
        .collect();

    stars.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    stars.truncate(limit);
    Ok(stars)
}
