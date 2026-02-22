use crate::db::Brain;
use crate::nlp;

/// Answer a question by searching the knowledge graph.
pub fn ask(brain: &Brain, question: &str) -> Result<Vec<String>, rusqlite::Error> {
    let tokens = nlp::tokenize(question);
    let mut results = Vec::new();

    // Search entities matching question terms
    for token in &tokens {
        if token.len() > 2 {
            let entities = brain.search_entities(token)?;
            for entity in entities {
                if entity.entity_type != "source" {
                    results.push(format!(
                        "[{:.0}%] {} ({})",
                        entity.confidence * 100.0,
                        entity.name,
                        entity.entity_type
                    ));

                    // Get related facts
                    let facts = brain.get_facts_for(entity.id)?;
                    for fact in facts.iter().take(3) {
                        results.push(format!(
                            "       {} = {} [{:.0}%]",
                            fact.key,
                            fact.value,
                            fact.confidence * 100.0
                        ));
                    }

                    // Get relations
                    let relations = brain.get_relations_for(entity.id)?;
                    for (subj, pred, obj, conf) in relations.iter().take(3) {
                        results.push(format!(
                            "       {subj} → {pred} → {obj} [{:.0}%]",
                            conf * 100.0
                        ));
                    }
                }
            }
        }
    }

    // Also search facts
    let fact_results = brain.search_facts(question)?;
    for (entity, key, value, conf) in fact_results.iter().take(5) {
        results.push(format!("[{:.0}%] {entity}: {key} = {value}", conf * 100.0));
    }

    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    results.retain(|r| seen.insert(r.clone()));

    Ok(results)
}

/// Show everything known about an entity.
pub fn about(brain: &Brain, name: &str) -> Result<Vec<String>, rusqlite::Error> {
    let mut info = Vec::new();

    if let Some(entity) = brain.get_entity_by_name(name)? {
        info.push(format!(
            "Type: {} | Confidence: {:.0}% | Seen {} times",
            entity.entity_type,
            entity.confidence * 100.0,
            entity.access_count
        ));
        info.push(format!(
            "First seen: {} | Last seen: {}",
            entity.first_seen, entity.last_seen
        ));

        let facts = brain.get_facts_for(entity.id)?;
        if !facts.is_empty() {
            info.push(String::new());
            info.push("Facts:".to_string());
            for fact in &facts {
                info.push(format!(
                    "  {} = {} [{:.0}%]",
                    fact.key,
                    fact.value,
                    fact.confidence * 100.0
                ));
            }
        }

        let relations = brain.get_relations_for(entity.id)?;
        if !relations.is_empty() {
            info.push(String::new());
            info.push("Relations:".to_string());
            for (subj, pred, obj, conf) in &relations {
                info.push(format!("  {subj} → {pred} → {obj} [{:.0}%]", conf * 100.0));
            }
        }
    } else {
        // Try fuzzy search
        let entities = brain.search_entities(name)?;
        if !entities.is_empty() {
            info.push("No exact match. Did you mean:".to_string());
            for e in entities.iter().take(5) {
                info.push(format!("  - {} ({})", e.name, e.entity_type));
            }
        }
    }

    Ok(info)
}

/// Show related entities.
pub fn related(brain: &Brain, name: &str) -> Result<Vec<String>, rusqlite::Error> {
    let mut results = Vec::new();

    if let Some(entity) = brain.get_entity_by_name(name)? {
        let related = brain.get_related_entities(entity.id)?;
        for (rel_name, predicate, confidence) in &related {
            results.push(format!(
                "{rel_name} (via {predicate}) [{:.0}%]",
                confidence * 100.0
            ));
        }
    }

    Ok(results)
}

/// Show recently learned facts.
pub fn recent(brain: &Brain, limit: usize) -> Result<Vec<String>, rusqlite::Error> {
    let facts = brain.recent_facts(limit)?;
    Ok(facts
        .into_iter()
        .map(|(entity, key, value, conf)| {
            format!("[{:.0}%] {entity}: {key} = {value}", conf * 100.0)
        })
        .collect())
}

/// Show top knowledge areas.
pub fn topics(brain: &Brain, limit: usize) -> Result<Vec<String>, rusqlite::Error> {
    let top = brain.top_entities(limit)?;
    Ok(top
        .into_iter()
        .map(|(name, etype, conf, count)| {
            format!(
                "{name} ({etype}) — confidence: {:.0}%, seen {count}×",
                conf * 100.0
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Brain;
    use crate::nlp::Extracted;

    fn setup_brain() -> Brain {
        let brain = Brain::open_in_memory().unwrap();
        let extracted = Extracted {
            entities: vec![
                ("Rust".to_string(), "language".to_string()),
                ("Mozilla".to_string(), "organization".to_string()),
            ],
            relations: vec![(
                "Mozilla".to_string(),
                "created".to_string(),
                "Rust".to_string(),
            )],
            keywords: vec!["programming".to_string(), "systems".to_string()],
            source_url: "https://rust-lang.org".to_string(),
        };
        brain.learn(&extracted).unwrap();
        brain
    }

    #[test]
    fn test_ask() {
        let brain = setup_brain();
        let results = ask(&brain, "Rust").unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_about() {
        let brain = setup_brain();
        let info = about(&brain, "Rust").unwrap();
        assert!(!info.is_empty());
    }

    #[test]
    fn test_related() {
        let brain = setup_brain();
        let related = related(&brain, "Rust").unwrap();
        assert!(!related.is_empty());
    }

    #[test]
    fn test_recent() {
        let brain = setup_brain();
        let recent = recent(&brain, 10).unwrap();
        assert!(!recent.is_empty());
    }

    #[test]
    fn test_topics() {
        let brain = setup_brain();
        let topics = topics(&brain, 10).unwrap();
        assert!(!topics.is_empty());
    }

    #[test]
    fn test_ask_no_results() {
        let brain = Brain::open_in_memory().unwrap();
        let results = ask(&brain, "nonexistent").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_about_no_match() {
        let brain = Brain::open_in_memory().unwrap();
        let info = about(&brain, "nonexistent").unwrap();
        assert!(info.is_empty());
    }
}
