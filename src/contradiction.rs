/// Contradiction detection — find conflicting facts in the knowledge graph.
use crate::db::Brain;

/// A detected contradiction between two facts.
#[derive(Debug, Clone)]
pub struct Contradiction {
    pub entity_name: String,
    pub key: String,
    pub value_a: String,
    pub value_b: String,
    pub confidence_a: f64,
    pub confidence_b: f64,
    pub source_a: String,
    pub source_b: String,
    pub severity: ContradictionSeverity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContradictionSeverity {
    /// Both sources are equally confident — real conflict
    Hard,
    /// One source is notably more confident — likely outdated info
    Soft,
    /// Values are slightly different but may be compatible (e.g. rounding)
    Minor,
}

impl ContradictionSeverity {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Hard => "HARD",
            Self::Soft => "SOFT",
            Self::Minor => "MINOR",
        }
    }
}

/// Keys known to have unique values (one correct answer per entity).
const UNIQUE_KEYS: &[&str] = &[
    "born",
    "died",
    "birth_date",
    "death_date",
    "founded",
    "capital",
    "population",
    "area",
    "currency",
    "president",
    "ceo",
    "headquarters",
    "founded_year",
    "language",
    "nationality",
    "country",
    "continent",
];

/// Detect contradictions in the knowledge graph.
/// Looks for entities with multiple conflicting values for the same key.
pub fn detect_contradictions(brain: &Brain) -> Result<Vec<Contradiction>, rusqlite::Error> {
    let mut contradictions = Vec::new();

    // Find all entities that have multiple values for the same key
    let entities = brain.all_entities()?;

    for entity in &entities {
        let facts = brain.get_facts_for(entity.id)?;

        // Group facts by key
        let mut by_key: std::collections::HashMap<String, Vec<_>> =
            std::collections::HashMap::new();
        for fact in &facts {
            by_key
                .entry(fact.key.clone())
                .or_default()
                .push(fact.clone());
        }

        // Check keys with multiple values
        for (key, values) in &by_key {
            if values.len() < 2 {
                continue;
            }

            // For "keyword" facts, multiple values are normal
            if key == "keyword" || key == "alias" || key == "also_known_as" {
                continue;
            }

            // For unique keys, any difference is a contradiction
            let is_unique = UNIQUE_KEYS.contains(&key.as_str());

            // Compare all pairs
            for i in 0..values.len() {
                for j in (i + 1)..values.len() {
                    let a = &values[i];
                    let b = &values[j];

                    // Skip if values are identical
                    if a.value.to_lowercase() == b.value.to_lowercase() {
                        continue;
                    }

                    let severity = if is_unique {
                        if are_numerically_close(&a.value, &b.value) {
                            ContradictionSeverity::Minor
                        } else {
                            classify_severity(a.confidence, b.confidence)
                        }
                    } else if are_numerically_close(&a.value, &b.value) {
                        ContradictionSeverity::Minor
                    } else {
                        // Non-unique keys with different values might be fine
                        // Only flag if they look contradictory
                        if looks_contradictory(&a.value, &b.value) {
                            classify_severity(a.confidence, b.confidence)
                        } else {
                            continue;
                        }
                    };

                    contradictions.push(Contradiction {
                        entity_name: entity.name.clone(),
                        key: key.clone(),
                        value_a: a.value.clone(),
                        value_b: b.value.clone(),
                        confidence_a: a.confidence,
                        confidence_b: b.confidence,
                        source_a: a.source_url.clone(),
                        source_b: b.source_url.clone(),
                        severity,
                    });
                }
            }
        }
    }

    // Sort by severity (hard first) then by entity name
    contradictions.sort_by(|a, b| {
        severity_rank(&a.severity)
            .cmp(&severity_rank(&b.severity))
            .then(a.entity_name.cmp(&b.entity_name))
    });

    Ok(contradictions)
}

fn severity_rank(s: &ContradictionSeverity) -> u8 {
    match s {
        ContradictionSeverity::Hard => 0,
        ContradictionSeverity::Soft => 1,
        ContradictionSeverity::Minor => 2,
    }
}

fn classify_severity(conf_a: f64, conf_b: f64) -> ContradictionSeverity {
    let diff = (conf_a - conf_b).abs();
    if diff < 0.15 {
        ContradictionSeverity::Hard
    } else {
        ContradictionSeverity::Soft
    }
}

/// Check if two numeric values are close enough to be rounding differences.
fn are_numerically_close(a: &str, b: &str) -> bool {
    if let (Some(na), Some(nb)) = (parse_number(a), parse_number(b)) {
        let max = na.abs().max(nb.abs());
        if max == 0.0 {
            return true;
        }
        ((na - nb).abs() / max) < 0.05 // within 5%
    } else {
        false
    }
}

fn parse_number(s: &str) -> Option<f64> {
    // Strip common formatting: commas, units
    let cleaned: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    cleaned.parse::<f64>().ok()
}

/// Heuristic: do two values look contradictory?
fn looks_contradictory(a: &str, b: &str) -> bool {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    // Boolean contradictions
    let true_words = ["true", "yes", "active", "alive", "open"];
    let false_words = ["false", "no", "inactive", "dead", "closed"];

    let a_true = true_words.iter().any(|w| a_lower == *w);
    let a_false = false_words.iter().any(|w| a_lower == *w);
    let b_true = true_words.iter().any(|w| b_lower == *w);
    let b_false = false_words.iter().any(|w| b_lower == *w);

    if (a_true && b_false) || (a_false && b_true) {
        return true;
    }

    // Numeric contradiction (significantly different numbers)
    if let (Some(na), Some(nb)) = (parse_number(a), parse_number(b)) {
        let max = na.abs().max(nb.abs());
        if max > 0.0 && ((na - nb).abs() / max) > 0.2 {
            return true;
        }
    }

    false
}

/// Format contradictions for display.
pub fn format_contradictions(contradictions: &[Contradiction]) -> String {
    if contradictions.is_empty() {
        return "✅ No contradictions detected.".to_string();
    }

    let mut out = format!("⚠️  Found {} contradiction(s):\n\n", contradictions.len());

    for (i, c) in contradictions.iter().enumerate() {
        out.push_str(&format!(
            "  {}. [{}] {}: {}\n",
            i + 1,
            c.severity.as_str(),
            c.entity_name,
            c.key
        ));
        out.push_str(&format!(
            "     Value A: \"{}\" [{:.0}%] ({})\n",
            c.value_a,
            c.confidence_a * 100.0,
            if c.source_a.is_empty() {
                "unknown"
            } else {
                &c.source_a
            }
        ));
        out.push_str(&format!(
            "     Value B: \"{}\" [{:.0}%] ({})\n",
            c.value_b,
            c.confidence_b * 100.0,
            if c.source_b.is_empty() {
                "unknown"
            } else {
                &c.source_b
            }
        ));
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_contradictions_empty() {
        let brain = Brain::open_in_memory().unwrap();
        let results = detect_contradictions(&brain).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_detects_contradicting_facts() {
        let brain = Brain::open_in_memory().unwrap();
        let eid = brain.upsert_entity("Paris", "city").unwrap();
        brain
            .upsert_fact(eid, "capital", "France", "source1.com")
            .unwrap();
        brain
            .upsert_fact(eid, "capital", "Germany", "source2.com")
            .unwrap();
        let results = detect_contradictions(&brain).unwrap();
        assert!(
            !results.is_empty(),
            "Should detect conflicting capital values"
        );
        assert_eq!(results[0].entity_name, "Paris");
        assert_eq!(results[0].key, "capital");
    }

    #[test]
    fn test_ignores_keyword_duplicates() {
        let brain = Brain::open_in_memory().unwrap();
        let eid = brain.upsert_entity("Rust", "language").unwrap();
        brain
            .upsert_fact(eid, "keyword", "systems", "url1")
            .unwrap();
        brain.upsert_fact(eid, "keyword", "safety", "url2").unwrap();
        let results = detect_contradictions(&brain).unwrap();
        assert!(results.is_empty(), "Keywords shouldn't be contradictions");
    }

    #[test]
    fn test_numeric_close_values_are_minor() {
        assert!(are_numerically_close("1000", "1005"));
        assert!(are_numerically_close("100.0", "99.5"));
        assert!(!are_numerically_close("100", "200"));
    }

    #[test]
    fn test_boolean_contradiction() {
        assert!(looks_contradictory("true", "false"));
        assert!(looks_contradictory("active", "inactive"));
        assert!(looks_contradictory("alive", "dead"));
        assert!(!looks_contradictory("red", "blue")); // not boolean
    }

    #[test]
    fn test_format_contradictions_empty() {
        let out = format_contradictions(&[]);
        assert!(out.contains("No contradictions"));
    }

    #[test]
    fn test_format_contradictions_with_data() {
        let c = vec![Contradiction {
            entity_name: "Earth".to_string(),
            key: "population".to_string(),
            value_a: "7 billion".to_string(),
            value_b: "8 billion".to_string(),
            confidence_a: 0.5,
            confidence_b: 0.6,
            source_a: "old.com".to_string(),
            source_b: "new.com".to_string(),
            severity: ContradictionSeverity::Soft,
        }];
        let out = format_contradictions(&c);
        assert!(out.contains("Earth"));
        assert!(out.contains("population"));
    }

    #[test]
    fn test_severity_classification() {
        assert_eq!(classify_severity(0.5, 0.5), ContradictionSeverity::Hard);
        assert_eq!(classify_severity(0.5, 0.8), ContradictionSeverity::Soft);
    }
}
