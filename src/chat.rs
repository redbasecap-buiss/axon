use crate::db::Brain;
use std::collections::BTreeMap;

/// Direction of a relation relative to the queried entity.
#[derive(Debug, Clone, PartialEq)]
pub enum RelDirection {
    Outgoing,
    Incoming,
}

/// A single formatted relation for display.
#[derive(Debug, Clone)]
pub struct FormattedRelation {
    pub predicate: String,
    pub target: String,
    pub direction: RelDirection,
    pub confidence: f64,
}

/// A structured answer synthesised from the knowledge graph.
#[derive(Debug, Clone)]
pub struct Answer {
    pub entity: String,
    pub entity_type: String,
    pub confidence: f64,
    pub summary: String,
    pub facts: Vec<(String, String)>,
    pub relations: Vec<FormattedRelation>,
    pub related_entities: Vec<String>,
}

// ---------------------------------------------------------------------------
// Question parsing
// ---------------------------------------------------------------------------

/// Strip common question prefixes (English + German) and return the subject.
pub fn parse_question(question: &str) -> String {
    let q = question.trim();

    // Ordered longest-first so greedy prefixes win.
    let prefixes: &[&str] = &[
        // English
        "tell me about ",
        "what do you know about ",
        "what can you tell me about ",
        "explain ",
        "describe ",
        "who is ",
        "who was ",
        "who are ",
        "who were ",
        "what is ",
        "what was ",
        "what are ",
        "what were ",
        "where is ",
        "where was ",
        "related to ",
        // German
        "erzähl mir über ",
        "erzähl mir von ",
        "was weisst du über ",
        "was weißt du über ",
        "wer ist ",
        "wer war ",
        "was ist ",
        "was war ",
        "was sind ",
        "wo ist ",
        "wo war ",
    ];

    let lower = q.to_lowercase();
    for prefix in prefixes {
        if lower.starts_with(prefix) {
            let rest = &q[prefix.len()..];
            // Strip trailing question mark / punctuation
            return strip_trailing_punct(rest).to_string();
        }
    }

    strip_trailing_punct(q).to_string()
}

fn strip_trailing_punct(s: &str) -> &str {
    s.trim_end_matches(|c: char| c == '?' || c == '!' || c == '.' || c.is_whitespace())
}

// ---------------------------------------------------------------------------
// Predicate formatting
// ---------------------------------------------------------------------------

/// Turn `snake_case` predicates into `Title Case` labels.
pub fn humanize_predicate(pred: &str) -> String {
    pred.replace('_', " ")
        .split_whitespace()
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    upper + chars.as_str()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Relation helpers
// ---------------------------------------------------------------------------

fn direction_arrow(dir: &RelDirection) -> &'static str {
    match dir {
        RelDirection::Outgoing => "→",
        RelDirection::Incoming => "←",
    }
}

/// Group relations by predicate and format for display.
pub fn format_relations(relations: &[FormattedRelation]) -> String {
    if relations.is_empty() {
        return String::new();
    }

    // Group by predicate (preserve insertion order with BTreeMap for determinism)
    let mut groups: BTreeMap<String, Vec<&FormattedRelation>> = BTreeMap::new();
    for rel in relations {
        groups.entry(rel.predicate.clone()).or_default().push(rel);
    }

    let mut out = String::new();
    for (predicate, members) in &groups {
        let label = humanize_predicate(predicate);
        out.push_str(&format!("  {}:\n", label));
        for rel in members.iter().take(5) {
            out.push_str(&format!(
                "    {} {} [{:.0}%]\n",
                direction_arrow(&rel.direction),
                rel.target,
                rel.confidence * 100.0
            ));
        }
        if members.len() > 5 {
            out.push_str(&format!("    ... and {} more\n", members.len() - 5));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Summary generation
// ---------------------------------------------------------------------------

/// Build a natural-language summary from entity metadata, facts, and relations.
pub fn generate_summary(
    entity: &str,
    entity_type: &str,
    facts: &[(String, String)],
    relations: &[FormattedRelation],
) -> String {
    let etype = entity_type.to_lowercase();
    let fact_snippet = summarize_facts(facts);
    let rel_snippet = summarize_top_relations(relations);

    match etype.as_str() {
        "person" | "people" | "scientist" | "artist" | "politician" | "author" | "musician" => {
            let mut s = format!("{entity} is a {etype}.");
            if !fact_snippet.is_empty() {
                s.push(' ');
                s.push_str(&fact_snippet);
            }
            if !rel_snippet.is_empty() {
                s.push_str(&format!(" Connected to: {rel_snippet}."));
            }
            s
        }
        "place" | "city" | "country" | "location" | "region" | "continent" => {
            let mut s = format!("{entity} is a {etype}.");
            if !fact_snippet.is_empty() {
                s.push(' ');
                s.push_str(&fact_snippet);
            }
            if !rel_snippet.is_empty() {
                s.push_str(&format!(" Notable for: {rel_snippet}."));
            }
            s
        }
        "concept" | "theory" | "field" | "discipline" | "topic" => {
            let mut s = format!("{entity} is a {etype}.");
            if !fact_snippet.is_empty() {
                s.push(' ');
                s.push_str(&fact_snippet);
            }
            if !rel_snippet.is_empty() {
                s.push_str(&format!(" Related to: {rel_snippet}."));
            }
            s
        }
        "organization" | "company" | "institution" | "org" => {
            let mut s = format!("{entity} is an organization.");
            if !fact_snippet.is_empty() {
                s.push(' ');
                s.push_str(&fact_snippet);
            }
            if !rel_snippet.is_empty() {
                s.push_str(&format!(" Associated with: {rel_snippet}."));
            }
            s
        }
        _ => {
            let mut s = format!("{entity} ({etype}).");
            if !fact_snippet.is_empty() {
                s.push(' ');
                s.push_str(&fact_snippet);
            }
            if !rel_snippet.is_empty() {
                s.push_str(&format!(" Related to: {rel_snippet}."));
            }
            s
        }
    }
}

fn summarize_facts(facts: &[(String, String)]) -> String {
    if facts.is_empty() {
        return String::new();
    }
    let items: Vec<String> = facts
        .iter()
        .take(4)
        .map(|(k, v)| format!("{}: {}", humanize_predicate(k), v))
        .collect();
    items.join(". ")
}

fn summarize_top_relations(relations: &[FormattedRelation]) -> String {
    if relations.is_empty() {
        return String::new();
    }
    let names: Vec<&str> = relations
        .iter()
        .take(5)
        .map(|r| r.target.as_str())
        .collect();
    names.join(", ")
}

// ---------------------------------------------------------------------------
// Main answer engine
// ---------------------------------------------------------------------------

/// Resolve a free-text question to a structured `Answer` from the knowledge graph.
pub fn answer_question(brain: &Brain, question: &str) -> Result<Option<Answer>, rusqlite::Error> {
    let subject = parse_question(question);
    if subject.is_empty() {
        return Ok(None);
    }

    // Try exact match first, then fuzzy
    let entity = if let Some(e) = brain.get_entity_by_name(&subject)? {
        e
    } else {
        let candidates = brain.search_entities(&subject)?;
        // Filter noise types
        let good: Vec<_> = candidates
            .into_iter()
            .filter(|e| {
                !matches!(
                    e.entity_type.as_str(),
                    "source"
                        | "url"
                        | "phrase"
                        | "relative_date"
                        | "number_unit"
                        | "date"
                        | "year"
                        | "currency"
                        | "email"
                        | "compound_noun"
                )
            })
            .collect();
        match good.into_iter().next() {
            Some(e) => e,
            None => return Ok(None),
        }
    };

    // Load facts
    let raw_facts = brain.get_facts_for(entity.id)?;
    let facts: Vec<(String, String)> = raw_facts
        .iter()
        .map(|f| (f.key.clone(), f.value.clone()))
        .collect();

    // Load relations (both directions), deduplicate, sort by confidence
    let raw_rels = brain.get_relations_for(entity.id)?;
    let mut relations: Vec<FormattedRelation> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (subj, pred, obj, conf) in &raw_rels {
        let (target, direction) = if subj == &entity.name {
            (obj.clone(), RelDirection::Outgoing)
        } else {
            (subj.clone(), RelDirection::Incoming)
        };
        let key = (pred.clone(), target.clone());
        if seen.insert(key) {
            relations.push(FormattedRelation {
                predicate: pred.clone(),
                target,
                direction,
                confidence: *conf,
            });
        }
    }
    relations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    // Related entities (just names, top unique ones not already in relations)
    let raw_related = brain.get_related_entities(entity.id)?;
    let rel_names: std::collections::HashSet<&str> =
        relations.iter().map(|r| r.target.as_str()).collect();
    let related_entities: Vec<String> = raw_related
        .iter()
        .map(|(name, _, _)| name.clone())
        .filter(|n| !rel_names.contains(n.as_str()) && n != &entity.name)
        .take(8)
        .collect();

    let summary = generate_summary(&entity.name, &entity.entity_type, &facts, &relations);

    Ok(Some(Answer {
        entity: entity.name,
        entity_type: entity.entity_type,
        confidence: entity.confidence,
        summary,
        facts,
        relations,
        related_entities,
    }))
}

/// Format an `Answer` for terminal display.
pub fn format_answer(answer: &Answer) -> String {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "🧠 {} ({}) — confidence: {:.0}%\n\n",
        answer.entity,
        answer.entity_type,
        answer.confidence * 100.0
    ));

    // Summary
    out.push_str(&answer.summary);
    out.push_str("\n\n");

    // Facts
    if !answer.facts.is_empty() {
        out.push_str("📋 Facts:\n");
        for (key, value) in &answer.facts {
            out.push_str(&format!("  • {}: {}\n", humanize_predicate(key), value));
        }
        out.push('\n');
    }

    // Relations
    if !answer.relations.is_empty() {
        out.push_str(&format!("🔗 Relations ({}):\n", answer.relations.len()));
        out.push_str(&format_relations(&answer.relations));
        out.push('\n');
    }

    // Related entities
    if !answer.related_entities.is_empty() {
        out.push_str(&format!("💡 Related: {}\n", answer.related_entities.join(", ")));
    }

    out
}

/// Run an interactive chat REPL.
pub fn run_chat(brain: &Brain) -> Result<(), Box<dyn std::error::Error>> {
    let version = env!("CARGO_PKG_VERSION");
    println!("🧠 axon v{version} — Crystalline Intelligence");
    println!("Type a question, or 'quit' to exit.\n");

    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        eprint!("> ");
        line.clear();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            break; // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed.to_lowercase().as_str(), "quit" | "exit" | "q") {
            println!("👋 Goodbye!");
            break;
        }

        match answer_question(brain, trimmed) {
            Ok(Some(answer)) => {
                println!();
                print!("{}", format_answer(&answer));
            }
            Ok(None) => {
                println!("🤷 I don't know anything about that yet. Try feeding me some URLs!\n");
            }
            Err(e) => {
                eprintln!("⚠️  Error: {e}\n");
            }
        }
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Brain;

    fn test_brain() -> Brain {
        Brain::open_in_memory().unwrap()
    }

    fn populated_brain() -> Brain {
        let brain = test_brain();
        let eid = brain.upsert_entity("Alan Turing", "person").unwrap();
        brain.upsert_fact(eid, "birth_year", "1912", "src").unwrap();
        brain
            .upsert_fact(eid, "nationality", "British", "src")
            .unwrap();

        let cs_id = brain.upsert_entity("Computer Science", "field").unwrap();
        let ai_id = brain
            .upsert_entity("Artificial Intelligence", "field")
            .unwrap();
        let bp_id = brain.upsert_entity("Bletchley Park", "place").unwrap();

        brain
            .upsert_relation(eid, "pioneered", cs_id, "src")
            .unwrap();
        brain
            .upsert_relation(eid, "pioneered", ai_id, "src")
            .unwrap();
        brain
            .upsert_relation(eid, "worked_at", bp_id, "src")
            .unwrap();
        brain
    }

    // ---- Question parsing (English) ----

    #[test]
    fn test_parse_who_is() {
        assert_eq!(parse_question("Who is Alan Turing?"), "Alan Turing");
    }

    #[test]
    fn test_parse_who_was() {
        assert_eq!(parse_question("Who was Newton?"), "Newton");
    }

    #[test]
    fn test_parse_what_is() {
        assert_eq!(parse_question("What is CRISPR?"), "CRISPR");
    }

    #[test]
    fn test_parse_tell_me_about() {
        assert_eq!(parse_question("Tell me about quantum physics"), "quantum physics");
    }

    #[test]
    fn test_parse_explain() {
        assert_eq!(parse_question("Explain photosynthesis"), "photosynthesis");
    }

    #[test]
    fn test_parse_related_to() {
        assert_eq!(
            parse_question("Related to quantum entanglement"),
            "quantum entanglement"
        );
    }

    #[test]
    fn test_parse_bare_subject() {
        assert_eq!(parse_question("Rust"), "Rust");
    }

    #[test]
    fn test_parse_strips_trailing_punctuation() {
        assert_eq!(parse_question("What is DNA??!"), "DNA");
    }

    // ---- Question parsing (German) ----

    #[test]
    fn test_parse_wer_ist() {
        assert_eq!(parse_question("Wer ist Einstein?"), "Einstein");
    }

    #[test]
    fn test_parse_wer_war() {
        assert_eq!(parse_question("Wer war Goethe?"), "Goethe");
    }

    #[test]
    fn test_parse_was_ist() {
        assert_eq!(parse_question("Was ist Quantenmechanik?"), "Quantenmechanik");
    }

    #[test]
    fn test_parse_erzaehl_mir_ueber() {
        assert_eq!(parse_question("Erzähl mir über CERN"), "CERN");
    }

    // ---- Predicate formatting ----

    #[test]
    fn test_humanize_predicate() {
        assert_eq!(humanize_predicate("worked_at"), "Worked At");
        assert_eq!(humanize_predicate("pioneered"), "Pioneered");
        assert_eq!(humanize_predicate("is_part_of"), "Is Part Of");
    }

    // ---- Summary generation ----

    #[test]
    fn test_summary_person() {
        let facts = vec![("birth_year".into(), "1912".into())];
        let rels = vec![FormattedRelation {
            predicate: "pioneered".into(),
            target: "CS".into(),
            direction: RelDirection::Outgoing,
            confidence: 0.8,
        }];
        let s = generate_summary("Turing", "person", &facts, &rels);
        assert!(s.contains("Turing"));
        assert!(s.contains("person"));
        assert!(s.contains("CS"));
    }

    #[test]
    fn test_summary_place() {
        let s = generate_summary("Zurich", "city", &[], &[]);
        assert!(s.contains("Zurich"));
        assert!(s.contains("city"));
    }

    #[test]
    fn test_summary_concept() {
        let s = generate_summary("Entropy", "concept", &[], &[]);
        assert!(s.contains("Entropy"));
        assert!(s.contains("concept"));
    }

    #[test]
    fn test_summary_org() {
        let s = generate_summary("CERN", "organization", &[], &[]);
        assert!(s.contains("CERN"));
        assert!(s.contains("organization"));
    }

    #[test]
    fn test_summary_unknown_type() {
        let s = generate_summary("Foo", "widget", &[], &[]);
        assert!(s.contains("Foo"));
        assert!(s.contains("widget"));
    }

    // ---- Relation formatting ----

    #[test]
    fn test_format_relations_grouped() {
        let rels = vec![
            FormattedRelation {
                predicate: "pioneered".into(),
                target: "CS".into(),
                direction: RelDirection::Outgoing,
                confidence: 0.8,
            },
            FormattedRelation {
                predicate: "pioneered".into(),
                target: "AI".into(),
                direction: RelDirection::Outgoing,
                confidence: 0.7,
            },
            FormattedRelation {
                predicate: "worked_at".into(),
                target: "Bletchley Park".into(),
                direction: RelDirection::Outgoing,
                confidence: 0.9,
            },
        ];
        let out = format_relations(&rels);
        assert!(out.contains("Pioneered:"));
        assert!(out.contains("→ CS"));
        assert!(out.contains("→ AI"));
        assert!(out.contains("Worked At:"));
        assert!(out.contains("→ Bletchley Park"));
    }

    #[test]
    fn test_format_relations_empty() {
        assert_eq!(format_relations(&[]), "");
    }

    // ---- Answer engine ----

    #[test]
    fn test_answer_question_exact_match() {
        let brain = populated_brain();
        let answer = answer_question(&brain, "Who was Alan Turing?").unwrap();
        assert!(answer.is_some());
        let a = answer.unwrap();
        assert_eq!(a.entity, "Alan Turing");
        assert_eq!(a.entity_type, "person");
        assert!(!a.facts.is_empty());
        assert!(!a.relations.is_empty());
    }

    #[test]
    fn test_answer_question_fuzzy_match() {
        let brain = populated_brain();
        let answer = answer_question(&brain, "Turing").unwrap();
        assert!(answer.is_some());
    }

    #[test]
    fn test_answer_question_not_found() {
        let brain = test_brain();
        let answer = answer_question(&brain, "Who is Xyzzy?").unwrap();
        assert!(answer.is_none());
    }

    #[test]
    fn test_answer_question_empty_input() {
        let brain = test_brain();
        let answer = answer_question(&brain, "").unwrap();
        assert!(answer.is_none());
    }

    #[test]
    fn test_format_answer_output() {
        let brain = populated_brain();
        let answer = answer_question(&brain, "Alan Turing").unwrap().unwrap();
        let text = format_answer(&answer);
        assert!(text.contains("🧠"));
        assert!(text.contains("Alan Turing"));
        assert!(text.contains("📋 Facts:"));
        assert!(text.contains("🔗 Relations"));
    }

    // ---- Edge cases ----

    #[test]
    fn test_entity_no_facts_no_relations() {
        let brain = test_brain();
        brain.upsert_entity("Lonely", "concept").unwrap();
        let answer = answer_question(&brain, "Lonely").unwrap().unwrap();
        assert!(answer.facts.is_empty());
        assert!(answer.relations.is_empty());
        assert!(answer.summary.contains("Lonely"));
    }
}
