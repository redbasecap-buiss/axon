use crate::db::Brain;
use serde_json::{json, Value};

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn mermaid_escape(s: &str) -> String {
    s.replace('"', "#quot;")
        .replace('[', "#lbrack;")
        .replace(']', "#rbrack;")
}

pub fn to_json(brain: &Brain) -> Result<String, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut ent_list: Vec<Value> = Vec::new();
    for e in &entities {
        let facts = brain.get_facts_for(e.id)?;
        let fact_values: Vec<Value> = facts
            .iter()
            .map(|f| {
                json!({
                    "key": f.key,
                    "value": f.value,
                    "confidence": f.confidence,
                })
            })
            .collect();
        ent_list.push(json!({
            "id": e.id,
            "name": e.name,
            "type": e.entity_type,
            "confidence": e.confidence,
            "access_count": e.access_count,
            "facts": fact_values,
        }));
    }
    let rel_list: Vec<Value> = relations
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "subject_id": r.subject_id,
                "predicate": r.predicate,
                "object_id": r.object_id,
                "confidence": r.confidence,
                "source_url": r.source_url,
            })
        })
        .collect();
    let doc = json!({
        "entities": ent_list,
        "relations": rel_list,
    });
    Ok(serde_json::to_string_pretty(&doc).unwrap_or_default())
}

pub fn to_json_ld(brain: &Brain) -> Result<String, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut nodes: Vec<Value> = Vec::new();
    for e in &entities {
        let facts = brain.get_facts_for(e.id)?;
        let mut node = json!({
            "@id": format!("urn:axon:entity:{}", e.id),
            "@type": e.entity_type,
            "name": e.name,
            "confidence": e.confidence,
        });
        for f in &facts {
            node[&f.key] = json!(f.value);
        }
        nodes.push(node);
    }
    let mut edges: Vec<Value> = Vec::new();
    for r in &relations {
        edges.push(json!({
            "@type": "Relationship",
            "subject": format!("urn:axon:entity:{}", r.subject_id),
            "predicate": r.predicate,
            "object": format!("urn:axon:entity:{}", r.object_id),
            "confidence": r.confidence,
        }));
    }
    let doc = json!({
        "@context": { "name": "http://schema.org/name", "confidence": "http://schema.org/Float" },
        "@graph": nodes.into_iter().chain(edges).collect::<Vec<_>>(),
    });
    Ok(serde_json::to_string_pretty(&doc).unwrap_or_default())
}

pub fn to_mermaid(brain: &Brain) -> Result<String, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut out = String::from("graph LR\n");
    for e in &entities {
        out.push_str(&format!("    n{}[\"{}\"]\n", e.id, mermaid_escape(&e.name)));
    }
    for r in &relations {
        out.push_str(&format!(
            "    n{} -->|\"{}\"| n{}\n",
            r.subject_id,
            mermaid_escape(&r.predicate),
            r.object_id
        ));
    }
    Ok(out)
}

pub fn to_dot(brain: &Brain) -> Result<String, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut out = String::from("digraph axon {\n    rankdir=LR;\n    node [shape=box];\n");
    for e in &entities {
        let label = e.name.replace('"', "\\\"");
        out.push_str(&format!("    n{} [label=\"{}\"];\n", e.id, label));
    }
    for r in &relations {
        let label = r.predicate.replace('"', "\\\"");
        out.push_str(&format!(
            "    n{} -> n{} [label=\"{}\"];\n",
            r.subject_id, r.object_id, label
        ));
    }
    out.push_str("}\n");
    Ok(out)
}

pub fn to_csv(brain: &Brain) -> Result<String, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut out = String::from("# Entities\nid,name,type,confidence,access_count\n");
    for e in &entities {
        out.push_str(&format!(
            "{},\"{}\",\"{}\",{},{}\n",
            e.id,
            e.name.replace('"', "\"\""),
            e.entity_type.replace('"', "\"\""),
            e.confidence,
            e.access_count
        ));
    }
    out.push_str("\n# Relations\nid,subject_id,predicate,object_id,confidence,source_url\n");
    for r in &relations {
        out.push_str(&format!(
            "{},{},\"{}\",{},{},\"{}\"\n",
            r.id,
            r.subject_id,
            r.predicate.replace('"', "\"\""),
            r.object_id,
            r.confidence,
            r.source_url.replace('"', "\"\"")
        ));
    }
    Ok(out)
}

pub fn to_graphml(brain: &Brain) -> Result<String, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <graphml xmlns=\"http://graphml.graphstruct.org/graphml\">\n\
         \x20 <key id=\"name\" for=\"node\" attr.name=\"name\" attr.type=\"string\"/>\n\
         \x20 <key id=\"type\" for=\"node\" attr.name=\"type\" attr.type=\"string\"/>\n\
         \x20 <key id=\"confidence\" for=\"all\" attr.name=\"confidence\" attr.type=\"double\"/>\n\
         \x20 <key id=\"predicate\" for=\"edge\" attr.name=\"predicate\" attr.type=\"string\"/>\n\
         \x20 <graph id=\"axon\" edgedefault=\"directed\">\n",
    );
    for e in &entities {
        xml.push_str(&format!(
            "    <node id=\"n{}\">\n\
             \x20     <data key=\"name\">{}</data>\n\
             \x20     <data key=\"type\">{}</data>\n\
             \x20     <data key=\"confidence\">{}</data>\n\
             \x20   </node>\n",
            e.id,
            xml_escape(&e.name),
            xml_escape(&e.entity_type),
            e.confidence
        ));
    }
    for r in &relations {
        xml.push_str(&format!(
            "    <edge id=\"e{}\" source=\"n{}\" target=\"n{}\">\n\
             \x20     <data key=\"predicate\">{}</data>\n\
             \x20     <data key=\"confidence\">{}</data>\n\
             \x20   </edge>\n",
            r.id,
            r.subject_id,
            r.object_id,
            xml_escape(&r.predicate),
            r.confidence
        ));
    }
    xml.push_str("  </graph>\n</graphml>\n");
    Ok(xml)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Brain;

    fn setup() -> Brain {
        let brain = Brain::open_in_memory().unwrap();
        let a = brain.upsert_entity("Alice", "person").unwrap();
        let b = brain.upsert_entity("Bob", "person").unwrap();
        brain
            .upsert_relation(a, "knows", b, "http://test.com")
            .unwrap();
        brain
            .upsert_fact(a, "age", "30", "http://test.com")
            .unwrap();
        brain
    }

    #[test]
    fn test_to_json() {
        let brain = setup();
        let out = to_json(&brain).unwrap();
        assert!(out.contains("Alice"));
        assert!(out.contains("Bob"));
        assert!(out.contains("knows"));
        assert!(out.contains("\"entities\""));
        assert!(out.contains("\"relations\""));
    }

    #[test]
    fn test_to_json_ld() {
        let brain = setup();
        let out = to_json_ld(&brain).unwrap();
        assert!(out.contains("@context"));
        assert!(out.contains("Alice"));
    }

    #[test]
    fn test_to_mermaid() {
        let brain = setup();
        let out = to_mermaid(&brain).unwrap();
        assert!(out.starts_with("graph LR"));
        assert!(out.contains("Alice"));
        assert!(out.contains("knows"));
    }

    #[test]
    fn test_to_dot() {
        let brain = setup();
        let out = to_dot(&brain).unwrap();
        assert!(out.starts_with("digraph axon"));
        assert!(out.contains("Alice"));
        assert!(out.contains("knows"));
    }

    #[test]
    fn test_to_csv() {
        let brain = setup();
        let out = to_csv(&brain).unwrap();
        assert!(out.contains("# Entities"));
        assert!(out.contains("# Relations"));
        assert!(out.contains("Alice"));
        assert!(out.contains("knows"));
    }

    #[test]
    fn test_to_graphml() {
        let brain = setup();
        let out = to_graphml(&brain).unwrap();
        assert!(out.contains("<graphml"));
        assert!(out.contains("Alice"));
    }

    #[test]
    fn test_to_json_empty() {
        let brain = Brain::open_in_memory().unwrap();
        let out = to_json(&brain).unwrap();
        assert!(out.contains("\"entities\""));
    }

    #[test]
    fn test_to_mermaid_empty() {
        let brain = Brain::open_in_memory().unwrap();
        let out = to_mermaid(&brain).unwrap();
        assert!(out.starts_with("graph LR"));
    }

    #[test]
    fn test_to_dot_special_chars() {
        let brain = Brain::open_in_memory().unwrap();
        brain.upsert_entity("O'Brien", "person").unwrap();
        let out = to_dot(&brain).unwrap();
        assert!(out.contains("O'Brien"));
    }

    #[test]
    fn test_to_csv_quoting() {
        let brain = Brain::open_in_memory().unwrap();
        brain.upsert_entity("Alice \"A\"", "person").unwrap();
        let out = to_csv(&brain).unwrap();
        assert!(out.contains("Alice \"\"A\"\""));
    }
}
