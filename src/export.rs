use crate::db::Brain;
use serde_json::{json, Value};

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
        for f in &facts { node[&f.key] = json!(f.value); }
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

pub fn to_graphml(brain: &Brain) -> Result<String, rusqlite::Error> {
    let entities = brain.all_entities()?;
    let relations = brain.all_relations()?;
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<graphml xmlns=\"http://graphml.graphstruct.org/graphml\">
  <key id=\"name\" for=\"node\" attr.name=\"name\" attr.type=\"string\"/>
  <key id=\"type\" for=\"node\" attr.name=\"type\" attr.type=\"string\"/>
  <key id=\"confidence\" for=\"all\" attr.name=\"confidence\" attr.type=\"double\"/>
  <key id=\"predicate\" for=\"edge\" attr.name=\"predicate\" attr.type=\"string\"/>
  <graph id=\"axon\" edgedefault=\"directed\">
");
    for e in &entities {
        xml.push_str(&format!("    <node id=\"n{}\">
      <data key=\"name\">{}</data>
      <data key=\"type\">{}</data>
      <data key=\"confidence\">{}</data>
    </node>
", e.id, xml_escape(&e.name), xml_escape(&e.entity_type), e.confidence));
    }
    for r in &relations {
        xml.push_str(&format!("    <edge id=\"e{}\" source=\"n{}\" target=\"n{}\">
      <data key=\"predicate\">{}</data>
      <data key=\"confidence\">{}</data>
    </edge>
", r.id, r.subject_id, r.object_id, xml_escape(&r.predicate), r.confidence));
    }
    xml.push_str("  </graph>
</graphml>
");
    Ok(xml)
}
