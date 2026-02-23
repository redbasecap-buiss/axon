#![allow(dead_code)]
use chrono::{NaiveDateTime, Utc};
use rusqlite::{params, Connection, Result};
use std::path::Path;

use crate::nlp::Extracted;

pub struct Brain {
    conn: Connection,
}

pub struct Stats {
    pub entity_count: usize,
    pub relation_count: usize,
    pub fact_count: usize,
    pub source_count: usize,
    pub db_size: String,
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: i64,
    pub name: String,
    pub entity_type: String,
    pub confidence: f64,
    pub first_seen: NaiveDateTime,
    pub last_seen: NaiveDateTime,
    pub access_count: i64,
}

#[derive(Debug, Clone)]
pub struct Relation {
    pub id: i64,
    pub subject_id: i64,
    pub predicate: String,
    pub object_id: i64,
    pub confidence: f64,
    pub source_url: String,
    pub learned_at: NaiveDateTime,
}

#[derive(Debug, Clone)]
pub struct Fact {
    pub id: i64,
    pub entity_id: i64,
    pub key: String,
    pub value: String,
    pub confidence: f64,
    pub source_url: String,
}

#[derive(Debug, Clone)]
pub struct FrontierEntry {
    pub url: String,
    pub priority: i32,
    pub last_crawled: Option<NaiveDateTime>,
}

impl Brain {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        let brain = Brain { conn };
        brain.init_schema()?;
        Ok(brain)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let brain = Brain { conn };
        brain.init_schema()?;
        Ok(brain)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL DEFAULT 0.5,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 1,
                UNIQUE(name, entity_type)
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                predicate TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_url TEXT NOT NULL DEFAULT '',
                learned_at TEXT NOT NULL,
                FOREIGN KEY(subject_id) REFERENCES entities(id),
                FOREIGN KEY(object_id) REFERENCES entities(id),
                UNIQUE(subject_id, predicate, object_id)
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_url TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(entity_id) REFERENCES entities(id),
                UNIQUE(entity_id, key, value)
            );
            CREATE TABLE IF NOT EXISTS frontier (
                url TEXT PRIMARY KEY,
                priority INTEGER NOT NULL DEFAULT 0,
                last_crawled TEXT,
                content_hash TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
            CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
            CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);
            ",
        )
    }

    pub fn upsert_entity(&self, name: &str, entity_type: &str) -> Result<i64> {
        let now = Utc::now()
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        self.conn.execute(
            "INSERT INTO entities (name, entity_type, confidence, first_seen, last_seen, access_count)
             VALUES (?1, ?2, 0.5, ?3, ?3, 1)
             ON CONFLICT(name, entity_type) DO UPDATE SET
                confidence = MIN(1.0, confidence + 0.1),
                last_seen = ?3,
                access_count = access_count + 1",
            params![name, entity_type, now],
        )?;
        let id = self.conn.query_row(
            "SELECT id FROM entities WHERE name = ?1 AND entity_type = ?2",
            params![name, entity_type],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    pub fn upsert_relation(
        &self,
        subject_id: i64,
        predicate: &str,
        object_id: i64,
        source_url: &str,
    ) -> Result<bool> {
        let now = Utc::now()
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let changes = self.conn.execute(
            "INSERT INTO relations (subject_id, predicate, object_id, confidence, source_url, learned_at)
             VALUES (?1, ?2, ?3, 0.5, ?4, ?5)
             ON CONFLICT(subject_id, predicate, object_id) DO UPDATE SET
                confidence = MIN(1.0, confidence + 0.1),
                learned_at = ?5",
            params![subject_id, predicate, object_id, source_url, now],
        )?;
        Ok(changes > 0)
    }

    pub fn upsert_fact(
        &self,
        entity_id: i64,
        key: &str,
        value: &str,
        source_url: &str,
    ) -> Result<bool> {
        let changes = self.conn.execute(
            "INSERT INTO facts (entity_id, key, value, confidence, source_url)
             VALUES (?1, ?2, ?3, 0.5, ?4)
             ON CONFLICT(entity_id, key, value) DO UPDATE SET
                confidence = MIN(1.0, confidence + 0.1)",
            params![entity_id, key, value, source_url],
        )?;
        Ok(changes > 0)
    }

    pub fn learn(&self, extracted: &Extracted) -> Result<(usize, usize, usize)> {
        let mut entity_count = 0;
        let mut relation_count = 0;
        let mut fact_count = 0;

        // Insert entities
        let mut entity_ids = std::collections::HashMap::new();
        for (name, etype) in &extracted.entities {
            let id = self.upsert_entity(name, etype)?;
            entity_ids.insert(name.clone(), id);
            entity_count += 1;
        }

        // Insert relations
        for (subj, pred, obj) in &extracted.relations {
            let subj_type = extracted
                .entities
                .iter()
                .find(|(n, _)| n == subj)
                .map(|(_, t)| t.as_str())
                .unwrap_or("unknown");
            let obj_type = extracted
                .entities
                .iter()
                .find(|(n, _)| n == obj)
                .map(|(_, t)| t.as_str())
                .unwrap_or("unknown");
            let subj_id = *entity_ids
                .entry(subj.clone())
                .or_insert_with(|| self.upsert_entity(subj, subj_type).unwrap_or(0));
            let obj_id = *entity_ids
                .entry(obj.clone())
                .or_insert_with(|| self.upsert_entity(obj, obj_type).unwrap_or(0));
            if subj_id > 0 && obj_id > 0 {
                self.upsert_relation(subj_id, pred, obj_id, &extracted.source_url)?;
                relation_count += 1;
            }
        }

        // Insert facts (keywords as facts about a source entity)
        if !extracted.keywords.is_empty() {
            let source_id = self.upsert_entity(&extracted.source_url, "source")?;
            for kw in &extracted.keywords {
                self.upsert_fact(source_id, "keyword", kw, &extracted.source_url)?;
                fact_count += 1;
            }
        }

        Ok((entity_count, relation_count, fact_count))
    }

    pub fn search_entities(&self, query: &str) -> Result<Vec<Entity>> {
        let pattern = format!("%{query}%");
        let mut stmt = self.conn.prepare(
            "SELECT id, name, entity_type, confidence, first_seen, last_seen, access_count
             FROM entities WHERE name LIKE ?1
             ORDER BY confidence DESC, access_count DESC LIMIT 20",
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
            Ok(Entity {
                id: row.get(0)?,
                name: row.get(1)?,
                entity_type: row.get(2)?,
                confidence: row.get(3)?,
                first_seen: parse_dt(&row.get::<_, String>(4)?),
                last_seen: parse_dt(&row.get::<_, String>(5)?),
                access_count: row.get(6)?,
            })
        })?;
        rows.collect()
    }

    pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, entity_type, confidence, first_seen, last_seen, access_count
             FROM entities WHERE LOWER(name) = LOWER(?1)
             ORDER BY confidence DESC LIMIT 1",
        )?;
        let mut rows = stmt.query_map(params![name], |row| {
            Ok(Entity {
                id: row.get(0)?,
                name: row.get(1)?,
                entity_type: row.get(2)?,
                confidence: row.get(3)?,
                first_seen: parse_dt(&row.get::<_, String>(4)?),
                last_seen: parse_dt(&row.get::<_, String>(5)?),
                access_count: row.get(6)?,
            })
        })?;
        match rows.next() {
            Some(Ok(e)) => Ok(Some(e)),
            _ => Ok(None),
        }
    }

    pub fn get_relations_for(&self, entity_id: i64) -> Result<Vec<(String, String, String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT e1.name, r.predicate, e2.name, r.confidence
             FROM relations r
             JOIN entities e1 ON r.subject_id = e1.id
             JOIN entities e2 ON r.object_id = e2.id
             WHERE r.subject_id = ?1 OR r.object_id = ?1
             ORDER BY r.confidence DESC",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        rows.collect()
    }

    pub fn get_facts_for(&self, entity_id: i64) -> Result<Vec<Fact>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, entity_id, key, value, confidence, source_url
             FROM facts WHERE entity_id = ?1
             ORDER BY confidence DESC",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok(Fact {
                id: row.get(0)?,
                entity_id: row.get(1)?,
                key: row.get(2)?,
                value: row.get(3)?,
                confidence: row.get(4)?,
                source_url: row.get(5)?,
            })
        })?;
        rows.collect()
    }

    pub fn get_related_entities(&self, entity_id: i64) -> Result<Vec<(String, String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT e.name, r.predicate, r.confidence
             FROM relations r
             JOIN entities e ON (e.id = r.object_id AND r.subject_id = ?1)
                             OR (e.id = r.subject_id AND r.object_id = ?1)
             ORDER BY r.confidence DESC LIMIT 20",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?;
        rows.collect()
    }

    pub fn recent_facts(&self, limit: usize) -> Result<Vec<(String, String, String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT e.name, f.key, f.value, f.confidence
             FROM facts f JOIN entities e ON f.entity_id = e.id
             ORDER BY f.id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        rows.collect()
    }

    pub fn top_entities(&self, limit: usize) -> Result<Vec<(String, String, f64, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, entity_type, confidence, access_count
             FROM entities
             WHERE entity_type != 'source'
             ORDER BY access_count DESC, confidence DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        rows.collect()
    }

    pub fn forget(&self, threshold: f64, min_age_days: i64) -> Result<usize> {
        let cutoff = (Utc::now() - chrono::Duration::days(min_age_days))
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let mut total = 0;
        total += self.conn.execute(
            "DELETE FROM facts WHERE confidence < ?1 AND entity_id IN (
                SELECT id FROM entities WHERE last_seen < ?2
            )",
            params![threshold, cutoff],
        )?;
        total += self.conn.execute(
            "DELETE FROM relations WHERE confidence < ?1 AND learned_at < ?2",
            params![threshold, cutoff],
        )?;
        total += self.conn.execute(
            "DELETE FROM entities WHERE confidence < ?1 AND last_seen < ?2
             AND id NOT IN (SELECT subject_id FROM relations)
             AND id NOT IN (SELECT object_id FROM relations)
             AND id NOT IN (SELECT entity_id FROM facts)",
            params![threshold, cutoff],
        )?;
        Ok(total)
    }

    pub fn apply_decay(&self, rate: f64) -> Result<usize> {
        let changed = self.conn.execute(
            "UPDATE entities SET confidence = MAX(0.0, confidence - ?1)
             WHERE confidence > 0.0",
            params![rate],
        )?;
        self.conn.execute(
            "UPDATE relations SET confidence = MAX(0.0, confidence - ?1)
             WHERE confidence > 0.0",
            params![rate],
        )?;
        Ok(changed)
    }

    pub fn stats(&self) -> Result<Stats> {
        let entity_count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))?;
        let relation_count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM relations", [], |r| r.get(0))?;
        let fact_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))?;
        let source_count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM entities WHERE entity_type = 'source'",
            [],
            |r| r.get(0),
        )?;
        let page_size: i64 = self.conn.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        let page_count: i64 = self.conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let bytes = page_size * page_count;
        let db_size = if bytes > 1_048_576 {
            format!("{:.1} MB", bytes as f64 / 1_048_576.0)
        } else {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        };
        Ok(Stats {
            entity_count,
            relation_count,
            fact_count,
            source_count,
            db_size,
        })
    }

    // Frontier management
    pub fn add_to_frontier(&self, url: &str, priority: i32) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO frontier (url, priority) VALUES (?1, ?2)",
            params![url, priority],
        )?;
        Ok(())
    }

    pub fn get_frontier(&self, limit: usize) -> Result<Vec<FrontierEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT url, priority, last_crawled FROM frontier
             ORDER BY priority DESC, last_crawled ASC NULLS FIRST
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |row| {
            let lc: Option<String> = row.get(2)?;
            Ok(FrontierEntry {
                url: row.get(0)?,
                priority: row.get(1)?,
                last_crawled: lc.map(|s| parse_dt(&s)),
            })
        })?;
        rows.collect()
    }

    pub fn mark_crawled(&self, url: &str, content_hash: &str) -> Result<()> {
        let now = Utc::now()
            .naive_utc()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        self.conn.execute(
            "UPDATE frontier SET last_crawled = ?1, content_hash = ?2 WHERE url = ?3",
            params![now, content_hash, url],
        )?;
        Ok(())
    }

    pub fn get_content_hash(&self, url: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT content_hash FROM frontier WHERE url = ?1")?;
        let mut rows = stmt.query_map(params![url], |row| row.get::<_, Option<String>>(0))?;
        match rows.next() {
            Some(Ok(h)) => Ok(h),
            _ => Ok(None),
        }
    }

    pub fn all_entities(&self) -> Result<Vec<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, entity_type, confidence, first_seen, last_seen, access_count
             FROM entities ORDER BY name ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Entity {
                id: row.get(0)?,
                name: row.get(1)?,
                entity_type: row.get(2)?,
                confidence: row.get(3)?,
                first_seen: parse_dt(&row.get::<_, String>(4)?),
                last_seen: parse_dt(&row.get::<_, String>(5)?),
                access_count: row.get(6)?,
            })
        })?;
        rows.collect()
    }

    pub fn all_relations(&self) -> Result<Vec<Relation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, subject_id, predicate, object_id, confidence, source_url, learned_at
             FROM relations ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Relation {
                id: row.get(0)?,
                subject_id: row.get(1)?,
                predicate: row.get(2)?,
                object_id: row.get(3)?,
                confidence: row.get(4)?,
                source_url: row.get(5)?,
                learned_at: parse_dt(&row.get::<_, String>(6)?),
            })
        })?;
        rows.collect()
    }

    pub fn get_entity_by_id(&self, id: i64) -> Result<Option<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, entity_type, confidence, first_seen, last_seen, access_count
             FROM entities WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(Entity {
                id: row.get(0)?,
                name: row.get(1)?,
                entity_type: row.get(2)?,
                confidence: row.get(3)?,
                first_seen: parse_dt(&row.get::<_, String>(4)?),
                last_seen: parse_dt(&row.get::<_, String>(5)?),
                access_count: row.get(6)?,
            })
        })?;
        match rows.next() {
            Some(Ok(e)) => Ok(Some(e)),
            _ => Ok(None),
        }
    }

    pub fn merge_entities(&self, from_id: i64, into_id: i64) -> Result<()> {
        // Delete relations that would violate the UNIQUE constraint before updating
        self.conn.execute(
            "DELETE FROM relations WHERE subject_id = ?1
             AND EXISTS (SELECT 1 FROM relations r2
                         WHERE r2.subject_id = ?2
                           AND r2.predicate = relations.predicate
                           AND r2.object_id = relations.object_id)",
            params![from_id, into_id],
        )?;
        self.conn.execute(
            "DELETE FROM relations WHERE object_id = ?1
             AND EXISTS (SELECT 1 FROM relations r2
                         WHERE r2.object_id = ?2
                           AND r2.predicate = relations.predicate
                           AND r2.subject_id = relations.subject_id)",
            params![from_id, into_id],
        )?;
        // Also delete self-referential relations that would result
        self.conn.execute(
            "DELETE FROM relations WHERE subject_id = ?1 AND object_id = ?2",
            params![from_id, into_id],
        )?;
        self.conn.execute(
            "DELETE FROM relations WHERE subject_id = ?2 AND object_id = ?1",
            params![from_id, into_id],
        )?;
        self.conn.execute(
            "UPDATE relations SET subject_id = ?2 WHERE subject_id = ?1",
            params![from_id, into_id],
        )?;
        self.conn.execute(
            "UPDATE relations SET object_id = ?2 WHERE object_id = ?1",
            params![from_id, into_id],
        )?;
        self.conn.execute(
            "UPDATE OR IGNORE facts SET entity_id = ?2 WHERE entity_id = ?1",
            params![from_id, into_id],
        )?;
        self.conn
            .execute("DELETE FROM facts WHERE entity_id = ?1", params![from_id])?;
        self.conn
            .execute("DELETE FROM entities WHERE id = ?1", params![from_id])?;
        Ok(())
    }

    /// Provide access to the underlying connection for extensions.
    pub fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        f(&self.conn)
    }

    pub fn search_facts(&self, query: &str) -> Result<Vec<(String, String, String, f64)>> {
        let pattern = format!("%{query}%");
        let mut stmt = self.conn.prepare(
            "SELECT e.name, f.key, f.value, f.confidence
             FROM facts f JOIN entities e ON f.entity_id = e.id
             WHERE f.value LIKE ?1 OR f.key LIKE ?1 OR e.name LIKE ?1
             ORDER BY f.confidence DESC LIMIT 20",
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        rows.collect()
    }
}

fn parse_dt(s: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap_or_else(|_| Utc::now().naive_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_brain() -> Brain {
        Brain::open_in_memory().unwrap()
    }

    #[test]
    fn test_upsert_entity() {
        let brain = test_brain();
        let id1 = brain.upsert_entity("Rust", "language").unwrap();
        let id2 = brain.upsert_entity("Rust", "language").unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_entity_confidence_increases() {
        let brain = test_brain();
        brain.upsert_entity("Rust", "language").unwrap();
        let e1 = brain.get_entity_by_name("Rust").unwrap().unwrap();
        assert!((e1.confidence - 0.5).abs() < f64::EPSILON);

        brain.upsert_entity("Rust", "language").unwrap();
        let e2 = brain.get_entity_by_name("Rust").unwrap().unwrap();
        assert!((e2.confidence - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_upsert_relation() {
        let brain = test_brain();
        let s = brain.upsert_entity("Rust", "language").unwrap();
        let o = brain.upsert_entity("Mozilla", "organization").unwrap();
        let created = brain
            .upsert_relation(s, "created_by", o, "https://example.com")
            .unwrap();
        assert!(created);
    }

    #[test]
    fn test_upsert_fact() {
        let brain = test_brain();
        let eid = brain.upsert_entity("Rust", "language").unwrap();
        brain
            .upsert_fact(eid, "year", "2010", "https://example.com")
            .unwrap();
        let facts = brain.get_facts_for(eid).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "2010");
    }

    #[test]
    fn test_search_entities() {
        let brain = test_brain();
        brain.upsert_entity("Rust", "language").unwrap();
        brain.upsert_entity("Python", "language").unwrap();
        let results = brain.search_entities("Rust").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Rust");
    }

    #[test]
    fn test_get_related_entities() {
        let brain = test_brain();
        let s = brain.upsert_entity("Rust", "language").unwrap();
        let o = brain.upsert_entity("Mozilla", "organization").unwrap();
        brain.upsert_relation(s, "created_by", o, "test").unwrap();
        let related = brain.get_related_entities(s).unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].0, "Mozilla");
    }

    #[test]
    fn test_recent_facts() {
        let brain = test_brain();
        let eid = brain.upsert_entity("Test", "test").unwrap();
        brain.upsert_fact(eid, "k1", "v1", "url").unwrap();
        brain.upsert_fact(eid, "k2", "v2", "url").unwrap();
        let recent = brain.recent_facts(10).unwrap();
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_forget() {
        let brain = test_brain();
        let eid = brain.upsert_entity("OldStuff", "test").unwrap();
        brain.upsert_fact(eid, "k", "v", "url").unwrap();
        // Set last_seen to the past so forget can prune it
        brain
            .conn
            .execute(
                "UPDATE entities SET last_seen = '2020-01-01 00:00:00' WHERE id = ?1",
                params![eid],
            )
            .unwrap();
        let pruned = brain.forget(0.6, 0).unwrap();
        assert!(pruned > 0);
    }

    #[test]
    fn test_apply_decay() {
        let brain = test_brain();
        brain.upsert_entity("Test", "test").unwrap();
        let decayed = brain.apply_decay(0.1).unwrap();
        assert_eq!(decayed, 1);
        let e = brain.get_entity_by_name("Test").unwrap().unwrap();
        assert!((e.confidence - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let brain = test_brain();
        brain.upsert_entity("A", "test").unwrap();
        let stats = brain.stats().unwrap();
        assert_eq!(stats.entity_count, 1);
    }

    #[test]
    fn test_frontier() {
        let brain = test_brain();
        brain.add_to_frontier("https://example.com", 1).unwrap();
        let frontier = brain.get_frontier(10).unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].url, "https://example.com");
    }

    #[test]
    fn test_mark_crawled() {
        let brain = test_brain();
        brain.add_to_frontier("https://example.com", 1).unwrap();
        brain.mark_crawled("https://example.com", "abc123").unwrap();
        let hash = brain.get_content_hash("https://example.com").unwrap();
        assert_eq!(hash, Some("abc123".to_string()));
    }

    #[test]
    fn test_search_facts() {
        let brain = test_brain();
        let eid = brain.upsert_entity("Rust", "language").unwrap();
        brain
            .upsert_fact(eid, "paradigm", "systems programming", "url")
            .unwrap();
        let results = brain.search_facts("systems").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_learn_from_extracted() {
        let extracted = Extracted {
            entities: vec![
                ("Rust".to_string(), "phrase".to_string()),
                ("Mozilla".to_string(), "phrase".to_string()),
            ],
            relations: vec![(
                "Rust".to_string(),
                "created".to_string(),
                "Mozilla".to_string(),
            )],
            keywords: vec!["programming".to_string(), "language".to_string()],
            source_url: "https://example.com".to_string(),
            language: crate::nlp::Language::English,
        };
        let brain = test_brain();
        let (e, r, f) = brain.learn(&extracted).unwrap();
        assert_eq!(e, 2);
        assert_eq!(r, 1);
        assert_eq!(f, 2);
    }

    #[test]
    fn test_top_entities() {
        let brain = test_brain();
        brain.upsert_entity("Rust", "language").unwrap();
        brain.upsert_entity("Rust", "language").unwrap(); // bump access
        brain.upsert_entity("Python", "language").unwrap();
        let top = brain.top_entities(10).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "Rust"); // higher access count
    }

    #[test]
    fn test_duplicate_frontier_entry() {
        let brain = test_brain();
        brain.add_to_frontier("https://example.com", 1).unwrap();
        brain.add_to_frontier("https://example.com", 2).unwrap(); // should be ignored
        let frontier = brain.get_frontier(10).unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].priority, 1); // first insert wins
    }

    #[test]
    fn test_confidence_caps_at_one() {
        let brain = test_brain();
        for _ in 0..20 {
            brain.upsert_entity("Test", "test").unwrap();
        }
        let e = brain.get_entity_by_name("Test").unwrap().unwrap();
        assert!(e.confidence <= 1.0);
    }
}
