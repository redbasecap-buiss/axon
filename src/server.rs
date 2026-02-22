use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::crawler;
use crate::db::Brain;
use crate::nlp;
use crate::query;

/// Shared application state.
struct AppState {
    db_path: PathBuf,
}

impl AppState {
    fn brain(&self) -> Result<Brain, StatusCode> {
        Brain::open(&self.db_path).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }
}

// ---------- request / response types ----------

#[derive(Deserialize)]
pub struct AskParams {
    q: Option<String>,
}

#[derive(Serialize)]
pub struct AskResponse {
    query: String,
    results: Vec<String>,
}

#[derive(Serialize)]
pub struct StatsResponse {
    entities: usize,
    relations: usize,
    facts: usize,
    sources: usize,
    db_size: String,
}

#[derive(Deserialize)]
pub struct FeedRequest {
    url: String,
}

#[derive(Serialize)]
pub struct FeedResponse {
    url: String,
    entities: usize,
    relations: usize,
    facts: usize,
}

#[derive(Serialize)]
pub struct TopicEntry {
    name: String,
    entity_type: String,
    confidence: f64,
    access_count: i64,
}

#[derive(Serialize)]
pub struct TopicsResponse {
    topics: Vec<TopicEntry>,
}

// ---------- handlers ----------

async fn handle_ask(
    State(state): State<Arc<AppState>>,
    Query(params): Query<AskParams>,
) -> Result<Json<AskResponse>, StatusCode> {
    let q = params.q.unwrap_or_default();
    let brain = state.brain()?;
    let results = query::ask(&brain, &q).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(AskResponse { query: q, results }))
}

async fn handle_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<StatsResponse>, StatusCode> {
    let brain = state.brain()?;
    let s = brain
        .stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(StatsResponse {
        entities: s.entity_count,
        relations: s.relation_count,
        facts: s.fact_count,
        sources: s.source_count,
        db_size: s.db_size,
    }))
}

async fn handle_feed(
    State(state): State<Arc<AppState>>,
    Json(body): Json<FeedRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let text = crawler::fetch_and_extract(&body.url)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let extracted = nlp::process_text(&text, &body.url);
    let brain = state.brain()?;
    let (entities, relations, facts) = brain
        .learn(&extracted)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok((
        StatusCode::OK,
        Json(FeedResponse {
            url: body.url,
            entities,
            relations,
            facts,
        }),
    ))
}

async fn handle_topics(
    State(state): State<Arc<AppState>>,
) -> Result<Json<TopicsResponse>, StatusCode> {
    let brain = state.brain()?;
    let top = brain
        .top_entities(20)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let topics = top
        .into_iter()
        .map(|(name, entity_type, confidence, access_count)| TopicEntry {
            name,
            entity_type,
            confidence,
            access_count,
        })
        .collect();
    Ok(Json(TopicsResponse { topics }))
}

async fn handle_dashboard(State(state): State<Arc<AppState>>) -> Result<Html<String>, StatusCode> {
    let brain = state.brain()?;
    let s = brain
        .stats()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>axon — knowledge engine</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; padding: 0 1rem; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; text-align: center; }}
  .stat .num {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
  .stat .label {{ font-size: 0.85rem; color: #8b949e; }}
  input[type=text] {{ width: 100%; padding: 0.75rem; font-size: 1rem; background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; box-sizing: border-box; }}
  #results {{ margin-top: 1rem; white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>🧠 axon</h1>
<div class="stats">
  <div class="stat"><div class="num">{entities}</div><div class="label">Entities</div></div>
  <div class="stat"><div class="num">{relations}</div><div class="label">Relations</div></div>
  <div class="stat"><div class="num">{facts}</div><div class="label">Facts</div></div>
  <div class="stat"><div class="num">{sources}</div><div class="label">Sources</div></div>
</div>
<input type="text" id="q" placeholder="Ask axon something…" autofocus>
<div id="results"></div>
<script>
const q = document.getElementById('q');
const r = document.getElementById('results');
let t;
q.addEventListener('input', () => {{
  clearTimeout(t);
  t = setTimeout(async () => {{
    if (!q.value.trim()) {{ r.textContent = ''; return; }}
    try {{
      const resp = await fetch('/api/ask?q=' + encodeURIComponent(q.value));
      const data = await resp.json();
      r.textContent = data.results.length ? data.results.join('\n') : 'No results found.';
    }} catch(e) {{ r.textContent = 'Error: ' + e; }}
  }}, 300);
}});
</script>
</body>
</html>"#,
        entities = s.entity_count,
        relations = s.relation_count,
        facts = s.fact_count,
        sources = s.source_count,
    );
    Ok(Html(html))
}

// ---------- router ----------

pub fn build_router(db_path: &Path) -> Router {
    let state = Arc::new(AppState {
        db_path: db_path.to_path_buf(),
    });

    Router::new()
        .route("/", get(handle_dashboard))
        .route("/api/ask", get(handle_ask))
        .route("/api/stats", get(handle_stats))
        .route("/api/feed", post(handle_feed))
        .route("/api/topics", get(handle_topics))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Run the HTTP API server on the given port.
pub async fn run_server(db_path: PathBuf, port: u16) -> anyhow::Result<()> {
    let app = build_router(&db_path);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    println!("🌐 Serving on http://0.0.0.0:{port}");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::util::ServiceExt;

    fn test_router() -> Router {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        // Create and populate brain
        let brain = Brain::open(&db_path).unwrap();
        brain.upsert_entity("Rust", "language").unwrap();
        brain.upsert_entity("Mozilla", "organization").unwrap();
        let s = brain.upsert_entity("Rust", "language").unwrap();
        let o = brain.upsert_entity("Mozilla", "organization").unwrap();
        brain
            .upsert_relation(s, "created_by", o, "https://example.com")
            .unwrap();
        brain
            .upsert_fact(s, "paradigm", "systems", "https://example.com")
            .unwrap();
        drop(brain);
        // Leak tempdir so it lives for the duration of tests
        let db_path_owned = db_path.to_path_buf();
        std::mem::forget(dir);
        build_router(&db_path_owned)
    }

    #[tokio::test]
    async fn test_dashboard() {
        let app = test_router();
        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("axon"));
        assert!(text.contains("Entities"));
    }

    #[tokio::test]
    async fn test_api_stats() {
        let app = test_router();
        let resp = app
            .oneshot(Request::get("/api/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["entities"].as_u64().unwrap() >= 2);
    }

    #[tokio::test]
    async fn test_api_ask() {
        let app = test_router();
        let resp = app
            .oneshot(Request::get("/api/ask?q=Rust").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["query"], "Rust");
        assert!(json["results"].as_array().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn test_api_topics() {
        let app = test_router();
        let resp = app
            .oneshot(Request::get("/api/topics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["topics"].as_array().is_some());
    }

    #[tokio::test]
    async fn test_api_ask_empty() {
        let app = test_router();
        let resp = app
            .oneshot(
                Request::get("/api/ask?q=nonexistent_xyz_123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["results"].as_array().unwrap().is_empty());
    }
}
