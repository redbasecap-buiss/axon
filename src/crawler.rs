#![allow(dead_code)]
use crate::db::Brain;
use crate::nlp;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use url::Url;

/// Fetch a URL and extract readable text.
pub async fn fetch_and_extract(url: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::builder()
        .user_agent("axon/0.1 (knowledge-engine; +https://github.com/redbasecap-buiss/axon)")
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // Check robots.txt
    let parsed = Url::parse(url)?;
    let robots_url = format!(
        "{}://{}/robots.txt",
        parsed.scheme(),
        parsed.host_str().unwrap_or("")
    );
    if let Ok(resp) = client.get(&robots_url).send().await {
        if resp.status().is_success() {
            if let Ok(body) = resp.text().await {
                let robot = texting_robots::Robot::new("axon", body.as_bytes())
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                if !robot.allowed(url) {
                    return Err(anyhow::anyhow!("Blocked by robots.txt: {url}"));
                }
            }
        }
    }

    let resp = client.get(url).send().await?;
    let html = resp.text().await?;

    Ok(extract_text_from_html(&html))
}

/// Fetch a URL and return both the raw HTML and extracted text.
pub async fn fetch_html_and_text(url: &str) -> anyhow::Result<(String, String)> {
    let client = reqwest::Client::builder()
        .user_agent("axon/0.1 (knowledge-engine; +https://github.com/redbasecap-buiss/axon)")
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let parsed = Url::parse(url)?;
    let robots_url = format!(
        "{}://{}/robots.txt",
        parsed.scheme(),
        parsed.host_str().unwrap_or("")
    );
    if let Ok(resp) = client.get(&robots_url).send().await {
        if resp.status().is_success() {
            if let Ok(body) = resp.text().await {
                let robot = texting_robots::Robot::new("axon", body.as_bytes())
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                if !robot.allowed(url) {
                    return Err(anyhow::anyhow!("Blocked by robots.txt: {url}"));
                }
            }
        }
    }

    let resp = client.get(url).send().await?;
    let html = resp.text().await?;
    let text = extract_text_from_html(&html);
    Ok((html, text))
}

/// Classes and IDs that indicate non-content (navigation, chrome, etc.)
const NOISE_CLASSES: &[&str] = &[
    "nav",
    "navbar",
    "sidebar",
    "footer",
    "header",
    "menu",
    "toc",
    "catlinks",
    "mw-navigation",
    "mw-panel",
    "mw-footer",
    "siteSub",
    "contentSub",
    "mw-jump-link",
    "mw-editsection",
    "navbox",
    "noprint",
    "printfooter",
    "mw-indicators",
    "mw-head",
    // "mw-page-container-inner",  // removed: this is a layout wrapper, not noise
    "vector-header",
    "vector-column-start",
    "vector-sticky-header",
    "mw-footer-container",
    "portlet",
    "portal",
];

/// Tags to always skip regardless of class.
const SKIP_TAGS: &[&str] = &["script", "style", "noscript", "nav", "footer", "header"];

/// Content tags we want to keep text from.
const CONTENT_TAGS: &[&str] = &[
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "td",
    "th",
    "blockquote",
    "caption",
    "figcaption",
    "dt",
    "dd",
    "span",
    "a",
    "em",
    "strong",
    "b",
    "i",
    "sup",
    "sub",
    "code",
];

/// Check if an element matches noise patterns.
fn is_noise_element(el: &scraper::node::Element) -> bool {
    let tag = el.name();
    if SKIP_TAGS.contains(&tag) {
        return true;
    }
    for class in el.attr("class").unwrap_or("").split_whitespace() {
        if NOISE_CLASSES.contains(&class) {
            return true;
        }
    }
    if let Some(id) = el.attr("id") {
        if NOISE_CLASSES.contains(&id) {
            return true;
        }
    }
    if el.attr("role").is_some_and(|r| r == "navigation") {
        return true;
    }
    false
}

/// Extract readable text from HTML, filtering out navigation/chrome noise.
/// Uses CSS selectors to first remove noise, then extract from content tags.
pub fn extract_text_from_html(html: &str) -> String {
    let document = scraper::Html::parse_document(html);

    let content_tags: std::collections::HashSet<&str> = CONTENT_TAGS.iter().copied().collect();

    let mut text = String::new();
    for node in document.tree.nodes() {
        if let scraper::node::Node::Text(t) = node.value() {
            // Check parent is a content tag
            let in_content = node.parent().is_some_and(|p| {
                p.value()
                    .as_element()
                    .is_some_and(|el| content_tags.contains(el.name()))
            });
            if !in_content {
                continue;
            }

            // Walk ancestors to check for noise
            let mut in_noise = false;
            let mut ancestor = node.parent();
            while let Some(anc) = ancestor {
                if let Some(el) = anc.value().as_element() {
                    if is_noise_element(el) {
                        in_noise = true;
                        break;
                    }
                }
                ancestor = anc.parent();
            }
            if in_noise {
                continue;
            }

            let s = t.text.trim();
            if !s.is_empty() {
                text.push_str(s);
                text.push(' ');
            }
        }
    }
    text
}

/// Extract links from HTML.
pub fn extract_links(html: &str, base_url: &str) -> Vec<String> {
    let document = scraper::Html::parse_document(html);
    let selector = scraper::Selector::parse("a[href]").unwrap();
    let base = Url::parse(base_url).ok();

    let mut links = Vec::new();
    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            let resolved = if let Some(base) = &base {
                base.join(href).map(|u| u.to_string()).ok()
            } else {
                None
            };
            if let Some(link) = resolved {
                if link.starts_with("http") {
                    links.push(link);
                }
            }
        }
    }
    links
}

fn hash_content(content: &str) -> String {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Crawl pages from the frontier.
pub async fn crawl(brain: &Brain, max_pages: usize) -> anyhow::Result<usize> {
    let frontier = brain.get_frontier(max_pages)?;
    let mut crawled = 0;

    for entry in frontier {
        // Politeness delay
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        match fetch_and_extract(&entry.url).await {
            Ok(text) => {
                let content_hash = hash_content(&text);

                // Skip if content hasn't changed
                if let Ok(Some(existing_hash)) = brain.get_content_hash(&entry.url) {
                    if existing_hash == content_hash {
                        brain.mark_crawled(&entry.url, &content_hash)?;
                        continue;
                    }
                }

                let extracted = nlp::process_text(&text, &entry.url);
                brain.learn(&extracted)?;
                brain.mark_crawled(&entry.url, &content_hash)?;

                // Add discovered links to frontier
                let client = reqwest::Client::builder()
                    .user_agent("axon/0.1")
                    .timeout(std::time::Duration::from_secs(30))
                    .build()?;
                if let Ok(resp) = client.get(&entry.url).send().await {
                    if let Ok(html) = resp.text().await {
                        let links = extract_links(&html, &entry.url);
                        for link in links.into_iter().take(5) {
                            let _ = brain.add_to_frontier(&link, 0);
                        }
                    }
                }

                crawled += 1;
            }
            Err(e) => {
                eprintln!("   Failed to crawl {}: {e}", entry.url);
            }
        }
    }

    Ok(crawled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_html() {
        let html = r#"
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is a test paragraph.</p>
            <script>var x = 1;</script>
            <style>.foo { color: red; }</style>
        </body>
        </html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Hello World"));
        assert!(text.contains("test paragraph"));
        assert!(!text.contains("var x"));
        assert!(!text.contains("color: red"));
    }

    #[test]
    fn test_extract_text_strips_nav_by_class() {
        let html = r#"
        <html><body>
            <div class="mw-navigation"><a>Jump to navigation</a></div>
            <div class="sidebar"><span>Menu items</span></div>
            <p>Real content here.</p>
        </body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Real content"), "text: {text}");
        assert!(!text.contains("Jump to navigation"), "text: {text}");
        assert!(!text.contains("Menu items"), "text: {text}");
    }

    #[test]
    fn test_extract_text_strips_nav_by_id() {
        let html = r#"
        <html><body>
            <div id="mw-panel"><span>Sidebar stuff</span></div>
            <div id="catlinks"><span>Categories</span></div>
            <div id="mw-footer"><span>Footer stuff</span></div>
            <p>Article content is here.</p>
        </body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Article content"), "text: {text}");
        assert!(!text.contains("Sidebar stuff"), "text: {text}");
        assert!(!text.contains("Categories"), "text: {text}");
        assert!(!text.contains("Footer stuff"), "text: {text}");
    }

    #[test]
    fn test_extract_text_strips_footer_tag() {
        let html = r#"
        <html><body>
            <h2>Section Title</h2>
            <p>Good content.</p>
            <footer><p>Copyright Wikimedia Foundation</p></footer>
        </body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Good content"), "text: {text}");
        assert!(!text.contains("Wikimedia"), "text: {text}");
    }

    #[test]
    fn test_extract_text_keeps_content_tags() {
        let html = r#"
        <html><body>
            <h1>Title</h1>
            <h2>Subtitle</h2>
            <p>Paragraph text.</p>
            <ul><li>List item</li></ul>
            <blockquote>A famous quote.</blockquote>
            <table><tr><td>Cell data</td></tr></table>
        </body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Title"), "text: {text}");
        assert!(text.contains("Paragraph text"), "text: {text}");
        assert!(text.contains("List item"), "text: {text}");
        assert!(text.contains("famous quote"), "text: {text}");
        assert!(text.contains("Cell data"), "text: {text}");
    }

    #[test]
    fn test_extract_text_strips_role_navigation() {
        let html = r#"
        <html><body>
            <div role="navigation"><a>Upload file</a></div>
            <p>Main article text.</p>
        </body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Main article"), "text: {text}");
        assert!(!text.contains("Upload file"), "text: {text}");
    }

    #[test]
    fn test_extract_text_wikipedia_realistic() {
        let html = r#"
        <html><body>
            <div id="mw-navigation">
                <div class="portlet"><span>Main page</span></div>
                <div class="portlet"><span>Random article</span></div>
                <div class="portlet"><span>Upload</span></div>
                <div class="portlet"><a>Community portal</a></div>
            </div>
            <div id="content">
                <h1>Albert Einstein</h1>
                <p>Albert Einstein was a German-born theoretical physicist.</p>
                <p>He developed the theory of relativity.</p>
            </div>
            <div id="catlinks"><span>Categories: Physicists</span></div>
            <div id="mw-footer"><span>Wikimedia Foundation</span></div>
        </body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Albert Einstein"), "text: {text}");
        assert!(text.contains("theoretical physicist"), "text: {text}");
        assert!(!text.contains("Main page"), "text: {text}");
        assert!(!text.contains("Random article"), "text: {text}");
        assert!(!text.contains("Upload"), "text: {text}");
        assert!(!text.contains("Community portal"), "text: {text}");
        assert!(!text.contains("Wikimedia Foundation"), "text: {text}");
    }

    #[test]
    fn test_extract_links() {
        let html = r#"
        <html><body>
            <a href="/about">About</a>
            <a href="https://example.com/page">Page</a>
            <a href="mailto:test@test.com">Email</a>
        </body></html>"#;
        let links = extract_links(html, "https://example.com");
        assert!(links.contains(&"https://example.com/about".to_string()));
        assert!(links.contains(&"https://example.com/page".to_string()));
        assert!(!links.iter().any(|l| l.contains("mailto")));
    }

    #[test]
    fn test_extract_text_empty_html() {
        let text = extract_text_from_html("");
        assert!(text.trim().is_empty());
    }

    #[test]
    fn test_hash_content() {
        let h1 = hash_content("hello");
        let h2 = hash_content("hello");
        let h3 = hash_content("world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
