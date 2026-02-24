/// Sitemap parser — discovers URLs from sitemap.xml and sitemap index files.
use url::Url;

/// A URL entry from a sitemap.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SitemapEntry {
    pub url: String,
    pub lastmod: Option<String>,
    pub priority: Option<f64>,
}

/// Parse a sitemap XML string and extract URLs.
/// Handles both <urlset> sitemaps and <sitemapindex> files.
pub fn parse_sitemap(xml: &str) -> SitemapResult {
    // Check if this is a sitemap index
    if xml.contains("<sitemapindex") {
        let sub_sitemaps = extract_tag_contents(xml, "sitemap")
            .into_iter()
            .filter_map(|block| extract_first_tag_content(&block, "loc"))
            .collect();
        SitemapResult::Index(sub_sitemaps)
    } else {
        let entries = extract_tag_contents(xml, "url")
            .into_iter()
            .filter_map(|block| {
                let url = extract_first_tag_content(&block, "loc")?;
                let lastmod = extract_first_tag_content(&block, "lastmod");
                let priority = extract_first_tag_content(&block, "priority")
                    .and_then(|s| s.parse::<f64>().ok());
                Some(SitemapEntry {
                    url,
                    lastmod,
                    priority,
                })
            })
            .collect();
        SitemapResult::UrlSet(entries)
    }
}

/// Result of parsing a sitemap.
#[derive(Debug)]
pub enum SitemapResult {
    /// Direct URL set
    UrlSet(Vec<SitemapEntry>),
    /// Sitemap index pointing to sub-sitemaps
    Index(Vec<String>),
}

/// Discover sitemap URLs for a domain.
/// Checks /sitemap.xml and /robots.txt for Sitemap: directives.
pub async fn discover_sitemaps(base_url: &str) -> anyhow::Result<Vec<String>> {
    let parsed = Url::parse(base_url)?;
    let origin = format!("{}://{}", parsed.scheme(), parsed.host_str().unwrap_or(""));
    let mut sitemap_urls = Vec::new();

    // Check robots.txt for Sitemap: lines
    let client = reqwest::Client::builder()
        .user_agent("axon/0.2 (knowledge-engine)")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let robots_url = format!("{origin}/robots.txt");
    if let Ok(resp) = client.get(&robots_url).send().await {
        if resp.status().is_success() {
            if let Ok(body) = resp.text().await {
                for line in body.lines() {
                    let trimmed = line.trim();
                    if let Some(url) = trimmed
                        .strip_prefix("Sitemap:")
                        .or_else(|| trimmed.strip_prefix("sitemap:"))
                    {
                        let url = url.trim();
                        if url.starts_with("http") {
                            sitemap_urls.push(url.to_string());
                        }
                    }
                }
            }
        }
    }

    // Always try /sitemap.xml as fallback
    let default = format!("{origin}/sitemap.xml");
    if !sitemap_urls.contains(&default) {
        sitemap_urls.push(default);
    }

    Ok(sitemap_urls)
}

/// Fetch and parse all sitemaps, recursively following sitemap indexes.
/// Returns all discovered URL entries.
pub async fn fetch_all_entries(
    sitemap_urls: &[String],
    max_entries: usize,
) -> anyhow::Result<Vec<SitemapEntry>> {
    let client = reqwest::Client::builder()
        .user_agent("axon/0.2 (knowledge-engine)")
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut all_entries = Vec::new();
    let mut to_fetch: Vec<String> = sitemap_urls.to_vec();
    let mut fetched = std::collections::HashSet::new();

    while let Some(url) = to_fetch.pop() {
        if fetched.contains(&url) || all_entries.len() >= max_entries {
            break;
        }
        fetched.insert(url.clone());

        let resp = match client.get(&url).send().await {
            Ok(r) if r.status().is_success() => r,
            _ => continue,
        };
        let body = match resp.text().await {
            Ok(b) => b,
            Err(_) => continue,
        };

        match parse_sitemap(&body) {
            SitemapResult::UrlSet(entries) => {
                let remaining = max_entries.saturating_sub(all_entries.len());
                all_entries.extend(entries.into_iter().take(remaining));
            }
            SitemapResult::Index(sub_urls) => {
                for sub in sub_urls {
                    if !fetched.contains(&sub) {
                        to_fetch.push(sub);
                    }
                }
            }
        }
    }

    Ok(all_entries)
}

// Simple XML tag extraction (no external XML dependency needed for sitemaps)

fn extract_tag_contents(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut results = Vec::new();
    let mut search_from = 0;

    while let Some(start) = xml[search_from..].find(&open) {
        let abs_start = search_from + start;
        // Find the end of the opening tag
        if let Some(tag_end) = xml[abs_start..].find('>') {
            let content_start = abs_start + tag_end + 1;
            if let Some(end) = xml[content_start..].find(&close) {
                results.push(xml[content_start..content_start + end].to_string());
                search_from = content_start + end + close.len();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    results
}

fn extract_first_tag_content(xml: &str, tag: &str) -> Option<String> {
    let contents = extract_tag_contents(xml, tag);
    contents.into_iter().next().map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_urlset() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
                <lastmod>2024-01-15</lastmod>
                <priority>0.8</priority>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
                <priority>0.5</priority>
            </url>
        </urlset>"#;

        match parse_sitemap(xml) {
            SitemapResult::UrlSet(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].url, "https://example.com/page1");
                assert_eq!(entries[0].lastmod.as_deref(), Some("2024-01-15"));
                assert_eq!(entries[0].priority, Some(0.8));
                assert_eq!(entries[1].url, "https://example.com/page2");
                assert!(entries[1].lastmod.is_none());
            }
            SitemapResult::Index(_) => panic!("Expected UrlSet"),
        }
    }

    #[test]
    fn test_parse_sitemap_index() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap>
                <loc>https://example.com/sitemap1.xml</loc>
            </sitemap>
            <sitemap>
                <loc>https://example.com/sitemap2.xml</loc>
            </sitemap>
        </sitemapindex>"#;

        match parse_sitemap(xml) {
            SitemapResult::Index(urls) => {
                assert_eq!(urls.len(), 2);
                assert_eq!(urls[0], "https://example.com/sitemap1.xml");
                assert_eq!(urls[1], "https://example.com/sitemap2.xml");
            }
            SitemapResult::UrlSet(_) => panic!("Expected Index"),
        }
    }

    #[test]
    fn test_parse_empty() {
        match parse_sitemap("<urlset></urlset>") {
            SitemapResult::UrlSet(entries) => assert!(entries.is_empty()),
            _ => panic!("Expected empty UrlSet"),
        }
    }

    #[test]
    fn test_extract_tag_contents() {
        let xml = "<a>hello</a><a>world</a>";
        let results = extract_tag_contents(xml, "a");
        assert_eq!(results, vec!["hello", "world"]);
    }

    #[test]
    fn test_extract_first_tag_content() {
        let xml = "<loc>  https://example.com  </loc>";
        assert_eq!(
            extract_first_tag_content(xml, "loc"),
            Some("https://example.com".to_string())
        );
    }

    #[test]
    fn test_parse_sitemap_with_attributes() {
        let xml = r#"<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/</loc>
            </url>
        </urlset>"#;
        match parse_sitemap(xml) {
            SitemapResult::UrlSet(entries) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].url, "https://example.com/");
            }
            _ => panic!("Expected UrlSet"),
        }
    }
}
