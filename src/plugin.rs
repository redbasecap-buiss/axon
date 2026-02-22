//! Plugin system for domain-specific entity/relation/fact extraction.

use scraper::{Html, Selector};
use std::collections::HashMap;

/// Extracted output from a plugin.
#[derive(Debug, Clone, Default)]
pub struct PluginOutput {
    pub entities: Vec<(String, String)>,
    pub relations: Vec<(String, String, String)>,
    pub facts: Vec<(String, String, String)>,
}

impl PluginOutput {
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty() && self.relations.is_empty() && self.facts.is_empty()
    }
}

/// Trait that all domain-specific extractors implement.
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn matches(&self, url: &str) -> bool;
    fn extract(&self, url: &str, html: &str, text: &str) -> PluginOutput;
}

// ─── Built-in Plugins ────────────────────────────────────────────────────────

pub struct WikipediaPlugin;

impl Plugin for WikipediaPlugin {
    fn name(&self) -> &str {
        "wikipedia"
    }

    fn matches(&self, url: &str) -> bool {
        url.contains("wikipedia.org/wiki/")
    }

    fn extract(&self, url: &str, html: &str, _text: &str) -> PluginOutput {
        let mut out = PluginOutput::default();
        let doc = Html::parse_document(html);

        let page_title = url
            .rsplit("/wiki/")
            .next()
            .unwrap_or("")
            .replace('_', " ")
            .split('#')
            .next()
            .unwrap_or("")
            .to_string();
        let page_title = percent_decode(&page_title);
        if page_title.is_empty() {
            return out;
        }

        out.entities
            .push((page_title.clone(), "wikipedia_article".to_string()));

        // Infobox key-value pairs
        if let Ok(sel) = Selector::parse(".infobox th, .infobox td") {
            let elements: Vec<_> = doc.select(&sel).collect();
            let mut i = 0;
            while i + 1 < elements.len() {
                let key = element_text(&elements[i]).trim().to_string();
                let value = element_text(&elements[i + 1]).trim().to_string();
                if !key.is_empty() && !value.is_empty() && key.len() < 100 && value.len() < 500 {
                    out.facts
                        .push((page_title.clone(), key.clone(), value.clone()));
                    if value.chars().next().is_some_and(|c| c.is_uppercase())
                        && !value
                            .chars()
                            .all(|c| c.is_numeric() || c == ',' || c == '.')
                    {
                        out.entities
                            .push((value.clone(), "infobox_value".to_string()));
                        out.relations.push((page_title.clone(), key, value));
                    }
                }
                i += 2;
            }
        }

        // Categories
        if let Ok(sel) = Selector::parse("#mw-normal-catlinks ul li a") {
            for el in doc.select(&sel) {
                let cat = element_text(&el).trim().to_string();
                if !cat.is_empty() {
                    out.entities.push((cat.clone(), "category".to_string()));
                    out.relations
                        .push((page_title.clone(), "category".to_string(), cat));
                }
            }
        }

        // "See also" links - parse from the full HTML text
        if let Some(see_also_pos) = html.find("id=\"See_also\"") {
            let after = &html[see_also_pos..];
            // Find the next <ul> ... </ul> block
            if let Some(ul_start) = after.find("<ul>") {
                if let Some(ul_end) = after[ul_start..].find("</ul>") {
                    let ul_html = &after[ul_start..ul_start + ul_end + 5];
                    let frag = Html::parse_fragment(ul_html);
                    if let Ok(a_sel) = Selector::parse("li a") {
                        for a in frag.select(&a_sel) {
                            let link_text = element_text(&a).trim().to_string();
                            if !link_text.is_empty() && link_text.len() < 200 {
                                out.entities
                                    .push((link_text.clone(), "related_topic".to_string()));
                                out.relations.push((
                                    page_title.clone(),
                                    "see_also".to_string(),
                                    link_text,
                                ));
                            }
                        }
                    }
                }
            }
        }

        out
    }
}

pub struct GitHubPlugin;

impl Plugin for GitHubPlugin {
    fn name(&self) -> &str {
        "github"
    }

    fn matches(&self, url: &str) -> bool {
        let u = url.trim_end_matches('/');
        if !u.contains("github.com/") {
            return false;
        }
        let after = u.split("github.com/").nth(1).unwrap_or("");
        let parts: Vec<&str> = after.split('/').filter(|s| !s.is_empty()).collect();
        parts.len() == 2
    }

    fn extract(&self, url: &str, html: &str, _text: &str) -> PluginOutput {
        let mut out = PluginOutput::default();
        let doc = Html::parse_document(html);

        let repo_name = url
            .split("github.com/")
            .nth(1)
            .unwrap_or("")
            .trim_end_matches('/')
            .to_string();
        if repo_name.is_empty() {
            return out;
        }

        out.entities
            .push((repo_name.clone(), "github_repo".to_string()));

        if let Some(owner) = repo_name.split('/').next() {
            out.entities
                .push((owner.to_string(), "github_user".to_string()));
            out.relations
                .push((repo_name.clone(), "owned_by".to_string(), owner.to_string()));
        }

        if let Some(desc) = extract_og_or_meta(&doc, "og:description") {
            out.facts
                .push((repo_name.clone(), "description".to_string(), desc));
        }

        if let Ok(sel) = Selector::parse("#repo-stars-counter-star, .social-count") {
            for el in doc.select(&sel) {
                let text = element_text(&el).trim().to_string();
                if !text.is_empty() {
                    out.facts
                        .push((repo_name.clone(), "stars".to_string(), text));
                    break;
                }
            }
        }

        if let Ok(sel) =
            Selector::parse("[data-ga-click*='language'], .d-inline .color-fg-default.text-bold")
        {
            for el in doc.select(&sel) {
                let lang = element_text(&el).trim().to_string();
                if !lang.is_empty() && lang.len() < 50 {
                    out.facts
                        .push((repo_name.clone(), "language".to_string(), lang.clone()));
                    out.entities
                        .push((lang, "programming_language".to_string()));
                    break;
                }
            }
        }

        if let Ok(sel) = Selector::parse(".topic-tag") {
            for el in doc.select(&sel) {
                let topic = element_text(&el).trim().to_string();
                if !topic.is_empty() {
                    out.entities.push((topic.clone(), "topic".to_string()));
                    out.relations
                        .push((repo_name.clone(), "topic".to_string(), topic));
                }
            }
        }

        out
    }
}

pub struct HackerNewsPlugin;

impl Plugin for HackerNewsPlugin {
    fn name(&self) -> &str {
        "hackernews"
    }

    fn matches(&self, url: &str) -> bool {
        url.contains("news.ycombinator.com/item")
    }

    fn extract(&self, url: &str, html: &str, _text: &str) -> PluginOutput {
        let mut out = PluginOutput::default();
        let doc = Html::parse_document(html);

        let item_id = url
            .split("id=")
            .nth(1)
            .unwrap_or("unknown")
            .split('&')
            .next()
            .unwrap_or("unknown")
            .to_string();
        let entity_name = format!("hn:{}", item_id);
        out.entities
            .push((entity_name.clone(), "hn_item".to_string()));

        if let Ok(sel) = Selector::parse(".titleline > a") {
            if let Some(el) = doc.select(&sel).next() {
                let title = element_text(&el).trim().to_string();
                if !title.is_empty() {
                    out.facts
                        .push((entity_name.clone(), "title".to_string(), title));
                }
            }
        }

        if let Ok(sel) = Selector::parse(".score") {
            if let Some(el) = doc.select(&sel).next() {
                let score = element_text(&el).trim().to_string();
                if !score.is_empty() {
                    out.facts
                        .push((entity_name.clone(), "score".to_string(), score));
                }
            }
        }

        if let Ok(sel) = Selector::parse(".hnuser") {
            if let Some(el) = doc.select(&sel).next() {
                let author = element_text(&el).trim().to_string();
                if !author.is_empty() {
                    out.entities.push((author.clone(), "hn_user".to_string()));
                    out.relations
                        .push((entity_name.clone(), "submitted_by".to_string(), author));
                }
            }
        }

        if let Ok(sel) = Selector::parse(".subtext a") {
            for el in doc.select(&sel) {
                let text = element_text(&el);
                if text.contains("comment") {
                    out.facts.push((
                        entity_name.clone(),
                        "comments".to_string(),
                        text.trim().to_string(),
                    ));
                    break;
                }
            }
        }

        out
    }
}

pub struct NewsArticlePlugin;

impl Plugin for NewsArticlePlugin {
    fn name(&self) -> &str {
        "news"
    }

    fn matches(&self, url: &str) -> bool {
        let patterns = [
            "reuters.com",
            "bbc.com",
            "bbc.co.uk",
            "nytimes.com",
            "theguardian.com",
            "washingtonpost.com",
            "cnn.com",
            "apnews.com",
            "bloomberg.com",
            "techcrunch.com",
            "arstechnica.com",
            "theverge.com",
        ];
        patterns.iter().any(|p| url.contains(p))
    }

    fn extract(&self, url: &str, html: &str, _text: &str) -> PluginOutput {
        let mut out = PluginOutput::default();
        let doc = Html::parse_document(html);

        let entity_name = extract_og_or_meta(&doc, "og:title")
            .or_else(|| extract_meta_name(&doc, "title"))
            .or_else(|| extract_tag_text(&doc, "title"))
            .unwrap_or_else(|| url.to_string());

        out.entities
            .push((entity_name.clone(), "news_article".to_string()));
        out.facts
            .push((entity_name.clone(), "url".to_string(), url.to_string()));

        if let Some(headline) = extract_og_or_meta(&doc, "og:title") {
            out.facts
                .push((entity_name.clone(), "headline".to_string(), headline));
        }

        if let Some(author) =
            extract_meta_name(&doc, "author").or_else(|| extract_og_or_meta(&doc, "article:author"))
        {
            out.entities.push((author.clone(), "person".to_string()));
            out.relations.push((
                entity_name.clone(),
                "authored_by".to_string(),
                author.clone(),
            ));
            out.facts
                .push((entity_name.clone(), "author".to_string(), author));
        }

        if let Some(date) = extract_meta_name(&doc, "date")
            .or_else(|| extract_og_or_meta(&doc, "article:published_time"))
            .or_else(|| extract_meta_name(&doc, "publish_date"))
        {
            out.facts
                .push((entity_name.clone(), "publish_date".to_string(), date));
        }

        if let Some(desc) = extract_og_or_meta(&doc, "og:description")
            .or_else(|| extract_meta_name(&doc, "description"))
        {
            out.facts
                .push((entity_name.clone(), "summary".to_string(), desc));
        }

        if let Some(site) = extract_og_or_meta(&doc, "og:site_name") {
            out.entities.push((site.clone(), "news_source".to_string()));
            out.relations
                .push((entity_name.clone(), "published_by".to_string(), site));
        }

        out
    }
}

pub struct RSSPlugin;

impl Plugin for RSSPlugin {
    fn name(&self) -> &str {
        "rss"
    }

    fn matches(&self, url: &str) -> bool {
        let lower = url.to_lowercase();
        lower.contains("/rss")
            || lower.contains("/feed")
            || lower.contains("/atom")
            || lower.ends_with(".rss")
            || lower.ends_with(".xml")
            || lower.ends_with("/atom.xml")
    }

    fn extract(&self, url: &str, html: &str, _text: &str) -> PluginOutput {
        let mut out = PluginOutput::default();
        let feed_name = format!("feed:{}", url);
        out.entities
            .push((feed_name.clone(), "rss_feed".to_string()));

        for item in parse_feed_items(html) {
            if let Some(ref title) = item.title {
                let entry_name = title.clone();
                out.entities
                    .push((entry_name.clone(), "feed_entry".to_string()));
                out.relations.push((
                    entry_name.clone(),
                    "from_feed".to_string(),
                    feed_name.clone(),
                ));
                if let Some(ref link) = item.link {
                    out.facts
                        .push((entry_name.clone(), "url".to_string(), link.clone()));
                }
                if let Some(ref date) = item.pub_date {
                    out.facts
                        .push((entry_name.clone(), "pub_date".to_string(), date.clone()));
                }
                if let Some(ref desc) = item.description {
                    let clean = strip_html_tags(desc);
                    if !clean.is_empty() && clean.len() < 1000 {
                        out.facts
                            .push((entry_name.clone(), "description".to_string(), clean));
                    }
                }
                if let Some(ref author) = item.author {
                    out.facts
                        .push((entry_name.clone(), "author".to_string(), author.clone()));
                }
            }
        }

        out
    }
}

// ─── Feed parsing helpers ────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct FeedItem {
    title: Option<String>,
    link: Option<String>,
    description: Option<String>,
    pub_date: Option<String>,
    author: Option<String>,
}

fn parse_feed_items(xml: &str) -> Vec<FeedItem> {
    let mut items = Vec::new();

    for item_str in split_xml_tags(xml, "item") {
        items.push(FeedItem {
            title: extract_xml_tag(&item_str, "title"),
            link: extract_xml_tag(&item_str, "link"),
            description: extract_xml_tag(&item_str, "description"),
            pub_date: extract_xml_tag(&item_str, "pubDate")
                .or_else(|| extract_xml_tag(&item_str, "dc:date")),
            author: extract_xml_tag(&item_str, "author")
                .or_else(|| extract_xml_tag(&item_str, "dc:creator")),
        });
    }

    if items.is_empty() {
        for entry_str in split_xml_tags(xml, "entry") {
            items.push(FeedItem {
                title: extract_xml_tag(&entry_str, "title"),
                link: extract_atom_link(&entry_str),
                description: extract_xml_tag(&entry_str, "summary")
                    .or_else(|| extract_xml_tag(&entry_str, "content")),
                pub_date: extract_xml_tag(&entry_str, "published")
                    .or_else(|| extract_xml_tag(&entry_str, "updated")),
                author: extract_xml_tag(&entry_str, "name"),
            });
        }
    }

    items
}

fn split_xml_tags(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let mut results = Vec::new();
    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find(&open) {
        let abs_start = search_from + start;
        if let Some(end) = xml[abs_start..].find(&close) {
            results.push(xml[abs_start..abs_start + end + close.len()].to_string());
            search_from = abs_start + end + close.len();
        } else {
            break;
        }
    }
    results
}

fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)?;
    let content_start = xml[start..].find('>')? + start + 1;
    let end = xml[content_start..].find(&close)?;
    let content = xml[content_start..content_start + end].trim();
    let content = if content.starts_with("<![CDATA[") && content.ends_with("]]>") {
        &content[9..content.len() - 3]
    } else {
        content
    };
    let result = content.trim().to_string();
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

fn extract_atom_link(xml: &str) -> Option<String> {
    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find("<link") {
        let abs_start = search_from + start;
        let tag_end = xml[abs_start..].find('>')? + abs_start;
        let tag = &xml[abs_start..=tag_end];
        if tag.contains("rel=") && !tag.contains("rel=\"alternate\"") {
            search_from = tag_end + 1;
            continue;
        }
        if let Some(href_start) = tag.find("href=\"") {
            let val_start = href_start + 6;
            if let Some(val_end) = tag[val_start..].find('"') {
                return Some(tag[val_start..val_start + val_end].to_string());
            }
        }
        search_from = tag_end + 1;
    }
    None
}

fn strip_html_tags(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    result.trim().to_string()
}

fn element_text(el: &scraper::ElementRef) -> String {
    el.text().collect::<Vec<_>>().join(" ")
}

fn extract_og_or_meta(doc: &Html, property: &str) -> Option<String> {
    let sel = Selector::parse(&format!("meta[property='{}']", property)).ok()?;
    let el = doc.select(&sel).next()?;
    let v = el.value().attr("content")?.trim().to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

fn extract_meta_name(doc: &Html, name: &str) -> Option<String> {
    let sel = Selector::parse(&format!("meta[name='{}']", name)).ok()?;
    let el = doc.select(&sel).next()?;
    let v = el.value().attr("content")?.trim().to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

fn extract_tag_text(doc: &Html, tag: &str) -> Option<String> {
    let sel = Selector::parse(tag).ok()?;
    let el = doc.select(&sel).next()?;
    let text = element_text(&el).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn percent_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
            {
                result.push(byte as char);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

// ─── Plugin Registry ─────────────────────────────────────────────────────────

pub struct PluginRegistry {
    plugins: Vec<Box<dyn Plugin>>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: vec![
                Box::new(WikipediaPlugin),
                Box::new(GitHubPlugin),
                Box::new(HackerNewsPlugin),
                Box::new(NewsArticlePlugin),
                Box::new(RSSPlugin),
            ],
        }
    }

    pub fn detect(&self, url: &str) -> Option<&dyn Plugin> {
        self.plugins
            .iter()
            .find(|p| p.matches(url))
            .map(|p| p.as_ref())
    }

    pub fn get_by_name(&self, name: &str) -> Option<&dyn Plugin> {
        self.plugins
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    pub fn list_names(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    pub fn extract(
        &self,
        url: &str,
        html: &str,
        text: &str,
        force_plugin: Option<&str>,
    ) -> Option<PluginOutput> {
        let plugin = if let Some(name) = force_plugin {
            self.get_by_name(name)?
        } else {
            self.detect(url)?
        };
        let output = plugin.extract(url, html, text);
        if output.is_empty() {
            None
        } else {
            Some(output)
        }
    }
}

/// Plugin configuration loaded from ~/.axon/config.toml
#[derive(Debug, Clone, Default)]
pub struct PluginConfig {
    pub enabled_plugins: Vec<String>,
    pub custom_patterns: HashMap<String, Vec<String>>,
}

impl PluginConfig {
    pub fn from_toml(value: &toml::Value) -> Self {
        let mut config = Self::default();
        if let Some(plugins) = value.get("plugins") {
            if let Some(enabled) = plugins.get("enabled") {
                if let Some(arr) = enabled.as_array() {
                    config.enabled_plugins = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                }
            }
            if let Some(patterns) = plugins.get("patterns") {
                if let Some(table) = patterns.as_table() {
                    for (name, val) in table {
                        if let Some(arr) = val.as_array() {
                            config.custom_patterns.insert(
                                name.clone(),
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect(),
                            );
                        }
                    }
                }
            }
        }
        config
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_detect_wikipedia() {
        let reg = PluginRegistry::new();
        let p = reg.detect("https://en.wikipedia.org/wiki/Rust_(programming_language)");
        assert_eq!(p.unwrap().name(), "wikipedia");
    }

    #[test]
    fn test_registry_detect_github() {
        let reg = PluginRegistry::new();
        assert_eq!(
            reg.detect("https://github.com/rust-lang/rust")
                .unwrap()
                .name(),
            "github"
        );
    }

    #[test]
    fn test_registry_detect_hackernews() {
        let reg = PluginRegistry::new();
        assert_eq!(
            reg.detect("https://news.ycombinator.com/item?id=12345")
                .unwrap()
                .name(),
            "hackernews"
        );
    }

    #[test]
    fn test_registry_detect_news() {
        let reg = PluginRegistry::new();
        assert!(reg.detect("https://www.bbc.com/news/article-123").is_some());
        assert!(reg.detect("https://techcrunch.com/2024/01/story").is_some());
    }

    #[test]
    fn test_registry_detect_rss() {
        let reg = PluginRegistry::new();
        assert!(reg.detect("https://example.com/rss").is_some());
        assert!(reg.detect("https://example.com/feed").is_some());
        assert!(reg.detect("https://example.com/blog.xml").is_some());
    }

    #[test]
    fn test_registry_detect_none() {
        let reg = PluginRegistry::new();
        assert!(reg.detect("https://example.com/random-page").is_none());
    }

    #[test]
    fn test_registry_get_by_name() {
        let reg = PluginRegistry::new();
        assert!(reg.get_by_name("wikipedia").is_some());
        assert!(reg.get_by_name("github").is_some());
        assert!(reg.get_by_name("hackernews").is_some());
        assert!(reg.get_by_name("news").is_some());
        assert!(reg.get_by_name("rss").is_some());
        assert!(reg.get_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_registry_list_names() {
        let reg = PluginRegistry::new();
        let names = reg.list_names();
        assert_eq!(names.len(), 5);
        assert!(names.contains(&"wikipedia"));
    }

    #[test]
    fn test_wikipedia_matches() {
        let p = WikipediaPlugin;
        assert!(p.matches("https://en.wikipedia.org/wiki/Rust_(programming_language)"));
        assert!(p.matches("https://de.wikipedia.org/wiki/Berlin"));
        assert!(!p.matches("https://github.com/rust-lang/rust"));
    }

    #[test]
    fn test_wikipedia_extract_infobox() {
        let html = r#"<html><body>
            <table class="infobox">
                <tr><th>Capital</th><td>Berlin</td></tr>
                <tr><th>Population</th><td>83 million</td></tr>
            </table>
            <div id="mw-normal-catlinks"><ul>
                <li><a>European countries</a></li>
            </ul></div>
        </body></html>"#;
        let out = WikipediaPlugin.extract("https://en.wikipedia.org/wiki/Germany", html, "");
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "Capital" && v == "Berlin"));
        assert!(out.facts.iter().any(|(_, k, _)| k == "Population"));
        assert!(out.entities.iter().any(|(n, _)| n == "European countries"));
    }

    #[test]
    fn test_wikipedia_percent_decode_title() {
        let out = WikipediaPlugin.extract(
            "https://en.wikipedia.org/wiki/S%C3%A3o_Paulo",
            "<html><body></body></html>",
            "",
        );
        assert!(out.entities.iter().any(|(n, _)| n.contains("Paulo")));
    }

    #[test]
    fn test_github_matches() {
        let p = GitHubPlugin;
        assert!(p.matches("https://github.com/rust-lang/rust"));
        assert!(!p.matches("https://github.com/rust-lang/rust/issues"));
        assert!(!p.matches("https://github.com/rust-lang"));
        assert!(!p.matches("https://example.com/foo/bar"));
    }

    #[test]
    fn test_github_extract_metadata() {
        let html = r#"<html><head>
            <meta property="og:description" content="A systems programming language" />
        </head><body>
            <span class="social-count">45.2k</span>
            <span class="topic-tag">systems-programming</span>
        </body></html>"#;
        let out = GitHubPlugin.extract("https://github.com/rust-lang/rust", html, "");
        assert!(out
            .entities
            .iter()
            .any(|(n, t)| n == "rust-lang/rust" && t == "github_repo"));
        assert!(out
            .entities
            .iter()
            .any(|(n, t)| n == "rust-lang" && t == "github_user"));
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "description" && v.contains("systems")));
        assert!(out.facts.iter().any(|(_, k, _)| k == "stars"));
    }

    #[test]
    fn test_hackernews_matches() {
        let p = HackerNewsPlugin;
        assert!(p.matches("https://news.ycombinator.com/item?id=12345"));
        assert!(!p.matches("https://news.ycombinator.com/"));
    }

    #[test]
    fn test_hackernews_extract() {
        let html = r#"<html><body>
            <span class="titleline"><a>Show HN: My Cool Project</a></span>
            <span class="score">142 points</span>
            <a class="hnuser">pg</a>
            <span class="subtext"><a>85 comments</a></span>
        </body></html>"#;
        let out = HackerNewsPlugin.extract("https://news.ycombinator.com/item?id=42", html, "");
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "title" && v.contains("Cool Project")));
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "score" && v.contains("142")));
        assert!(out
            .entities
            .iter()
            .any(|(n, t)| n == "pg" && t == "hn_user"));
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "comments" && v.contains("85")));
    }

    #[test]
    fn test_news_matches() {
        let p = NewsArticlePlugin;
        assert!(p.matches("https://www.bbc.com/news/article-123"));
        assert!(!p.matches("https://example.com/blog"));
    }

    #[test]
    fn test_news_extract_og_tags() {
        let html = r#"<html><head>
            <meta property="og:title" content="Breaking: Rust 2.0 Released" />
            <meta property="og:description" content="The Rust team announces version 2.0" />
            <meta property="og:site_name" content="TechCrunch" />
            <meta name="author" content="Jane Doe" />
            <meta property="article:published_time" content="2024-06-15T10:00:00Z" />
        </head><body></body></html>"#;
        let out = NewsArticlePlugin.extract("https://techcrunch.com/2024/rust-2", html, "");
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "headline" && v.contains("Rust 2.0")));
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "author" && v == "Jane Doe"));
        assert!(out.facts.iter().any(|(_, k, _)| k == "publish_date"));
        assert!(out
            .facts
            .iter()
            .any(|(_, k, v)| k == "summary" && v.contains("version 2.0")));
        assert!(out.entities.iter().any(|(n, _)| n == "TechCrunch"));
    }

    #[test]
    fn test_rss_matches() {
        let p = RSSPlugin;
        assert!(p.matches("https://example.com/rss"));
        assert!(p.matches("https://example.com/feed"));
        assert!(p.matches("https://example.com/atom.xml"));
        assert!(!p.matches("https://example.com/page"));
    }

    #[test]
    fn test_rss_extract_items() {
        let xml = r#"<?xml version="1.0"?>
        <rss version="2.0"><channel><title>My Blog</title>
            <item>
              <title>First Post</title>
              <link>https://example.com/first</link>
              <description>This is the first post</description>
              <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
              <author>Alice</author>
            </item>
            <item>
              <title>Second Post</title>
              <link>https://example.com/second</link>
              <description><![CDATA[<p>HTML content</p>]]></description>
            </item>
        </channel></rss>"#;
        let out = RSSPlugin.extract("https://example.com/rss", xml, "");
        assert!(out
            .entities
            .iter()
            .any(|(n, t)| n == "First Post" && t == "feed_entry"));
        assert!(out
            .entities
            .iter()
            .any(|(n, t)| n == "Second Post" && t == "feed_entry"));
        assert!(out
            .facts
            .iter()
            .any(|(n, k, _)| n == "First Post" && k == "url"));
        assert!(out
            .facts
            .iter()
            .any(|(n, k, _)| n == "First Post" && k == "pub_date"));
        assert!(out
            .facts
            .iter()
            .any(|(n, k, _)| n == "First Post" && k == "author"));
        assert!(out
            .facts
            .iter()
            .any(|(n, k, v)| n == "Second Post" && k == "description" && !v.contains("<p>")));
    }

    #[test]
    fn test_atom_extract_entries() {
        let xml = r#"<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>My Atom Feed</title>
          <entry>
            <title>Atom Entry</title>
            <link rel="alternate" href="https://example.com/atom-entry"/>
            <summary>An atom summary</summary>
            <published>2024-01-15T12:00:00Z</published>
          </entry>
        </feed>"#;
        let out = RSSPlugin.extract("https://example.com/atom.xml", xml, "");
        assert!(out
            .entities
            .iter()
            .any(|(n, t)| n == "Atom Entry" && t == "feed_entry"));
        assert!(out
            .facts
            .iter()
            .any(|(n, k, v)| n == "Atom Entry" && k == "url" && v.contains("atom-entry")));
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<p>Hello <b>world</b></p>"), "Hello world");
        assert_eq!(strip_html_tags("no tags"), "no tags");
    }

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("Hello%20World"), "Hello World");
    }

    #[test]
    fn test_split_xml_tags() {
        let xml = "<item><title>A</title></item><item><title>B</title></item>";
        assert_eq!(split_xml_tags(xml, "item").len(), 2);
    }

    #[test]
    fn test_extract_xml_tag() {
        let xml = "<item><title>Hello World</title><link>http://example.com</link></item>";
        assert_eq!(
            extract_xml_tag(xml, "title"),
            Some("Hello World".to_string())
        );
        assert_eq!(
            extract_xml_tag(xml, "link"),
            Some("http://example.com".to_string())
        );
        assert_eq!(extract_xml_tag(xml, "missing"), None);
    }

    #[test]
    fn test_extract_xml_tag_cdata() {
        let xml = "<description><![CDATA[<p>Rich content</p>]]></description>";
        assert_eq!(
            extract_xml_tag(xml, "description"),
            Some("<p>Rich content</p>".to_string())
        );
    }

    #[test]
    fn test_plugin_output_is_empty() {
        assert!(PluginOutput::default().is_empty());
        let non_empty = PluginOutput {
            entities: vec![("t".into(), "t".into())],
            ..Default::default()
        };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_plugin_config_from_toml() {
        let val: toml::Value = toml::from_str(
            r#"
            [plugins]
            enabled = ["wikipedia", "github"]
            [plugins.patterns]
            news = ["mysite.com/news", "blog.example.com"]
        "#,
        )
        .unwrap();
        let config = PluginConfig::from_toml(&val);
        assert_eq!(config.enabled_plugins, vec!["wikipedia", "github"]);
        assert_eq!(config.custom_patterns["news"].len(), 2);
    }

    #[test]
    fn test_registry_extract_with_force() {
        let reg = PluginRegistry::new();
        let html = r#"<html><head>
            <meta property="og:title" content="Test Article" />
            <meta property="og:description" content="A test" />
        </head><body></body></html>"#;
        let out = reg.extract("https://random.example.com/page", html, "", Some("news"));
        assert!(out.is_some());
    }

    #[test]
    fn test_registry_extract_auto_detect() {
        let reg = PluginRegistry::new();
        let html = r#"<html><body>
            <span class="titleline"><a>Test</a></span>
            <span class="score">10 points</span>
            <a class="hnuser">user1</a>
        </body></html>"#;
        let out = reg.extract("https://news.ycombinator.com/item?id=1", html, "", None);
        assert!(out.is_some());
    }

    #[test]
    fn test_github_no_match_subpages() {
        assert!(!GitHubPlugin.matches("https://github.com/rust-lang/rust/issues/123"));
        assert!(!GitHubPlugin.matches("https://github.com/rust-lang/rust/pulls"));
    }
}
