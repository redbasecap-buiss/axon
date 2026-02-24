/// Markdown file ingestion — feed local .md files into the knowledge graph.
use std::path::Path;

/// Extract readable text from markdown by stripping formatting.
pub fn extract_text_from_markdown(md: &str) -> String {
    let mut text = String::with_capacity(md.len());

    for line in md.lines() {
        let trimmed = line.trim();

        // Skip empty lines (preserve paragraph breaks)
        if trimmed.is_empty() {
            if !text.ends_with('\n') {
                text.push('\n');
            }
            continue;
        }

        // Strip heading markers
        let line = if let Some(rest) = trimmed.strip_prefix("######") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("#####") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("####") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("###") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("##") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix('#') {
            rest.trim()
        } else {
            trimmed
        };

        // Skip horizontal rules
        if line
            .chars()
            .all(|c| c == '-' || c == '*' || c == '_' || c == ' ')
            && line.len() >= 3
        {
            continue;
        }

        // Strip list markers
        let line = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .or_else(|| line.strip_prefix("+ "))
            .or_else(|| {
                // Numbered lists: "1. ", "2. ", etc.
                let dot = line.find(". ")?;
                if dot <= 3 && line[..dot].chars().all(|c| c.is_ascii_digit()) {
                    Some(&line[dot + 2..])
                } else {
                    None
                }
            })
            .unwrap_or(line);

        // Strip inline code fences (skip code blocks)
        if line.starts_with("```") {
            continue;
        }

        // Strip inline formatting: **bold**, *italic*, `code`, ~~strikethrough~~, [links](url)
        let line = strip_inline_formatting(line);

        if !line.is_empty() {
            text.push_str(&line);
            text.push(' ');
        }
    }

    text
}

/// Strip inline markdown formatting from a line.
fn strip_inline_formatting(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            // Bold/italic: **, *, __, _
            '*' | '_' => {
                // Skip formatting markers
                if i + 1 < chars.len() && chars[i + 1] == chars[i] {
                    i += 2; // skip **
                } else {
                    i += 1; // skip *
                }
            }
            // Strikethrough: ~~
            '~' if i + 1 < chars.len() && chars[i + 1] == '~' => {
                i += 2;
            }
            // Inline code: `
            '`' => {
                i += 1;
                // Extract content until closing `
                while i < chars.len() && chars[i] != '`' {
                    result.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    i += 1; // skip closing `
                }
            }
            // Links: [text](url) — keep text, drop url
            '[' => {
                i += 1;
                let mut link_text = String::new();
                while i < chars.len() && chars[i] != ']' {
                    link_text.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    i += 1; // skip ]
                }
                // Skip (url) part
                if i < chars.len() && chars[i] == '(' {
                    i += 1;
                    while i < chars.len() && chars[i] != ')' {
                        i += 1;
                    }
                    if i < chars.len() {
                        i += 1; // skip )
                    }
                }
                result.push_str(&link_text);
            }
            // Images: ![alt](url) — keep alt text
            '!' if i + 1 < chars.len() && chars[i + 1] == '[' => {
                i += 2; // skip ![
                let mut alt = String::new();
                while i < chars.len() && chars[i] != ']' {
                    alt.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    i += 1;
                }
                if i < chars.len() && chars[i] == '(' {
                    i += 1;
                    while i < chars.len() && chars[i] != ')' {
                        i += 1;
                    }
                    if i < chars.len() {
                        i += 1;
                    }
                }
                result.push_str(&alt);
            }
            c => {
                result.push(c);
                i += 1;
            }
        }
    }

    result
}

/// Read a markdown file from disk and extract text.
pub fn read_markdown_file(path: &Path) -> anyhow::Result<String> {
    let content = std::fs::read_to_string(path)?;
    Ok(extract_text_from_markdown(&content))
}

/// Ingest a markdown file: extract text, run NLP, store in brain.
pub fn ingest_file(brain: &crate::db::Brain, path: &Path) -> anyhow::Result<(usize, usize, usize)> {
    let text = read_markdown_file(path)?;
    let source = format!("file://{}", path.display());
    let extracted = crate::nlp::process_text(&text, &source);
    let counts = brain.learn(&extracted)?;
    Ok(counts)
}

/// Ingest all markdown files from a directory (non-recursive by default).
pub fn ingest_directory(
    brain: &crate::db::Brain,
    dir: &Path,
    recursive: bool,
) -> anyhow::Result<(usize, usize, usize, usize)> {
    let mut total_files = 0;
    let mut total_entities = 0;
    let mut total_relations = 0;
    let mut total_facts = 0;

    let entries: Vec<_> = if recursive {
        walkdir(dir)?
    } else {
        std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect()
    };

    for path in entries {
        if path.extension().and_then(|e| e.to_str()) == Some("md") {
            match ingest_file(brain, &path) {
                Ok((e, r, f)) => {
                    total_files += 1;
                    total_entities += e;
                    total_relations += r;
                    total_facts += f;
                }
                Err(err) => {
                    eprintln!("   Warning: failed to ingest {}: {}", path.display(), err);
                }
            }
        }
    }

    Ok((total_files, total_entities, total_relations, total_facts))
}

/// Recursively walk a directory and collect all file paths.
fn walkdir(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            paths.extend(walkdir(&path)?);
        } else {
            paths.push(path);
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_headings() {
        let md = "# Title\n## Subtitle\nParagraph text.";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("Title"));
        assert!(text.contains("Subtitle"));
        assert!(text.contains("Paragraph text"));
        assert!(!text.contains('#'));
    }

    #[test]
    fn test_strip_formatting() {
        let md = "This is **bold** and *italic* and `code` text.";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("bold"));
        assert!(text.contains("italic"));
        assert!(text.contains("code"));
        assert!(!text.contains('*'));
        assert!(!text.contains('`'));
    }

    #[test]
    fn test_strip_links() {
        let md = "Visit [Rust](https://rust-lang.org) for more info.";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("Rust"));
        assert!(!text.contains("https://rust-lang.org"));
        assert!(!text.contains('['));
    }

    #[test]
    fn test_strip_images() {
        let md = "Here is ![a photo](image.png) of something.";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("a photo"));
        assert!(!text.contains("image.png"));
    }

    #[test]
    fn test_strip_lists() {
        let md = "- Item one\n- Item two\n1. Numbered\n2. Also numbered";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("Item one"));
        assert!(text.contains("Item two"));
        assert!(text.contains("Numbered"));
        assert!(text.contains("Also numbered"));
    }

    #[test]
    fn test_code_blocks_skipped() {
        let md = "Text before\n```rust\nfn main() {}\n```\nText after";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("Text before"));
        assert!(text.contains("Text after"));
        // Code fence lines should be skipped, but content between them
        // is still processed as regular text (simple parser)
        assert!(!text.contains("```"));
    }

    #[test]
    fn test_horizontal_rules_skipped() {
        let md = "Above\n---\nBelow\n***\nEnd";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("Above"));
        assert!(text.contains("Below"));
        assert!(text.contains("End"));
    }

    #[test]
    fn test_empty_markdown() {
        let text = extract_text_from_markdown("");
        assert!(text.trim().is_empty());
    }

    #[test]
    fn test_strikethrough() {
        let md = "This is ~~deleted~~ text.";
        let text = extract_text_from_markdown(md);
        assert!(text.contains("deleted"));
        assert!(!text.contains('~'));
    }

    #[test]
    fn test_ingest_file() {
        let dir = tempfile::tempdir().unwrap();
        let md_path = dir.path().join("test.md");
        std::fs::write(
            &md_path,
            "# Rust\nRust is a programming language created by Mozilla.",
        )
        .unwrap();

        let brain = crate::db::Brain::open_in_memory().unwrap();
        let (e, r, f) = ingest_file(&brain, &md_path).unwrap();
        assert!(e + r + f > 0, "Should learn something from the file");
    }
}
