use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Extracted {
    pub entities: Vec<(String, String)>,          // (name, type)
    pub relations: Vec<(String, String, String)>, // (subject, predicate, object)
    pub keywords: Vec<String>,
    pub source_url: String,
}

/// Process raw text and extract entities, relations, and keywords.
pub fn process_text(text: &str, source_url: &str) -> Extracted {
    let sentences = split_sentences(text);
    let tokens = tokenize(text);

    let entities = extract_entities(&sentences);
    let relations = extract_relations(&sentences);
    let keywords = extract_keywords(&tokens, 10);

    let deduped_entities = deduplicate_entities(entities);

    Extracted {
        entities: deduped_entities,
        relations,
        keywords,
        source_url: source_url.to_string(),
    }
}

/// Split text into sentences.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if trimmed.len() > 3 {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if trimmed.len() > 3 {
        sentences.push(trimmed);
    }
    sentences
}

/// Tokenize text into lowercase words.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.len() > 1)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Extract entities: capitalized phrases, numbers, dates, URLs.
pub fn extract_entities(sentences: &[String]) -> Vec<(String, String)> {
    let mut entities = Vec::new();
    let stop_words: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    for sentence in sentences {
        // Capitalized phrases (2+ words or single capitalized non-stop word)
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut i = 0;
        while i < words.len() {
            let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
            if !word.is_empty()
                && word.chars().next().is_some_and(|c| c.is_uppercase())
                && !stop_words.contains(word.to_lowercase().as_str())
                && i > 0
            // skip sentence-initial capitalization
            {
                let mut phrase = vec![word.to_string()];
                let mut j = i + 1;
                while j < words.len() {
                    let next = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                    if !next.is_empty() && next.chars().next().is_some_and(|c| c.is_uppercase()) {
                        phrase.push(next.to_string());
                        j += 1;
                    } else {
                        break;
                    }
                }
                let name = phrase.join(" ");
                if name.len() > 1 {
                    entities.push((name, "phrase".to_string()));
                }
                i = j;
            } else {
                i += 1;
            }
        }

        // Numbers (years, quantities)
        for word in &words {
            let clean = word.trim_matches(|c: char| !c.is_numeric());
            if clean.len() == 4 {
                if let Ok(n) = clean.parse::<u32>() {
                    if (1800..=2100).contains(&n) {
                        entities.push((clean.to_string(), "year".to_string()));
                    }
                }
            }
        }

        // URLs in text
        for word in &words {
            if word.starts_with("http://") || word.starts_with("https://") {
                entities.push((word.to_string(), "url".to_string()));
            }
        }
    }

    entities
}

/// Extract subject-verb-object relations from sentences.
pub fn extract_relations(sentences: &[String]) -> Vec<(String, String, String)> {
    let relation_verbs: HashSet<&str> = [
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "created",
        "built",
        "made",
        "wrote",
        "developed",
        "founded",
        "invented",
        "discovered",
        "launched",
        "acquired",
        "bought",
        "sold",
        "uses",
        "using",
        "contains",
        "includes",
        "supports",
        "runs",
        "produces",
        "generates",
        "provides",
        "offers",
    ]
    .iter()
    .copied()
    .collect();

    let mut relations = Vec::new();

    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 3 {
            continue;
        }

        for i in 1..words.len() - 1 {
            let verb = words[i]
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if relation_verbs.contains(verb.as_str()) {
                // Look backward for subject (capitalized phrase)
                let mut subj_parts = Vec::new();
                let mut j = i as isize - 1;
                while j >= 0 {
                    let w = words[j as usize].trim_matches(|c: char| !c.is_alphanumeric());
                    if !w.is_empty() && w.chars().next().is_some_and(|c| c.is_uppercase()) {
                        subj_parts.push(w);
                        j -= 1;
                    } else {
                        break;
                    }
                }
                subj_parts.reverse();

                // Look forward for object (capitalized phrase)
                let mut obj_parts = Vec::new();
                let mut k = i + 1;
                while k < words.len() {
                    let w = words[k].trim_matches(|c: char| !c.is_alphanumeric());
                    if !w.is_empty() && w.chars().next().is_some_and(|c| c.is_uppercase()) {
                        obj_parts.push(w);
                        k += 1;
                    } else {
                        break;
                    }
                }

                if !subj_parts.is_empty() && !obj_parts.is_empty() {
                    let subject = subj_parts.join(" ");
                    let object = obj_parts.join(" ");
                    if subject != object {
                        relations.push((subject, verb.clone(), object));
                    }
                }
            }
        }
    }

    relations
}

/// Extract top keywords using TF-IDF-like scoring.
pub fn extract_keywords(tokens: &[String], max: usize) -> Vec<String> {
    let stop_words: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in tokens {
        if !stop_words.contains(t.as_str()) && t.len() > 2 {
            *freq.entry(t.as_str()).or_insert(0) += 1;
        }
    }

    let mut scored: Vec<(&&str, f64)> = freq
        .iter()
        .map(|(word, count)| {
            // Simple TF score weighted by word length (proxy for specificity)
            let tf = *count as f64 / tokens.len().max(1) as f64;
            let len_bonus = (word.len() as f64 / 10.0).min(1.0);
            (word, tf * (1.0 + len_bonus))
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(max)
        .map(|(w, _)| w.to_string())
        .collect()
}

/// Deduplicate entities using Levenshtein distance.
pub fn deduplicate_entities(entities: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut result: Vec<(String, String)> = Vec::new();

    for (name, etype) in entities {
        let is_dup = result.iter().any(|(existing, existing_type)| {
            existing_type == &etype
                && levenshtein(&name.to_lowercase(), &existing.to_lowercase()) <= 2
        });
        if !is_dup {
            result.push((name, etype));
        }
    }

    result
}

/// Levenshtein distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

const STOP_WORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "our",
    "their",
    "not",
    "no",
    "nor",
    "so",
    "if",
    "then",
    "than",
    "too",
    "very",
    "just",
    "about",
    "above",
    "after",
    "before",
    "between",
    "into",
    "through",
    "during",
    "until",
    "against",
    "among",
    "throughout",
    "despite",
    "towards",
    "upon",
    "concerning",
    "also",
    "however",
    "therefore",
    "furthermore",
    "moreover",
    "although",
    "nevertheless",
    "whereas",
    "whereby",
    "hereby",
    "therein",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "any",
    "here",
    "there",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. This is a test! How are you?";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is Rust.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
    }

    #[test]
    fn test_extract_entities_capitalized() {
        let sentences = vec!["The company Google was founded in California.".to_string()];
        let entities = extract_entities(&sentences);
        let names: Vec<&str> = entities.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Google"));
        assert!(names.contains(&"California"));
    }

    #[test]
    fn test_extract_entities_years() {
        let sentences = vec!["Rust was released in 2015 by Mozilla.".to_string()];
        let entities = extract_entities(&sentences);
        let has_year = entities.iter().any(|(_, t)| t == "year");
        assert!(has_year);
    }

    #[test]
    fn test_extract_entities_urls() {
        let sentences = vec!["Visit https://rust-lang.org for more info.".to_string()];
        let entities = extract_entities(&sentences);
        let has_url = entities.iter().any(|(_, t)| t == "url");
        assert!(has_url);
    }

    #[test]
    fn test_extract_relations() {
        let sentences = vec!["Google created Android for mobile devices.".to_string()];
        let relations = extract_relations(&sentences);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].0, "Google");
        assert_eq!(relations[0].1, "created");
        assert_eq!(relations[0].2, "Android");
    }

    #[test]
    fn test_extract_keywords() {
        let tokens = tokenize("Rust is a systems programming language focused on safety and performance and concurrency");
        let keywords = extract_keywords(&tokens, 5);
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 5);
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_different() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "hello"), 5);
        assert_eq!(levenshtein("hello", ""), 5);
    }

    #[test]
    fn test_deduplicate_entities() {
        let entities = vec![
            ("Google".to_string(), "phrase".to_string()),
            ("Gogle".to_string(), "phrase".to_string()), // typo
            ("Mozilla".to_string(), "phrase".to_string()),
        ];
        let deduped = deduplicate_entities(entities);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_process_text() {
        let text =
            "Google was founded by Larry Page and Sergey Brin in 1998. Google created Android.";
        let extracted = process_text(text, "https://example.com");
        assert!(!extracted.entities.is_empty());
        assert_eq!(extracted.source_url, "https://example.com");
    }

    #[test]
    fn test_stop_words_filtered() {
        let tokens = tokenize("the quick brown fox jumps over the lazy dog");
        let keywords = extract_keywords(&tokens, 10);
        assert!(!keywords.contains(&"the".to_string()));
    }

    #[test]
    fn test_empty_text() {
        let extracted = process_text("", "https://example.com");
        assert!(extracted.entities.is_empty());
        assert!(extracted.relations.is_empty());
    }

    #[test]
    fn test_multi_word_entity() {
        let sentences = vec!["The project was led by New York University researchers.".to_string()];
        let entities = extract_entities(&sentences);
        let names: Vec<&str> = entities.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("New York University")));
    }
}
