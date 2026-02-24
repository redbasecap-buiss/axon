/// Fuzzy string matching with Levenshtein distance and typo tolerance.
/// Compute Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1].to_lowercase().eq(b[j - 1].to_lowercase()) {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);

            // Damerau transposition
            if i > 1
                && j > 1
                && a[i - 1].to_lowercase().eq(b[j - 2].to_lowercase())
                && a[i - 2].to_lowercase().eq(b[j - 1].to_lowercase())
            {
                curr[j] = curr[j].min(prev[j - 1]); // transposition cost = 1 (already added via substitution path)
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Compute normalized similarity (0.0 to 1.0) between two strings.
pub fn similarity(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

/// Check if a string is a fuzzy match within a given max edit distance.
pub fn is_fuzzy_match(query: &str, target: &str, max_distance: usize) -> bool {
    levenshtein(query, target) <= max_distance
}

/// Compute the max allowed edit distance based on query length.
/// Short words (1-3): 0 edits (exact only)
/// Medium words (4-6): 1 edit
/// Long words (7+): 2 edits
pub fn auto_max_distance(query: &str) -> usize {
    match query.len() {
        0..=3 => 0,
        4..=6 => 1,
        _ => 2,
    }
}

/// Search a list of candidates for fuzzy matches, returning (candidate, distance, similarity)
/// sorted by distance then alphabetically.
pub fn fuzzy_search<'a>(
    query: &str,
    candidates: &'a [String],
    max_distance: Option<usize>,
) -> Vec<(&'a str, usize, f64)> {
    let max_dist = max_distance.unwrap_or_else(|| auto_max_distance(query));
    let query_lower = query.to_lowercase();

    let mut results: Vec<(&str, usize, f64)> = candidates
        .iter()
        .filter_map(|c| {
            let c_lower = c.to_lowercase();

            // Exact substring match gets distance 0
            if c_lower.contains(&query_lower) {
                return Some((c.as_str(), 0, 1.0));
            }

            let dist = levenshtein(&query_lower, &c_lower);
            if dist <= max_dist {
                let sim = similarity(&query_lower, &c_lower);
                Some((c.as_str(), dist, sim))
            } else {
                // Also try matching against individual words in multi-word candidates
                let words: Vec<&str> = c_lower.split_whitespace().collect();
                for word in &words {
                    let word_dist = levenshtein(&query_lower, word);
                    if word_dist <= max_dist {
                        let sim = similarity(&query_lower, word);
                        return Some((c.as_str(), word_dist, sim));
                    }
                }
                None
            }
        })
        .collect();

    results.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    results
}

/// Bigram-based similarity for longer strings (good for phrase matching).
pub fn bigram_similarity(a: &str, b: &str) -> f64 {
    fn bigrams(s: &str) -> Vec<(char, char)> {
        let chars: Vec<char> = s.to_lowercase().chars().collect();
        if chars.len() < 2 {
            return vec![];
        }
        chars.windows(2).map(|w| (w[0], w[1])).collect()
    }

    let a_bigrams = bigrams(a);
    let b_bigrams = bigrams(b);

    if a_bigrams.is_empty() && b_bigrams.is_empty() {
        return 1.0;
    }
    if a_bigrams.is_empty() || b_bigrams.is_empty() {
        return 0.0;
    }

    let matches = a_bigrams.iter().filter(|bg| b_bigrams.contains(bg)).count();
    (2.0 * matches as f64) / (a_bigrams.len() + b_bigrams.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "hello"), 5);
        assert_eq!(levenshtein("hello", ""), 5);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_levenshtein_single_edit() {
        assert_eq!(levenshtein("kitten", "sitten"), 1); // substitution
        assert_eq!(levenshtein("hello", "helo"), 1); // deletion
        assert_eq!(levenshtein("helo", "hello"), 1); // insertion
    }

    #[test]
    fn test_levenshtein_transposition() {
        assert_eq!(levenshtein("ab", "ba"), 1);
    }

    #[test]
    fn test_levenshtein_case_insensitive() {
        assert_eq!(levenshtein("Hello", "hello"), 0);
        assert_eq!(levenshtein("RUST", "rust"), 0);
    }

    #[test]
    fn test_similarity() {
        assert!((similarity("hello", "hello") - 1.0).abs() < f64::EPSILON);
        assert!(similarity("hello", "world") < 0.5);
        assert!(similarity("rust", "ruts") > 0.5);
    }

    #[test]
    fn test_auto_max_distance() {
        assert_eq!(auto_max_distance("ab"), 0);
        assert_eq!(auto_max_distance("abc"), 0);
        assert_eq!(auto_max_distance("rust"), 1);
        assert_eq!(auto_max_distance("python"), 1);
        assert_eq!(auto_max_distance("javascript"), 2);
    }

    #[test]
    fn test_fuzzy_search_exact() {
        let candidates = vec!["Rust".to_string(), "Python".to_string(), "Ruby".to_string()];
        let results = fuzzy_search("Rust", &candidates, None);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "Rust");
    }

    #[test]
    fn test_fuzzy_search_typo() {
        let candidates = vec!["Rust".to_string(), "Python".to_string(), "Ruby".to_string()];
        let results = fuzzy_search("Rsut", &candidates, Some(2));
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "Rust");
    }

    #[test]
    fn test_fuzzy_search_substring() {
        let candidates = vec!["Albert Einstein".to_string(), "Niels Bohr".to_string()];
        let results = fuzzy_search("Einstein", &candidates, None);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "Albert Einstein");
    }

    #[test]
    fn test_bigram_similarity() {
        assert!(bigram_similarity("hello", "hello") > 0.99);
        assert!(bigram_similarity("night", "nacht") < 0.5);
        assert!(bigram_similarity("", "") > 0.99);
    }

    #[test]
    fn test_fuzzy_search_no_match() {
        let candidates = vec!["Rust".to_string()];
        let results = fuzzy_search("zzzzz", &candidates, Some(1));
        assert!(results.is_empty());
    }

    #[test]
    fn test_is_fuzzy_match() {
        assert!(is_fuzzy_match("rust", "rust", 0));
        assert!(is_fuzzy_match("rust", "ruts", 2));
        assert!(!is_fuzzy_match("rust", "python", 2));
    }
}
