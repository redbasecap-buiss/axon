#![allow(dead_code)]
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Extracted {
    pub entities: Vec<(String, String)>,
    pub relations: Vec<(String, String, String)>,
    pub keywords: Vec<String>,
    pub source_url: String,
    pub language: Language,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
    German,
    French,
    Italian,
    Spanish,
    Unknown,
}

const STOP_WORDS_EN: &[&str] = &[
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

const STOP_WORDS_DE: &[&str] = &[
    "den", "der", "die", "das", "ein", "eine", "einer", "eines", "einem", "einen", "und", "oder",
    "aber", "in", "im", "an", "am", "auf", "aus", "bei", "mit", "nach", "seit", "von", "vor", "zu",
    "zum", "zur", "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden", "hat",
    "haben", "hatte", "hatten", "sein", "gewesen", "kann", "muss", "soll", "sollen", "sollte",
    "darf", "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "unser", "euer", "nicht",
    "kein", "keine", "keinem", "keinen", "keiner", "sich", "als", "auch", "noch", "schon", "so",
    "wie", "was", "wer", "wo", "wenn", "weil", "dass", "ob", "denn", "doch", "nur", "sehr", "dann",
    "da", "hier", "dort", "dieser", "diese", "dieses", "jeder", "jede", "jedes", "alle", "man",
    "mehr", "viel", "einige", "andere",
];

const STOP_WORDS_FR: &[&str] = &[
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "et", "ou", "mais", "en",
    "dans", "sur", "sous", "avec", "pour", "par", "sans", "vers", "chez", "entre", "est", "sont",
    "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on", "ce", "cette", "ces", "mon",
    "ton", "son", "notre", "votre", "leur", "leurs", "ne", "pas", "plus", "si", "que", "qui",
    "quoi", "dont", "aussi", "tout", "tous", "toute", "toutes", "ici",
];

const STOP_WORDS_IT: &[&str] = &[
    "il", "lo", "la", "le", "li", "gli", "un", "uno", "una", "di", "del", "della", "dei", "delle",
    "da", "dal", "dalla", "in", "nel", "nella", "a", "al", "alla", "con", "su", "sul", "sulla",
    "per", "tra", "fra", "e", "o", "ma", "non", "che", "sono", "era", "erano", "ha", "hanno", "io",
    "tu", "lui", "lei", "noi", "voi", "loro", "mi", "ti", "ci", "si", "questo", "questa", "questi",
    "queste", "quello", "quella", "come", "dove", "quando", "anche", "tutto", "tutti",
];

const STOP_WORDS_ES: &[&str] = &[
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en", "con", "por",
    "para", "sin", "sobre", "entre", "y", "o", "pero", "que", "es", "son", "era", "eran", "fue",
    "fueron", "ha", "han", "ser", "estar", "tiene", "tienen", "yo", "no", "como", "este", "esta",
    "estos", "estas", "ese", "esa", "todo", "todos", "toda", "todas", "otro", "otra", "otros",
    "donde", "cuando",
];

/// Blacklist of web/Wikipedia chrome that should never be entities.
const ENTITY_BLACKLIST: &[&str] = &[
    "wikipedia",
    "wikimedia",
    "wikimedia foundation",
    "wikisource",
    "jump",
    "navigation",
    "main page",
    "main",
    "contents",
    "random",
    "upload",
    "community",
    "special",
    "recent",
    "recent changes",
    "help learn",
    "help",
    "contact",
    "community portal",
    "current events",
    "random article",
    "about wikipedia",
    "donate",
    "what links here",
    "related changes",
    "permanent link",
    "page information",
    "cite this page",
    "create account",
    "log in",
    "talk",
    "contributions",
    "read",
    "view source",
    "view history",
    "edit",
    "search",
    "printable version",
    "download",
    "tools",
    "languages",
    "sidebar",
    "toggle",
    "menu",
    "jump to content",
    "jump to navigation",
    "jump to search",
    "personal tools",
    "namespaces",
    "variants",
    "views",
    "more",
    "general",
    "statistics",
    "cookie statement",
    "mobile view",
    "developers",
    "privacy policy",
    "terms of use",
    "desktop",
    "powered by mediawiki",
    "footer",
    "header",
    "skip to content",
    "cookie",
    "accept",
    "dismiss",
    "subscribe",
    "sign in",
    "sign up",
    "log out",
    "loading",
    "please wait",
    "click here",
    "read more",
    "learn more",
    "continue reading",
    "close",
    "open",
    "show",
    "hide",
    "expand",
    "collapse",
    "previous",
    "next",
    "back",
    "forward",
    "home",
    "settings",
    "preferences",
    "notifications",
    "appearance",
    "copyright",
    "all rights reserved",
    "terms of service",
    "external links",
    "see also",
    "see",
    "citations",
    "references notes",
    "categories",
    "further reading",
    "bibliography",
    "notes",
    "from wikipedia",
    "free encyclopedia",
    "retrieved",
    "archived",
    "cite this",
    "page",
    "article",
    "section",
    "chapter",
    "index",
    "glossary",
    "appendix",
    "abstract",
    "show",
    "shows",
    "frequently",
    "brochure",
    "faq",
    "click",
    "click here",
    "portal",
    "stub",
    "redirect",
    "disambiguation",
    "wikidata",
    "wikidata item",
    "namespace",
    "template",
    "user page",
    "bibcode",
    "s2cid",
    "oclc",
    "jstor",
    "wayback machine",
    "arxiv",
    "accessed",
    "published",
    "original",
    // German Wikipedia UI
    "abstimmungen",
    "agglomerationen",
    "anmelden",
    "bearbeiten",
    "benutzerkonto",
    "diskussion",
    "druckversion",
    "einzelnachweise",
    "hauptseite",
    "kategorien",
    "literatur",
    "mitmachen",
    "quellenangaben",
    "quelltext",
    "seiteninformationen",
    "spenden",
    "verlinkte",
    "versionsgeschichte",
    "weblinks",
    "werkzeuge",
    "zufälliger artikel",
    // Generic UI/navigation terms
    "account",
    "outline",
    "overview",
    "summary",
    "introduction",
    "background",
    "description",
    "details",
    "information",
    "resources",
    "documentation",
    "feedback",
    "share",
    "bookmark",
    "print",
    "copy",
    "paste",
    "delete",
    "remove",
    "add",
    "create",
    "update",
    "submit",
    "cancel",
    "confirm",
    "apply",
    "filter",
    "sort",
    "browse",
    "explore",
    "researchers",
    "researchgate",
    "mathpages",
    "mathworld",
    "wolfram",
    "pubmed",
    "academia.edu",
    "scholarpedia",
    "springer",
    "elsevier",
    "wiley",
    // Generic standalone words that aren't meaningful entities
    "number",
    "response",
    "geometry",
    "course",
    "british",
    "illusion",
    "internet",
    "control",
    "evidence",
    "force",
    "growth",
    "practice",
    "council",
    "empire",
    "language",
    "philosophical",
    "school",
    "senate",
    "university",
    "closed",
    "ancient",
    "central",
    "eastern",
    "western",
    "northern",
    "southern",
    "national",
    "international",
    "royal",
    "imperial",
    "period",
    "region",
    "population",
    "religion",
    "culture",
    "science",
    "war",
    "battle",
    "treaty",
    "kingdom",
    "dynasty",
    "republic",
    // Common English words that get capitalized at sentence starts
    "claim",
    "crank",
    "exile",
    "fewer",
    "older",
    "overall",
    "faculty",
    "pursuit",
    "scholar",
    "rapport",
    "miracle",
    "crusade",
    "courage",
    "extract",
    "kitchen",
    "anatomy",
    "alchemy",
    "descent",
    "eclipse",
    "suicide",
    "altitude",
    "episode",
    "charter",
    "foreign",
    "freedom",
    "justice",
    "journey",
    "context",
    "immense",
    "divided",
    "fashion",
    "coastal",
    "longest",
    "imagine",
    "reports",
    "present",
    "quantum",
    "opinion",
    "revised",
    // Added 2026-02-24: more generic words
    "simulator",
    "discoverers",
    "calendar",
    "collage",
    "continent",
    "continuum",
    "continua",
    "epigram",
    "revisit",
    "tribune",
    "preschool",
    "saltwater",
    "brotherhood",
    "commander",
    "chariot",
    "intelligencer",
    "existence",
    "server",
    "straits",
    "dynamique",
    "gebiete",
    "gladiatori",
    "militare",
    "tidskrift",
    "epistulae",
    "partie",
    // French Wikipedia UI
    "accueil",
    "afficher",
    "contributions",
    "historique",
    "modifier",
    "outils",
    "portail",
    "military history",
    "civil order",
    "human pressure",
    "personnel colours",
    // Transitional/conjunctive adverbs that get picked up as entity tails
    "meanwhile",
    "nachlass",
    "elsewhere",
    "perhaps",
    "apparently",
    "regardless",
    "nonetheless",
    "subsequently",
    "accordingly",
    "consequently",
    "simultaneously",
    "alternatively",
    "approximately",
    "predominantly",
    "traditionally",
    "essentially",
    "respectively",
    "particularly",
    "significantly",
    "substantially",
    "independently",
    "occasionally",
    "ultimately",
    // Added 2026-02-25: generic words and concatenation patterns from DB cleanup
    "difficulties",
    "eleventh",
    "end-user",
    "forgotten",
    "fresh-water",
    "respect",
    "society",
    "year time",
    "sea-bottom",
    "nachbarn",
    "raumentwicklung",
    "memoria",
    "further",
    // Added 2026-02-25: generic terms and hyphenated non-entities
    "floating-point",
    "divide-by-zero",
    "inter-war period",
    "town-centre",
    "turing-complete",
    "gram-interval",
    "spectra",
    "harmony",
    "lighter",
    "tangram",
    "skylark",
    "acquaintanceship",
    // Added 2026-02-24: generic words found in DB cleanup
    "transcript",
    "biographer",
    "persuasion",
    "subsurface",
    "rainforest",
    "nanoscale",
    "bilayer",
    "moderne",
    "heuristica",
    "applicazioni",
    "lettres",
    "controversy",
    "companion",
    "democracy",
    "conspiracy",
    "marketplace",
    "hierarchy",
    "biography",
    "reptile",
    "theatre",
    "limit",
    "monasteries",
    "meantime",
    "thermodynamic",
    "invasion",
    "currency",
    "catalogue",
    // Added 2026-02-24 (cron round 2): more noise
    "doi",
    "indian",
    "germanic",
    "marquis",
    "grant",
    "levantine",
    // Added 2026-02-24 (cron round 3): demonyms and generics
    "italian",
    "bengali",
    "israeli",
    "kyrgyz",
    "persian",
    "tajik",
    "territorial",
    "piracy",
    "peacock",
    // Added 2026-02-24 (cron round 4): generic nouns/adjectives cleaned from DB
    "eyesight",
    "sticky",
    "blacksmith",
    "sandwich",
    "nowhere",
    "whatever",
    "whenever",
    "himself",
    "welcome",
    "excellent",
    "perfect",
    "strange",
    "smallest",
    "highest",
    "softer",
    "machine",
    "written",
    "revealed",
    "preceded",
    "translated",
    "integrated",
    "distributed",
    "convergent",
    "divergent",
    "heaven",
    "monster",
    "dragon",
    "travel",
    "visit",
    "founders",
    "master",
    "student",
    "teacher",
    "startup",
    "outbreak",
    "downfall",
    "daytime",
    "everybody",
    "concrete",
    "multiple",
    "improper",
    "birthday",
    "childhood",
    "girlhood",
    "personhood",
    "deanship",
    "merchant",
    "consultant",
    "surveyor",
    "thinker",
    "liberator",
    "peacemaker",
    "creator",
    "inventor",
    "designer",
    "programmer",
    "principal",
    "coordinator",
    "consumer",
    "explorer",
    "wanderer",
    "postage",
    "records",
    "session",
    "product",
    "component",
    "pattern",
    "constraint",
    "constraints",
    "examples",
    "types",
    "properties",
    "subjects",
    "responses",
    "remarks",
    "theme",
    "proposal",
    "progress",
    "rationale",
    "exercise",
    "percentage",
    "trial",
    "relationship",
    "guidelines",
    "terminology",
    "participants",
    "comparisons",
    "discussions",
    "consequences",
    "equivalents",
    "hallmark",
    "casualty",
    "emergency",
    "contingency",
    "insurgency",
    "suffrage",
    "regency",
    "legacy",
    "destiny",
    "fantasy",
    "panorama",
    "monolith",
    "catapult",
    "gauntlet",
    "labyrinth",
    "pendulum",
    "terrace",
    "pyramid",
    "horseshoe",
    "fishtail",
    "rattlesnake",
    "hedgehog",
    "mockingbird",
    "patriot",
    "tempest",
    "panther",
    "scream",
    "fallout",
    "blitzkrieg",
    "wartime",
    "surrender",
    "leaflet",
    "paperback",
    "photograph",
    "watercolour",
    "whitepaper",
    "newsgroup",
    "newspaper",
    "satellite",
    "mainstream",
    "lifestyle",
    "healthcare",
    "coinage",
    "marriage",
    "rainfall",
    "erosion",
    "knockout",
    "breakdown",
    "breakthrough",
    "shrinkage",
    "merge",
    "shuffle",
    "iterate",
    "snapshot",
    "scatter",
    "border",
    "capital",
    "domain",
    "bucket",
    "piece",
    "write",
    "detail",
    "robot",
    "banquet",
    "caravan",
    "defense",
    "grammar",
    "lattice",
    "lexicon",
    "scalar",
    "logic",
    "scheme",
    "bottom",
    "friends",
    "friendships",
    "verge",
    "woven",
    "reprinted",
    "formally",
    "investigate",
    "communicate",
    "general-purpose",
    "generalprobe",
    "concurrency concurrent",
    "sorry",
    "dark",
    "collected",
    // Added 2026-02-25: generic terms from DB cleanup
    "elector",
    "biotechnology",
    "oberbürgermeister",
    "barbarians",
    "centurion",
    "fundamentalsatzes",
    "bastante",
    "slovene",
    "supercomputer",
    "elektronik",
    "decurion",
    "posguerra",
    "urkunden",
    "sammelregeln",
    "wärmestrahlung",
    "augustinians",
    "philistines",
    // Added 2026-02-24 (brain cleaner round 2)
    "carryout",
    "vorschläge",
    "perusine",
    // Added 2026-02-24 (brain cleaner round 3)
    "refugee",
    "neurosurgeons",
    "intervalles",
    "atomkerne",
    "filmportrait",
    "stadion",
    // Added 2026-02-24 (brain cleaner round 4): generic/academic/foreign noise
    "hauptidealsatz",
    "verwandlungsinhalt",
    "regionalportraits",
    "gemeindewappen",
    "confoederatio",
    "controversial",
    "methanosarcina",
    "nicotiana",
    "hatcheria",
    "leptotrichia",
    "neogene",
    "universitas",
    "escutcheons",
    "triarii",
    "falange",
    "heimskringla",
    "characteristics",
    "phylogeography",
    "planetology",
    "microplastics",
    "bacchanalia",
    "photoemission",
    "phyllotaxis",
    "commentariolus",
    "sechseläutenmarsch",
    "ptolemäerreich",
    "rettung",
    "klarheit",
    "biodiversidade",
    "poblamiento",
    "bioinformatics",
    "mathematika",
    // Added 2026-02-24 (brain cleaner round 5)
    "temperature",
    "harmonic",
    "supersonic",
    "attosecond",
    "telegraph",
    "niemand",
    "hermanos",
    "campéon",
    "issues",
    "ballistics",
    "computation",
    "admiralty",
];

/// Common person name prefixes/titles for entity classification.
const PERSON_TITLES: &[&str] = &[
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "professor",
    "sir",
    "lord",
    "lady",
    "king",
    "queen",
    "prince",
    "princess",
    "president",
    "chancellor",
    "minister",
    "senator",
    "governor",
    "general",
    "admiral",
    "captain",
    "colonel",
    "saint",
    "pope",
    "bishop",
    "rabbi",
    "imam",
    "reverend",
    "father",
    "brother",
    "sister",
    "mother",
];

/// Common place suffixes/keywords for entity classification.
const PLACE_INDICATORS: &[&str] = &[
    "city",
    "town",
    "village",
    "county",
    "state",
    "province",
    "region",
    "district",
    "island",
    "islands",
    "river",
    "lake",
    "mountain",
    "mountains",
    "valley",
    "sea",
    "ocean",
    "bay",
    "gulf",
    "strait",
    "peninsula",
    "desert",
    "forest",
    "park",
    "street",
    "avenue",
    "boulevard",
    "road",
    "bridge",
    "airport",
    "port",
    "harbor",
    "station",
    "square",
    "plaza",
    "cathedral",
    "church",
    "mosque",
    "temple",
    "canton",
    "republic",
    "kingdom",
    "empire",
    "territory",
    "colony",
    "prefecture",
    "coast",
    "cape",
    "creek",
    "plateau",
    "basin",
    "gorge",
    "glacier",
    "springs",
    "falls",
    "heights",
    "hills",
    "reef",
    "trench",
    "rift",
    "ridge",
    "atoll",
    "fjord",
    "lagoon",
    "oasis",
    "crater",
    "caldera",
    "volcano",
    "geyser",
    "dune",
    "dunes",
    "canyon",
    "ravine",
    "delta",
    // German/Swiss place terms
    "stadt",
    "altstadt",
    "stadtteil",
    "gemeinde",
    "bezirk",
    "kanton",
    "bundesland",
    "tal",
    "see",
    "fluss",
    "berg",
    "gebirge",
    "wald",
    "feld",
    "hafen",
    "platz",
    "strasse",
    "gasse",
    "brücke",
];

/// Words that indicate a concept/thing rather than a person.
const CONCEPT_INDICATORS: &[&str] = &[
    "operation",
    "project",
    "protocol",
    "system",
    "theory",
    "model",
    "algorithm",
    "process",
    "method",
    "technique",
    "architecture",
    "framework",
    "standard",
    "specification",
    "extension",
    "instruction",
    "interface",
    "register",
    "memory",
    "processor",
    "compiler",
    "hardware",
    "software",
    "network",
    "security",
    "data",
    "device",
    "machine",
    "engine",
    "design",
    "platform",
    "application",
    "performance",
    "technology",
    "revolution",
    "expedition",
    "movement",
    "treaty",
    "agreement",
    "conference",
    "summit",
    "war",
    "battle",
    "crisis",
    "economy",
    "market",
    "trade",
    "policy",
    "reform",
    "law",
    "act",
    "code",
    "plan",
    "program",
    "initiative",
    "campaign",
    "era",
    "period",
    "age",
    "century",
    "empire",
    "kingdom",
    "republic",
    "dynasty",
    "civilization",
    "culture",
    "language",
    "religion",
    "philosophy",
    "science",
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "medicine",
    "engineering",
    "computing",
    "research",
    "study",
    "analysis",
    "review",
    "report",
    "survey",
    "index",
    "ranking",
    "overview",
    "speech",
    "building",
    "palace",
    "castle",
    "tower",
    "hall",
    "manor",
    "abbey",
    "priory",
    "monument",
    "memorial",
    "museum",
    "gallery",
    "theatre",
    "theater",
    "cinema",
    "stadium",
    "arena",
    "wine",
    "journal",
    "gazette",
    "times",
    "post",
    "herald",
    "tribune",
    "observer",
    "chronicle",
    "telegraph",
    "mail",
    "press",
    "media",
    "news",
    "award",
    "prize",
    "medal",
    "trophy",
    "coordinates",
    "nano",
    "pigs",
    "court",
    "settlements",
    "rankings",
    "education",
    "europa",
    "available",
    "algebra",
    "theorem",
    "equation",
    "equations",
    "hypothesis",
    "paradox",
    "principle",
    "principles",
    "lemma",
    "conjecture",
    "axiom",
    "postulate",
    "formula",
    "calculus",
    "geometry",
    "topology",
    "mechanics",
    "dynamics",
    "thermodynamics",
    "relativity",
    "electromagnetism",
    "optics",
    "acoustics",
    "index",
    "report",
    "aftermath",
    "spectrum",
    "distribution",
    "function",
    "transform",
    "operator",
    "integral",
    "derivative",
    "approximation",
    "simulation",
    "optimization",
    "iteration",
    "recursion",
    "complexity",
    "automaton",
    "automata",
    "grammar",
    "syntax",
    "semantics",
    "logic",
    "inference",
    "regression",
    "classification",
    "clustering",
    "manifold",
    "invariant",
    "symmetry",
    "tensor",
    "vector",
    "matrix",
    "polynomial",
    "series",
    "sequence",
    "convergence",
    "divergence",
    "inequality",
    "identity",
    "duality",
    "correspondence",
    "isomorphism",
    "homomorphism",
    "morphism",
    "computer",
    "computers",
    "scientist",
    "scientists",
    "engineers",
    "engines",
    "rockets",
    "episode",
    "episodes",
    "podcast",
    "podcasts",
    "blog",
    "magazine",
    "newspaper",
    "subdivision",
    "subdivisions",
    "governance",
    "house",
    "layer",
    "net",
    "effect",
    "rule",
    "problem",
    "river",
    "lake",
    "mountain",
    "valley",
    "bridge",
    "gate",
    "square",
    "street",
    "road",
    "avenue",
    "park",
    "garden",
    "library",
    "church",
    "cathedral",
    "temple",
    "mosque",
    "school",
    "college",
    "cup",
    "league",
    "championship",
    "congress",
    "council",
    "committee",
    "board",
    "bureau",
    "office",
    "department",
    "ministry",
    "canal",
    "cockade",
    "cockades",
    "history",
    "lectures",
    "boom",
    "citizenship",
    "curtain",
    "coat",
    "wars",
    "army",
    "navy",
    "fleet",
    "regiment",
    "cavalry",
    "infantry",
    "artillery",
    "cuirassiers",
    "dragoons",
    "guards",
    "brigade",
    "corps",
    "division",
    "energy",
    "reviews",
    "theoretical",
    "experimental",
    "practical",
    "peace",
    "truce",
    "armistice",
    "legacy",
    "heritage",
    "tradition",
    "anthem",
    "flag",
    "seal",
    "crest",
    "heraldry",
    "doctrine",
    "manifesto",
    "declaration",
    "proclamation",
    "charter",
    "constitution",
    "amendment",
    "referendum",
    "elections",
    "massacre",
    "siege",
    "rebellion",
    "revolt",
    "uprising",
    "resistance",
    "occupation",
    "liberation",
    "invasion",
    "conquest",
    "annexation",
    "collapse",
    "dissolution",
    "drought",
    "disc",
    "disk",
    "matter",
    "galaxies",
    "galaxy",
    "stars",
    "planets",
    "atoms",
    "molecules",
    "particles",
    "electrons",
    "neutrons",
    "photons",
    "fear",
    "belief",
    "commerce",
    "patch",
    "clause",
    "transfers",
    "appointments",
    "ships",
    "people",
    "places",
    "things",
    "centuries",
    "decades",
    "lands",
    "territories",
    "seas",
    "waters",
    "forests",
    "fields",
    "structures",
    "systems",
    "networks",
    "connections",
    "relationships",
    "patterns",
    "formations",
    "compositions",
    "measurements",
    "observations",
    "discoveries",
    "innovations",
    "inventions",
    "developments",
    "experiments",
    "calculations",
    "constructions",
    "productions",
    "distributions",
    "collections",
    "selections",
    "reactions",
    "interactions",
    "operations",
    "transitions",
    "transformations",
    "migrations",
    "expansions",
    "contractions",
    "oscillations",
    "fluctuations",
    "configurations",
    "classifications",
    "representations",
    "implementations",
    "applications",
    "interpretations",
    "considerations",
    "determinations",
    "contributions",
    "communications",
    "negotiations",
    "celebrations",
    "demonstrations",
    "organizations",
    "civilizations",
    "institutions",
    "populations",
    "generations",
    "publications",
    "regulations",
    "violations",
    "competitions",
    "exhibitions",
    "dimensions",
    "conditions",
    "positions",
    "propositions",
    "traditions",
    "ambitions",
    "expeditions",
    "missions",
    "emissions",
    "permissions",
    "submissions",
    "commissions",
    "caliphate",
    "encyclopedia",
    "worst",
    "best",
    "domain",
    "domains",
    "malicious",
    "cognitive",
    "semantic",
    "lexical",
    "syntactic",
    "phonetic",
    "galactic",
    "cosmic",
    "planetary",
    "stellar",
    "orbital",
    "magnetic",
    "electric",
    "acoustic",
    "atomic",
    "quantum",
    "digital",
    "analog",
    "binary",
    "linear",
    "nonlinear",
    "scalar",
    "differential",
    "integral",
    "algebraic",
    "geometric",
    "topological",
    "stochastic",
    "deterministic",
    "probabilistic",
    "asymptotic",
    "environmental",
    "ecological",
    "biological",
    "geological",
    "archaeological",
    "anthropological",
    "philosophical",
    "theological",
    "ideological",
    "mythological",
    "sociological",
    "psychological",
    "cultural",
    "political",
    "economic",
    "historical",
    "geographical",
    "mathematical",
    "scientific",
    "technological",
    "industrial",
    "agricultural",
    "commercial",
    "financial",
    "statistical",
    "analytical",
    "nautical",
    "medieval",
    "colonial",
    "imperial",
    "papal",
    "feudal",
    "tribal",
    "ethnic",
    "demographic",
    "diplomatic",
    "military",
    "naval",
    "aerial",
    "strategic",
    "tactical",
    "defensive",
    "offensive",
    "continental",
    "transatlantic",
    "mediterranean",
    "byzantine",
    "ottoman",
    "catholic",
    "protestant",
    "orthodox",
    "islamic",
    "muslim",
    "christian",
    "jewish",
    "buddhist",
    "hindu",
    "fellow",
    "fellows",
    "caliphate",
    "sultanate",
    "emirate",
    "khanate",
    "shogunate",
    "speech",
    "speeches",
    "awardees",
    "prizes",
    "chronicles",
    "divergence",
    "hippie",
    "containers",
    // Added 2026-02-24: generic concepts found in DB cleanup
    "archaeologist",
    "atheist",
    "botanist",
    "complexity",
    "computability",
    "correspondence",
    "distance",
    "economist",
    "equivalence",
    "homosexuality",
    "nationalist",
    "naturalist",
    "orientalist",
    "pathologist",
    "patience",
    "playlist",
    "prominence",
    "propagandist",
    "realist",
    "reconnaissance",
    "royalist",
    "temperature",
    "tourist",
    "stagecoaches",
    "committees",
    "carabineers",
    "constant",
    "constants",
    "phenomenon",
    "phenomena",
    "corollary",
    "postulates",
    "notation",
    "expansion",
    "approximations",
    "factorization",
    "decomposition",
    "interpolation",
    "extrapolation",
    "classification",
    "formulation",
    "generalization",
    "specialization",
    "normalization",
    "regularization",
    "discretization",
    "linearization",
    "diagonalization",
];

/// Common organization suffixes for entity classification.
const ORG_INDICATORS: &[&str] = &[
    "inc",
    "ltd",
    "llc",
    "corp",
    "corporation",
    "company",
    "co",
    "group",
    "gmbh",
    "ag",
    "sa",
    "foundation",
    "institute",
    "university",
    "college",
    "school",
    "academy",
    "association",
    "society",
    "organization",
    "organisation",
    "council",
    "committee",
    "commission",
    "agency",
    "bureau",
    "department",
    "ministry",
    "bank",
    "fund",
    "alliance",
    "federation",
    "union",
    "league",
    "team",
    "club",
    "party",
    "network",
    "lab",
    "labs",
    "laboratory",
    "laboratories",
    "technologies",
    "systems",
    "solutions",
    "services",
    "industries",
    "enterprises",
    "senate",
    "parliament",
    "congress",
    "centre",
    "center",
    "forces",
    "corps",
    "command",
    "authority",
    "tribunal",
    "court",
    "exchange",
    "trust",
    "board",
    "observatory",
    "conservatory",
    "seminary",
    "consortium",
    "cooperative",
];

const ABBREVIATIONS: &[&str] = &[
    "dr", "mr", "mrs", "ms", "prof", "jr", "sr", "inc", "ltd", "co", "corp", "vs", "etc", "al",
    "approx", "dept", "est", "vol", "fig", "ref", "st", "ave", "blvd",
];

const GERMAN_MONTHS: &[(&str, u32)] = &[
    ("januar", 1),
    ("februar", 2),
    ("märz", 3),
    ("april", 4),
    ("mai", 5),
    ("juni", 6),
    ("juli", 7),
    ("august", 8),
    ("september", 9),
    ("oktober", 10),
    ("november", 11),
    ("dezember", 12),
];

const FRENCH_MONTHS: &[(&str, u32)] = &[
    ("janvier", 1),
    ("février", 2),
    ("mars", 3),
    ("avril", 4),
    ("mai", 5),
    ("juin", 6),
    ("juillet", 7),
    ("août", 8),
    ("septembre", 9),
    ("octobre", 10),
    ("novembre", 11),
    ("décembre", 12),
];

const ITALIAN_MONTHS: &[(&str, u32)] = &[
    ("gennaio", 1),
    ("febbraio", 2),
    ("marzo", 3),
    ("aprile", 4),
    ("maggio", 5),
    ("giugno", 6),
    ("luglio", 7),
    ("agosto", 8),
    ("settembre", 9),
    ("ottobre", 10),
    ("novembre", 11),
    ("dicembre", 12),
];

pub fn detect_language(text: &str) -> Language {
    let tokens: HashSet<String> = text
        .split(|c: char| !c.is_alphanumeric() && c != '\'' && !"äöüßéèêàùñìò".contains(c))
        .filter(|w| w.len() > 1)
        .map(|w| w.to_lowercase())
        .collect();
    let count = |list: &[&str]| -> usize { list.iter().filter(|w| tokens.contains(**w)).count() };
    let scores = [
        (Language::English, count(STOP_WORDS_EN)),
        (Language::German, count(STOP_WORDS_DE)),
        (Language::French, count(STOP_WORDS_FR)),
        (Language::Italian, count(STOP_WORDS_IT)),
        (Language::Spanish, count(STOP_WORDS_ES)),
    ];
    scores
        .iter()
        .max_by_key(|(_, s)| *s)
        .filter(|(_, s)| *s >= 2)
        .map(|(l, _)| *l)
        .unwrap_or(Language::Unknown)
}

pub fn split_sentences(text: &str) -> Vec<String> {
    let abbrevs: HashSet<&str> = ABBREVIATIONS.iter().copied().collect();
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        let ch = chars[i];
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            if ch == '.' {
                let before: String = current
                    .trim_end_matches('.')
                    .chars()
                    .rev()
                    .take_while(|c| c.is_alphabetic())
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                if abbrevs.contains(before.to_lowercase().as_str()) {
                    i += 1;
                    continue;
                }
                if before.len() <= 1 {
                    i += 1;
                    continue;
                }
                if i + 1 < len && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    continue;
                }
            }
            let trimmed = current.trim().to_string();
            if trimmed.len() > 3 {
                sentences.push(trimmed);
            }
            current.clear();
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if trimmed.len() > 3 {
        sentences.push(trimmed);
    }
    sentences
}

pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'' && !"äöüßéèêà".contains(c))
        .filter(|w| w.len() > 1)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Check if an entity passes minimum quality filters.
/// Single-word terms too generic to be useful entities.
const GENERIC_SINGLE_WORDS: &[&str] = &[
    "output",
    "input",
    "overview",
    "subscribe",
    "download",
    "upload",
    "search",
    "login",
    "logout",
    "submit",
    "cancel",
    "delete",
    "update",
    "create",
    "select",
    "click",
    "default",
    "example",
    "test",
    "demo",
    "sample",
    "note",
    "notes",
    "summary",
    "introduction",
    "conclusion",
    "appendix",
    "table",
    "figure",
    "chapter",
    "section",
    "part",
    "volume",
    "issue",
    "item",
    "list",
    "array",
    "object",
    "string",
    "integer",
    "float",
    "boolean",
    "function",
    "class",
    "method",
    "variable",
    "parameter",
    "argument",
    "return",
    "import",
    "export",
    "module",
    "package",
    "library",
    "framework",
    "august",
    "march",
    "may",
    "imperial",
    "modern",
    "various",
    "early",
    "late",
    "first",
    "second",
    "third",
    "new",
    "old",
    "great",
    "large",
    "small",
    "high",
    "low",
    "long",
    "short",
    "major",
    "minor",
    "general",
    "special",
    "national",
    "international",
    "western",
    "eastern",
    "northern",
    "southern",
    "central",
    "united",
    "free",
    "open",
    "public",
    "private",
    "former",
    "current",
    "original",
    "standard",
    "common",
    "popular",
    "traditional",
    "classical",
    "main",
    "time",
    "rise",
    "lane",
    "mote",
    "acre",
    "blue",
    "vale",
    "done",
    "idiot",
    "dissect",
    "substitute",
    "layout",
    "summer",
    "seventeen",
    "limbo",
    "coalition",
    "evolution",
    "dissipation",
    "transition",
    "engineering",
    "missing",
    "introducing",
    "translate",
    "browse",
    "altogether",
    "likewise",
    "chair",
    "many",
    "several",
    "certain",
    "according",
    "following",
    "including",
    "related",
    "based",
    "known",
    "called",
    "named",
    "used",
    "given",
    "made",
    "found",
    "however",
    "although",
    "despite",
    "during",
    "between",
    "within",
    "without",
    "after",
    "before",
    "since",
    "until",
    "above",
    "below",
    "under",
    "over",
    "through",
    "among",
    "against",
    "towards",
    "along",
    "different",
    "important",
    "possible",
    "available",
    "necessary",
    "particular",
    "significant",
    "similar",
    "specific",
    "various",
    "recent",
    "previous",
    "additional",
    "potential",
    "certain",
    "particular",
    "primary",
    "secondary",
    "final",
    "initial",
    "entire",
    "basic",
    "typical",
    "separate",
    "individual",
    "alternative",
    "corresponding",
    "equivalent",
    "respective",
    "subsequent",
    "remaining",
    "existing",
    "proposed",
    "suggested",
    "described",
    "considered",
    "required",
    "expected",
    "observed",
    "obtained",
    "produced",
    "presented",
    "compared",
    "associated",
    "combined",
    "applied",
    "performed",
    "developed",
    "designed",
    "implemented",
    "established",
    "introduced",
    "published",
    "reported",
    "discussed",
    "explained",
    "defined",
    "determined",
    "provided",
    "represented",
    "illustrated",
    "demonstrated",
    "mentioned",
    "achieved",
    "maintained",
    "supported",
    "selected",
    "identified",
    "evaluated",
    "estimated",
    "measured",
    "calculated",
    "generated",
    "contributed",
    "contributed",
    "commemorated",
    "interpreted",
    "research",
    "journal",
    "review",
    "proceedings",
    "frequently",
    "evaluation",
    "shows",
    "advanced",
    "application",
    "applications",
    "computing",
    "market",
    "social",
    "energy",
    "division",
    "editing",
    "september",
    "october",
    "november",
    "december",
    "january",
    "february",
    "april",
    "june",
    "july",
    "nutshell",
    "peopling",
    "downloads",
    "reviews",
    "contributions",
    "variants",
    "printable",
    "sections",
    "references",
    "categories",
    "languages",
    "interactions",
    "navigation",
    "contents",
    "sources",
    "links",
    "options",
    "settings",
    "features",
    "results",
    "details",
    "items",
    "pages",
    "files",
    "images",
    "tables",
    "lists",
    "fight",
    "roots",
    "kings",
    "adage",
    "adieu",
    "african",
    "alpine",
    "always",
    "analog",
    "analyst",
    "ancient",
    "angry",
    "angular",
    "annual",
    "answer",
    "apart",
    "abstract",
    "active",
    "actual",
    "average",
    "aware",
    "broad",
    "capable",
    "complex",
    "compound",
    "complete",
    "critical",
    "crucial",
    "direct",
    "distinct",
    "diverse",
    "domestic",
    "dominant",
    "dramatic",
    "dynamic",
    "effective",
    "efficient",
    "elaborate",
    "elegant",
    "evident",
    "exact",
    "explicit",
    "external",
    "extreme",
    "formal",
    "frequent",
    "fundamental",
    "genuine",
    "global",
    "gradual",
    "hostile",
    "ideal",
    "identical",
    "implicit",
    "independent",
    "indirect",
    "infinite",
    "inherent",
    "innocent",
    "instant",
    "intense",
    "internal",
    "inverse",
    "isolated",
    "lateral",
    "legitimate",
    "liberal",
    "literal",
    "logical",
    "marginal",
    "massive",
    "mature",
    "maximum",
    "medieval",
    "medium",
    "mental",
    "minimal",
    "moderate",
    "molecular",
    "moral",
    "mutual",
    "native",
    "natural",
    "negative",
    "neutral",
    "noble",
    "nominal",
    "normal",
    "nuclear",
    "numerical",
    "obvious",
    "organic",
    "parallel",
    "partial",
    "passive",
    "permanent",
    "persistent",
    "physical",
    "plural",
    "polar",
    "portable",
    "positive",
    "precise",
    "profound",
    "progressive",
    "prominent",
    "proper",
    "proportional",
    "pure",
    "radical",
    "random",
    "rational",
    "raw",
    "realistic",
    "reasonable",
    "regular",
    "relevant",
    "remote",
    "rigid",
    "robust",
    "rough",
    "royal",
    "rural",
    "secular",
    "severe",
    "shallow",
    "sharp",
    "simple",
    "singular",
    "slight",
    "solar",
    "solid",
    "sparse",
    "spatial",
    "stable",
    "static",
    "steep",
    "strict",
    "structural",
    "subtle",
    "sudden",
    "sufficient",
    "superior",
    "supreme",
    "symbolic",
    "synthetic",
    "temporal",
    "terminal",
    "thermal",
    "tight",
    "tiny",
    "total",
    "tough",
    "trivial",
    "tropical",
    "ultimate",
    "uniform",
    "unique",
    "universal",
    "urban",
    "urgent",
    "valid",
    "vertical",
    "virtual",
    "visible",
    "visual",
    "vital",
    "volatile",
    "wealthy",
    // Additional single-word noise
    "absolute",
    "acoustic",
    "adventure",
    "algorithm",
    "capture",
    "cockade",
    "cookery",
    "highway",
    "viewer",
    "reference",
    "reflections",
    "breakup",
    "breakout",
    "emperors",
    "subtitle",
    "aftermath",
    "wave",
    "edge",
    "fame",
    "flag",
    "gate",
    "horn",
    "look",
    "mind",
    "coat",
    "tool",
    // Short single-word noise that keeps appearing
    "area",
    "bang",
    "blog",
    "book",
    "bulk",
    "call",
    "case",
    "code",
    "cold",
    "drop",
    "dust",
    "fall",
    "fine",
    "food",
    "glow",
    "guru",
    "halt",
    "hand",
    "keen",
    "know",
    "laws",
    "lazy",
    "life",
    "link",
    "load",
    "lust",
    "mock",
    "moon",
    "much",
    "neon",
    "past",
    "path",
    "post",
    "push",
    "rear",
    "role",
    "roof",
    "sept",
    "sons",
    "thus",
    "tour",
    "true",
    "west",
    "wiki",
    "zero",
    "icon",
    "flow",
    "slim",
    "play",
    "ones",
    "aeon",
    "bold",
    "mine",
    "chap",
    // Generic nouns/adjectives that appear as single-word false entities
    "atoms",
    "audio",
    "beast",
    "birds",
    "black",
    "block",
    "boots",
    "canal",
    "cheap",
    "check",
    "chips",
    "civil",
    "clubs",
    "coral",
    "crime",
    "crown",
    "daily",
    "death",
    "delta",
    "draft",
    "drawn",
    "drove",
    "dutch",
    "earth",
    "edict",
    "elder",
    "entry",
    "error",
    "essay",
    "faith",
    "farms",
    "fiber",
    "fifth",
    "flaws",
    "fluid",
    "foods",
    "fuzzy",
    "games",
    "gamma",
    "glass",
    "globe",
    "golem",
    "greek",
    "green",
    "guard",
    "guide",
    "guild",
    "haiku",
    "happy",
    "harsh",
    "hence",
    "hindu",
    "honor",
    "house",
    "human",
    "icons",
    "inner",
    "light",
    "lines",
    "looks",
    "lords",
    "maple",
    "mania",
    "mayor",
    "means",
    "media",
    "merit",
    "mines",
    "motto",
    "music",
    "names",
    "noise",
    "norse",
    "north",
    "opera",
    "order",
    "paper",
    "peace",
    "penny",
    "phase",
    "photo",
    "place",
    "posts",
    "pound",
    "power",
    "price",
    "pride",
    "print",
    "probe",
    "proto",
    "psalm",
    "radio",
    "rapid",
    "realm",
    "reign",
    "relay",
    "risks",
    "roman",
    "round",
    "rumor",
    "scope",
    "shape",
    "shift",
    "siege",
    "sigma",
    "sixty",
    "smart",
    "smoke",
    "south",
    "stage",
    "stars",
    "steam",
    "stone",
    "story",
    "study",
    "sugar",
    "sunni",
    "swiss",
    "taste",
    "tatar",
    "terra",
    "tests",
    "texts",
    "thief",
    "times",
    "torus",
    "trade",
    "turks",
    "usage",
    "value",
    "vedic",
    "visas",
    "welsh",
    "wheel",
    "while",
    "winds",
    "words",
    "world",
    "years",
    "eulogy",
    "losses",
    "memory",
    "statue",
    "encode",
    "series",
    "causes",
    "period",
    "center",
    "postal",
    "carbon",
    "vortex",
    "winter",
    "ethnic",
    "upland",
    // Added 2026-02-23: more generic words appearing as false concepts
    // Month names as standalone entities
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "annals",
    "alternatively",
    "asylum",
    "assume",
    "anyone",
    "around",
    "agent",
    "alpha",
    "atomic",
    "attack",
    "author",
    "bacon",
    "binary",
    "except",
    "inference",
    "labour",
    "lowell",
    "matador",
    "medicine",
    "members",
    "origin",
    "pirate",
    "mirror",
    "making",
    "others",
    "binder",
    "action",
    "adding",
    "agile",
    "reader",
    "reason",
    "romanesque",
    "speculum",
    "strata",
    "sunset",
    "terror",
    "thread",
    "spread",
    "reading",
    "vortrag",
    "iconoclasts",
    "complicite",
    "counterexamples",
    "parallelogram",
    "semipalatinsk",
    "wiedervereinigung",
    // Added 2026-02-23: more noise from DB analysis
    "abundance",
    "academia",
    "accademia",
    "admissibility",
    "advances",
    "transactions",
    "zeitschrift",
    "speculum",
    // Added 2026-02-23 round 2: generic nouns/adjectives from DB cleanup
    "generator",
    "toolkit",
    "twin",
    "theory",
    "peasantry",
    "geopolitics",
    "hardware",
    "formula",
    "discrete",
    "cornerstones",
    "railways",
    "model",
    "concepts",
    "diagram",
    "sequence",
    "geschichte",
    "studium",
    "schule",
    "kernelemente",
    "ottoman",
    "germans",
    "salle",
    "cliometrica",
    "oligocene",
    "mappae",
    "typically",
    "underscores",
    "coherent",
    "preliminaries",
    "refrigerator",
    "circle",
    "nations",
    "cavalry",
    "poverty",
    "pharmacy",
    "climate",
    "transmission",
    "priest",
    "variety",
    "paradigms",
    "consistency",
    "connectionist",
    "cuprate",
    "diplomacy",
    "primates",
    "oecologia",
    "gallipoli",
    "turkic",
    "biogeography",
    "messiah",
    "neurodynamics",
    "janissaries",
    "malaria",
    "delivering",
    // Added 2026-02-23 round 3: more noise from DB analysis
    "database",
    "conceptually",
    "convolutional",
    "uncertainty",
    "thrift",
    "conjugate",
    "misfortunes",
    "chronicle",
    "proton",
    "surface",
    "frontline",
    "decision",
    "climatology",
    "peasant",
    "purchase",
    "dandelion",
    "tetrarchy",
    "nestorianism",
    // Added 2026-02-23 round 4: more single-word noise from DB cleanup
    "expansion",
    "evaporate",
    "training",
    "elephant",
    "mining",
    "workers",
    "revolution",
    "governors",
    "clusters",
    "russian",
    "despotate",
    "infantry",
    "zufriedenheit",
    "küstenmorphologie",
    "naturwissenschaftler",
    "affair",
    "agriculture",
    "aircraft",
    "algebraic",
    "alternatives",
    "apprenticeship",
    "arguments",
    "assembly",
    "assess",
    "assistant",
    "asteroid",
    "astronaut",
    "astronomical",
    "astrophysics",
    "atmosphere",
    "attacks",
    "attention",
    "attitude",
    "autobiography",
    "bachelor",
    "bandwagon",
    "baroque",
    "behavior",
    "beneath",
    "difficulty",
    "hydrogen",
    "magnesium",
    "aneutronic",
    "christendom",
    "collider",
    "piezoelectric",
    "francophone",
    "sociology",
    "microbiology",
    "electronics",
    "countess",
    "interstellar",
    "provincial",
    "approximate",
    "challenge",
    "country",
    "impulse",
    "accident",
    "marine",
    "forget",
    "thank",
    "police",
    "faster",
    "thereafter",
    "peasantry",
    "geopolitics",
    "cornerstones",
    "railways",
    "studium",
    "schule",
    "kernelemente",
    "salle",
    "oligocene",
    "typically",
    "underscores",
    "coherent",
    "preliminaries",
    "refrigerator",
    "circle",
    "nations",
    "poverty",
    "pharmacy",
    "climate",
    "transmission",
    "priest",
    "variety",
    "paradigms",
    "consistency",
    "connectionist",
    "cuprate",
    "diplomacy",
    "primates",
    "biogeography",
    "messiah",
    "neurodynamics",
    "janissaries",
    "malaria",
    // Added 2026-02-23 round 5: capitalized common English words appearing as false entities
    "access",
    "across",
    "again",
    "another",
    "because",
    "finally",
    "however",
    "indeed",
    "instead",
    "later",
    "longer",
    "often",
    "still",
    "therefore",
    "today",
    "together",
    // Added 2026-02-23 round 6: more generic words from DB cleanup
    "beauty",
    "congress",
    "theology",
    "slavery",
    "dispel",
    "swamp",
    "conversely",
    "remnants",
    "fokker",
    "gegenwart",
    "komfortrouten",
    "conventionnels",
    "deepest",
    "predecessor",
    "bronze",
    // Added 2026-02-23 round 7: more noise from DB analysis
    "continent",
    "hemisphere",
    "latitude",
    "longitude",
    "altitude",
    "gradient",
    "amplitude",
    "frequency",
    "wavelength",
    "bandwidth",
    "voltage",
    "resistance",
    "capacitance",
    "inductance",
    "momentum",
    "velocity",
    "acceleration",
    "friction",
    "gravity",
    "density",
    "pressure",
    "viscosity",
    "elasticity",
    "conductivity",
    "permeability",
    "diffusion",
    "absorption",
    "emission",
    "reflection",
    "refraction",
    "diffraction",
    "polarization",
    "resonance",
    "vibration",
    "turbulence",
    "corrosion",
    "erosion",
    "sediment",
    "substrate",
    "catalyst",
    "reagent",
    "solvent",
    "polymer",
    "isotope",
    "neutron",
    "electron",
    "photon",
    "nucleus",
    "chromosome",
    "genome",
    "protein",
    "enzyme",
    "antibody",
    "pathogen",
    "parasite",
    "predator",
    "herbivore",
    "omnivore",
    "ecosystem",
    "biosphere",
    "lithosphere",
    "atmosphere",
    "hydrosphere",
    "cryosphere",
    "tectonics",
    "volcanic",
    "seismic",
    "glacier",
    "watershed",
    "aquifer",
    "estuary",
    "archipelago",
    "peninsula",
    "isthmus",
    "plateau",
    "steppe",
    "tundra",
    "savanna",
    "grassland",
    "wetland",
    "marshland",
    "woodland",
    "undergrowth",
    "canopy",
    "vegetation",
    "foliage",
    "specimen",
    "taxonomy",
    "morphology",
    "physiology",
    "pathology",
    "neurology",
    "cardiology",
    "oncology",
    "radiology",
    "pediatrics",
    "obstetrics",
    "orthopedics",
    "dermatology",
    "psychiatry",
    "anesthesia",
    "prognosis",
    "diagnosis",
    "symptom",
    "syndrome",
    "pandemic",
    "epidemic",
    "endemic",
    "quarantine",
    "vaccine",
    "antibiotic",
    "therapeutic",
    "palliative",
    "prosthetic",
    "surgical",
    "clinical",
    "forensic",
    "testimony",
    "verdict",
    "indictment",
    "prosecution",
    "defendant",
    "plaintiff",
    "jurisdiction",
    "legislation",
    "arbitration",
    "mediation",
    "litigation",
    "statute",
    "ordinance",
    "sovereign",
    "monarchy",
    "autocracy",
    "aristocracy",
    "plutocracy",
    "theocracy",
    "bureaucracy",
    "meritocracy",
    "oligarchy",
    "anarchy",
    "communism",
    "capitalism",
    "socialism",
    "fascism",
    "nationalism",
    "imperialism",
    "colonialism",
    "feudalism",
    "mercantilism",
    "liberalism",
    "conservatism",
    "populism",
    "authoritarianism",
    "totalitarianism",
    "secularism",
    "fundamentalism",
    "extremism",
    "radicalism",
    "terrorism",
    "insurgency",
    "guerrilla",
    "propaganda",
    "censorship",
    "espionage",
    "sabotage",
    "blockade",
    "embargo",
    "sanction",
    "reparations",
    "restitution",
    "compensation",
    "subsidy",
    "tariff",
    "quota",
    "monopoly",
    "oligopoly",
    "cartel",
    "conglomerate",
    "merger",
    "acquisition",
    "bankruptcy",
    "insolvency",
    "inflation",
    "deflation",
    "recession",
    "depression",
    "stagnation",
    "austerity",
    "stimulus",
    "devaluation",
    "speculation",
    "dividend",
    "commodity",
    "derivative",
    "portfolio",
    "collateral",
    "mortgage",
    "annuity",
    "pension",
    "endowment",
    "scholarship",
    "fellowship",
    "internship",
    "apprentice",
    "curriculum",
    "pedagogy",
    "syllabus",
    "dissertation",
    "manuscript",
    "anthology",
    "monograph",
    "treatise",
    "pamphlet",
    "broadsheet",
    "tabloid",
    "editorial",
    "obituary",
    "eulogy",
    "satire",
    "parody",
    "allegory",
    "metaphor",
    "analogy",
    "paradox",
    "irony",
    "rhetoric",
    "discourse",
    "dialectic",
    "epistemology",
    "ontology",
    "metaphysics",
    "aesthetics",
    "hermeneutics",
    "phenomenology",
    "existentialism",
    "pragmatism",
    "empiricism",
    "rationalism",
    "stoicism",
    "nihilism",
    "skepticism",
    "relativism",
    "determinism",
    "dualism",
    "monism",
    "pluralism",
    "materialism",
    "idealism",
    "realism",
    "nominalism",
    "utilitarianism",
    "hedonism",
    "altruism",
    "pacifism",
    "anarchism",
    "syndicalism",
    // Added 2026-02-24: more generic words from DB cleanup
    "honour",
    "genius",
    "sterile",
    "dispute",
    "cuisine",
    "quest",
    "oxygen",
    "décor",
    "revolutionary",
    "recurrence",
    "experience",
    "poetry",
    "overhead",
    "likewise",
    "buggy",
    "undergraduate",
    "survey",
    "doctrine",
    "troops",
    "factory",
    "decline",
    "oblique",
    "semiconductors",
    "decrees",
    "enrich",
    "camels",
    "argent",
    "patriarch",
    "sovereign",
    "prolog",
    "rivalry",
    "decree",
    "thermae",
    // Added 2026-02-24: more noise from DB cleanup
    "afterword",
    "afficionado",
    "additionally",
    "invented",
    "condensate",
    "waterfall",
    "diciembre",
    "marseillois",
    "bonapartist",
    "bektashi",
    "tecnología",
    "rainbow",
    "insight",
    "profile",
    "version",
    "utilize",
    "unknown",
    "water",
    "saturday",
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    // Added 2026-02-24 round 2: more noise from DB cleanup
    "econocide",
    "panamax",
    "incense",
    "correspondents",
    "lighthouse",
    "conversion",
    "sarcophagus",
    "landgraviate",
    "devshirme",
    "reconnaissance",
    "existence",
    "residence",
    // Added 2026-02-25: more generic words from DB cleanup
    "butterfly",
    "conformational",
    "governorates",
    "transformer",
    "congregational",
    "institutional",
    "magistrates",
    "nineteenth",
    "thirteenth",
    "goldhill",
    "westwood",
    // Added 2026-02-25 round 2: more noise from DB cleanup
    "hegemon",
    "emission",
    "graphen",
    "locustella",
    "gleichmässig",
    "wertschöpfung",
    "antinomien",
    "mathesi",
    // Added 2026-02-25 round 3: more noise from DB cleanup
    "residence",
    "hyperbolic",
    "carnage",
    "weaker",
    "multicore",
    "superweapons",
    "symbolist",
    "egyptomania",
    "albanians",
    "sustainability",
    "marble",
    "inspirefest",
    "confoederatio",
    "uranoscopia",
    "mechanik",
    "netflix",
    // Added 2026-02-24 (cron): more generic words from DB cleanup
    "reduce",
    "gauge",
    "dipole",
    "prose",
    "ultra",
    "either",
    "stormy",
    "artist",
    "narrow",
    "right",
    "bound",
    "color",
    "minute",
    "jewish",
    "giant",
    "nobody",
    "essai",
    // Added 2026-02-24 (cron round 2): more noise from DB cleanup
    "academic",
    "acids",
    "adventures",
    "afterwards",
    "algebra",
    "algorithms",
    "altering",
    "america",
    "amplitude",
    "analysis",
    "analytik",
    "ancients",
    "angeles",
    "antarctic",
    "antiquity",
    "apocalypse",
    "afterlife",
    "diseases",
    "degenerate",
    "economic",
    "empires",
    "passion",
    "transformers",
    "majesty",
    "combatants",
    "contributors",
    "merchants",
    "buffer",
    "tragedy",
    "onomatopoeia",
    "citizenship",
    "milestone",
    "estimate",
    "supplementum",
    "besides",
    "propulsion",
    "runaway",
    "consumers",
    "mathematicae",
    "aufgabe",
    "stochastic",
    "personal",
    "algorithmica",
    "mystery",
    "percussion",
    "performer",
    "biographie",
    "periodico",
    "immediate",
    "european",
    "venetians",
    "banknotenserie",
    "superintendent",
    "presbyterians",
    "terahertz",
    "apologia",
    "testnet",
    "discourse",
    "structure",
    "portrait",
    "frequency",
    // Added 2026-02-24: generic English words found as false concept entities
    "photographs",
    "fundamentals",
    "succession",
    "instruments",
    "formulae",
    "engines",
    "practitioners",
    "spectroscopy",
    "discoveries",
    "astronomers",
    "approaches",
    "artifacts",
    "selenium",
    "aristocrats",
    "bodyguard",
    "interferometer",
    "conjectures",
    "curriculum",
    "variational",
    "biggest",
    "breakthroughs",
    "omega",
    "autumn",
    "yeast",
    "attempt",
    // Added 2026-02-24 (brain cleaner round 5)
    "celebrity",
    "formulas",
    "harmonic",
    "supersonic",
    "attosecond",
    "telegraph",
    "temperature",
    "computation",
    "ballistics",
    "admiralty",
];

/// Trailing words that indicate bad phrase boundary (Wikipedia sentence fragments).
const TRAILING_JUNK: &[&str] = &[
    "if",
    "in",
    "the",
    "a",
    "of",
    "and",
    "or",
    "at",
    "to",
    "for",
    "by",
    "from",
    "with",
    "on",
    "is",
    "are",
    "was",
    "were",
    "its",
    "his",
    "her",
    "their",
    "an",
    "as",
    "but",
    "not",
    "out",
    "since",
    "has",
    "had",
    "have",
    "been",
    "be",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "into",
    "than",
    "then",
    "also",
    "that",
    "this",
    "these",
    "those",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "how",
    "why",
    "what",
    "both",
    "such",
    "some",
    "all",
    "each",
    "every",
    // Month names — prevent "Person January" style entities
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    // Non-name trailing words found in DB analysis 2026-02-24
    "role",
    "wall",
    "regime",
    "root",
    "most",
    "results",
    "deduced",
    "investopedia",
    "navigable",
    "blockchain",
    "spektakel",
    // Citation/publisher/academic trailing fragments
    "some",
    "advances",
    "birds",
    "geometry",
    "notices",
    "proceedings",
    "magnitude",
    "revue",
    "praeger",
    "univ",
    "blackwell",
    "veit",
    "jahr",
    "sometimes",
    "expression",
    // Added 2026-02-24: more trailing junk from DB cleanup
    "died",
    "fellow",
    "member",
    "presenting",
    "optimized",
    "cook",
    "dictionary",
    "handbook",
    "notation",
    "annals",
    "software",
    "hardware",
    "colours",
    "pressure",
    "available",
    "women",
    "workers",
    "detectors",
    "studies",
    "cast",
    "reconsidering",
    "works",
    "nobelprize",
    "specifications",
    "methods",
    "atlas",
    "environment",
    "week",
    "machinery",
    "production",
    "conference",
    "intelligence",
    "history",
    "sorting",
    "quicksort",
    "astronomy",
    "pivot",
    "chart",
    "scientist",
    "mathematician",
    "physicist",
    "philosopher",
    "historian",
    "biologist",
    "chemist",
    "geologist",
    "anthropologist",
    "sociologist",
    "psychologist",
    "economist",
    "linguist",
    "archaeologist",
];

fn is_valid_entity(name: &str, etype: &str) -> bool {
    let trimmed = name.trim();

    // Length bounds
    if trimmed.len() < 3 || trimmed.len() > 100 {
        return false;
    }

    // Multi-word concepts over 40 chars are almost always citation/article title fragments
    if etype == "concept" && trimmed.contains(' ') && trimmed.len() > 40 {
        return false;
    }

    // Person names over 35 chars are typically citation fragments
    if etype == "person" && trimmed.len() > 35 {
        return false;
    }

    // Reject fragments where any word is a single letter (e.g. "B It", "K It", "Q Pa")
    {
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() >= 2
            && words
                .iter()
                .any(|w| w.len() == 1 && w.chars().next().is_some_and(|c| c.is_alphabetic()))
        {
            // Allow well-known patterns like "C++" or initials in names with 3+ words
            if words.len() <= 2 {
                return false;
            }
        }
    }

    // Structured types (dates, urls, emails, currency, number_unit) skip text-based filters
    if matches!(
        etype,
        "date" | "relative_date" | "url" | "email" | "currency" | "number_unit" | "year"
    ) {
        return true;
    }

    // Reject entities containing math/code symbols (likely formula fragments)
    if trimmed
        .chars()
        .any(|c| "Σ→←≈≤≥∈∀∃∫∂∇∆∞±÷×∝∑∏√∩∪⊂⊃⊆⊇∧∨¬⟨⟩{}[]|\\".contains(c))
    {
        return false;
    }

    // Reject citation-like patterns (e.g. "I:361 Robert Millikan", "Tacitus Annales IV.5")
    if trimmed.contains(':') && trimmed.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }

    // Reject slash-separated compound entities (e.g. "Karatsuba/Voronin", "Ocean/sea")
    // but allow known patterns like "AdS/CFT", "TCP/IP"
    if trimmed.contains('/') {
        let parts: Vec<&str> = trimmed.split('/').collect();
        if parts.len() == 2 && parts.iter().all(|p| p.len() > 3) {
            return false;
        }
    }

    // Reject entities with 5+ words unless they contain linking prepositions
    // (e.g. "University of California at Berkeley" is valid)
    {
        let word_count = trimmed.split_whitespace().count();
        if word_count >= 6 {
            return false;
        }
        if word_count == 5 {
            let lower_str = trimmed.to_lowercase();
            let lower_words: Vec<&str> = lower_str.split_whitespace().collect();
            let linking = [
                "of", "the", "de", "del", "di", "du", "von", "van", "la", "le", "at", "in", "for",
                "und", "and",
            ];
            let has_linking = lower_words.iter().any(|w| linking.contains(w));
            if !has_linking {
                return false;
            }
        }
    }

    // Reject Cyrillic-only entries (Wikipedia cross-language artifacts)
    if trimmed
        .chars()
        .all(|c| !c.is_ascii_alphabetic() || c.is_whitespace())
        && trimmed
            .chars()
            .any(|c| ('\u{0400}'..='\u{04FF}').contains(&c))
    {
        return false;
    }

    // Blacklist check (case-insensitive)
    let lower = trimmed.to_lowercase();
    if ENTITY_BLACKLIST.contains(&lower.as_str()) {
        return false;
    }

    // Reject single generic words
    if !lower.contains(' ') && GENERIC_SINGLE_WORDS.contains(&lower.as_str()) {
        return false;
    }

    // Reject single-word entities that look like truncated citation references (e.g. "Annu", "Beig", "Plut")
    // These are typically 4-char fragments ending in consonant clusters that aren't real words
    if !lower.contains(' ') && trimmed.len() == 4 && etype == "concept" {
        // If it's a 4-char capitalized word with default confidence, likely truncated
        let chars: Vec<char> = trimmed.chars().collect();
        if chars[0].is_uppercase()
            && chars[1..].iter().all(|c| c.is_lowercase())
            && !lower.ends_with('a')
            && !lower.ends_with('e')
            && !lower.ends_with('i')
            && !lower.ends_with('o')
            && !lower.ends_with('y')
        {
            // Words ending in consonants that are 4 chars and classified as concept are likely noise
            // (real 4-letter concepts like "Yoga" or "Dojo" end in vowels)
            return false;
        }
    }

    // Reject taxonomic/scientific family names (e.g. Candonidae, Baicaliinae, Spongillidae)
    if !lower.contains(' ') {
        let suffixes = [
            "aceae", "idae", "inae", "ales", "oidea", "iformes", "opsida",
            // Extinct animal genus suffixes
            "therium", "saurus", "pithecus", "cetus", "odon",
        ];
        if suffixes.iter().any(|s| lower.ends_with(s)) && trimmed.len() > 6 {
            return false;
        }
    }

    // Reject single-word person titles used alone (President, Queen, etc.)
    if !lower.contains(' ') && PERSON_TITLES.contains(&lower.as_str()) {
        return false;
    }

    // Reject single-word capitalized adjectives ending in common suffixes (e.g. "Equivariant", "Archaeological")
    if !lower.contains(' ') && etype == "concept" && trimmed.len() > 6 {
        let adj_suffixes = [
            "iant", "ible", "able", "ical", "ious", "eous", "ular", "ular", "atic", "etic", "otic",
            "ural", "inal", "idal", "imal", "ival",
        ];
        if adj_suffixes.iter().any(|s| lower.ends_with(s)) {
            return false;
        }
    }

    // Reject entities ending with trailing junk words
    if let Some(last_word) = lower.split_whitespace().last() {
        if lower.contains(' ') && TRAILING_JUNK.contains(&last_word) {
            return false;
        }
    }

    // Reject entities starting with trailing junk or verb/participle prefixes (leftover fragments)
    if let Some(first_word) = lower.split_whitespace().next() {
        if lower.contains(' ') && TRAILING_JUNK.contains(&first_word) {
            return false;
        }
        // Reject entities starting with past participles, gerunds, or verbs
        let leading_verbs: &[&str] = &[
            "born",
            "died",
            "introducing",
            "making",
            "creating",
            "building",
            "showing",
            "acquiring",
            "dissolving",
            "discovering",
            "combining",
            "starting",
            "ending",
            "increasing",
            "decreasing",
            "becoming",
            "getting",
            "having",
            "being",
            "doing",
            "going",
            "coming",
            "taking",
            "giving",
            "finding",
            "telling",
            "asking",
            "hidden",
            "broken",
            "forgotten",
            "chosen",
            "proven",
            "driven",
            "written",
            "radioactive",
            "geeks",
            "meanwhile",
            // Adverbs/conjunctions that start sentence fragments
            "additionally",
            "accordingly",
            "afterwards",
            "apparently",
            "closely",
            "conversely",
            "currently",
            "especially",
            "eventually",
            "henceforth",
            "nonetheless",
            "similarly",
            "specifically",
            "thereafter",
            "announces",
            "regardless",
            "perhaps",
            "certainly",
            "evidently",
            "presumably",
            "supposedly",
            "allegedly",
            "merely",
            "roughly",
            "briefly",
            "broadly",
            "collectively",
            "consequently",
            "exclusively",
            "formally",
            "fundamentally",
            "generally",
            "gradually",
            "historically",
            "independently",
            "initially",
            "internally",
            "ironically",
            "largely",
            "literally",
            "locally",
            "notably",
            "officially",
            "originally",
            "particularly",
            "physically",
            "politically",
            "potentially",
            "practically",
            "primarily",
            "privately",
            "professionally",
            "publicly",
            "rapidly",
            "recently",
            "relatively",
            "reportedly",
            "significantly",
            "strictly",
            "subsequently",
            "successfully",
            "technically",
            "temporarily",
            "traditionally",
            "typically",
            "ultimately",
            "unfortunately",
            "virtually",
            "properties",
            "environment",
            "environmental",
            "spaces",
            "balanced",
            "cosmic",
            "march",
            "miniature",
            "literacy",
            "regeneration",
            "democracy",
            "hospital",
            "discover",
            "evening",
            "historic",
            "refugee",
            "cryptographic",
            "functional",
            "optimized",
            "pioneering",
            "highly",
            "loved",
            "spartan",
            "variant",
            "cotton",
            "serial",
            "reflection",
            "historical",
            "medieval",
            "commemorations",
            "business",
            "using",
            "according",
            "including",
            "following",
            "based",
            "known",
            "called",
            "named",
            "related",
            "located",
            "described",
            "considered",
            "established",
            "published",
            // Academic/citation leading fragments
            "pattern",
            "cultural",
            "analysis",
            "glacier",
            "mathematic",
            "theory",
            "history",
            "given",
            "publications",
            "konferenz",
            "parallellinien",
            "mengenlehre",
            "klimabulletin",
            "bioingenieria",
            "denver",
        ];
        if lower.contains(' ') && leading_verbs.contains(&first_word) {
            return false;
        }
        // Reject verb-phrase entities like "Uses MapReduce", "Has Many", "Is Known"
        let verb_prefixes: &[&str] = &[
            "uses", "has", "is", "was", "are", "were", "can", "could", "would", "should", "does",
            "did", "will", "shall", "may", "might", "must", "needs", "gets", "makes", "takes",
            "gives", "keeps", "lets", "shows", "says", "tells", "asks", "runs", "sets", "puts",
        ];
        if lower.contains(' ') && verb_prefixes.contains(&first_word) {
            return false;
        }
    }

    // All-uppercase check: reject unless it's a short acronym (≤6 chars)
    if trimmed
        .chars()
        .all(|c| c.is_uppercase() || !c.is_alphabetic())
    {
        let alpha_len = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        if alpha_len > 6 {
            return false;
        }
    }

    // Don't start with numbers unless it looks like a measurement/date
    if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return false;
    }

    // Reject "Category:" prefixed entries (Wikipedia metadata)
    if lower.starts_with("category:") {
        return false;
    }

    // Reject "See X" cross-references (e.g. "See Gibbs", "See Analytic")
    if lower.starts_with("see ") && lower.split_whitespace().count() <= 3 {
        return false;
    }

    // Reject "List of" patterns (Wikipedia list article titles)
    if lower.starts_with("list of ") {
        return false;
    }

    // Reject multi-word entities containing publication/reference title keywords
    // e.g. "Bernardo José-Miguel Handbook", "Complete Patents of Nikola Tesla"
    if lower.contains(' ') {
        const PUB_TITLE_WORDS: &[&str] = &[
            "handbook",
            "dictionary",
            "encyclopedia",
            "encyclopaedia",
            "annals",
            "patents",
            "inventions",
            "brochure",
            "proceedings",
            "transactions",
            "letters correspondence",
            "homepage",
            "travelchinaguide",
            "encyclopædia britannica",
        ];
        for kw in PUB_TITLE_WORDS {
            if lower.contains(kw) {
                return false;
            }
        }
    }

    // Reject "-language" suffix entries (Wikipedia language metadata like "French-language")
    if lower.ends_with("-language") {
        return false;
    }

    // Reject compound adjectives used as standalone entities (e.g. "Self-supervised", "Object-oriented")
    if !lower.contains(' ') && lower.contains('-') {
        let parts: Vec<&str> = lower.split('-').collect();
        if let Some(last) = parts.last() {
            let adjective_suffixes = [
                "oriented",
                "supported",
                "influenced",
                "specific",
                "supervised",
                "augmented",
                "based",
                "driven",
                "related",
                "powered",
                "enabled",
                "aware",
                "like",
                "ready",
                "free",
                "rich",
                "poor",
                "dependent",
                "dimensional",
                "controlled",
                "martial",
                "equilibrium",
                "state",
                "time",
                "domain",
                "level",
                "scale",
                "order",
                "body",
                "step",
                "valued",
                "valued",
                "connected",
                "coupled",
                "resolved",
                "born",
                "made",
                "led",
                "dominated",
                "speaking",
                "style",
                "type",
                "case",
            ];
            if adjective_suffixes.iter().any(|s| last == s) {
                return false;
            }
        }
        // Reject X-to-Y patterns (e.g. "Source-to-source", "Sun-to-Earth")
        if parts.len() == 3 && parts[1] == "to" {
            return false;
        }
        // Also reject "Nationality-Nationality" patterns (e.g. "Chinese-Soviet", "Sino-American")
        if let Some(first) = parts.first() {
            let nationality_stems = [
                "sino", "anglo", "franco", "russo", "austro", "indo", "chinese", "american",
                "soviet", "british", "french", "german", "italian", "spanish", "russian",
                "japanese", "african", "european", "axis", "allied",
            ];
            if nationality_stems.contains(first) {
                return false;
            }
        }
    }

    // Reject CITEREF citation keys (e.g. "CITEREFWilkinson2012")
    if lower.starts_with("citeref") || lower.contains("citeref") {
        return false;
    }

    // Reject Wikipedia language sidebar entries (e.g. "Afrikaans Alemannisch", "Lombard Latviešu")
    const WIKI_LANG_FRAGMENTS: &[&str] = &[
        "afrikaans",
        "alemannisch",
        "azərbaycanca",
        "беларус",
        "čeština",
        "latviešu",
        "lombard",
        "bosanski",
        "català",
        "nedersaksies",
        "plattdüütsch",
        "asturianu",
        "esperanto",
        "galego",
        "interlingua",
        "occitan",
        "piemontèis",
        "sardu",
        "scots",
        "shqip",
        "sicilianu",
        "ślůnski",
        "srpskohrvatski",
        "tatarça",
        "walon",
        "žemaitėška",
    ];
    for frag in WIKI_LANG_FRAGMENTS {
        if lower.contains(frag) {
            return false;
        }
    }

    // Reject citation/reference fragments commonly misidentified as person names
    const CITATION_FRAGMENTS: &[&str] = &[
        "chapter",
        "vol ",
        " vol",
        "théorie",
        "abhandlungen",
        "gesammelte",
        "handbuch",
        "integrals",
        "operators",
        "thermodynamik",
        "annalen",
        "zeitschrift",
        "proceedings",
        "transactions",
        "bulletin",
        "comptes rendus",
    ];
    for frag in CITATION_FRAGMENTS {
        if lower.contains(frag) {
            return false;
        }
    }

    // Reject sentence fragments (contain verbs/articles in long phrases)
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() >= 4 {
        let sentence_markers = [
            "the", "a", "an", "is", "was", "were", "are", "have", "had", "has", "if", "that",
            "which", "by", "from", "into", "also", "then", "when", "where", "who", "whom", "whose",
            "not", "but", "yet", "so", "because", "although", "though", "since", "until", "while",
            "after", "before", "during", "about", "against", "between", "through", "without",
        ];
        let marker_count = words
            .iter()
            .filter(|w| sentence_markers.contains(w))
            .count();
        if marker_count >= 2 {
            return false;
        }
    }

    // Reject multi-word phrases starting with lowercase (not proper nouns)
    if trimmed.contains(' ')
        && trimmed.len() > 20
        && trimmed.chars().next().is_some_and(|c| c.is_lowercase())
    {
        return false;
    }

    // Reject entities with ISBN, citation markers, or caret references
    if lower.contains("isbn") || lower.contains(" ^ ") || trimmed.contains('^') {
        return false;
    }

    // Reject "Edition" fragments (book metadata, not entities)
    if lower.contains("edition") {
        return false;
    }

    // Reject citation title fragments ("X Research The Y", "Overview of the Z")
    if lower.contains(" research the ") || lower.contains(" overview ") {
        return false;
    }

    // Reject incomplete institutional fragments (e.g. "University Press", "Fellow of X")
    if lower == "university press"
        || lower.starts_with("fellow of ")
        || lower.starts_with("physics of ")
    {
        return false;
    }

    // Reject publisher/journal names (citation noise, not knowledge entities)
    const PUBLISHER_NAMES: &[&str] = &[
        "routledge",
        "addison-wesley",
        "addison–wesley",
        "springer",
        "wiley",
        "elsevier",
        "mcgraw-hill",
        "mcgraw–hill",
        "prentice hall",
        "prentice-hall",
        "o'reilly",
        "cambridge university press",
        "oxford university press",
        "mit press",
        "princeton university press",
        "academic press",
        "john wiley",
        "de gruyter",
        "brill",
        "kluwer",
        "pergamon",
        "birkhäuser",
        "vieweg",
        "teubner",
        "race point publishing",
        "packt publishing",
        "iop publishing",
    ];
    for pub_name in PUBLISHER_NAMES {
        if lower == *pub_name
            || lower.starts_with(&format!("{} ", pub_name))
            || lower.ends_with(&format!(" {}", pub_name))
        {
            return false;
        }
    }

    // Reject "X Press" publisher patterns (e.g. "Toronto Press", "Finsbury Press")
    if lower.ends_with(" press") && !lower.contains("associated") && !lower.contains("freedom") {
        return false;
    }

    // Reject "X Publishing" / "X Education" patterns (publisher names)
    if lower.ends_with(" publishing") || lower.ends_with(" education") {
        return false;
    }

    // Reject entities containing Big-O / Theta notation fragments
    if lower.contains("θ(")
        || lower.contains("o(")
        || trimmed.ends_with('Θ')
        || trimmed.ends_with('θ')
    {
        return false;
    }

    // Reject URLs that somehow get classified as entities
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return false;
    }

    // Reject entities containing brackets, special parsing artifacts, or fragment markers
    if trimmed.contains('[')
        || trimmed.contains(']')
        || trimmed.contains('{')
        || trimmed.contains('}')
        || trimmed.contains('(')
        || trimmed.contains(')')
        || trimmed.contains('#')
        || trimmed.contains('=')
    {
        return false;
    }

    // Reject single long words (code identifiers, not knowledge entities)
    if !lower.contains(' ') && trimmed.len() > 20 {
        return false;
    }

    // Reject camelCase code identifiers (e.g. BTreeMap, CoreMarks, DevTerm)
    if !trimmed.contains(' ') && trimmed.len() > 3 {
        let mut transitions = 0;
        let chars_vec: Vec<char> = trimmed.chars().collect();
        for k in 1..chars_vec.len() {
            if chars_vec[k].is_uppercase() && chars_vec[k - 1].is_lowercase() {
                transitions += 1;
            }
        }
        // True camelCase has lowercase→uppercase transitions mid-word
        // But allow known patterns like "McDonald" (1 transition is fine for names)
        if transitions >= 2 {
            return false;
        }
    }

    // Reject nationality-adjective phrases that aren't real entities
    // e.g. "Dutch American", "Austrian Netherlands" as person names
    let nationality_adjectives: &[&str] = &[
        "dutch",
        "austrian",
        "canadian",
        "serbian",
        "croatian",
        "russian",
        "french",
        "british",
        "german",
        "italian",
        "spanish",
        "portuguese",
        "swedish",
        "norwegian",
        "danish",
        "finnish",
        "polish",
        "czech",
        "hungarian",
        "romanian",
        "bulgarian",
        "greek",
        "turkish",
        "chinese",
        "japanese",
        "korean",
        "indian",
        "brazilian",
        "mexican",
        "american",
        "african",
        "asian",
        "european",
        "soviet",
        "western",
        "eastern",
        "northern",
        "southern",
        "central",
        "modern",
        "ancient",
        "early",
        "imperial",
        "royal",
        "upper",
        "lower",
        "holy",
        "national",
        "old",
        "new",
        "united",
        "arab",
        "irish",
        "scottish",
        "belgian",
        "persian",
        "ottoman",
        "latin",
    ];
    if lower.contains(' ') {
        let first_word = lower.split_whitespace().next().unwrap_or("");
        let last_word = lower.split_whitespace().last().unwrap_or("");
        // Reject "Nationality/Adjective + Nationality/Generic" patterns
        if nationality_adjectives.contains(&first_word)
            && (nationality_adjectives.contains(&last_word)
                || GENERIC_SINGLE_WORDS.contains(&last_word))
        {
            return false;
        }
    }

    // Reject German compound nouns that are generic terms (not proper nouns)
    if !lower.contains(' ') && trimmed.len() > 12 && etype == "concept" {
        let german_generic_suffixes = [
            "verteilung",
            "geschichte",
            "bereichen",
            "festspiele",
            "wissenschaft",
            "forschung",
            "entwicklung",
            "verwaltung",
            "gesellschaft",
            "morphologie",
            "verhältnis",
            "beziehung",
            "darstellung",
            "beschreibung",
            "behandlung",
            "berechnung",
            "untersuchung",
            "zusammenfassung",
            "erklärung",
            "ungen",
            "schaft",
            "ierung",
        ];
        if german_generic_suffixes.iter().any(|s| lower.ends_with(s)) {
            return false;
        }
    }

    // Reject 3+-word "person" entities where ALL words are capitalized and NONE are linking words
    // These are typically lists of names/places mashed together (e.g. "Bulgaria Serbia Montenegro")
    if etype == "person" && lower.contains(' ') {
        let w: Vec<&str> = trimmed.split_whitespace().collect();
        if w.len() >= 3 {
            let linking = [
                "of", "the", "de", "del", "di", "du", "von", "van", "la", "le", "el", "al", "das",
                "des", "der", "den", "und", "and", "bin", "ibn", "ben", "y", "e",
            ];
            let all_cap = w
                .iter()
                .all(|word| word.chars().next().is_some_and(|c| c.is_uppercase()));
            let has_link = w
                .iter()
                .any(|word| linking.contains(&word.to_lowercase().as_str()));
            // If all words are capitalized with no linking words, and the last word looks like
            // a standalone noun (not a typical surname suffix), it's probably a list/fragment
            if all_cap && !has_link {
                // Allow if it matches common name patterns (First Middle Last)
                // But reject if it has 4+ words with no linking — almost certainly noise
                if w.len() >= 4 {
                    return false;
                }
            }
        }
    }

    // Reject single-word entities classified as "person" — real people have at least two words
    if etype == "person" && !lower.contains(' ') {
        return false;
    }

    // Reject "person" entities containing publishing/academic/concept terms
    if etype == "person" {
        let person_blacklist_words = [
            "publishers",
            "verlag",
            "buchverlag",
            "thesis",
            "github",
            "sourceforge",
            "youtube",
            "proquest",
            "docs",
            "robot",
            "stack",
            "chip",
            "monthly",
            "notices",
            "recurrent",
            "falcon",
            "mirror",
            "foundry",
            "battalion",
            "regiment",
            "brigade",
            "squadron",
            "flotilla",
            "division",
            "corps",
            "particle",
            "interview",
            "textbook",
            "progress",
            "context",
            "polities",
            "chaos",
            "premier",
            "civilization",
            "checkpoints",
            "attractors",
            "westerners",
            "development board",
            "authorization",
            "university",
            "institute",
            "academy",
            "college",
            "school",
            "committee",
            "department",
            "museum",
            "foundation",
            "laboratory",
            "observatory",
            // Country/region names — "X China", "Y Rome" etc. are not people
            "empire",
            "republic",
            "dynasty",
            "kingdom",
            "initiative",
            "commonwealth",
            "international",
            "program ",
            "programs",
            "bibliotheca",
            "doctoral",
            "patreon",
            "communications",
            "mongol",
            "storm",
            "dark",
            "state",
        ];
        // Also reject person entities ending with a country/region name
        const PERSON_COUNTRY_BLACKLIST: &[&str] = &[
            "china", "japan", "india", "russia", "france", "germany", "italy", "spain", "turkey",
            "iran", "iraq", "egypt", "rome", "greece", "persia", "arabia", "mongolia", "tibet",
            "burma", "siam", "brazil", "mexico",
        ];
        if let Some(last) = lower.split_whitespace().last() {
            if lower.contains(' ') && PERSON_COUNTRY_BLACKLIST.contains(&last) {
                return false;
            }
        }
        if person_blacklist_words.iter().any(|w| lower.contains(w)) {
            return false;
        }
    }

    // Reject "person" entities containing tech acronyms (3+ consecutive uppercase letters)
    if etype == "person" {
        let has_acronym = trimmed.split_whitespace().any(|w| {
            let upper_run = w.chars().filter(|c| c.is_uppercase()).count();
            upper_run >= 3
                && w.len() >= 3
                && w.chars()
                    .all(|c| c.is_uppercase() || c.is_ascii_digit() || c == '-')
        });
        if has_acronym {
            return false;
        }
    }

    // Reject single-word concepts under 4 chars (too ambiguous to be useful)
    if etype == "concept" && !lower.contains(' ') && trimmed.len() < 4 {
        return false;
    }

    // Reject 3+-word entities ending with a single uppercase letter (citation fragments like "Berndt Bruce C")
    // but allow Roman numerals (I, V, X, L, C, D, M) for historical figures like "Charles V"
    {
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() >= 3 {
            if let Some(last) = words.last() {
                if last.len() == 1 && last.chars().next().is_some_and(|c| c.is_uppercase()) {
                    let ch = last.chars().next().unwrap();
                    if !matches!(ch, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M') {
                        return false;
                    }
                }
            }
        }
    }

    // Reject journal/encyclopedia citation fragments
    if lower.contains("journal of")
        || lower.contains("researchgate")
        || lower.contains("encyclopedia")
        || lower.contains("mathpages")
    {
        return false;
    }

    // Reject entities containing Wikipedia reference/citation fragments
    if lower.contains("archived")
        || lower.contains("isbn")
        || lower.contains(" pdf ")
        || lower.starts_with("pdf ")
        || lower.ends_with(" pdf")
        || lower.contains("citeseer")
        || lower.contains("weatherbase")
        || lower.contains("issn")
        || lower.contains("pmid")
        || lower.contains("retrieved")
        || lower.contains("bibcode")
        || lower.contains("s2cid")
        || lower.contains("oclc")
        || lower.contains("jstor")
        || lower.contains("wayback machine")
        || lower.contains("arxiv")
        || lower == "accessed"
        || lower == "published"
    {
        return false;
    }

    // Reject entities with words concatenated without space (parsing errors like "TelescopeThis")
    if trimmed.len() > 5 {
        let chars: Vec<char> = trimmed.chars().collect();
        for i in 1..chars.len() {
            if chars[i].is_uppercase() && chars[i - 1].is_lowercase() {
                // Check if the suffix after the uppercase is a common word like "This", "The", "And"
                let suffix: String = chars[i..].iter().collect();
                let suffix_lower = suffix.to_lowercase();
                if matches!(
                    suffix_lower.as_str(),
                    "this" | "the" | "and" | "that" | "here" | "there" | "from" | "with"
                ) {
                    return false;
                }
            }
        }
    }

    // Reject entities that look like sentence fragments (contain common verbs/prepositions sequences)
    if (lower.contains("noted ") || lower.contains("known ") || lower.contains("called "))
        && (lower.starts_with("noted ")
            || lower.starts_with("known ")
            || lower.starts_with("called "))
    {
        return false;
    }

    // Reject multi-word entities starting with compound-adjective prefixes
    // e.g. "Non-Euclidean Style", "Pre-British India", "Cross-Cultural Contacts"
    if lower.contains(' ') {
        let compound_prefixes = [
            "non-", "pre-", "post-", "anti-", "multi-", "cross-", "semi-", "self-", "co-", "pan-",
            "inter-", "intra-", "trans-", "sub-", "super-", "pseudo-", "quasi-", "counter-",
            "over-", "under-", "re-", "de-", "un-",
        ];
        let first_word = lower.split_whitespace().next().unwrap_or("");
        if compound_prefixes.iter().any(|p| first_word.starts_with(p)) {
            // Allow if it's a well-known proper noun (e.g. "Non-Aligned Movement" is a concept anyway)
            // but reject as person
            if etype == "person" {
                return false;
            }
        }
    }

    // Reject multi-word concepts ending with adjective/fragment words (parsing artifacts)
    if etype == "concept" && lower.contains(' ') {
        let last_word = lower.split_whitespace().last().unwrap_or("");
        let fragment_endings = [
            "electronic",
            "virtual",
            "aerial",
            "mechanical",
            "optical",
            "digital",
            "classical",
            "musical",
            "physical",
            "chemical",
            "biological",
            "geological",
            "astronomical",
            "mathematical",
            "statistical",
            "computational",
            "experimental",
            "theoretical",
            "numerical",
            "analytical",
            "structural",
            "functional",
        ];
        let first_word = lower.split_whitespace().next().unwrap_or("");
        let fragment_starts = [
            "applications",
            "properties",
            "interpreter",
            "compiler",
            "processor",
            "introducing",
            "acquiring",
            "dissolving",
            "combining",
            "meanwhile",
        ];
        if fragment_endings.contains(&last_word) && !lower.contains(" and ") {
            // Only reject if it looks like a fragment (2 words, no connectors)
            let word_count = lower.split_whitespace().count();
            if word_count <= 3 {
                return false;
            }
        }
        if fragment_starts.contains(&first_word) {
            return false;
        }
    }

    // Reject multi-word "person" entities where first word ends in common adjective suffixes
    if etype == "person" && lower.contains(' ') {
        let first_word = lower.split_whitespace().next().unwrap_or("");
        let adj_suffixes = [
            "ized", "ised", "ated", "ting", "ling", "ning", "ring", "ding", "ical", "ious", "eous",
            "ular", "ular", "ible", "able",
        ];
        if adj_suffixes.iter().any(|s| first_word.ends_with(s)) {
            return false;
        }
    }

    true
}

/// Classify an entity name into person/place/org/concept based on heuristics.
fn classify_entity_type(name: &str) -> &'static str {
    let lower = name.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // Known company/brand prefixes — "Google Translate", "Apple Music" etc. are products, not people
    const PRODUCT_PREFIXES: &[&str] = &[
        "google",
        "apple",
        "microsoft",
        "amazon",
        "meta",
        "nvidia",
        "intel",
        "samsung",
        "adobe",
        "oracle",
        "ibm",
        "cisco",
        "dell",
        "hp",
        "sony",
        "tesla",
        "openai",
        "anthropic",
    ];
    if let Some(first) = words.first() {
        if words.len() >= 2 && PRODUCT_PREFIXES.contains(first) {
            // Check if last word is an org indicator
            if let Some(last) = words.last() {
                let clean = last.trim_matches(|c: char| !c.is_alphanumeric());
                if ORG_INDICATORS.contains(&clean) {
                    return "organization";
                }
            }
            return "concept";
        }
    }

    // Known places (single-word) that keep getting misclassified
    const KNOWN_PLACES: &[&str] = &[
        "africa",
        "antarctica",
        "arctic",
        "asia",
        "europe",
        "eurasia",
        "bolivia",
        "azerbaijan",
        "bessarabia",
        "birmingham",
        "barcelona",
        "baghdad",
        "belgrade",
        "berlin",
        "bergen",
        "marseille",
        "trieste",
        "tabriz",
        "portland",
        "adrianople",
        "alexandretta",
        "trafalgar",
        "lausanne",
        "medina",
        "kashgar",
        "phoenicia",
        "dodona",
        "dakhla",
        "mediolanum",
        "partick",
        "badsey",
        "passchendaele",
        "harrogate",
        "greenford",
        "xiangyang",
        "austerlitz",
        "auerstädt",
        "badajoz",
        "bastille",
        "borodino",
        "gallipoli",
        "constantinople",
        "jerusalem",
        "damascus",
        "cairo",
        "alexandria",
        "carthage",
        "rome",
        "athens",
        "sparta",
        "corinth",
        "thebes",
        "babylon",
        "nineveh",
        "persepolis",
        "samarkand",
        "timbuktu",
        "istanbul",
        "ankara",
        "tehran",
        "kabul",
        "delhi",
        "mumbai",
        "kolkata",
        "chennai",
        "bangalore",
        "hyderabad",
        "shanghai",
        "beijing",
        "guangzhou",
        "shenzhen",
        "tokyo",
        "osaka",
        "kyoto",
        "seoul",
        "pyongyang",
        "hanoi",
        "bangkok",
        "singapore",
        "jakarta",
        "manila",
        "sydney",
        "melbourne",
        "auckland",
        "wellington",
        "nairobi",
        "lagos",
        "kinshasa",
        "johannesburg",
        "casablanca",
        "tunis",
        "algiers",
        "tripoli",
        "khartoum",
        "mogadishu",
        "havana",
        "bogotá",
        "lima",
        "santiago",
        "montevideo",
        "strasbourg",
        "toulouse",
        "lyon",
        "bordeaux",
        "nice",
        "nantes",
        "munich",
        "hamburg",
        "cologne",
        "frankfurt",
        "stuttgart",
        "dresden",
        "leipzig",
        "nuremberg",
        "vienna",
        "salzburg",
        "innsbruck",
        "graz",
        "zurich",
        "zürich",
        "geneva",
        "genève",
        "bern",
        "basel",
        "lausanne",
        "lucerne",
        "lugano",
        "winterthur",
        // Countries often misclassified as concepts
        "tajikistan",
        "turkmenistan",
        "uzbekistan",
        "kyrgyzstan",
        "kazakhstan",
        "afghanistan",
        "bangladesh",
        "madagascar",
        "mozambique",
        "zimbabwe",
        "botswana",
        "namibia",
        "ethiopia",
        "tanzania",
        "morocco",
        "tunisia",
        "algeria",
        "somalia",
        "cameroon",
        "senegal",
        "mauritania",
        "guatemala",
        "honduras",
        "nicaragua",
        "paraguay",
        "uruguay",
        "venezuela",
        "suriname",
        "reykjavik",
        "bratislava",
        "bucharest",
        "budapest",
        "helsinki",
        "tallinn",
        "vilnius",
        "pristina",
        "tirana",
        "podgorica",
        "skopje",
        "chisinau",
        "minsk",
        "sarajevo",
        "zagreb",
        "belfast",
        "edinburgh",
        "cardiff",
        "lisbon",
        "madrid",
        "warsaw",
        "prague",
        "copenhagen",
        "stockholm",
        "oslo",
        "amsterdam",
        "brussels",
        "dublin",
        "london",
        "paris",
        "moscow",
        "kyiv",
        // Added 2026-02-23: more cities/places
        "perth",
        "malaga",
        "schengen",
        "ferrara",
        "ionia",
        "wallachia",
        "chalcedon",
        "piombino",
        // Added 2026-02-23: more places from DB analysis
        "cremona",
        "swansea",
        "guildford",
        "stonehenge",
        "salamanca",
        "hangzhou",
        "camargue",
        "eifel",
        "antwerp",
        "somerville",
        "wrocław",
        "visby",
        "megara",
        "sinope",
        "fezzan",
        "abyssinia",
        // Added 2026-02-24: more places
        "kaliningrad",
        "nagasaki",
        "mannheim",
        "fukuoka",
        "oswaldtwistle",
        "leskovac",
        "ouargla",
        "bratsk",
        "wolfeboro",
        "shanhaiguan",
        "hondschoote",
        "montmédy",
        "seaburn",
        "kinross",
        "bosworth",
        "littlefield",
        "arcadia",
        "assyria",
        "pithekoussai",
        "tondidarou",
        "magnesia",
        "paphos",
        "thessaloniki",
        "savannah",
        "kythera",
        "macedon",
        "waterloo",
        "epirus",
        "bithynia",
        "cappadocia",
        "cilicia",
        "phrygia",
        "lydia",
        "caria",
        "lycia",
        "pamphylia",
        "galatia",
        "pontus",
        "thrace",
        "epirus",
        "boeotia",
        "attica",
        "arcadia",
        "messenia",
        "laconia",
        "argolis",
        "achaea",
        // Added 2026-02-25: places from DB reclassification
        "warszawa",
        "taganrog",
        "kildare",
        "bergfelde",
        "almopia",
        "machupicchu",
        "kurkut",
        "neuquén",
        // Added 2026-02-25 round 2: more places from DB reclassification
        "polotsk",
        "etruria",
        "hyphasis",
        "yarkand",
        "norpatagonia",
        "gotland",
        "mauguio",
        "demerara",
        "patavium",
        "wuchang",
        "chillán",
        "chivirkuy",
        "edgerton",
        // Added 2026-02-25 round 3
        "helmstedt",
        "lexington",
        "bukhara",
        "villupuram",
        "puteoli",
        "pentagon",
        "grenoble",
        // Added 2026-02-24: more places from DB reclassification
        "artaxata",
        "bruttium",
        "chiasso",
        "elimiotae",
        "extremadura",
        "fregellae",
        "glasgow",
        "lacedaemon",
        "perpinya",
        "sichuan",
        "timoneion",
        "ushuaia",
        "calafate",
        "ivigtut",
        "worcester",
        "badakhshan",
        "paraguay",
        "vitoria",
        "navarre",
        "tephrike",
        "potidaea",
        "lindisfarne",
        "alexandreia",
        "chaldiran",
        "yuanshi",
        "acarnania",
        "aetolia",
        "aegospotamos",
        "ahobamba",
        "ailana",
        // Added 2026-02-24 (cron): places from DB reclassification
        "vistula",
        "lintao",
        "jersey",
        "muscat",
        "gospić",
        "fayum",
        "zeitun",
        "magog",
        // Added 2026-02-24 (brain cleaner): reclassified from concept
        "bologna",
        // Added 2026-02-25 (brain cleaner round)
        "hakodate",
        "sheppey",
        "sphakteria",
        "lauterbrunnental",
        "glenmama",
        "perriertoppen",
        "moghulistan",
        "holland",
        "kurdistan",
        "dagestan",
        "jutland",
        "shetland",
        "uppland",
        "coihaique",
        "almaty",
        "altaussee",
        "mignone",
        "moorhead",
        "wilmslow",
        // Added 2026-02-24 (brain cleaner round 2)
        "rumelia",
        "arabia",
        "martigny",
        "marengo",
        "cerrado",
        "gondwana",
        "talgar",
        // Added 2026-02-24 (brain cleaner round 4)
        "rangoon",
        "gurganj",
        "kangju",
        "nellore",
        "yingpan",
        "gipuzkoa",
        "dearborn",
        "brindisi",
        "yokohama",
        "catoira",
        "civitella",
        "ralswiek",
        "chaitén",
        "halicarnassus",
        "ollantaytambo",
        "nordaustlandet",
        "chadwickryggen",
        "keshengzhuang",
        "orihuela",
        "chaeronea",
        "pelousion",
        "pingxingguan",
        "pampeluna",
        "griebnitzsee",
        "chalcidice",
        "lyskamm",
        "praeneste",
        "jungfraubahn",
        "pilsoto",
        "pedropunt",
        "talayata",
        "whitsunday",
        // Added 2026-02-24 (brain cleaner round 5)
        "cyrenaica",
        "tuscany",
        "rosaspata",
        "samnium",
    ];
    if !lower.contains(' ') && KNOWN_PLACES.contains(&lower.as_str()) {
        return "place";
    }

    // Known historical/ancient persons often misclassified as concepts
    const KNOWN_PERSONS: &[&str] = &[
        "herodotus",
        "callisthenes",
        "thucydides",
        "xenophon",
        "polybius",
        "plutarch",
        "tacitus",
        "suetonius",
        "livy",
        "sallust",
        "josephus",
        "eusebius",
        "procopius",
        "ammianus",
        "diodorus",
        "strabo",
        "pausanias",
        "arrian",
        "appian",
        "cassius",
        "eratosthenes",
        "hipparchus",
        "aristarchus",
        "archimedes",
        "euclid",
        "pythagoras",
        "democritus",
        "empedocles",
        "anaxagoras",
        "anaximander",
        "parmenides",
        "zeno",
        "epicurus",
        "epictetus",
        "seneca",
        "cicero",
        "virgil",
        "ovid",
        "horace",
        "juvenal",
        "lucretius",
        "hammurapi",
        "nebuchadnezzar",
        "ashurbanipal",
        "sargon",
        "confucius",
        "avicenna",
        "averroes",
        "maimonides",
        "alhazen",
        "fibonacci",
        "copernicus",
        "kepler",
        "galileo",
        "brahe",
        // Added 2026-02-24: more historical persons
        "beethoven",
        "tokugawa",
        "newcomen",
        "wedgwood",
        "ptolemy",
        "marcellus",
        "imhotep",
        "kernighan",
        "heraclides",
        "callisthenes",
        "narasimha",
        // Added 2026-02-25: persons from DB reclassification
        "fairbank",
        "walecka",
        "lippold",
        "dicke",
        "skrabec",
        "lemmermeyer",
        "bolloten",
        "senemut",
        "thorgest",
        "mithridates",
        "phaedra",
        // Added 2026-02-25 round 2: more persons from DB reclassification
        "sorghaghtani",
        "ottaviani",
        "górecki",
        "gagarin",
        "guralnik",
        "kitsikis",
        "radukovskii",
        "sheehan",
        "apastamba",
        "nicolet",
        "haspar",
        // Added 2026-02-25 round 3
        "khalatnikov",
        "eichengreen",
        "cimabue",
        "pisistratus",
        "mirabeau",
        "poiseuille",
        "whitehorne",
        "ariqboke",
        "miscamble",
        "kemble",
        // Added 2026-02-24: more persons from DB reclassification
        "donadoni",
        "garrard",
        "hammersley",
        "hásteinn",
        "jacobson",
        "khurshah",
        "klooster",
        "mcfarland",
        "nightingale",
        "pinarius",
        "redheffer",
        "rosseland",
        "sanjurjo",
        "spencer",
        "spergel",
        "treadgold",
        "verlinde",
        "komnenos",
        "milner-barry",
        "farrell",
        "matschke",
        "howarth",
        "karmarkar",
        "kaldellis",
        "bottari",
        "calpurnia",
        "wysession",
        "calvera",
        "tanimoto",
        "albrecht",
        "kendall",
        "borcherds",
        "hippias",
        "garland",
        "eudoxus",
        "barnave",
        "skeat",
        "weyuker",
        "donoghue",
        "horemheb",
        "mesha",
        "ezana",
        "aaronson",
        "aaserud",
        "abdulla",
        "abrikosov",
        "aczel",
        "adamthwaite",
        "addington",
        "ahenobarbus",
        "akhenaten",
        "kramers",
        "kapitza",
        "medawar",
        "stuewer",
        "michell",
        "ackermann",
        "agathoclea",
        "akashdeep",
        // Added 2026-02-24 (brain cleaner round 4)
        "ashtekar",
        "bhattacharya",
        "quirke",
        "sekunda",
        "feldhay",
        "salucci",
        "elgot",
        "middlekauff",
        "parameswara",
        "bellamy",
        "padmanabhan",
        "bonfante",
        "maccone",
        "townshend",
        "silvestre",
        "viviani",
        "cordry",
        "perella",
        "macmullen",
        "loescher",
        "kratoska",
        "pećanac",
        // Added 2026-02-24 (cron): persons from DB reclassification
        "pompey",
        "putnam",
        "lamarck",
        "colbert",
        "bullock",
        "shelah",
        "tannaka",
        "kershaw",
        "mazower",
        "pietsch",
        "gefter",
        "coullet",
        "udwadia",
        "minsky",
        "barak",
        "pucci",
        "kummer",
        "debray",
        "lovell",
        "cormen",
        "vogel",
        "borel",
        "munch",
        "loczy",
        "stoney",
        "bivar",
        // Added 2026-02-24 (brain cleaner): reclassified from concept
        "voltaire",
        "leibniz",
        "montesquieu",
        "descartes",
        "attucks",
        "balfour",
        "churchill",
        "wordsworth",
        "minkowski",
        "alzheimer",
        "aristotle",
        "avogadro",
        "barnett",
        "ashton",
        // Added 2026-02-25 (brain cleaner round)
        "haworth",
        "potheinos",
        "ehrenfest",
        "törnqvist",
        "rybczyńska",
        "ramsey",
        "morgenstern",
        "mccarty",
        "yanukovych",
        "taharqa",
        "strommer",
        "mohanty",
        "valerii",
        "balcells",
        "giannantonio",
        "delorme",
        "brightwell",
        "seidman",
        "marston",
        "catulus",
        "ameilhon",
        "anastasio",
        "allender",
        "copeland",
        "rowland",
        "sutherland",
        "richelieu",
        "desargues",
        "bouguer",
        "alcubierre",
        "aleksandrov",
        "alma-tadema",
        "sieczkowska",
        "bemmelen",
        "zauzmer",
        // Added 2026-02-24 (brain cleaner)
        "riemann",
        "brahmagupta",
        "duchenne",
        "curtius",
        "dummett",
        "mackinnon",
        // Added 2026-02-24 (brain cleaner round 5)
        "blanton",
        "papademetriou",
        "albers",
        "horrocks",
        "englert",
        "mcclusky",
        "ostergård",
        "leyzorek",
        "barthold",
        "ruffini",
        "kolmogorov",
        "harden",
        "tiberius",
        "augustus",
    ];
    if !lower.contains(' ') && KNOWN_PERSONS.contains(&lower.as_str()) {
        return "person";
    }
    // Multi-word entities ending with a known place/country → place
    if lower.contains(' ') {
        let last_word = lower.split_whitespace().last().unwrap_or("");
        const COUNTRY_NAMES: &[&str] = &[
            "austria",
            "germany",
            "france",
            "italy",
            "spain",
            "portugal",
            "switzerland",
            "belgium",
            "netherlands",
            "luxembourg",
            "denmark",
            "sweden",
            "norway",
            "finland",
            "poland",
            "czechia",
            "hungary",
            "romania",
            "bulgaria",
            "greece",
            "turkey",
            "russia",
            "ukraine",
            "belarus",
            "serbia",
            "croatia",
            "bosnia",
            "albania",
            "ireland",
            "scotland",
            "england",
            "wales",
            "iceland",
            "estonia",
            "latvia",
            "lithuania",
            "slovakia",
            "slovenia",
            "moldova",
            "montenegro",
            "macedonia",
            "china",
            "japan",
            "korea",
            "india",
            "pakistan",
            "iran",
            "iraq",
            "egypt",
            "brazil",
            "mexico",
            "argentina",
            "chile",
            "colombia",
            "peru",
            "cuba",
            "canada",
            "australia",
            "indonesia",
            "thailand",
            "vietnam",
            "malaysia",
            "philippines",
            "singapore",
            "mongolia",
            "myanmar",
            "cambodia",
            "laos",
            "nepal",
            "afghanistan",
            "syria",
            "jordan",
            "lebanon",
            "israel",
            "palestine",
            "morocco",
            "tunisia",
            "algeria",
            "libya",
            "sudan",
            "ethiopia",
            "kenya",
            "nigeria",
            "ghana",
            "tanzania",
            "uganda",
            "mozambique",
            "madagascar",
            "zimbabwe",
            "zambia",
            "botswana",
            "namibia",
            "angola",
            "congo",
        ];
        if KNOWN_PLACES.contains(&last_word) || COUNTRY_NAMES.contains(&last_word) {
            return "place";
        }
    }

    // "Mount X", "Lake X", "Cape X", "Fort X" → place
    const PLACE_PREFIXES: &[&str] = &[
        "mount",
        "lake",
        "cape",
        "fort",
        "port",
        "isle",
        "gulf",
        "bay",
        "rio",
        "san",
        "santa",
        "saint",
        "st",
        "sea of",
        "strait of",
        "isle of",
    ];
    if let Some(first) = words.first() {
        if words.len() >= 2 && PLACE_PREFIXES.contains(first) {
            return "place";
        }
    }

    // Check for person title prefix (e.g. "Dr. Smith", "President Obama")
    if let Some(first) = words.first() {
        let clean = first.trim_matches('.');
        if PERSON_TITLES.contains(&clean) {
            return "person";
        }
    }

    // Check for organization indicators (last word or any word)
    if let Some(last) = words.last() {
        let clean = last.trim_matches(|c: char| !c.is_alphanumeric());
        if ORG_INDICATORS.contains(&clean) {
            return "organization";
        }
    }
    // Also check second-to-last for patterns like "Apple Inc."
    if words.len() >= 2 {
        let second_last = words[words.len() - 2].trim_matches(|c: char| !c.is_alphanumeric());
        if ORG_INDICATORS.contains(&second_last) {
            return "organization";
        }
    }

    // Check for place indicators
    for w in &words {
        let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
        if PLACE_INDICATORS.contains(&clean) {
            return "place";
        }
    }

    // German compound place names (suffixes like -strasse, -platz, -brücke, etc.)
    const GERMAN_PLACE_SUFFIXES: &[&str] = &[
        "strasse",
        "straße",
        "platz",
        "gasse",
        "brücke",
        "kirche",
        "burg",
        "dorf",
        "heim",
        "stadt",
        "berg",
        "feld",
        "wald",
        "hafen",
        "turm",
        "tor",
        "hof",
        "allee",
        "weg",
        "bahnhof",
        "hauptbahnhof",
        "friedhof",
        "horn",
    ];
    if !lower.contains(' ') && lower.len() > 6 {
        if GERMAN_PLACE_SUFFIXES.iter().any(|s| lower.ends_with(s)) {
            // Exclude known person names ending in these (e.g. Ginzburg, Hausdorff)
            const PERSON_EXCEPTIONS: &[&str] = &[
                "ginzburg",
                "hausdorff",
                "hamburg",
                "salzburg",
                "heidelberg",
                "nuremberg",
                "gutenberg",
                "goldberg",
                "rosenberg",
                "weinberg",
                "spielberg",
                "zuckerberg",
                "bloomberg",
                "sandberg",
                "kirchhoff",
                "kirchner",
                "waghorn",
                "blinkhorn",
                "longhorn",
                "elkhorn",
                "buckhorn",
                "leghorn",
                "inkhorn",
                "foghorn",
                "alphorn",
                "flügelhorn",
            ];
            if !PERSON_EXCEPTIONS.contains(&lower.as_str()) {
                return "place";
            }
        }
    }

    // Check for concept indicators (before person heuristic)
    for w in &words {
        let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
        if CONCEPT_INDICATORS.contains(&clean) {
            return "concept";
        }
    }

    // All-caps short acronyms are likely organizations (NASA, UNESCO, NATO)
    if name.len() >= 3
        && name.len() <= 6
        && name.chars().all(|c| c.is_uppercase() || !c.is_alphabetic())
    {
        return "organization";
    }

    // Reject nationality/direction-prefixed phrases as person names
    let nationality_prefixes: &[&str] = &[
        "french",
        "british",
        "german",
        "italian",
        "spanish",
        "portuguese",
        "swedish",
        "russian",
        "chinese",
        "japanese",
        "korean",
        "indian",
        "american",
        "african",
        "european",
        "soviet",
        "dutch",
        "austrian",
        "belgian",
        "polish",
        "greek",
        "turkish",
        "persian",
        "ottoman",
        "arab",
        "irish",
        "scottish",
        "mexican",
        "brazilian",
        "canadian",
        "serbian",
        "croatian",
        "western",
        "eastern",
        "northern",
        "southern",
        "central",
        "modern",
        "ancient",
        "early",
        "imperial",
        "royal",
        "upper",
        "lower",
        "holy",
        "national",
        "old",
        "new",
        "united",
        "latin",
    ];
    if let Some(first) = words.first() {
        if nationality_prefixes.contains(first) && words.len() >= 2 {
            // Check if remaining words indicate place/org before defaulting to concept
            if let Some(last) = words.last() {
                let clean = last.trim_matches(|c: char| !c.is_alphanumeric());
                if PLACE_INDICATORS.contains(&clean) {
                    return "place";
                }
                if ORG_INDICATORS.contains(&clean) {
                    return "organization";
                }
            }
            return "concept";
        }
    }

    // If any word is an all-caps acronym (3+ chars), it's likely a concept/tech term, not a person
    if name.split_whitespace().any(|w| {
        w.len() >= 3
            && w.chars()
                .all(|c| c.is_uppercase() || c.is_ascii_digit() || c == '-')
    }) {
        return "concept";
    }

    // Single-word place detection by geographic suffix
    if words.len() == 1 {
        const GEO_SUFFIXES: &[&str] = &[
            "burg", "burgh", "bury", "stadt", "town", "ville", "polis", "grad", "abad", "port",
            "haven", "ford", "mouth", "minster", "chester", "cester", "bridge", "wick", "stead",
            "heim", "dorf", "berg",
        ];
        for suffix in GEO_SUFFIXES {
            if lower.len() > suffix.len() + 2 && lower.ends_with(suffix) {
                return "place";
            }
        }
    }

    // Two or three capitalized words with no indicators → likely person name
    // But only if words look like actual names (no long compound words)
    if words.len() >= 2
        && words.len() <= 3
        && name
            .split_whitespace()
            .all(|w| w.chars().next().is_some_and(|c| c.is_uppercase()))
    {
        // Heuristic: person names have short words (< 15 chars each)
        // and typically don't contain words ending in common suffixes
        if name.split_whitespace().all(|w| w.len() < 15) {
            let has_noun_suffix = words.iter().any(|w| {
                let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
                clean.ends_with("tion")
                    || clean.ends_with("ment")
                    || clean.ends_with("ness")
                    || clean.ends_with("ity")
                    || clean.ends_with("ism")
                    || clean.ends_with("ing")
                    || clean.ends_with("ics")
                    || clean.ends_with("ogy")
                    || clean.ends_with("phy")
                    || clean.ends_with("ence")
                    || clean.ends_with("ance")
                    || clean.ends_with("ure")
                    || clean.ends_with("ery")
                    || clean.ends_with("ory")
                    || clean.ends_with("ary")
                    || clean.ends_with("ous")
                    || clean.ends_with("ive")
                    || clean.ends_with("ons")
                    || clean.ends_with("als")
                    || clean.ends_with("ems")
                    || clean.ends_with("ies")
                    || clean.ends_with("ures")
                    || clean.ends_with("ths")
                    || clean.ends_with("nce")
                    || clean.ends_with("age")
                    || clean.ends_with("ual")
                    || clean.ends_with("cal")
                    || clean.ends_with("lar")
                    || clean.ends_with("ble")
            });
            // Also check if any word is a concept/place/org indicator
            let has_indicator = words.iter().any(|w| {
                let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
                CONCEPT_INDICATORS.contains(&clean)
                    || PLACE_INDICATORS.contains(&clean)
                    || ORG_INDICATORS.contains(&clean)
            });
            // Common words that indicate concept, not person
            const NOT_PERSON_WORDS: &[&str] = &[
                "summary",
                "control",
                "profile",
                "demise",
                "review",
                "analysis",
                "report",
                "system",
                "theory",
                "model",
                "process",
                "method",
                "design",
                "network",
                "protocol",
                "standard",
                "format",
                "module",
                "engine",
                "platform",
                "automatic",
                "executive",
                "general",
                "special",
                "primary",
                "advanced",
                "basic",
                "applied",
                "abstract",
                "digital",
                "dynamic",
                "static",
                "global",
                "local",
                "virtual",
                "pulsar",
                "stellar",
                "atomic",
                "quantum",
                "neural",
                "nubian",
                "mitteilungen",
                "pacific",
                "art",
                "glacier",
                "station",
                "manor",
                "horn",
                "girls",
                "groups",
                "phase",
                "need",
                "one",
                "chase",
                "legion",
                "saga",
                "papyrus",
                "horde",
                "conspiracy",
                "campaign",
                "expedition",
                "rebellion",
                "revolution",
                "massacre",
                "crisis",
                "conflict",
                "conquest",
                "siege",
                "famine",
                "plague",
                "canal",
                "strait",
                "peninsula",
                "archipelago",
                "corridor",
                "frontier",
                "border",
                "map",
                "maps",
                "plate",
                "plates",
                "newspaper",
                "sciences",
                "effects",
                "models",
                "mapper",
                "dynasty",
                "empire",
                "kingdom",
                "republic",
                "federation",
                "commonwealth",
                "caliphate",
                "sultanate",
                "khanate",
                "principality",
                "oblast",
                "voivodeship",
                // Geographic features / generic terms that prevent person classification
                "trench",
                "rift",
                "atlas",
                "ecology",
                "geographic",
                "web",
                "feed",
                "resource",
                "loss",
                "destruction",
                "developers",
                "destroyers",
                "explained",
                "piled",
                "geographica",
                "oceanographic",
                "oceania",
                "procellarum",
                "reef",
                "atoll",
                "fjord",
                "delta",
                "ridge",
                "tropic",
                "tropics",
                "shelf",
                "vent",
                "geyser",
                "lagoon",
                "oasis",
                "dune",
                "dunes",
                "crater",
                "caldera",
                "volcano",
                "codex",
                "encoder",
                "autoencoder",
                "decoder",
                "transformer",
                "classifier",
                "regressor",
                "discriminator",
                "generator",
                "optimizer",
                "scheduler",
                "benchmark",
                "dataset",
                "corpus",
                "anthology",
                "coalition",
                "crusade",
                "invasion",
                "occupation",
                "armistice",
                "restoration",
                "reformation",
                "enlightenment",
                "renaissance",
                "inquisition",
                "continent",
                "gradient",
                "palette",
                "instruments",
                "writings",
                "vorlesungen",
                "maxivan",
                "franca",
                "allied",
                "crusader",
                "lit",
            ];
            // Also reject if any word is a common English word (not a name)
            let has_common_word = words.iter().any(|w| {
                let clean_raw = w.trim_matches(|c: char| !c.is_alphanumeric());
                let clean = clean_raw.to_lowercase();
                let clean = clean.as_str();
                GENERIC_SINGLE_WORDS.contains(&clean)
                    || ENTITY_BLACKLIST.contains(&clean)
                    || TRAILING_JUNK.contains(&clean)
                    || NOT_PERSON_WORDS.contains(&clean)
            });
            if !has_noun_suffix && !has_indicator && !has_common_word {
                // Reject if any word contains digits (e.g. "XE6", "S15", "Pile-1")
                let has_code = name
                    .split_whitespace()
                    .any(|w| w.chars().any(|c| c.is_ascii_digit()));
                if !has_code {
                    return "person";
                }
            }
        }
    }

    "concept"
}

pub fn extract_entities(sentences: &[String]) -> Vec<(String, String)> {
    let mut entities = Vec::new();
    let stop_en: HashSet<&str> = STOP_WORDS_EN.iter().copied().collect();
    let stop_de: HashSet<&str> = STOP_WORDS_DE.iter().copied().collect();
    let stop_fr: HashSet<&str> = STOP_WORDS_FR.iter().copied().collect();
    let stop_it: HashSet<&str> = STOP_WORDS_IT.iter().copied().collect();
    let stop_es: HashSet<&str> = STOP_WORDS_ES.iter().copied().collect();
    let all_stops: Vec<&HashSet<&str>> = vec![&stop_en, &stop_de, &stop_fr, &stop_it, &stop_es];
    for sentence in sentences {
        extract_capitalized(sentence, &all_stops, &mut entities);
        extract_dates(sentence, &mut entities);
        extract_numbers_units(sentence, &mut entities);
        extract_emails(sentence, &mut entities);
        extract_urls(sentence, &mut entities);
        extract_years(sentence, &mut entities);
    }
    // Apply quality filter
    entities.retain(|(name, etype)| is_valid_entity(name, etype));
    entities
}

fn extract_capitalized(
    sentence: &str,
    all_stops: &[&HashSet<&str>],
    entities: &mut Vec<(String, String)>,
) {
    // Split on em-dashes and en-dashes before processing to avoid cross-clause entities
    let sub_sentences: Vec<&str> = sentence.split(['—', '–', '|']).collect();
    for sub in sub_sentences {
        extract_capitalized_inner(sub, all_stops, entities);
    }
}

fn extract_capitalized_inner(
    sentence: &str,
    all_stops: &[&HashSet<&str>],
    entities: &mut Vec<(String, String)>,
) {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let is_stop = |w: &str| -> bool {
        let lower = w.to_lowercase();
        all_stops.iter().any(|set| set.contains(lower.as_str()))
    };
    let mut i = 0;
    while i < words.len() {
        let raw_word = words[i];
        // Strip possessives ('s, 's) before processing
        let word = raw_word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .trim_end_matches("'s")
            .trim_end_matches("\u{2019}s");
        if !word.is_empty()
            && word.chars().next().is_some_and(|c| c.is_uppercase())
            && !is_stop(word)
            && i > 0
        {
            let mut phrase = vec![word.to_string()];
            let mut j = i + 1;
            // Linking words that can appear inside entity names (e.g. "University of California")
            const ENTITY_LINKING_WORDS: &[&str] = &[
                "of", "the", "de", "del", "di", "du", "von", "van", "la", "le", "les", "el", "al",
                "das", "des", "der", "den", "für", "for", "sur", "en", "upon", "on", "in", "at",
            ];
            // Max 6 words per entity phrase to capture "University of California at Berkeley"
            while j < words.len() && phrase.len() < 6 {
                let next = words[j]
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .trim_end_matches("'s")
                    .trim_end_matches("\u{2019}s");
                if !next.is_empty() && next.chars().next().is_some_and(|c| c.is_uppercase()) {
                    phrase.push(next.to_string());
                    j += 1;
                } else if !next.is_empty()
                    && ENTITY_LINKING_WORDS.contains(&next.to_lowercase().as_str())
                    && j + 1 < words.len()
                {
                    // Look ahead: only include linking word if followed by a capitalized word
                    let after = words[j + 1]
                        .trim_matches(|c: char| !c.is_alphanumeric())
                        .trim_end_matches("'s")
                        .trim_end_matches("\u{2019}s");
                    if !after.is_empty() && after.chars().next().is_some_and(|c| c.is_uppercase()) {
                        phrase.push(next.to_string());
                        phrase.push(after.to_string());
                        j += 2;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            // Strip trailing words that are blacklisted (e.g. "Tibet Meanwhile" → "Tibet")
            // But keep org/place indicators as valid trailing words
            while phrase.len() > 1 {
                let last_lower = phrase.last().unwrap().to_lowercase();
                let last_clean = last_lower.trim_matches(|c: char| !c.is_alphanumeric());
                if ORG_INDICATORS.contains(&last_clean) || PLACE_INDICATORS.contains(&last_clean) {
                    break;
                }
                if ENTITY_BLACKLIST.contains(&last_lower.as_str())
                    || GENERIC_SINGLE_WORDS.contains(&last_lower.as_str())
                    || CONCEPT_INDICATORS.contains(&last_clean)
                {
                    phrase.pop();
                } else {
                    break;
                }
            }
            // Strip leading words that are blacklisted navigation/UI terms
            // (e.g. "Toggle Einstein" → "Einstein") but NOT org/place indicators
            // since those start valid entities ("University of X", "New York")
            while phrase.len() > 1 {
                let first_lower = phrase[0].to_lowercase();
                // Don't strip if it's an org/place indicator (starts valid multi-word entities)
                let first_clean = first_lower.trim_matches(|c: char| !c.is_alphanumeric());
                if ORG_INDICATORS.contains(&first_clean) || PLACE_INDICATORS.contains(&first_clean)
                {
                    break;
                }
                if ENTITY_BLACKLIST.contains(&first_lower.as_str()) {
                    phrase.remove(0);
                } else {
                    break;
                }
            }
            let name = phrase.join(" ");
            if name.len() > 1 {
                let etype = if name.len() > 12 && !name.contains(' ') {
                    "concept"
                } else {
                    classify_entity_type(&name)
                };
                entities.push((name, etype.to_string()));
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

fn extract_years(sentence: &str, entities: &mut Vec<(String, String)>) {
    for word in sentence.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_numeric());
        if clean.len() == 4 {
            if let Ok(n) = clean.parse::<u32>() {
                if (1800..=2100).contains(&n) {
                    entities.push((clean.to_string(), "year".to_string()));
                }
            }
        }
    }
}

fn extract_dates(sentence: &str, entities: &mut Vec<(String, String)>) {
    let text = sentence;
    let lower = text.to_lowercase();

    // ISO: 2024-01-15
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 9 < bytes.len() {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
            && bytes[i + 4] == b'-'
            && bytes[i + 5].is_ascii_digit()
            && bytes[i + 6].is_ascii_digit()
            && bytes[i + 7] == b'-'
            && bytes[i + 8].is_ascii_digit()
            && bytes[i + 9].is_ascii_digit()
        {
            entities.push((text[i..i + 10].to_string(), "date".to_string()));
            i += 10;
        } else {
            i += 1;
        }
    }

    // US: MM/DD/YYYY
    for cap in find_date_patterns(text, true) {
        entities.push((cap, "date".to_string()));
    }
    // EU: DD.MM.YYYY
    for cap in find_date_patterns(text, false) {
        entities.push((cap, "date".to_string()));
    }

    // Helper: safely extract localized month dates using char-aware slicing.
    // We search in the lowercased text, then map byte positions back to the
    // original using char_indices() to avoid panicking on multi-byte chars
    // like en-dash '–'.
    let chars_orig: Vec<(usize, char)> = text.char_indices().collect();
    let chars_lower: Vec<(usize, char)> = lower.char_indices().collect();

    // Build a byte-offset mapping from lower -> original (same char index)
    // Both have the same number of chars, so we can zip them.
    // We need: given a byte offset in `lower`, find the corresponding byte offset in `text`.
    let byte_map: HashMap<usize, usize> = chars_lower
        .iter()
        .zip(chars_orig.iter())
        .map(|((lb, _), (ob, _))| (*lb, *ob))
        .collect();

    fn safe_slice(src: &str, byte_start: usize, byte_end: usize) -> Option<&str> {
        if byte_start <= byte_end
            && byte_end <= src.len()
            && src.is_char_boundary(byte_start)
            && src.is_char_boundary(byte_end)
        {
            Some(&src[byte_start..byte_end])
        } else {
            None
        }
    }

    // German month dates
    for (month_name, _) in GERMAN_MONTHS {
        if let Some(lower_pos) = lower.find(month_name) {
            let orig_pos = byte_map.get(&lower_pos).copied().unwrap_or(lower_pos);
            let month_end_lower = lower_pos + month_name.len();
            let orig_end = byte_map
                .get(&month_end_lower)
                .copied()
                .unwrap_or(orig_pos + month_name.len());
            if let Some(before) = safe_slice(text, 0, orig_pos) {
                let before = before.trim_end();
                if let Some(day_start) = before.rfind(|c: char| !c.is_ascii_digit() && c != '.') {
                    let mut next_boundary = day_start + 1;
                    while next_boundary < before.len() && !before.is_char_boundary(next_boundary) {
                        next_boundary += 1;
                    }
                    let day_part = &before[next_boundary..];
                    let day_part = day_part.trim_matches('.');
                    if let Ok(d) = day_part.trim().parse::<u32>() {
                        if (1..=31).contains(&d) {
                            if let Some(after) = safe_slice(text, orig_end, text.len()) {
                                let year_str: String = after
                                    .trim()
                                    .chars()
                                    .take_while(|c| c.is_ascii_digit())
                                    .collect();
                                if year_str.len() == 4 {
                                    let orig_month =
                                        safe_slice(text, orig_pos, orig_end).unwrap_or(month_name);
                                    entities.push((
                                        format!("{}. {} {}", d, orig_month, year_str),
                                        "date".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // French month dates
    for (month_name, _) in FRENCH_MONTHS {
        if let Some(lower_pos) = lower.find(month_name) {
            let orig_pos = byte_map.get(&lower_pos).copied().unwrap_or(lower_pos);
            let month_end_lower = lower_pos + month_name.len();
            let orig_end = byte_map
                .get(&month_end_lower)
                .copied()
                .unwrap_or(orig_pos + month_name.len());
            if let Some(before) = safe_slice(text, 0, orig_pos) {
                let before = before.trim_end();
                if let Some(day_str) = before.split_whitespace().last() {
                    if let Ok(d) = day_str.parse::<u32>() {
                        if (1..=31).contains(&d) {
                            if let Some(after) = safe_slice(text, orig_end, text.len()) {
                                let year_str: String = after
                                    .trim()
                                    .chars()
                                    .take_while(|c| c.is_ascii_digit())
                                    .collect();
                                if year_str.len() == 4 {
                                    let orig_month =
                                        safe_slice(text, orig_pos, orig_end).unwrap_or(month_name);
                                    entities.push((
                                        format!("{} {} {}", d, orig_month, year_str),
                                        "date".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Italian month dates
    for (month_name, _) in ITALIAN_MONTHS {
        if let Some(lower_pos) = lower.find(month_name) {
            let orig_pos = byte_map.get(&lower_pos).copied().unwrap_or(lower_pos);
            let month_end_lower = lower_pos + month_name.len();
            let orig_end = byte_map
                .get(&month_end_lower)
                .copied()
                .unwrap_or(orig_pos + month_name.len());
            if let Some(before) = safe_slice(text, 0, orig_pos) {
                let before = before.trim_end();
                if let Some(day_str) = before.split_whitespace().last() {
                    if let Ok(d) = day_str.parse::<u32>() {
                        if (1..=31).contains(&d) {
                            if let Some(after) = safe_slice(text, orig_end, text.len()) {
                                let year_str: String = after
                                    .trim()
                                    .chars()
                                    .take_while(|c| c.is_ascii_digit())
                                    .collect();
                                if year_str.len() == 4 {
                                    let orig_month =
                                        safe_slice(text, orig_pos, orig_end).unwrap_or(month_name);
                                    entities.push((
                                        format!("{} {} {}", d, orig_month, year_str),
                                        "date".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Relative dates (EN)
    for rel in &[
        "yesterday",
        "today",
        "tomorrow",
        "last week",
        "next week",
        "last month",
        "next month",
        "last year",
        "next year",
    ] {
        if lower.contains(rel) {
            entities.push((rel.to_string(), "relative_date".to_string()));
        }
    }
    // Relative dates (DE)
    for rel in &["gestern", "heute", "morgen", "letzte woche", "vorgestern"] {
        if lower.contains(rel) {
            entities.push((rel.to_string(), "relative_date".to_string()));
        }
    }
}

fn find_date_patterns(text: &str, us_format: bool) -> Vec<String> {
    let mut results = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if chars[i].is_ascii_digit() {
            let start = i;
            let p1: String = chars[i..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            i += p1.len();
            if i < len
                && (chars[i] == '/'
                    || (us_format && chars[i] == '-')
                    || (!us_format && chars[i] == '.'))
            {
                let sep = chars[i];
                i += 1;
                let p2: String = chars[i..]
                    .iter()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                i += p2.len();
                if i < len && chars[i] == sep {
                    i += 1;
                    let p3: String = chars[i..]
                        .iter()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    i += p3.len();
                    if p3.len() == 4 {
                        if let (Ok(a), Ok(b), Ok(y)) =
                            (p1.parse::<u32>(), p2.parse::<u32>(), p3.parse::<u32>())
                        {
                            let valid = if us_format {
                                (1..=12).contains(&a) && (1..=31).contains(&b)
                            } else {
                                (1..=31).contains(&a) && (1..=12).contains(&b)
                            };
                            if valid && (1800..=2100).contains(&y) {
                                let end = start + p1.len() + 1 + p2.len() + 1 + p3.len();
                                let s: String = chars[start..end].iter().collect();
                                results.push(s);
                            }
                        }
                    }
                }
            }
            continue;
        }
        i += 1;
    }
    results
}

fn extract_numbers_units(sentence: &str, entities: &mut Vec<(String, String)>) {
    let units: HashSet<&str> = [
        "gb", "mb", "tb", "kb", "ghz", "mhz", "khz", "hz", "kg", "mg", "g", "lb", "oz", "km", "mi",
        "m", "cm", "mm", "ft", "l", "ml", "gal", "w", "kw", "mw", "v", "mv",
    ]
    .iter()
    .copied()
    .collect();
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        let word = words[i].trim_matches(|c: char| c == ',' || c == ';');
        // Percentage
        if let Some(num) = word.strip_suffix('%') {
            if parse_num(num).is_some() {
                entities.push((word.to_string(), "number_unit".to_string()));
                i += 1;
                continue;
            }
        }
        // Currency prefix: $, €, £
        if word.starts_with('$') || word.starts_with('€') || word.starts_with('£') {
            let skip = word.char_indices().nth(1).map(|(i, _)| i).unwrap_or(1);
            let num = &word[skip..];
            if parse_num(num).is_some() {
                let mut full = word.to_string();
                if i + 1 < words.len() {
                    let nxt = words[i + 1]
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .to_lowercase();
                    if matches!(
                        nxt.as_str(),
                        "million"
                            | "billion"
                            | "trillion"
                            | "thousand"
                            | "millionen"
                            | "milliarden"
                    ) {
                        full = format!(
                            "{} {}",
                            word,
                            words[i + 1].trim_matches(|c: char| c == ',' || c == ';')
                        );
                        i += 1;
                    }
                }
                entities.push((full, "currency".to_string()));
                i += 1;
                continue;
            }
        }
        // Currency code prefix: CHF, EUR, USD, GBP
        let upper = word.to_uppercase();
        if matches!(upper.as_str(), "CHF" | "EUR" | "USD" | "GBP") && i + 1 < words.len() {
            let next = words[i + 1].trim_matches(|c: char| c == ',' || c == ';');
            let cleaned = next.replace(['\'', '\u{2019}'], "");
            if parse_num(&cleaned).is_some() {
                entities.push((format!("{} {}", word, next), "currency".to_string()));
                i += 2;
                continue;
            }
        }
        // Number + unit
        if parse_num(word).is_some() && i + 1 < words.len() {
            let next = words[i + 1]
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if units.contains(next.as_str()) {
                entities.push((
                    format!(
                        "{} {}",
                        word,
                        words[i + 1].trim_matches(|c: char| c == ',' || c == ';')
                    ),
                    "number_unit".to_string(),
                ));
                i += 2;
                continue;
            }
        }
        i += 1;
    }
}

fn parse_num(s: &str) -> Option<f64> {
    s.replace(['\'', '\u{2019}', ','], "").parse::<f64>().ok()
}

fn extract_emails(sentence: &str, entities: &mut Vec<(String, String)>) {
    for word in sentence.split_whitespace() {
        let w = word.trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '@' && c != '.' && c != '_' && c != '-' && c != '+'
        });
        if let Some(at) = w.find('@') {
            let local = &w[..at];
            let domain = &w[at + 1..];
            if !local.is_empty() && domain.contains('.') && domain.len() > 3 {
                entities.push((w.to_string(), "email".to_string()));
            }
        }
    }
}

fn extract_urls(sentence: &str, entities: &mut Vec<(String, String)>) {
    for word in sentence.split_whitespace() {
        let w = word.trim_end_matches([',', '.', ')', ']', ';']);
        if w.starts_with("http://") || w.starts_with("https://") {
            entities.push((w.to_string(), "url".to_string()));
        }
    }
}

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

        // Passive voice: "X was developed by Y"
        for i in 0..words.len().saturating_sub(3) {
            let w1 = words[i]
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            let w2 = words[i + 1]
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if matches!(w1.as_str(), "was" | "were" | "is" | "are")
                && (w2.ends_with("ed") || w2.ends_with("en"))
            {
                if let Some(by_pos) = words[i + 2..].iter().position(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                        == "by"
                }) {
                    let by_idx = i + 2 + by_pos;
                    let mut subj = Vec::new();
                    let mut j = i as isize - 1;
                    while j >= 0 {
                        let w = words[j as usize].trim_matches(|c: char| !c.is_alphanumeric());
                        if !w.is_empty() && w.chars().next().is_some_and(|c| c.is_uppercase()) {
                            subj.push(w);
                            j -= 1;
                        } else {
                            break;
                        }
                    }
                    subj.reverse();
                    let mut obj = Vec::new();
                    let mut k = by_idx + 1;
                    while k < words.len() {
                        let w = words[k].trim_matches(|c: char| !c.is_alphanumeric());
                        if !w.is_empty() && w.chars().next().is_some_and(|c| c.is_uppercase()) {
                            obj.push(w);
                            k += 1;
                        } else {
                            break;
                        }
                    }
                    if !subj.is_empty() && !obj.is_empty() {
                        let subject = subj.join(" ");
                        let object = obj.join(" ");
                        if subject != object {
                            relations.push((object, w2.clone(), subject));
                        }
                    }
                }
            }
        }

        // Appositions: "Berlin, the capital of Germany,"
        for i in 1..words.len().saturating_sub(3) {
            let w = words[i].trim_matches(|c: char| !c.is_alphanumeric());
            if !w.is_empty()
                && w.chars().next().is_some_and(|c| c.is_uppercase())
                && words[i].ends_with(',')
            {
                let next = words[i + 1]
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if matches!(
                    next.as_str(),
                    "the" | "a" | "an" | "die" | "der" | "das" | "le" | "la"
                ) {
                    let mut appo = Vec::new();
                    let mut k = i + 2;
                    while k < words.len() {
                        let part = words[k].trim_end_matches([',', '.']);
                        appo.push(part);
                        if words[k].ends_with(',') || words[k].ends_with('.') {
                            break;
                        }
                        k += 1;
                    }
                    if !appo.is_empty() {
                        relations.push((w.to_string(), "is".to_string(), appo.join(" ")));
                    }
                }
            }
        }

        // Standard SVO
        for i in 1..words.len() - 1 {
            let verb = words[i]
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if relation_verbs.contains(verb.as_str()) {
                let mut subj = Vec::new();
                let mut j = i as isize - 1;
                while j >= 0 {
                    let w = words[j as usize].trim_matches(|c: char| !c.is_alphanumeric());
                    if !w.is_empty() && w.chars().next().is_some_and(|c| c.is_uppercase()) {
                        subj.push(w);
                        j -= 1;
                    } else {
                        break;
                    }
                }
                subj.reverse();
                let mut obj = Vec::new();
                let mut k = i + 1;
                while k < words.len() {
                    let w = words[k].trim_matches(|c: char| !c.is_alphanumeric());
                    if !w.is_empty() && w.chars().next().is_some_and(|c| c.is_uppercase()) {
                        obj.push(w);
                        k += 1;
                    } else {
                        break;
                    }
                }
                if !subj.is_empty() && !obj.is_empty() {
                    let subject = subj.join(" ");
                    let object = obj.join(" ");
                    if subject != object {
                        relations.push((subject, verb.clone(), object));
                    }
                }
            }
        }
    }
    relations
}

pub fn extract_keywords(tokens: &[String], max: usize) -> Vec<String> {
    let all_stops: HashSet<&str> = STOP_WORDS_EN
        .iter()
        .chain(STOP_WORDS_DE.iter())
        .chain(STOP_WORDS_FR.iter())
        .chain(STOP_WORDS_IT.iter())
        .chain(STOP_WORDS_ES.iter())
        .copied()
        .collect();
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in tokens {
        if !all_stops.contains(t.as_str()) && t.len() > 2 {
            *freq.entry(t.as_str()).or_insert(0) += 1;
        }
    }
    let mut scored: Vec<(&&str, f64)> = freq
        .iter()
        .map(|(word, count)| {
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

pub fn deduplicate_entities(entities: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut result: Vec<(String, String)> = Vec::new();
    for (name, etype) in entities {
        let is_dup = result.iter().any(|(existing, et)| {
            et == &etype && levenshtein(&name.to_lowercase(), &existing.to_lowercase()) <= 2
        });
        if !is_dup {
            result.push((name, etype));
        }
    }
    result
}

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

pub fn process_text(text: &str, source_url: &str) -> Extracted {
    let language = detect_language(text);
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
        language,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. This is a test! How are you?");
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
        let sentences = vec!["The company Google was founded in California.".into()];
        let entities = extract_entities(&sentences);
        let names: Vec<&str> = entities.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Google"));
        assert!(names.contains(&"California"));
    }

    #[test]
    fn test_extract_entities_years() {
        let sentences = vec!["Rust was released in 2015 by Mozilla.".into()];
        assert!(extract_entities(&sentences)
            .iter()
            .any(|(_, t)| t == "year"));
    }

    #[test]
    fn test_extract_entities_urls() {
        let sentences = vec!["Visit https://rust-lang.org for more info.".into()];
        assert!(extract_entities(&sentences).iter().any(|(_, t)| t == "url"));
    }

    #[test]
    fn test_extract_relations() {
        let sentences = vec!["Google created Android for mobile devices.".into()];
        let relations = extract_relations(&sentences);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].0, "Google");
        assert_eq!(relations[0].1, "created");
        assert_eq!(relations[0].2, "Android");
    }

    #[test]
    fn test_extract_keywords() {
        let tokens = tokenize("Rust is a systems programming language focused on safety and performance and concurrency");
        let kw = extract_keywords(&tokens, 5);
        assert!(!kw.is_empty());
        assert!(kw.len() <= 5);
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
            ("Google".into(), "phrase".into()),
            ("Gogle".into(), "phrase".into()),
            ("Mozilla".into(), "phrase".into()),
        ];
        assert_eq!(deduplicate_entities(entities).len(), 2);
    }

    #[test]
    fn test_process_text() {
        let e = process_text(
            "Google was founded by Larry Page and Sergey Brin in 1998. Google created Android.",
            "https://example.com",
        );
        assert!(!e.entities.is_empty());
        assert_eq!(e.source_url, "https://example.com");
    }

    #[test]
    fn test_stop_words_filtered() {
        let kw = extract_keywords(&tokenize("the quick brown fox jumps over the lazy dog"), 10);
        assert!(!kw.contains(&"the".to_string()));
    }

    #[test]
    fn test_empty_text() {
        let e = process_text("", "https://example.com");
        assert!(e.entities.is_empty());
        assert!(e.relations.is_empty());
    }

    #[test]
    fn test_multi_word_entity() {
        let entities =
            extract_entities(&["The project was led by New York University researchers.".into()]);
        assert!(entities
            .iter()
            .any(|(n, _)| n.contains("New York University")));
    }

    // ===== NEW TESTS =====

    #[test]
    fn test_detect_english() {
        assert_eq!(
            detect_language(
                "The quick brown fox jumps over the lazy dog and is very happy about it."
            ),
            Language::English
        );
    }

    #[test]
    fn test_detect_german() {
        assert_eq!(
            detect_language("Der schnelle braune Fuchs springt und ist sehr hier dort auch nicht."),
            Language::German
        );
    }

    #[test]
    fn test_detect_french() {
        assert_eq!(
            detect_language(
                "Le renard brun rapide saute dans le jardin avec les enfants pour nous."
            ),
            Language::French
        );
    }

    #[test]
    fn test_detect_italian() {
        assert_eq!(
            detect_language("Il rapido volpe marrone salta con la porta nella casa della noi."),
            Language::Italian
        );
    }

    #[test]
    fn test_detect_spanish() {
        assert_eq!(detect_language("El zorro marrón salta sobre el perro perezoso en la casa con los niños para todos."), Language::Spanish);
    }

    #[test]
    fn test_german_date_format() {
        let entities = extract_entities(&["Das Treffen findet am 15. Januar 2024 statt.".into()]);
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "date" && n.contains("Januar") && n.contains("2024")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_german_compound_noun() {
        let entities =
            extract_entities(
                &["Die Softwareentwicklung ist in der Bundesrepublik wichtig.".into()],
            );
        // Softwareentwicklung is filtered as generic German compound noun (ends in -entwicklung)
        // Bundesrepublik should be extracted as a concept
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "concept" && n.contains("Bundesrepublik")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_french_date() {
        let entities = extract_entities(&["La date est le 5 février 2024 pour nous.".into()]);
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "date" && n.contains("février")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_italian_date() {
        let entities = extract_entities(&["La data del 10 gennaio 2024 per Roma.".into()]);
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "date" && n.contains("gennaio")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_iso_date() {
        let entities =
            extract_entities(&["The release date is 2024-01-15 for the new version.".into()]);
        assert!(entities
            .iter()
            .any(|(n, t)| t == "date" && n == "2024-01-15"));
    }

    #[test]
    fn test_us_date_format() {
        let entities = extract_entities(&["The event is on 01/15/2024 in the city.".into()]);
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "date" && n.contains("01/15/2024")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_eu_date_format() {
        let entities = extract_entities(&["Das Datum ist 15.01.2024 und das ist wichtig.".into()]);
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "date" && n.contains("15.01.2024")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_relative_date_english() {
        let entities =
            extract_entities(&["I saw it yesterday and will check next week again.".into()]);
        let rel: Vec<&str> = entities
            .iter()
            .filter(|(_, t)| t == "relative_date")
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(rel.contains(&"yesterday"));
        assert!(rel.contains(&"next week"));
    }

    #[test]
    fn test_relative_date_german() {
        let entities =
            extract_entities(&["Ich habe es gestern gesehen und vorgestern besprochen.".into()]);
        let rel: Vec<&str> = entities
            .iter()
            .filter(|(_, t)| t == "relative_date")
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(rel.contains(&"gestern"));
    }

    #[test]
    fn test_number_unit_gb() {
        let entities = extract_entities(&["The disk has 750 GB of storage available.".into()]);
        assert!(entities
            .iter()
            .any(|(n, t)| t == "number_unit" && n.contains("750") && n.contains("GB")));
    }

    #[test]
    fn test_number_unit_ghz() {
        let entities = extract_entities(&["The processor runs at 3.5 GHz for performance.".into()]);
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "number_unit" && n.contains("3.5") && n.contains("GHz")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_percentage() {
        let entities =
            extract_entities(&["Performance improved by 42% compared to before.".into()]);
        assert!(entities
            .iter()
            .any(|(n, t)| t == "number_unit" && n.contains("42%")));
    }

    #[test]
    fn test_currency_dollar() {
        let entities = extract_entities(&["The acquisition cost $1.2 million in total.".into()]);
        assert!(entities.iter().any(|(_, t)| t == "currency"));
    }

    #[test]
    fn test_currency_euro() {
        let entities = extract_entities(&["Das kostet nur etwa €500 pro Monat.".into()]);
        assert!(entities
            .iter()
            .any(|(n, t)| t == "currency" && n.contains("€500")));
    }

    #[test]
    fn test_currency_chf() {
        let entities = extract_entities(&["Der Preis ist CHF 1'200 pro Jahr.".into()]);
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "currency" && n.contains("CHF")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_email_extraction() {
        let entities = extract_entities(&["Contact us at info@example.com for details.".into()]);
        assert!(entities
            .iter()
            .any(|(n, t)| t == "email" && n == "info@example.com"));
    }

    #[test]
    fn test_url_extraction() {
        let entities = extract_entities(&["Visit https://www.rust-lang.org for docs.".into()]);
        assert!(entities
            .iter()
            .any(|(n, t)| t == "url" && n.contains("rust-lang.org")));
    }

    #[test]
    fn test_sentence_boundary_abbreviations() {
        let sentences = split_sentences("Dr. Smith went to the store. He bought milk.");
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("Dr."));
        assert!(sentences[0].contains("store"));
    }

    #[test]
    fn test_sentence_boundary_mr() {
        let sentences =
            split_sentences("Mr. Jones and Mrs. Smith attended the meeting. It was productive.");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_passive_voice_relation() {
        let relations = extract_relations(&[
            "The framework Android was developed by Google in California.".into(),
        ]);
        assert!(
            relations
                .iter()
                .any(|(s, v, o)| s.contains("Google") && v == "developed" && o.contains("Android")),
            "relations: {:?}",
            relations
        );
    }

    #[test]
    fn test_apposition_relation() {
        let relations =
            extract_relations(&["The city of Berlin, the capital of Germany, is vibrant.".into()]);
        assert!(
            relations
                .iter()
                .any(|(s, v, _)| s.contains("Berlin") && v == "is"),
            "relations: {:?}",
            relations
        );
    }

    #[test]
    fn test_process_text_language_detection() {
        let e = process_text(
            "Der schnelle braune Fuchs springt und ist sehr hier dort auch nicht mehr.",
            "https://example.de",
        );
        assert_eq!(e.language, Language::German);
    }

    #[test]
    fn test_german_stopwords_filtered() {
        let kw = extract_keywords(
            &tokenize("der schnelle braune Fuchs springt über den faulen Hund"),
            10,
        );
        assert!(!kw.contains(&"der".to_string()));
        assert!(!kw.contains(&"den".to_string()));
    }

    #[test]
    fn test_mixed_entities() {
        let entities = extract_entities(&[
            "Contact John at john@example.com about the 750 GB server costing $500 by 2024-03-15."
                .into(),
        ]);
        let types: HashSet<&str> = entities.iter().map(|(_, t)| t.as_str()).collect();
        assert!(types.contains("email"), "Missing email in {:?}", entities);
        assert!(
            types.contains("number_unit"),
            "Missing number_unit in {:?}",
            entities
        );
        assert!(
            types.contains("currency"),
            "Missing currency in {:?}",
            entities
        );
        assert!(types.contains("date"), "Missing date in {:?}", entities);
    }

    // ===== NLP NOISE FILTER TESTS =====

    #[test]
    fn test_blacklist_filters_wikipedia_chrome() {
        let sentences =
            vec!["The page shows Wikipedia and Navigation links to Random articles.".into()];
        let entities = extract_entities(&sentences);
        let names: Vec<String> = entities.iter().map(|(n, _)| n.to_lowercase()).collect();
        assert!(
            !names.contains(&"wikipedia".to_string()),
            "entities: {:?}",
            entities
        );
        assert!(
            !names.contains(&"navigation".to_string()),
            "entities: {:?}",
            entities
        );
        assert!(
            !names.contains(&"random".to_string()),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_blacklist_filters_wikimedia() {
        let sentences = vec!["Powered by Wikimedia Foundation and Community Portal links.".into()];
        let entities = extract_entities(&sentences);
        let names: Vec<String> = entities.iter().map(|(n, _)| n.to_lowercase()).collect();
        assert!(
            !names.iter().any(|n| n.contains("wikimedia")),
            "entities: {:?}",
            entities
        );
        assert!(
            !names.iter().any(|n| n.contains("community")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_blacklist_filters_upload_special() {
        let sentences = vec!["Click Upload or Special pages to find Help Learn more.".into()];
        let entities = extract_entities(&sentences);
        let names: Vec<String> = entities.iter().map(|(n, _)| n.to_lowercase()).collect();
        assert!(
            !names.contains(&"upload".to_string()),
            "entities: {:?}",
            entities
        );
        assert!(
            !names.contains(&"special".to_string()),
            "entities: {:?}",
            entities
        );
        assert!(
            !names.iter().any(|n| n.contains("help")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_entity_min_length() {
        // Entities shorter than 3 chars should be filtered
        let sentences = vec!["The AI is good at NLP research tasks.".into()];
        let entities = extract_entities(&sentences);
        let names: Vec<&str> = entities.iter().map(|(n, _)| n.as_str()).collect();
        // "AI" is only 2 chars, should be filtered
        assert!(!names.contains(&"AI"), "entities: {:?}", entities);
        // "NLP" is 3 chars, should be kept
        assert!(names.contains(&"NLP"), "entities: {:?}", entities);
    }

    #[test]
    fn test_entity_max_length() {
        // Entities over 100 chars should be filtered
        let long_name = "A".repeat(101);
        assert!(!is_valid_entity(&long_name, "phrase"));
        assert!(is_valid_entity("Normal Entity", "phrase"));
    }

    #[test]
    fn test_entity_all_uppercase_rejected() {
        // All-uppercase entities > 6 alpha chars should be rejected
        assert!(!is_valid_entity("NAVIGATION PANEL", "phrase"));
        assert!(!is_valid_entity("WIKIMEDIA", "phrase"));
        // Short acronyms are OK
        assert!(is_valid_entity("NASA", "phrase"));
        assert!(is_valid_entity("UNESCO", "phrase"));
    }

    #[test]
    fn test_entity_starting_with_number_rejected() {
        assert!(!is_valid_entity("123Company", "phrase"));
        assert!(!is_valid_entity("42nd Street", "phrase"));
        // But dates/measurements starting with numbers are fine
        assert!(is_valid_entity("2024-01-15", "date"));
        assert!(is_valid_entity("500 GB", "number_unit"));
    }

    #[test]
    fn test_phrase_max_6_words() {
        // A long sequence of capitalized words should be capped at 6
        let sentences = vec![
            "We visited The Very Long Name Organization Department Division Bureau Section yesterday.".into(),
        ];
        let entities = extract_entities(&sentences);
        for (name, etype) in &entities {
            if etype != "date"
                && etype != "url"
                && etype != "email"
                && etype != "currency"
                && etype != "number_unit"
                && etype != "year"
            {
                let word_count = name.split_whitespace().count();
                assert!(
                    word_count <= 6,
                    "Entity '{}' has {} words, max 6",
                    name,
                    word_count
                );
            }
        }
    }

    #[test]
    fn test_linking_words_bridge_entities() {
        // "University of California" should be captured as a single entity
        let sentences =
            vec!["He studied at the University of California at Berkeley in 1990.".into()];
        let entities = extract_entities(&sentences);
        let names: Vec<String> = entities.iter().map(|(n, _)| n.clone()).collect();
        assert!(
            names.iter().any(|n| n.contains("University of California")),
            "Expected 'University of California' entity, got: {:?}",
            names
        );
    }

    #[test]
    fn test_linking_words_require_capitalized_follower() {
        // Linking word "of" should NOT bridge if next word is lowercase
        let sentences = vec!["The Battle of the ages was fierce.".into()];
        let entities = extract_entities(&sentences);
        let names: Vec<String> = entities.iter().map(|(n, _)| n.clone()).collect();
        // Should not create "Battle of the ages"
        assert!(
            !names.iter().any(|n| n.contains("ages")),
            "Should not bridge to lowercase: {:?}",
            names
        );
    }

    #[test]
    fn test_unicode_en_dash_no_panic() {
        // This should NOT panic on en-dash '–' or other multi-byte chars
        let text = "The event – held on 5 février 2024 – was in Paris.";
        let sentences = split_sentences(text);
        let entities = extract_entities(&sentences);
        // Should not panic, and should extract the date
        assert!(
            entities
                .iter()
                .any(|(n, t)| t == "date" && n.contains("février")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_unicode_em_dash_no_panic() {
        let text = "Einstein—born 14 März 1879—changed physics forever.";
        let sentences = split_sentences(text);
        let _entities = extract_entities(&sentences);
        // Just verify no panic
    }

    #[test]
    fn test_unicode_mixed_no_panic() {
        // Text with various multi-byte characters
        let text = "Zürich's café – founded in März 2020 – serves crème brûlée.";
        let sentences = split_sentences(text);
        let _entities = extract_entities(&sentences);
        // No panic = success
    }

    #[test]
    fn test_real_entity_not_filtered() {
        // Real knowledge entities should pass all filters
        let sentences = vec![
            "Albert Einstein developed the theory of relativity at Princeton University.".into(),
        ];
        let entities = extract_entities(&sentences);
        let names: Vec<&str> = entities.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.iter().any(|n| n.contains("Einstein")),
            "entities: {:?}",
            entities
        );
        assert!(
            names.iter().any(|n| n.contains("Princeton")),
            "entities: {:?}",
            entities
        );
    }

    #[test]
    fn test_single_word_person_titles_rejected() {
        // Single-word person titles like "President", "Queen" should be rejected
        assert!(!is_valid_entity("President", "person"));
        assert!(!is_valid_entity("Queen", "person"));
        assert!(!is_valid_entity("Minister", "person"));
        // But multi-word names with titles are fine
        assert!(is_valid_entity("President Obama", "person"));
        assert!(is_valid_entity("Queen Elizabeth", "person"));
    }

    #[test]
    fn test_generic_adjectives_rejected() {
        // Single-word adjectives/adverbs should not be entities
        assert!(!is_valid_entity("Ancient", "concept"));
        assert!(!is_valid_entity("Dynamic", "concept"));
        assert!(!is_valid_entity("Massive", "concept"));
        assert!(!is_valid_entity("Noble", "concept"));
        // But real entity names that happen to be adjectives are fine in multi-word
        assert!(is_valid_entity("Noble Prize", "concept"));
    }

    #[test]
    fn test_process_text_filters_noise() {
        let e = process_text(
            "Wikipedia is a free encyclopedia. Navigation links help users. Albert Einstein was a physicist.",
            "https://en.wikipedia.org/wiki/Test",
        );
        let names: Vec<String> = e.entities.iter().map(|(n, _)| n.to_lowercase()).collect();
        assert!(
            !names.contains(&"wikipedia".to_string()),
            "entities: {:?}",
            e.entities
        );
        assert!(
            !names.contains(&"navigation".to_string()),
            "entities: {:?}",
            e.entities
        );
        assert!(
            names.iter().any(|n| n.contains("einstein")),
            "entities: {:?}",
            e.entities
        );
    }

    #[test]
    fn test_geo_suffix_classification() {
        assert_eq!(classify_entity_type("Middletown"), "place");
        assert_eq!(classify_entity_type("Stuttgart"), "place");
        assert_eq!(classify_entity_type("Hagerstown"), "place");
        assert_eq!(classify_entity_type("Amphipolis"), "place");
        assert_eq!(classify_entity_type("Freetown"), "place");
        assert_eq!(classify_entity_type("Heidelberg"), "place");
        assert_eq!(classify_entity_type("Düsseldorf"), "place");
        // Surnames ending in -ford should still be short enough to match,
        // but person classification takes priority for multi-word
        assert_eq!(classify_entity_type("Newport"), "place");
    }
}
