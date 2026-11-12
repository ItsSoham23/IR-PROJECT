"""
Text Preprocessing and Vocabulary Construction.

Enforces protocol requirements:
- Vocabulary is learned strictly from training documents.
- Lowercasing, tokenization, stopword removal, and minimum document frequency threshold.
- Generates sparse or dense document-term count matrices.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse as sp


# Restricted vocabulary for observed publication furniture, not campus prose.
_BITSCAN_LAYOUT_WORDS = set("bits bitscan pilani hyderabad goa campus k birla institute of technology and science semester sem notes issue no i ii iii jan january june jun july jul dec december august aug september sept october november march april may february feb".split())


def _bitscan_layout_signature(line):
    """Canonicalize observed page-edge furniture, including changing page numbers."""
    words = re.findall(r"[a-z]+", line.casefold())
    if not words or len(words) > 22 or not set(words) <= _BITSCAN_LAYOUT_WORDS:
        return None
    if not set(words) & {"bits", "bitscan", "campus", "semester", "institute"}:
        return None
    # Only boundary page numbers vary; retain issue numbers and year ranges.
    line = re.sub(r"^\s*[-\u2013]?\s*\d{1,3}\s*(?:[-\u2013]\s*|\s+)", "", line)
    line = re.sub(r"\s+\d{1,3}\s*$", "", line)
    return re.sub(r"[\W_]+", " ", line.casefold()).strip()


def clean_bitscan_pages(pages: List[str]) -> str:
    """Remove observed furniture; retain prose, paragraph and page boundaries.

    A restricted layout signature must recur at page edges on two or more pages.
    Changing boundary page numbers are ignored, while years/issues remain part
    of the signature. No substantive campus terms are globally blacklisted.
    Form feeds preserve pages for deterministic model-section selection.
    """
    lines_by_page = [[re.sub(r"[ \t]+", " ", line).strip()
                      for line in page.splitlines()] for page in pages]
    edges = Counter()
    for lines in lines_by_page:
        nonempty = [line for line in lines if line]
        signatures = {_bitscan_layout_signature(line)
                      for line in nonempty[:3] + nonempty[-3:]}
        edges.update(signature for signature in signatures if signature)
    repeated = {signature for signature, count in edges.items() if count >= 2}
    furniture = re.compile(
        r"(?:bitscan(?:\s*[-:|]?\s*\d+)?|(?:page\s+)?\d+(?:\s+of\s+\d+)?|"
        r"issue\s*(?:no\.?\s*)?\d+|"
        r"(?:[iv12]+\s+)?semester\s*(?:[iv12]+)?\s*[,/-]?\s*\d{4}[-\u2013/]\d{2,4}|"
        r"editorial (?:board|team)|(?:printed|published|edited) by\s*:.*|"
        r"(?:email|e-mail|phone|fax|contact us)\s*:.*)", re.IGNORECASE)
    cleaned_pages = []
    for lines in lines_by_page:
        nonempty_indices = [i for i, line in enumerate(lines) if line]
        edge_indices = set(nonempty_indices[:3] + nonempty_indices[-3:])
        kept = []
        for index, line in enumerate(lines):
            if (furniture.fullmatch(line)
                    or re.fullmatch(r".+\.{3,}\s*\d+(?:[-\u2013]\d+)?", line)
                    or (index in edge_indices and _bitscan_layout_signature(line) in repeated)):
                continue
            kept.append(line)
        cleaned_pages.append(re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip())
    return "\f".join(cleaned_pages)


_WIKINEWS_NONARTICLE_SECTIONS = {
    "source", "sources", "references", "related news", "related stories", "related",
    "related wikinews",
    "external link", "external links", "sister links", "see also",
}
_WIKINEWS_DATE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}")


def clean_wikinews_text(text: str) -> str:
    """Remove observed metadata/sections, never individual journalistic words.

    API headings and HTML headings (serialized to the same syntax) delimit
    sections. Skip only named non-article sections and their subsections;
    resume at the next peer article heading, preserving interviews and prose.
    """
    cleaned = []
    skipped_level = None
    for line in text.splitlines():
        stripped = line.strip()
        if not cleaned and _WIKINEWS_DATE.fullmatch(stripped):
            continue
        heading = re.fullmatch(r"(={2,6})\s*(.*?)\s*\1", stripped)
        if heading:
            level, title = len(heading[1]), heading[2].casefold()
            if skipped_level is not None and level <= skipped_level:
                skipped_level = None
            if title in _WIKINEWS_NONARTICLE_SECTIONS and skipped_level is None:
                skipped_level = level
        if skipped_level is not None:
            continue
        if stripped == "Have an opinion on this story? Share it!":
            continue
        if stripped or cleaned:
            cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()

# Standard English stopwords (self-contained to avoid external download dependencies)
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "will", "just", "also", "would", "one", "two", "like", "use", "using", "first"
}


class TextPreprocessor:
    """
    Tokenizer and document-term matrix builder.
    """

    def __init__(
        self,
        min_df: int = 2,
        max_df_ratio: float = 0.95,
        min_word_len: int = 3,
        stopwords: Optional[Set[str]] = None,
    ):
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.min_word_len = min_word_len
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.token_pattern = re.compile(r"\b[a-zA-Z]+\b")

        self.vocab_: List[str] = []
        self.word_to_idx_: Dict[str, int] = {}
        self.doc_freqs_: Dict[str, int] = {}

    def tokenize(self, text: str) -> List[str]:
        """Convert raw text to lowercased tokens filtering punctuation and stopwords."""
        text = self.clean_text(text)
        tokens = self.token_pattern.findall(text.lower())
        return [
            t
            for t in tokens
            if len(t) >= self.min_word_len and t not in self.stopwords
        ]

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove scraper metadata and repeated BITS navigation boilerplate."""
        lines = [line.strip() for line in text.splitlines()]
        if "Upload Screenshot" in lines:
            lines = lines[lines.index("Upload Screenshot") + 1 :]
        if "Latest news" in lines:
            lines = lines[: lines.index("Latest news")]

        boilerplate = {
            "Integrated First Degree",
            "Higher Degree",
            "Doctoral Programmes",
            "Give us your feedback",
            "Upload Screenshot",
            "Keep Exploring",
            "University Home",
            "BITSoM, Mumbai",
            "BITSLAW, Mumbai",
            "Academics",
            "Admission",
            "Research & Innovation",
            "International Admissions",
            "BITS Pilani Digital",
        }
        cleaned = []
        seen = set()
        for line in lines:
            if not line or line in boilerplate or line in seen:
                continue
            seen.add(line)
            cleaned.append(line)
        return "\n".join(cleaned)

    def fit(self, raw_documents: List[str]) -> "TextPreprocessor":
        """
        Build vocabulary strictly from training corpus documents.
        """
        doc_counts: Dict[str, int] = {}
        n_docs = len(raw_documents)

        for doc in raw_documents:
            unique_words = set(self.tokenize(doc))
            for w in unique_words:
                doc_counts[w] = doc_counts.get(w, 0) + 1

        self.doc_freqs_ = doc_counts
        max_df = int(self.max_df_ratio * n_docs)

        # Filter by min_df and max_df
        filtered_vocab = [
            w
            for w, df in doc_counts.items()
            if df >= self.min_df and df <= max_df
        ]
        # Sort vocabulary alphabetically for determinism
        self.vocab_ = sorted(filtered_vocab)
        self.word_to_idx_ = {w: i for i, w in enumerate(self.vocab_)}

        return self

    def transform(
        self, raw_documents: List[str], to_sparse: bool = False
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Transform raw documents into count matrix N of shape (D, V).
        Words not in training vocabulary are ignored (or mapped to OOV).
        """
        if not self.word_to_idx_:
            raise ValueError("Preprocessor must be fitted before transforming.")

        D = len(raw_documents)
        V = len(self.vocab_)

        if to_sparse:
            rows, cols, data = [], [], []
            for d, doc in enumerate(raw_documents):
                tokens = self.tokenize(doc)
                word_counts: Dict[int, int] = {}
                for t in tokens:
                    if t in self.word_to_idx_:
                        idx = self.word_to_idx_[t]
                        word_counts[idx] = word_counts.get(idx, 0) + 1
                for idx, count in word_counts.items():
                    rows.append(d)
                    cols.append(idx)
                    data.append(count)
            return sp.csr_matrix((data, (rows, cols)), shape=(D, V), dtype=np.float64)
        else:
            matrix = np.zeros((D, V), dtype=np.float64)
            for d, doc in enumerate(raw_documents):
                tokens = self.tokenize(doc)
                for t in tokens:
                    if t in self.word_to_idx_:
                        matrix[d, self.word_to_idx_[t]] += 1.0
            return matrix

    def fit_transform(
        self, raw_documents: List[str], to_sparse: bool = False
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """Fit vocabulary and return count matrix."""
        return self.fit(raw_documents).transform(raw_documents, to_sparse=to_sparse)
