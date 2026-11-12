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


def clean_bitscan_pages(pages: List[str]) -> str:
    """Remove identifiable PDF furniture, preserving prose and campus vocabulary.

    Repetition alone is insufficient: only short lines at page edges repeated
    on at least half the pages (minimum two) are treated as running furniture.
    TOC entries need dot leaders; editorial prose is deliberately retained.
    """
    lines_by_page = [
        [re.sub(r"[ \t]+", " ", line).strip() for line in page.splitlines()
         if line.strip()]
        for page in pages
    ]
    edges = Counter()
    for lines in lines_by_page:
        edges.update(set(line.casefold() for line in lines[:2] + lines[-2:]
                         if len(line.split()) <= 8
                         and not re.search(r"[.!?]$", line)
                         and not re.search(r"\b(students?|faculty)\b", line, re.I)))
    repeated = {line for line, count in edges.items()
                if count >= max(2, (len(pages) + 1) // 2)}
    furniture = re.compile(
        r"(?:bitscan(?:\s*[-:|]?\s*\d+)?|(?:page\s+)?\d+(?:\s+of\s+\d+)?|"
        r"issue\s*(?:no\.?\s*)?\d+|"
        r"(?:[iv12]+\s+)?semester\s*(?:[iv12]+)?\s*[,/-]?\s*\d{4}[-–/]\d{2,4}|"
        r"(?:table of )?contents|editorial (?:board|team)|"
        r"(?:printed|published|edited) by\s*:.*|"
        r"(?:email|e-mail|phone|fax|contact us)\s*:.*)",
        re.IGNORECASE,
    )
    cleaned_pages = []
    for lines in lines_by_page:
        kept = []
        for index, line in enumerate(lines):
            at_edge = index < 2 or index >= len(lines) - 2
            layout_line = (
                len(line.split()) <= 18
                and re.search(r"\bbitscan\b", line, re.I)
                and not re.search(r"\b(students?|faculty|research|events?|welcome)\b", line, re.I)
                and re.fullmatch(
                    r"[\d\s,.:|/–-]*(?:(?:bitscan|bits|pilani|hyderabad|goa|campus|"
                    r"k|birla|semester|sem|issue|notes|no|i|ii|jan|june|july|dec|"
                    r"january|december)[\d\s,.:|/–-]*)+", line, re.I)
            )
            if (furniture.fullmatch(line)
                    or re.fullmatch(r".+\.{3,}\s*\d+(?:[-–]\d+)?", line)
                    or layout_line
                    or (at_edge and line.casefold() in repeated)):
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept))
    return "\n\n".join(page for page in cleaned_pages if page).strip()

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
