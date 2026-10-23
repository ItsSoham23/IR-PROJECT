import numpy as np

from src.defense import DefenseFilter
from src.preprocess import TextPreprocessor


def test_bits_navigation_is_removed_from_text():
    text = """Upload Screenshot
    Integrated First Degree
    Article body about research methods
    Latest news
    Campus
    """
    cleaned = TextPreprocessor.clean_text(text)
    assert "Article body" in cleaned
    assert "Integrated First Degree" not in cleaned
    assert "Campus" not in cleaned


def test_deduplication_protects_real_documents():
    clean = np.eye(3)
    poison = np.array([[1.0, 0.0, 0.0]])
    filtered, kept = DefenseFilter.deduplicate_documents(
        np.vstack([clean, poison]), threshold=0.75, protected_indices=[0, 1, 2]
    )
    assert kept == [0, 1, 2]
    np.testing.assert_array_equal(filtered, clean)
