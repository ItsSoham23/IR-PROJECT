"""Pre-model candidate selection, shared by ingestion audits and preparation."""

DEFAULT_CATEGORIES = ("Campus", "Politics", "Sports", "Technology")
TARGET_PER_CATEGORY = 64
TRAIN_PER_CATEGORY = 51
SELECTION_POLICY = {
    "version": "centered_extracted_length_v1",
    "ordering": ["extracted_word_count", "document_id"],
    "subset": "centered contiguous subset; start = (valid_count - target_count) // 2",
    "uses_split_seed": False,
    "uses_model_results": False,
}


def select_candidates(documents, target_per_category=TARGET_PER_CATEGORY,
                      categories=DEFAULT_CATEGORIES):
    """Select by extraction length and ID before any split or model is fitted."""
    selected = []
    for category in categories:
        pool = sorted((d for d in documents if d["category"] == category),
                      key=lambda d: (d["word_count"], d["document_id"]))
        if len(pool) < target_per_category:
            raise ValueError(f"{category}: need {target_per_category} valid documents, found {len(pool)}")
        offset = (len(pool) - target_per_category) // 2
        selected.extend(sorted(pool[offset:offset + target_per_category],
                               key=lambda d: d["document_id"]))
    return selected
