"""
Adversarial Poisoning and Control Document Generators.

Implements:
- Baseline Poison Attack: Repeating top-N keywords from the target topic.
- Bridging Poison Attack: Blending keywords between target topic and other categories to blur boundaries.
- Condition C Random Control: Uniformly sampled words from the vocabulary to control for corpus growth.
"""

from typing import List, Optional, Set
import numpy as np


class AdversarialPoisonGenerator:
    """
    Constructs injected documents for Condition B (Poisoning) and Condition C (Random Control).
    """

    def __init__(self, vocab: List[str], random_state: Optional[int] = None):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.rng = np.random.RandomState(random_state)

    def generate_keyword_repetition_attack(
        self,
        target_words: List[str],
        n_docs: int = 10,
        doc_length: int = 100,
        noise_ratio: float = 0.05,
    ) -> List[str]:
        """
        Baseline poison attack:
        Repeats top keywords of the target topic with optional slight word noise.
        """
        poison_docs = []
        target_indices = [self.word_to_idx[w] for w in target_words if w in self.word_to_idx]
        if not target_indices:
            raise ValueError("None of the target words are present in the vocabulary.")

        for _ in range(n_docs):
            doc_words = []
            for _ in range(doc_length):
                if self.rng.rand() < noise_ratio:
                    # Random word noise to evade simple string exact match
                    word_idx = self.rng.randint(0, self.vocab_size)
                else:
                    word_idx = self.rng.choice(target_indices)
                doc_words.append(self.vocab[word_idx])
            poison_docs.append(" ".join(doc_words))

        return poison_docs

    def generate_bridging_attack(
        self,
        target_words: List[str],
        confuser_words: List[str],
        n_docs: int = 10,
        doc_length: int = 100,
        target_ratio: float = 0.5,
    ) -> List[str]:
        """
        Bridging poison attack:
        Mixes target topic vocabulary with vocabulary from a confusing category
        to pull the learned topic toward that category and blur topic boundaries.
        """
        target_indices = [self.word_to_idx[w] for w in target_words if w in self.word_to_idx]
        confuser_indices = [self.word_to_idx[w] for w in confuser_words if w in self.word_to_idx]

        poison_docs = []
        for _ in range(n_docs):
            doc_words = []
            for _ in range(doc_length):
                if self.rng.rand() < target_ratio:
                    w_idx = self.rng.choice(target_indices)
                else:
                    w_idx = self.rng.choice(confuser_indices)
                doc_words.append(self.vocab[w_idx])
            poison_docs.append(" ".join(doc_words))

        return poison_docs

    def generate_random_control(
        self,
        n_docs: int = 10,
        doc_length: int = 100,
    ) -> List[str]:
        """
        Condition C control:
        Generates documents with words chosen uniformly at random from the vocabulary.
        """
        control_docs = []
        for _ in range(n_docs):
            sampled_indices = self.rng.randint(0, self.vocab_size, size=doc_length)
            words = [self.vocab[idx] for idx in sampled_indices]
            control_docs.append(" ".join(words))

        return control_docs
