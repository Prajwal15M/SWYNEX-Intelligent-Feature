"""Offline test of the span-extraction code using a tiny RANDOM DistilBERT.
It checks shapes/offset handling, not answer quality (no weights are downloaded)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForQuestionAnswering

import dataqa

CONTEXT = "The region with the highest revenue was East, which generated 900 dollars."
QUESTION = "Which region generated the highest revenue?"


@pytest.fixture()
def tiny(tmp_path):
    words = sorted(set((CONTEXT + " " + QUESTION).replace(",", " , ").replace(".", " . ").replace("?", " ? ").split()))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    vf = tmp_path / "vocab.txt"
    vf.write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(vf), do_lower_case=False)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=1, n_heads=2, max_position_embeddings=512)
    model = DistilBertForQuestionAnswering(cfg).eval()
    return model, tok


def test_extract_returns_span_from_context_and_valid_probability(tiny):
    model, tok = tiny
    text, score = dataqa.extract_answer(model, tok, QUESTION, CONTEXT)
    assert text and text in CONTEXT          # answer is a real slice of the context
    assert 0.0 < score <= 1.0                # joint probability


def test_span_length_is_capped(tiny):
    model, tok = tiny
    text, _ = dataqa.extract_answer(model, tok, QUESTION, CONTEXT, max_answer_tokens=3)
    assert len(tok.tokenize(text)) <= 3
