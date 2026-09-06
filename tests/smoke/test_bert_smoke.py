"""Smoke test for the BERT prediction pipeline (src/modeling/dl/predict.py).

Wraps the module's own `_run_smoke_test()` rather than duplicating it: that function already
exercises train_final_model -> save_model -> load_model -> predict / predict_proba (both the
PyTorch LegalBertClassifier and an sklearn LogisticRegression head, via duck-typing) and
predict_bert / predict_bert_proba on 3 documents (a monkeypatched AutoModel, so no GPU or
HuggingFace download is required), for both case types.
"""

from __future__ import annotations


def test_bert_predict_pipeline_smoke():
    import predict  # from src/modeling/dl, added to sys.path by tests/conftest.py

    predict._run_smoke_test()
