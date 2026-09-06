"""Deep-learning pipeline for the Legal BERTimbau court-decision classifiers.

Bundles training (``train.py``), inference (``predict.py``), data/feature
loading (``features.py``), and embedding-extraction utilities
(``legal_bertimbau_tokenization_embedding.py``,
``extract_bert_embeddings.py``) for the two fine-tuned classification tasks:
DV (binary Kept/Altered) and BoC (ternary Favourable/Unfavourable/Partial),
predicted from Portuguese court decision text.
"""
