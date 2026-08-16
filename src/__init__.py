"""
ModernLayoutLM: LayoutLM-style 2D layout embeddings on a ModernBERT backbone

A four stage pipeline of data, model, train and eval, each importing only from stages before it
Submodules are imported by the caller rather than re-exported, so src.data never drags in torch
"""
