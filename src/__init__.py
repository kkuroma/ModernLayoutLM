"""
ModernLayoutLM: LayoutLM-style 2D layout embeddings on a ModernBERT backbone

The package is a four stage pipeline. Each stage imports only from the stages before it, a
rule enforced by the `import-linter` layers contract in `pyproject.toml`.

- `src.data`: acquire raw datasets and normalize them into one canonical schema.
- `src.model`: the config, modeling and processor code. Self contained, exportable to the Hub.
- `src.train`: pretraining and fine-tuning loops.
- `src.eval`: metrics and result reporting.

Submodules are imported explicitly by the caller rather than re-exported here, so that pulling
in `src.data` never drags torch along with it.
"""
