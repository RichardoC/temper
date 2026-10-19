"""MultiPL-E evaluation adapter (M2.0): export + container scoring + pass@k.

Submodules:
  export -- temper run records -> MultiPL-E completion files (+ manifest)
  score  -- drive the official MultiPL-E container, compute unbiased pass@k

Design contract (docs/m2-design.md M2.0): our steering code never touches
scoring internals — temper only *produces completion files* in the official
format and *consumes* the container's results files.
"""
