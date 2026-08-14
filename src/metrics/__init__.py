"""Derived metrics that are not part of the per-match analysis pipeline.

src/processor.py computes what is stored with a match. This package holds the
metrics computed across matches at request time — behavioural axes, role
classification — which had accumulated inside api.py between route handlers.
"""
