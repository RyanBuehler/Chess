"""Autotelic goal playground: value-agnostic, deadline-bounded goals.

Goals are expressed in the board's own rule-level feature vocabulary (so the
goal vocabulary equals the state vocabulary -> completeness by construction).
This package provides:

- features:  rule-level state-feature extraction from a chess.Board.
- templates: goal templates (the delta vocabulary) + canonical identity.
- verifier:  exact achieved-by-deadline? over a game record.
- encoding:  goal -> conditioning planes + deadline scalar.
"""
