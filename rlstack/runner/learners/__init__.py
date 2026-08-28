"""Learners: real training metal behind the Learner protocol, one per file.

torch_learner imports torch at module scope, so import it lazily — only where a
GPU run is being assembled, never from the rlstack package root (STYLE rule 7).
"""
