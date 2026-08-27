"""Learners: real training metal behind the Learner protocol, one per file.

torch_learner imports torch at module scope, so import it lazily
(`from rlstack.runner.learners.torch_learner import TorchLearner`) only where
a GPU run is being assembled — never from the rlstack package root.
"""
