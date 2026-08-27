"""Engines: real inference metal behind the Engine protocol, one per file.

vllm_engine imports vLLM at module scope, so import it lazily
(`from rlstack.runner.engines.vllm_engine import VllmEngine`) only where a
GPU run is being assembled — never from the rlstack package root.
"""
