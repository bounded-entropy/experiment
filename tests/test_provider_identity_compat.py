"""Finite aliases preserve legacy math; future operation edits retain new identity."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import rlstack
from rlstack.policy.adapters.identity_compat import EXTRACTED_DEFINITIONS
from rlstack.registry import ADAPTER_TYPES, source_hash


def combined(entry):
    return hashlib.sha256((source_hash(entry.cls) + entry.instance.definition.source_hash()).encode()).hexdigest()


class ProviderIdentityCompatibilityTest(unittest.TestCase):
    def test_guard_only_provider_edits_keep_the_published_legacy_digests(self):
        expected = {
            "attn_bias": "8787c22bbd90a20cfa66b85604c5b4056687d6e042ee2a0450cfabb93794c1cb",
            "lora": "f8696c474dd9774d92516a22a3730e9a25e9bde1b7fba9a7517f1d5ffc492c5d",
            "plora": "5637bcd111bc6a5039fcb4d972e2ff0983fbbe4b6c8a9c77ceba15fa04322630",
            "soft_prompt": "cf5ed42f944880c4e492002a6cb32630463c112d976fdd2e1b4a06fcdc2e2c96",
            "spectral": "b7bb2b6cb5c1734a6f60523f3d74851da6c0cb2eaa8ba1856dd58682a07e31b5",
            "spectral_latent": "c806e161f57dcae0de34e4dbca48b06b29359325898b3235af1d9afc558f4e0b",
        }
        for name, old in expected.items():
            with self.subTest(adapter=name):
                entry = ADAPTER_TYPES.get(name)
                self.assertEqual(EXTRACTED_DEFINITIONS[combined(entry)], old)
                self.assertEqual(entry.source_hash, old)

    def test_changed_lora_operation_does_not_inherit_legacy_identity(self):
        entry = ADAPTER_TYPES.get("lora")
        original = Path.read_text
        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path.name == "lora_torch.py":
                old = "return self.inner(x) + delta.to(x.dtype)"
                self.assertIn(old, text)
                return text.replace(old, "return self.inner(x) + 2 * delta.to(x.dtype)")
            return text
        with patch.object(Path, "read_text", changed):
            edited = combined(entry)
        self.assertNotIn(edited, EXTRACTED_DEFINITIONS)
        self.assertNotEqual(edited, entry.source_hash)



if __name__ == "__main__":
    unittest.main()
