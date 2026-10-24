"""Behavior controls retain old spec bytes and survive remote serialization."""
import json
import unittest
from rlstack.spec.specs import SamplingSpec
from rlstack.spec.canonical import canonical_json, content_hash, wire_json
from rlstack.runner.remote import encode_sampling, decode_sampling, spec_from_json

class SamplingControlsTest(unittest.TestCase):
    def test_legacy_defaults_are_byte_identical(self):
        self.assertEqual(canonical_json(SamplingSpec()),
            '{"__type__":"SamplingSpec","max_tokens":1024,"temperature":1.0,"top_p":1.0}')
        self.assertEqual(encode_sampling(SamplingSpec()),
                         dict(temperature=1.,top_p=1.,max_tokens=1024))
        self.assertEqual(decode_sampling(dict(temperature=.3,top_p=.8,max_tokens=12)),
                         SamplingSpec(.3,.8,12))

    def test_all_explicit_controls_change_identity_and_cross_both_wires(self):
        controls=dict(top_k=20,min_p=0.,presence_penalty=1.5,repetition_penalty=1.)
        value=SamplingSpec(1.,.95,32768,**controls)
        self.assertEqual(decode_sampling(json.loads(json.dumps(encode_sampling(value)))),value)
        self.assertEqual(spec_from_json(json.loads(wire_json(value))),value)
        for key,arg in controls.items():
            self.assertNotEqual(content_hash(SamplingSpec()),
                                content_hash(SamplingSpec(**{key:arg})))
