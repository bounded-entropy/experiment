"""Transports: real wire substrates behind the Transport protocol, one per file.

modal_cls imports the Modal SDK at module scope, so import it lazily — only
where a venue's door is being wired, never from the rlstack package root
(STYLE rule 7, pinned by tests/test_architecture.py). `transport_for`
(runner/remote.py) is the one caller: it reads an address's scheme and imports
the branch that scheme names.

LocalTransport is NOT here. It is the protocol's own enforcement — the json
round trip both ways — rather than a substrate, so it lives beside the
protocol in runner/remote.py, the same rule that keeps FakeEngine in
runner/fakes.py instead of in runner/engines/.
"""
