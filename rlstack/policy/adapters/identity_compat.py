"""Exact-source aliases for the behavior-preserving ADR 0028 extraction.

An alias covers one exact declaration AND its provider source digest. Editing
either stops matching and receives a new identity. This is not an exemption
from code identity and never rewrites a run/store. The old digests come from
the parent 483259c; only moving lifecycle forwarding was authorized here.
"""

EXTRACTED_DEFINITIONS: dict[str, str] = {
    # attn_bias: unchanged operation and provider bodies
    '0706bd3bb5363a6d7d58d54febffc3c9027ea3d06acdd1f26732930ac53a292c': '8787c22bbd90a20cfa66b85604c5b4056687d6e042ee2a0450cfabb93794c1cb',
    # dream_bank: unchanged operation and provider bodies
    '677a9c6a0239077795f032b8ee5f30578bb0c7aaccaa61c006af3f01cba6e239': 'b36a8e0f35b29e5a69cbff58b05ab3b79ad1392f01870ff4456d5ce273fd7a39',
    # learned_tasks: unchanged operation and provider bodies
    'f993de8556de862f08ce992bcd4790fa50fed509037a1cc7f7824adbca90e1aa': 'b55539ebc8b06eb0211b48bf5ac67c44d012e78fd93de51dfcd883f550974cde',
    # lora: unchanged operation and provider bodies
    '2c8c3bdf2111e4f14b66a65acab8f0fdbbc82b540cba89855a877121dc7b573e': 'f8696c474dd9774d92516a22a3730e9a25e9bde1b7fba9a7517f1d5ffc492c5d',
    # nsteer: unchanged operation and provider bodies
    '0ffcbe1bc8ef1abd2fbe28b467a3c3847bb258b2f14b2432d6c57e8a2927c26c': '3d18ad994ce0591fe407a1864441ecc0d0761fbadc2a6d6013eb4409afe0ae87',
    # plora: unchanged operation and provider bodies
    '87b81ea3eacb438ed55d9dce9efbd789c5f15e54a677a19800e7204eaac52557': '5637bcd111bc6a5039fcb4d972e2ff0983fbbe4b6c8a9c77ceba15fa04322630',
    # soft_prompt: unchanged operation and provider bodies
    '1d314e262d63b77f8a274fc5a95c93315c863282299661885832465111a7b69a': 'cf5ed42f944880c4e492002a6cb32630463c112d976fdd2e1b4a06fcdc2e2c96',
    # spectral: unchanged operation and provider bodies
    '1c5c929281e543eecc26bf50378d2b286d31ae413d89bb568fffbb2787037f47': 'b7bb2b6cb5c1734a6f60523f3d74851da6c0cb2eaa8ba1856dd58682a07e31b5',
    # spectral_latent: unchanged operation and provider bodies
    'fa6ea635e083e403bfda48dbc8150a9a3448bddc0548d6f1529903004b035ac1': 'c806e161f57dcae0de34e4dbca48b06b29359325898b3235af1d9afc558f4e0b',
    # spectral_tasks: unchanged operation and provider bodies
    '2e17de46f6be115671bec1aea2010f19d5607b27184d8488d381fa9f40ab94e2': 'dc2908a9b26f2187e490ecfced06e388971355a4903fd9f2186a80d26844a60a',
    # steer: unchanged operation and provider bodies
    'e7b8186e5eb2b2c913018e5d3275f1bf855e001fce7223e1a0e12ce8ccabc997': '4a065046ea875707d7f4dcec37081fcd9d6fc686cce3528ae73d80825d4d5fad',
    # value_head: unchanged operation and provider bodies
    '08408f0205c78ede6002baf8b4c29e3a689fc1c99c48077f62878875d959f626': 'e299e914626c386ac2878285093ac1a5bfa2144b154c2b51ec24fee4ec44aa7d',
    # ADR 0029: exact provider fixes only affect foreign mixed-row slots,
    # which legacy one-slot specs could not produce. Their owned-slot math,
    # initialization and checkpoint codecs remain unchanged. DreamBank is
    # deliberately excluded: its calibration algorithm also changed.
    # attn_bias
    '0e8dd8ac8ca41b7bff1ba74aa04b1131bf8bbf119f0ee5fa59ceaef1305c1607': '8787c22bbd90a20cfa66b85604c5b4056687d6e042ee2a0450cfabb93794c1cb',
    # learned_tasks
    '6a27d7435bf742d389d2dd609656a9ec1d6b81283163fe84f522debee17d1e91': 'b55539ebc8b06eb0211b48bf5ac67c44d012e78fd93de51dfcd883f550974cde',
    # lora
    '6519b318d5a6b9e5bababff64270631be81b476269ca03c6d67e934bcdf96ad2': 'f8696c474dd9774d92516a22a3730e9a25e9bde1b7fba9a7517f1d5ffc492c5d',
    # plora
    '952ab5387ec30655532a30795031a6c7332b9007147114a8ffde700f0387c3e3': '5637bcd111bc6a5039fcb4d972e2ff0983fbbe4b6c8a9c77ceba15fa04322630',
    # soft_prompt
    '9bee819955606a671d3658762bfb300a813653a9466f2f2421938ebc84a18202': 'cf5ed42f944880c4e492002a6cb32630463c112d976fdd2e1b4a06fcdc2e2c96',
    # spectral
    '7069f379209f9b53e5df11b729921e33203811527067a41a3869c24c24144d11': 'b7bb2b6cb5c1734a6f60523f3d74851da6c0cb2eaa8ba1856dd58682a07e31b5',
    # spectral_latent
    '6e22a7f24882a4b08ed09fc68e35e07dc44a2305a87b4f0e53c52bb72407ad34': 'c806e161f57dcae0de34e4dbca48b06b29359325898b3235af1d9afc558f4e0b',
    # spectral_tasks
    '370934e10814b1257e27f91ada47632be6d40598810bba2e2877db0367fba354': 'dc2908a9b26f2187e490ecfced06e388971355a4903fd9f2186a80d26844a60a',
}
