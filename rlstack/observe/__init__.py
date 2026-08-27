"""The observer: read-only derivations over stores and journals.

The region's one rule: NEVER attach, NEVER write. Everything here renders
committed bytes — host journals, run peeks (manifest / dictionary / ledger)
— into views. The CLI (python -m rlstack) is text over these; a UI is JSON
over the same *_data functions; both read the run's own dictionary.json
instead of re-deriving any declaration.

    locate.py   store_for(locator): a store locator only resolves somewhere —
                paths where mounted, s3:// anywhere (later), modal:// only
                inside a container beside the volume
    views.py    hosts/runs/gpu: the *_data functions (structured, for a UI)
                and render_* (text, for the CLI)
"""

from rlstack.observe.locate import store_for  # noqa: F401
from rlstack.observe.views import (  # noqa: F401
    gpu_data, hosts_data, render_gpu, render_hosts, render_runs, runs_data,
)
