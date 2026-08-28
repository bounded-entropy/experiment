"""The observer: read-only derivations over stores and journals.

The region's one rule: NEVER attach, NEVER write. Everything here renders
committed bytes — host journals, run peeks, and (since #57) SEALED wave
artifacts — into views, and reads each run's own dictionary.json rather than
re-deriving any declaration.

    locate.py       store_for(locator): a store is NAMED by a locator and the
                    reader must run somewhere it resolves — plus the ROOTS
                    under a top directory (#58: a folder IS a store root,
                    chosen at birth and never moved)
    views.py        hosts / runs / gpu, each as *_data (structured) and
                    render_* (terminal text)
    series.py       run_series: one run's dictionary joined with its committed
                    history — the graphs' data
    host_series.py  the per-HOST reading off the journals — birth facts,
                    residencies, gpu and traffic channels, run_timing —
                    plus fleet_data, the global join
    aggregate.py    the journals' numeric plane: a serving host's rails, one
                    run's clock, the fleet's two sums, the timeline's moments
    waves.py        the wave browser's reading: one sealed wave as
                    distributions and as chat
    panels.py       user-declared derived graphs, as expressions-as-data
    page.py         the reader of web/: THE document and its assets
    web/            the UI's static assets — index.html, style.css and native
                    ES modules; no build step, no CDN, no framework
    ui.py           the routes: a dependency-free WSGI app
"""

from rlstack.observe.aggregate import fleet_throughput  # noqa: F401
from rlstack.observe.host_series import (  # noqa: F401
    fleet_data, host_series, run_timing,
)
from rlstack.observe.locate import (  # noqa: F401
    Root, roots_for, roots_under, rooted, store_for,
)
from rlstack.observe.series import run_series  # noqa: F401
from rlstack.observe.ui import serve, ui_app  # noqa: F401
from rlstack.observe.waves import wave_detail, wave_list  # noqa: F401
from rlstack.observe.views import (  # noqa: F401
    gpu_data, hosts_data, matches, render_gpu, render_hosts, render_runs,
    runs_data,
)
