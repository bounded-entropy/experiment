"""The observer: read-only derivations over stores and journals.

The region's one rule: NEVER attach, NEVER write. Everything here renders
committed bytes — host journals and run peeks — into views, and reads each run's
own dictionary.json rather than re-deriving any declaration.

    locate.py       store_for(locator): a store is NAMED by a locator and the
                    reader must run somewhere it resolves
    views.py        hosts / runs / gpu, each as *_data (structured) and
                    render_* (terminal text)
    series.py       run_series: one run's dictionary joined with its committed
                    history — the graphs' data
    host_series.py  the per-HOST reading off the journals — birth facts,
                    residencies, gpu and traffic channels, run_timing —
                    plus fleet_data, the global join
    panels.py       user-declared derived graphs, as expressions-as-data
    page.py         THE document: one self-contained page, every route's
                    response
    ui.py           the routes: a dependency-free WSGI app
"""

from rlstack.observe.host_series import (  # noqa: F401
    fleet_data, host_series, run_timing,
)
from rlstack.observe.locate import store_for  # noqa: F401
from rlstack.observe.series import run_series  # noqa: F401
from rlstack.observe.ui import serve, ui_app  # noqa: F401
from rlstack.observe.views import (  # noqa: F401
    gpu_data, hosts_data, render_gpu, render_hosts, render_runs, runs_data,
)
