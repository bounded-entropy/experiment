"""tasks/ — turning a dataset into a task set.

`base.py` holds the dataset-blind verbs (write / load / split); one file per
dataset beside it holds the ONE function that reads that dataset's rows. A
dataset module imports its heavy dependencies inside its function (rule 7), so
this package root stays free to import.
"""

from rlstack.data.tasks.base import (
    draw_for, load_tasks, split_tasks, write_tasks,
)
from rlstack.data.tasks.dapo_math import dapo_math_tasks
