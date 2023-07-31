import os
from pathlib import Path
import sys

def get_project_root():
    return os.environ.get(
        'TAILNFLOWS_HOME', 
        Path(__file__).parent.parent
    )


class MarginalTailAdaptiveFlowImport():
    """
    Context manager for importing marginal tail adaptive code
    https://github.com/MikeLasz/marginalTailAdaptiveFlow/
    """
    def __init__(self):
        self.path = f'{get_project_root()}/../marginalTailAdaptiveFlow' # must reference local repo

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass