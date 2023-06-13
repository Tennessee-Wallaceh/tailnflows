import os
from pathlib import Path

def get_project_root():
    return os.environ.get(
        'TAILNFLOWS_HOME', 
        Path(__file__).parent.parent
    )