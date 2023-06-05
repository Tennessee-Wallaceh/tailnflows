"""
Variational targets
"""

from tailnflows.targets import neals_funnel
from tailnflows.targets.posteriordb import diamonds, eight_schools

targets = {
    'neals_funnel': neals_funnel.get_target,
    'diamonds': diamonds.get_target,
    'eight_schools': eight_schools.get_target,
}