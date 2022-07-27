# Import existing workflows from BTV commissioninfg first:
from BTVNanoCommissioning.workflows import workflows as wf

# Now add our additional workflows
# from ExampleWorkflow.workflows.mytestwf import (
#     NanoProcessor as TestProcessor,
# )

from Hpluscharm.workflows.hplusc_HWW2l2nu_process import (
    NanoProcessor as HWW2l2nu_new,
)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process_old import (
    NanoProcessor as HWW2l2nu,
)


workflows = {}

workflows = wf
# workflows["mytestwf"] = TestProcessor
workflows["HWW2l2nu"] = HWW2l2nu
workflows["HWW2l2nu_new"] = HWW2l2nu_new
__all__ = ["workflows"]
