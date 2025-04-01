# Import existing workflows from BTV commissioninfg first:


# Now add our additional workflows
# from ExampleWorkflow.workflows.mytestwf import (
#     NanoProcessor as TestProcessor,
# )
from Hpluscharm.workflows.hplusc_gen_process import (
    NanoProcessor as gen,
)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process_test_analysis import (
    NanoProcessor as HWW2l2nu_for_sherpa,
)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process_HLT import (
    NanoProcessor as HLT,
)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process_fake import (
    NanoProcessor as fake,
)

from Hpluscharm.workflows.hplusc_HWW2l2nu_process_test import (
    NanoProcessor as HWWtest,
)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process_hwwid import (
    NanoProcessor as HWWid,
)


workflows = {}

# workflows = wf
# workflows["mytestwf"] = TestProcessor
# workflows["HWW2l2nu_new"] = HWW2l2nu_new
workflows["HWWtest"] = HWWtest
workflows["HWWsherpa"] = HWW2l2nu_for_sherpa
workflows["HWWid"] = HWWid
workflows["HLT"] = HLT
workflows["fake"] = fake
workflows["gen"] = gen

# workflows["GEN"] = gen
__all__ = ["workflows"]
