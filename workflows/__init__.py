# Import existing workflows from BTV commissioninfg first:


# Now add our additional workflows
# from ExampleWorkflow.workflows.mytestwf import (
#     NanoProcessor as TestProcessor,
# )
#from Hpluscharm.workflows.hplusc_gen_process import (
#    NanoProcessor as gen,
#)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process import (
    NanoProcessor as HWW2l2nu_new,
)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process_HLT import (
    NanoProcessor as HLT,
)
from Hpluscharm.workflows.hplusc_HWW2l2nu_process_PDF import (
    NanoProcessor as PDF,
)

from Hpluscharm.workflows.hplusc_HWW2l2nu_process_test import (
    NanoProcessor as HWWtest,
)



workflows = {}

#workflows = wf
# workflows["mytestwf"] = TestProcessor
workflows["HWW2l2nu_new"] = HWW2l2nu_new
workflows["HWWtest"] = HWWtest
workflows["HLT"] = HLT
workflows["PDF"] = PDF

#workflows["GEN"] = gen
__all__ = ["workflows"]
