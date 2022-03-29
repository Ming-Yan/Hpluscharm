import parsl
from parsl.app.app import python_app, bash_app
from parsl.data_provider.files import File
import os

from parsl.providers import LocalProvider, CondorProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname, address_by_query

env_extra = [
            f'export PYTHONPATH=$PYTHONPATH:{os.getcwd()}',
        ]

condor_extra = [
            f'source {os.environ["HOME"]}/.bashrc',
            'source activate coffea'
            ]
htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label='coffea_parsl_condor',
                        address=address_by_query(),
                        max_workers=1,
                        provider=CondorProvider(
                            nodes_per_block=1,
                            init_blocks=1,
                            max_blocks=1,
                            worker_init="\n".join(env_extra + condor_extra),
                            walltime="00:10:00",
                        ),
                    )
                ]
            )
parsl.load(htex_config)

@bash_app
def sort(unsorted, outputs=[]):
    """Call sort executable on the input file"""
    return "sleep 30; sort -g {} > {}".format(unsorted.filepath, outputs[0].filepath)

s = sort(File(os.path.abspath("unsorted.txt")), 
         outputs=[File(os.path.abspath("sorted_c.txt"))])

output_file = s.outputs[0].result()

print("Contents of the unsorted.txt file:")
with open('input/unsorted.txt', 'r') as f:
    print(f.read().replace("\n",","))
    
print("\nContents of the sorted output file:")
with open(output_file, 'r') as f:
    print(f.read().replace("\n",","))

parsl.clear()
