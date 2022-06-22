# Hpluscharm
Higgs plus charm analysis framework coffea based


## Setup
! This enviroment should under `bash` shell! 
### Coffea installation with Miniconda
# only first time 
git clone git@github.com:Ming-Yan/Hpluscharm.git
For installing Miniconda, see also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Local-or-remote
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run and follow instructions on screen
bash Miniconda3-latest-Linux-x86_64.sh
```
NOTE: always make sure that conda, python, and pip point to local Miniconda installation (`which conda` etc.).

You can either use the default environment`base` or create a new one:
```
# create new environment named coffea with env.yml
conda coffea create -f env.yml
# activate environment coffea
conda activate coffea
```

Setup things with x509 proxy
```bash
# There are some dependencies which are currently not available as packages, so we'll have to copy them from `/cvmfs/grid.cern.ch/etc/grid-security/`
   
    mkdir ~/.grid-security
    scp -r lxplus:/cvmfs/grid.cern.ch/etc/grid-security/* ~/.grid-security
   
    
#All that's left is to point `voms-proxy-init` to the right vomses directory
   
    voms-proxy-init --voms cms --vomses ~/.grid-security/vomses
```


### Other installation options for coffea
See https://coffeateam.github.io/coffea/installation.html
### Running jupyter remotely
See also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Remote-jupyter
1. On your local machine, edit `.ssh/config`:
```
Host lxplus*
  HostName lxplus7.cern.ch
  User <your-user-name>
  ForwardX11 yes
  ForwardAgent yes
  ForwardX11Trusted yes
Host *_f
  LocalForward localhost:8800 localhost:8800
  ExitOnForwardFailure yes
```
2. Connect to remote with `ssh lxplus_f`
3. Start a jupyter notebook:
```
jupyter notebook --ip=127.0.0.1 --port 8800 --no-browser
```




#  activate enviroment once you have coffea framework 
1. activate coffea
`conda activate coffea`
2. setup the proxy
` voms-proxy-init --voms cms --vomses ~/.grid-security/vomses`
3. Run the framework



## Structure

Each workflow can be a separate "processor" file, creating the mapping from NanoAOD to
the histograms we need. Workflow processors can be passed to the `runner.py` script 
along with the fileset these should run over. Multiple executors can be chosen 
(for now iterative - one by one, uproot/futures - multiprocessing and dask-slurm). 

For now only the selections and histogram producer are available.

To test a small set of files to see whether the workflows run smoothly, run:
```
python runner.py  --json metadata/higgs.json --limit 1 --wf HWW2l2nu ( --export_array)
```
- Samples are in `metadata`
- Corrections in `utils/correction.py` 

### Scale out @lxplus (not recommended)

```
python runner.py  --json metadata/higgs.json --limit 1 --wf HWW2l2nu -j ${njobs} --executor dask/lxplus
``` 
### Scale out @DESY/lxcluster Aachen

```
python runner.py  --json metadata/higgs.json --limit 1 --wf HWW2l2nu  --year ${year}  -j ${cores} -s ${scaleout jobs} --executor parsl/condor --memory ${memory_per_jobs} --export_array
```
### Scale out @HPC Aachen

```
python runner.py  --json metadata/higgs.json --limit 1 --wf HWW2l2nu -j ${njobs} --executor parsl/slurm
```

## Make the json files

Use the `fetch.py` in `filefetcher`, the `$input_DAS_list` is the info extract from DAS, and output json files in `metadata/`

```
python fetch.py --input ${input_DAS_list} --output ${output_json_name} --site ${site}
```



## Create compiled corretions file(`pkl.gz`)

Use the `utils/compile_jec.py`, editted the path in the `dict` of `jet_factory`

```
python -m utils.compile_jec data/jec_compiled.pkl.gz
```
