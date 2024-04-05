from BTVNanoCommissioning.utils.xs_scaler import collate
from BTVNanoCommissioning.utils.plot_utils import load_coffea
import numpy as np
from BTVNanoCommissioning.helpers.xsection import xsection
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt, mplhep as hep
from BTVNanoCommissioning.utils.plot_utils import (
    load_coffea,
    load_default,
    rebin_and_xlabel,
    plotratio,
    errband_opts,
    autoranger,
)
import numpy as np

output_MC=load_coffea({'input':"MC_nsv/*.coffea",'lumi':41500},True)
output_higgs=load_coffea({'input':"higgs_nsv_v00/*.coffea",'lumi':41500},True)
output_data = load_coffea({'input':'data_nsv_v00/*.coffea'},False)
output={mc:output_MC[mc] for mc in output_MC.keys()}
for d in output_data.keys():output[d] = output_data[d]
for d in output_higgs.keys():output[d] = output_higgs[d]
mergemap_ws={
    "hc":['gchcWW2L2Nu_4f'],
    "data_obs":[
        "MuonEG_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1",
        "MuonEG_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1",
        "MuonEG_Run2017D-UL2017_MiniAODv2_NanoAODv9-v1",
        "MuonEG_Run2017E-UL2017_MiniAODv2_NanoAODv9-v1",
        "MuonEG_Run2017F-UL2017_MiniAODv2_NanoAODv9-v1"],
    "higgs":['ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8', 'HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'VBFHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8', 'GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'HWplusJ_HToWWTo2L2Nu_WTo2L_M-125_TuneCP5_13TeV-powheg-pythia8', 'GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8', 'HWminusJ_HToWW_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8','GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8', "GluGluToZH_HToZZTo4L_M125_TuneCP5_13TeV-jhugenv723-pythia8","VBF_HToZZTo4L_M125_TuneCP5_withDipoleRecoil_13TeV-powheg2-jhugenv7011-pythia8","WminusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8","WplusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8","bbH_HToZZTo4L_M125_TuneCP2_13TeV-jhugenv7011-pythia8","ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8","tqH_HToZZTo4L_M125_TuneCP5_13TeV-jhugenv7011-pythia8","ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8"],
    'ttbar':['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8'],
    'st':['ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8'],
#     'WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8'
    "zjets":['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8', 'DYJetsToTauTauToMuTauh_M-50_TuneCP5_13TeV-madgraphMLM-pythia8'],
    'vv':['WZ_TuneCP5_13TeV-pythia8', 'WW_TuneCP5_13TeV-pythia8', 'ZZ_TuneCP5_13TeV-pythia8']
}
collated = collate(output,mergemap_ws)

    
syst_list=['ttbar_weightDown','ttbar_weightUp', 'DeepJetC_LHEScaleWeight_muRDown', 'DeepJetC_PSWeightISRDown', 'PDFaS_weightUp', 'ele_IDUp', 'DeepJetC_XSec_BRUnc_WJets_cDown', 'scalevar_7ptDown', 'mu_IsoUp', 'aS_weightUp', 'ele_RecoUp', 'mu_RecoUp', 'DeepJetC_ExtrapDown', 'DeepJetC_InterpUp', 'mu_IDDown', 'scalevar_3ptUp', 'L1prefireweightDown', 'DeepJetC_PSWeightISRUp', 'DeepJetC_StatDown', 'DeepJetC_jesTotalUp', 'DeepJetC_StatUp', 'UEPS_ISRDown', 'UEPS_FSRUp', 'DeepJetC_PSWeightFSRUp', 'puweightUp', 'scalevar_3ptDown', 'mu_IDUp', 'DeepJetC_jerDown', 'PDFaS_weightDown', 'PDF_weightUp', 'puweightDown', 'DeepJetC_XSec_BRUnc_WJets_cUp', 'DeepJetC_XSec_BRUnc_DYJets_bUp', 'DeepJetC_ExtrapUp', 'DeepJetC_InterpDown', 'DeepJetC_LHEScaleWeight_muRUp', 'DeepJetC_PUWeightUp', 'DeepJetC_jerUp', 'DeepJetC_XSec_BRUnc_DYJets_cDown', 'DeepJetC_LHEScaleWeight_muFDown', 'DeepJetC_PUWeightDown', 'L1prefireweightUp', 'DeepJetC_XSec_BRUnc_DYJets_cUp', 'DeepJetC_PSWeightFSRDown', 'mu_IsoDown', 'ele_IDDown', 'DeepJetC_jesTotalDown', 'ele_RecoDown', 'mu_RecoDown', 'scalevar_7ptUp', 'UEPS_FSRDown', 'UEPS_ISRUp', 'DeepJetC_XSec_BRUnc_DYJets_bDown', 'aS_weightDown', 'DeepJetC_LHEScaleWeight_muFUp', 'PDF_weightDown', 'JESUp', 'JESDown', 'UESUp', 'UESDown', 'JERUp', 'JERDown']
# corr_sys=[]
axes={'flav':sum,'lepflav':'emu'}
var={'top_CR_lowmT':"template_mTh",'SR_HM':"template_BDT_SR_HM",'SR_LM':"template_BDT_SR_LM",'SR2_LM':"template_BDT_SR2_LM"}
corr_syst=["ttbar_weight","PDFaS_weight","aS_weight","DeepJetC_Interp","scalevar_3pt","DeepJetC_PSWeightISR","DeepJetC_jesTotal","UEPS_FSR","DeepJetC_PSWeightFSR","PDF_weight","DeepJetC_XSec_BRUnc_WJets_c","DeepJetC_XSec_BRUnc_DYJets_b","DeepJetC_Extrap","DeepJetC_LHEScaleWeight_muR","DeepJetC_PUWeight","DeepJetC_jer","L1prefireweight","DeepJetC_XSec_BRUnc_DYJets_c","scalevar_7pt","UEPS_ISR","DeepJetC_LHEScaleWeight_muF"]

import mplhep as hep
import uproot 

for r in var.keys():
    root_out=uproot.recreate(f"../card_maker/shape/{r}_{var[r]}.root")
    for s in collated.keys():
        axes['syst']='nominal'
        axes['region']=r
        print(s,r)
        root_out[s]=collated[s][var[r]][axes]
#         hep.histplot(collated[s][var[r]][axes],label='nominal')
        if s!="data_obs":
            for sys in syst_list:
                axes['syst']=sys
                prefix = "_13TeV"
                if sys.replace("Up","").replace("Down","")  in corr_syst:prefix = "_13TeV"
                else:prefix="_13TeV_2017"
                if 'ttbar_' in sys:
                    if s=='ttbar':root_out[f'{s}_CMS_{prefix}{sys}']=collated[s][var[r]][axes]
                    else :
                        axes['syst']='nominal'
                        root_out[f'{s}_CMS_{prefix}{sys}']=collated[s][var[r]][axes]

            