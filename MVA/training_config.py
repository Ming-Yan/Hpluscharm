import numpy as np
train_collect={
    'nj':["# Selected Jet",5,[1,6]],
    'njout':["# Selected Jet $|\\eta|>$2.4",6,[1,7]],
    'jetflav_btagDeepFlavCvL':["Jet1 CvL",45,[0.1,1]],
    'jetflav_btagDeepFlavCvB':["Jet1 CvB",26,[0.48,1]], 
    'jetflav2_btagDeepFlavCvL':["Jet2 CvL",45,[0.1,1]], 
    'jetflav2_btagDeepFlavCvB':["Jet2 CvB",26,[0.48,1]], 
    'h_pt':["Higgs $p_T$",50,[0,300]],
    'h_eta':["Higgs $\\eta$",50,[-2.5,2.5]], 
    'h_phi':["Higgs $\\phi$",30,[-np.pi,np.pi]], 
    'll_pt':["$\\ell\\ell p_T$",40,[20,300]], 
    'll_eta':["$\\ell\\ell \\eta$",50,[-2.5,2.5]],
    'll_phi':["$\\ell\ell \\phi$",30,[-np.pi,np.pi]], 
    'template_ll_mass':["$\\ell\\ell$ mass",34,[10,78]], 
     'll_mass':["$\\ell\\ell$ mass",34,[10,78]], 
    'lep1_pt':["$\\ell_1 p_T$",47,[18,300]],
    'lep1_eta':["$\\ell_1\\eta$",25,[-2.5,2.5]], 
    'lep1_phi':["$\\ell_1 \\phi$",30,[-np.pi,np.pi]], 
    'lep2_pt':["$\\ell_2 p_T$",45,[10,100]],
    'lep2_eta':["$\\ell_2\\eta$",25,[-2.5,2.5]], 
    'lep2_phi':["$\\ell_2 \\phi$",30,[-np.pi,np.pi]], 
    'jetflav_pt':["Jet 1 $p_T$",30,[0,300]],
    'jetflav_eta':["Jet 1 $\\eta$",25,[-2.5,2.5]], 
    'jetflav_phi':["Jet 1 $\\phi$",30,[-np.pi,np.pi]], 
    'jetflav2_pt':["Jet 2 $p_T$",48,[12,300]],
    'jetflav2_eta':["Jet 2 $\\eta$",50,[-2.5,2.5]], 
    'jetflav2_phi':["Jet 2 $\\phi$",30,[-np.pi,np.pi]], 
    'mT1':["$m_{T}^{\\ell 1}$",25,[0,300]], 
    'mT2':["$m_{T}^{\\ell 2}$",20,[30,150]], 
    'mTh':["$m_{T}^{\\ell\\ell}$",40,[60,300]], 
    'nselsv':["# Selected SV",7,[0,7]], 
    'nsv':["# SV",15,[0,15]], 
#     'nj':['# Jets',]
    'npvs':["# Selected SV",42,[0,84]], 
    'MET_pt':["MET $p_T$",47,[18,300]],
    'MET_significance':["MET significance",50,[0,100]], 
    'MET_sumEt':["$\\Sigma$ MET $E_T$",40,[200,2000]], 
    'MET_phi':["MET $\\phi$",30,[-np.pi,np.pi]], 
    'MET_covYY':["$Cov_{YY}$",50,[0,2000]], 
    'MET_covXY':["$Cov_{XY}$",50,[-100,100]], 
    'MET_covXX':["$Cov_{XX}$",50,[0,2000]], 
    'llc_dr':["$\\Delta R(\\ell\\ell,c)$",25,[0,5]], 
    'lll1_dr':["$\\Delta R(\\ell\\ell,\\ell_1)$",20,[0,2]], 
    'lll2_dr':["$\\Delta R(\\ell\\ell,\\ell_2)$",20,[0,5]], 
    'l1met_dphi':["$\\Delta\\phi(\\ell_1,MET)$",30,[-np.pi,np.pi]], 
    'l2met_dphi':["$\\Delta\\phi(\\ell_2,MET)$",30,[-np.pi,np.pi]], 
    'cW1_dphi':["$\\Delta\\phi(\\ell_1+MET,c)$",30,[-np.pi,np.pi]], 
    'METTkMETdphi':["$\\Delta\\phi(MET,TkMET)$",30,[-0.5,0.5]]
}
config2017 = {
    "binary_LM":
    {
        "input":
        {
            "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/bkg_*_array_top_v00/*.coffea"},
            "sig":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/signal_array_top_v00/output.coffea"}
        }
        ,
        "mergemap":{
            "HWW":['ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8', 'HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'VBFHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8', 'GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'HWplusJ_HToWWTo2L2Nu_WTo2L_M-125_TuneCP5_13TeV-powheg-pythia8', 'GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8', 'HWminusJ_HToWW_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8'],
            "vjets":['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8', 'WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToTauTauToMuTauh_M-50_TuneCP5_13TeV-madgraphMLM-pythia8'],
            'st':['ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8'],
            'tt':['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8'],
            'vv':['WZ_TuneCP5_13TeV-pythia8', 'WW_TuneCP5_13TeV-pythia8', 'ZZ_TuneCP5_13TeV-pythia8']
    },
        "varlist":
        [
            "ll_pt",
            "lll1_dr",
            "lll2_dr",
            "llc_dr",
            "lep1_pt",
            "lep2_pt",
            "ll_mass",
            "MET_pt",
            "jetflav_pt",
            "l1met_dphi",
            "l2met_dphi",
            "cW1_dphi",
            "mT1",
            "mT2",
            "jetflav_btagDeepFlavCvL",
            "jetflav_btagDeepFlavCvB"
        ]
    },
    "old":
    {
        "input":
        {
            "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/bkg_*_array_top_v00/*.coffea"},
            "sig":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/signal_array_top_v00/output.coffea"}
        }
        ,
        "mergemap":{
            "HWW":['ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8', 'HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'VBFHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8', 'GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'HWplusJ_HToWWTo2L2Nu_WTo2L_M-125_TuneCP5_13TeV-powheg-pythia8', 'GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8', 'HWminusJ_HToWW_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8'],
            "vjets":['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8', 'WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToTauTauToMuTauh_M-50_TuneCP5_13TeV-madgraphMLM-pythia8'],
            'st':['ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8'],
            'tt':['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8'],
            'vv':['WZ_TuneCP5_13TeV-pythia8', 'WW_TuneCP5_13TeV-pythia8', 'ZZ_TuneCP5_13TeV-pythia8']
    },
        "varlist":
         [
                "ll_mass",
                "MET_pt",
                "jetflav_pt",
                "lep1_pt",
                "lep2_pt",
                "ll_pt",
                "mT1",
                "mT2",
                "jetflav_btagDeepFlavCvL",
                "jetflav_btagDeepFlavCvB",
                "lll1_dr",
                "lll2_dr",
                "llc_dr",
                "l1met_dphi",
                "l2met_dphi",
                "cW1_dphi",
            ]
    },
    
    "binary":
    {
        "input":
        {   "UL16_preAPV":
            {   
               "data":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/data_nsv_preVFP_UL16_array_v00/*.coffea"},
                "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/MC_nsv_preVFP_UL16_array_v01/*.coffea"},
                "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_preVFP_UL16_array_v00/output_Hc.coffea"]}},
            "UL16_postAPV":
            {"data":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/data_nsv_postVFP_UL16_array_v00/*.coffea"},
                "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/MC_nsv_postVFP_UL16_array_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_postVFP_UL16_array_v00/output_Hc.coffea"]}},
            "UL17":
            {   "data":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/data_array_top_v00/*.coffea"},
                "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/UL17_bkg/MC_*_array_top_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/signal_fixHLT_UL17_array_v00/output.coffea"]}},
            "UL18":
            {
                "data":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/data_nsv_UL18_arr_only_v00/output_*.coffea"},
                "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/MC_nsv_UL18_array_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_UL18_array_v00/output_Hc.coffea"]}
            }
        }
        ,
        "mergemap":{
            "Higgs (WW+ZZ)":['ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8', 'HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'VBFHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8', 'GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'HWplusJ_HToWWTo2L2Nu_WTo2L_M-125_TuneCP5_13TeV-powheg-pythia8', 'GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8', 'HWminusJ_HToWW_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8'],
            "Z+jets":['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8'],
            'ST':['ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8'],
            'TT':['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8'],
            'VV':['WZ_TuneCP5_13TeV-pythia8', 'WW_TuneCP5_13TeV-pythia8', 'ZZ_TuneCP5_13TeV-pythia8']
    },
        "varlist":
        [
            "ll_pt",
            "lll1_dr",
            "lll2_dr",
            "llc_dr",
            "lep1_pt",
            "lep2_pt",
            "ll_mass",
            "MET_pt",
            "jetflav_pt",
            "l1met_dphi",
            "l2met_dphi",
            "cW1_dphi",
            "mT1",
            "mT2",
            "jetflav_btagDeepFlavCvL",
            "jetflav_btagDeepFlavCvB",
            "nsv",
        ]
    },
    "binary_bkg":
    {
        "input":
        {   "UL16_preAPV":
            {
                "data":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/data_nsv_preVFP_UL16_array_v00/*.coffea"},
                "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/MC_nsv_preVFP_UL16_array_v01/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_preVFP_UL16_array_v00/output_Hc.coffea"]}},
            "UL16_postAPV":
            {"bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/MC_nsv_postVFP_UL16_array_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_postVFP_UL16_array_v00/output_Hc.coffea"]}},
            "UL17":
            {"bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/UL17_bkg/MC_*_array_top_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/signal_fixHLT_UL17_array_v00/output.coffea"]}},
            "UL18":
            {"bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/MC_nsv_UL18_array_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_UL18_array_v00/output_Hc.coffea"]}
            }
        }
        ,
         "mergemap":{
            "Higgs (WW+ZZ)":['ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8', 'HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'VBFHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8', 'GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'HWplusJ_HToWWTo2L2Nu_WTo2L_M-125_TuneCP5_13TeV-powheg-pythia8', 'GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8', 'HWminusJ_HToWW_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8'],
            "Z+jets":['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8'],
            'ST':['ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8'],
            'TT':['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8'],
            'VV':['WZ_TuneCP5_13TeV-pythia8', 'WW_TuneCP5_13TeV-pythia8', 'ZZ_TuneCP5_13TeV-pythia8']
    },
        "varlist":
        [
            "ll_pt",
            "lll1_dr",
            "lll2_dr",
            "llc_dr",
            "lep1_pt",
            "lep2_pt",
            "ll_mass",
            "MET_pt",
            "jetflav_pt",
            "l1met_dphi",
            "l2met_dphi",
            "cW1_dphi",
            "mT1",
            "mT2",
            "jetflav_btagDeepFlavCvL",
            "jetflav_btagDeepFlavCvB",
            "nsv",      
        ]
    },
     "binary_bkg_skim":
    {
        "input":
        {   "UL16_preAPV":
            {
                "data":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/data_nsv_preVFP_UL16_array_v00/*.coffea"},
                "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/MC_nsv_preVFP_UL16_array_v01/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_preVFP_UL16_array_v00/output_Hc.coffea"]}}
        }
        ,
        "mergemap":{
            "Higgs (WW+ZZ)":['ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8', 'HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'VBFHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8', 'GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'HWplusJ_HToWWTo2L2Nu_WTo2L_M-125_TuneCP5_13TeV-powheg-pythia8', 'GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8', 'HWminusJ_HToWW_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8'],
            "Z+jets":['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8', 'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8'],
            'ST':['ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8', 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8', 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8'],
            'TT':['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8', 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8'],
            'VV':['WZ_TuneCP5_13TeV-pythia8', 'WW_TuneCP5_13TeV-pythia8', 'ZZ_TuneCP5_13TeV-pythia8']
    },
        "varlist":
        [
            "ll_pt",
            "lll1_dr",
            "lll2_dr",
            "llc_dr",
            "lep1_pt",
            "lep2_pt",
            "ll_mass",
            "MET_pt",
            "jetflav_pt",
            "l1met_edphi",
            "cW1_dphi",
            "mT1",
            "mT2",
            "jetflav_btagDeepFlavCvL",
            "jetflav_btagDeepFlavCvB",
            "nsv",      
        ]
    },
    "binary_higgs":
    {
        "input":
        {   
            "UL16_preAPV":
            {
                
                "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/MC_nsv_preVFP_UL16_array_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_preVFP_UL16_array_v00/output_Hc.coffea"]}},
            "UL16_postAPV":
            {"bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/MC_nsv_postVFP_UL16_array_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_postVFP_UL16_array_v00/output_Hc.coffea"]}},
            "UL17":
            {"bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/MC_higgs_array_top_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/signal_fixHLT_UL17_array_v00/output.coffea"]}},
            "UL18":
            {"bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/MC_nsv_UL18_array_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/signal_nsv_UL18_array_v00/output_Hc.coffea"]}
            }
        },
        "mergemap":{
             "HWW":[
                 #'ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8', 
                 'HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'VBFHToWWTo2L2Nu_M125_TuneCP5_13TeV_powheg2_JHUGenV714_pythia8', 'GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8', 'HWplusJ_HToWWTo2L2Nu_WTo2L_M-125_TuneCP5_13TeV-powheg-pythia8', 'GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8', 'HWminusJ_HToWW_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8'
             ]
    },
        "varlist":
        [
            "ll_pt",
            "lll1_dr",
            "lll2_dr",
            "llc_dr",
            "lep1_pt",
            "lep2_pt",
            "ll_mass",
            "MET_pt",
            "jetflav_pt",
            "l1met_dphi",
            "l2met_dphi",
            "cW1_dphi",
            "mT1",
            "mT2",
            "jetflav_btagDeepFlavCvL",
            "jetflav_btagDeepFlavCvB",
            "nsv",      
        ]
    },
    "binary_ggH":
    {
        "input":
        {
            "bkg":{"input":"/nfs/dust/cms/user/milee/CoffeaRunner/coffea_output/coffea_nsv_0503/MC_higgs_array_top_v00/*.coffea"},
            "sig":{"input":["/nfs/dust/cms/user/milee/CoffeaRunner/signal_fixHLT_UL17_array_v00/output.coffea"]}
        }
        ,
        "mergemap":{
             "HWW":['GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8']
    },
        "varlist":
        [
            "ll_pt",
            "lll1_dr",
            "lll2_dr",
            "llc_dr",
            "lep1_pt",
            "lep2_pt",
            "ll_mass",
            "MET_pt",
            "jetflav_pt",
            "l1met_dphi",
            "l2met_dphi",
            "cW1_dphi",
            "mT1",
            "mT2",
            "jetflav_btagDeepFlavCvL",
            "jetflav_btagDeepFlavCvB",
            "nsv",
        ]
    }
}
