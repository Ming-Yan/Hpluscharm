HLTmenu = {
    "mu1hlt": {
        "UL16_preVFP": ["IsoTkMu24"],
        "UL16_postVFP": ["IsoTkMu24"],
        "UL17": ["IsoMu27"],
        "UL18": ["IsoMu24"],
    },
    "mu2hlt": {
        "UL16_preVFP": [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",  # runH
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",  # runH
        ],
        "UL16_postVFP": [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",  # runH
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",  # runH
        ],
        "UL17": [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
        ],
        "UL18": [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        ],
    },
    "e1hlt": {
        "UL16_preVFP": ["Ele27_WPTight_Gsf", "Ele25_eta2p1_WPTight_Gsf"],
        "UL16_postVFP": ["Ele27_WPTight_Gsf", "Ele25_eta2p1_WPTight_Gsf"],
        "UL17": [
            "Ele35_WPTight_Gsf",
        ],
        "UL18": [
            "Ele32_WPTight_Gsf",
        ],
    },
    "e2hlt": {
        "UL16_preVFP": [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "UL16_postVFP": [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "UL17": [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
        ],
        "UL18": [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
        ],
    },
    "emuhlt": {
        "UL16_preVFP": [
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "UL16_postVFP": [
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "UL17": [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "UL18": [
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
    },
}
correction_config = {
    "UL16_preVFP": {
        "PU": "puwei_UL16.histo.root",
        "JME": "compile_jec.pkl.gz",
        "BTV": {
            "DeepCSVB": "DeepCSV_94XSF_V5_B_F.csv",
            "DeepCSVC": "DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root",
            "DeepJetB": "DeepFlavour_94XSF_V4_B_F.csv",
            "DeepJetC": "DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root",
        },
        "LSF": {
            # "ele_Trig TrigSF": "Ele32_L1DoubleEG_TrigSF_vhcc.histo.root",
            "ele_ID_pre EGamma_SF2D": "egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D.histo.root",
            "ele_Rereco_above20_pre EGamma_SF2D": "egammaEffi_ptAbove20.txt_EGM2D_UL2016preVFP.histo.root",
            "ele_Rereco_below20_pre EGamma_SF2D": "egammaEffi_ptBelow20.txt_EGM2D_UL2016preVFP.histo.root",
            "mu_ID_pre NUM_TightID_DEN_TrackerMuons_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.histo.root",
            "mu_Iso_pre NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.histo.root",
        },
    },
    "UL16_postVFP": {
        "PU": "puwei_UL16.histo.root",
        "JME": "compile_jec.pkl.gz",
        "BTV": {
            "DeepCSVB": "DeepCSV_94XSF_V5_B_F.csv",
            "DeepCSVC": "DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root",
            "DeepJetB": "DeepFlavour_94XSF_V4_B_F.csv",
            "DeepJetC": "DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root",
        },
        "LSF": {
            # "ele_Trig TrigSF": "Ele32_L1DoubleEG_TrigSF_vhcc.histo.root",
            "ele_ID_post EGamma_SF2D": "egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D.histo.root",
            "ele_Rereco_above20_post EGamma_SF2D": "egammaEffi_ptAbove20.txt_EGM2D_UL2016postVFP.histo.root",
            "ele_Rereco_below20_post EGamma_SF2D": "egammaEffi_ptBelow20.txt_EGM2D_UL2016postVFP.histo.root",
            "mu_ID_post NUM_TightID_DEN_TrackerMuons_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.histo.root",
            "mu_Iso_post NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.histo.root",
        },
    },
    "UL17": {
        "PU": "puweight_UL17.histo.root",
        "JME": "mc_compile_jec.pkl.gz",
        "BTV": {
            "DeepJetC": "DeepJet_ctagSF_Summer20UL17_interp.root",
        },
        "LSF": {
            # "ele_Trig TrigSF": "Ele32_L1DoubleEG_TrigSF_vhcc.histo.root",
            "ele_Rereco_above20 EGamma_SF2D": "egammaEffi_ptAbove20.txt_EGM2D_UL2017.histo.root",
            "ele_Rereco_below20 EGamma_SF2D": "egammaEffi_ptBelow20.txt_EGM2D_UL2017.histo.root",
            "ele_ID EGamma_SF2D": "egammaEffi.txt_EGM2D_MVA90iso_UL17.histo.root",
            "mu_ID NUM_TightID_DEN_TrackerMuons_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.histo.root",
            "mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.histo.root",
            "ele_Rereco_above20_error EGamma_SF2D_error": "egammaEffi_ptAbove20.txt_EGM2D_UL2017.histo.root",
            "ele_Rereco_below20_error EGamma_SF2D_error": "egammaEffi_ptBelow20.txt_EGM2D_UL2017.histo.root",
            "ele_ID_error EGamma_SF2D_error": "egammaEffi.txt_EGM2D_MVA90iso_UL17.histo.root",
            "mu_ID_error NUM_TightID_DEN_TrackerMuons_abseta_pt_error": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.histo.root",
            "mu_Iso_error NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_error": "Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.histo.root",
        },
    },
    "UL18": {
        "PU": "puwei_UL18.histo.root",
        "JME": "UL18_compile_jec_unc_AK4chs.pkl.gz",
        "BTV": {
            "DeepCSVB": "DeepCSV_94XSF_V5_B_F.csv",
            "DeepCSVC": "DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root",
            "DeepJetB": "DeepFlavour_94XSF_V4_B_F.csv",
            "DeepJetC": "DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root",
        },
        "LSF": {
            # "ele_Trig TrigSF": "Ele32_L1DoubleEG_TrigSF_vhcc.histo.root",
            "ele_ID EGamma_SF2D": "egammaEffi.txt_EGM2D_MVA90iso_UL18.histo.root",
            "ele_Rereco_above20 EGamma_SF2D": "egammaEffi_ptAbove20.txt_EGM2D_UL2018.histo.root",
            "ele_Rereco_below20 EGamma_SF2D": "egammaEffi_ptBelow20.txt_EGM2D_UL2018.histo.root",
            "mu_ID NUM_TightID_DEN_TrackerMuons_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.histo.root",
            "mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt": "Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.histo.root",
        },
    },
}
