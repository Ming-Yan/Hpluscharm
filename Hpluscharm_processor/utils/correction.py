from operator import truediv
import numpy as np
import awkward as ak
import gzip
import pickle

#from coffea.lookup_tools.lookup_base import lookup_base
from coffea.lookup_tools import extractor#,txt_converters,rochester_lookup


from coffea.btag_tools import BTagScaleFactor
from helpers.cTagSFReader import *
import correctionlib



def init_corr(year):
    ext = extractor()
    if year == "2016":
        ##Electron
        ext.add_weight_sets(["ele_Trig TrigSF data/Rereco17_94X/Lepton/Ele32_L1DoubleEG_TrigSF_vhcc.histo.histo.root"])
        ext.add_weight_sets(["ele_ID_post EGamma_SF2D data/Electron/UL16/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D.histo.root"])
        ext.add_weight_sets(["ele_Rereco_above20_post EGamma_SF2D data/Electron/UL16/egammaEffi_ptAbove20.txt_EGM2D_UL2016postVFP.histo.root"])
        ext.add_weight_sets(["ele_Rereco_below20_post EGamma_SF2D data/Electron/UL16/egammaEffi_ptBelow20.txt_EGM2D_UL2016postVFP.histo.root"])
        ext.add_weight_sets(["ele_ID_pre EGamma_SF2D data/Electron/UL16/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D.histo.root"])
        ext.add_weight_sets(["ele_Rereco_above20_pre EGamma_SF2D data/Electron/UL16/egammaEffi_ptAbove20.txt_EGM2D_UL2016preVFP.histo.root"])
        ext.add_weight_sets(["ele_Rereco_below20_pre EGamma_SF2D data/Electron/UL16/egammaEffi_ptBelow20.txt_EGM2D_UL2016preVFP.histo.root"]) 
        ####Muon 
        ext.add_weight_sets(["mu_ID_pre NUM_TightID_DEN_TrackerMuons_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.histo.root"])
        # ext.add_weight_sets(["mu_Trig_pre NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_SingleMuonTriggers.histo.root"])
        ext.add_weight_sets(["mu_Iso_pre NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.histo.root"])
        ext.add_weight_sets(["mu_ID_post NUM_TightID_DEN_TrackerMuons_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.histo.root"])
        # ext.add_weight_sets(["mu_Trig_post NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_SingleMuonTriggers.histo.root"])
        ext.add_weight_sets(["mu_Iso_post NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.histo.root"])        
        ## PU weight 
        ext.add_weight_sets(["* * data/PU/UL16/puwei_UL16.histo.root"])
        ext.add_weight_sets(["* * data/PUJetId/UL16/UL16_jmar.corr.json"])
        with gzip.open("data/JEC_JERSF/UL16/compile_jec.pkl.gz") as fin: jmestuff = pickle.load(fin)
        
    elif year == "2017":
        ## Electron 
        ext.add_weight_sets(["ele_Rereco_above20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptAbove20.txt_EGM2D_UL2017.histo.root"])
        ext.add_weight_sets(["ele_ID EGamma_SF2D data/Electron/UL17/egammaEffi.txt_EGM2D_MVA90iso_UL17.histo.root"])
        ext.add_weight_sets(["ele_Rereco_below20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptBelow20.txt_EGM2D_UL2017.histo.root"])
        ### Muon
        # ext.add_weight_sets(["mu_Trig NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL17/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers.histo.root"])
        ext.add_weight_sets(["mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL17/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.histo.root"])
        ext.add_weight_sets(["mu_ID NUM_TightID_DEN_TrackerMuons_abseta_pt data/Muon/UL17/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.histo.root"])
        ext.add_weight_sets(["* * data/PU/UL17/puwei_UL17.histo.root"])
        #correctionlib
        ext.add_weight_sets(["* * data/PUJetId/UL17/UL17_jmar.corr.json"])
        with gzip.open("data/JEC_JERSF/UL17/compile_jec.pkl.gz") as fin: jmestuff = pickle.load(fin)

    elif year == "2018":
        ### Electron 
        ext.add_weight_sets(["ele_ID EGamma_SF2D data/Electron/UL17/egammaEffi.txt_EGM2D_MVA90iso_UL18.histo.root"])
        ext.add_weight_sets(["ele_Rereco_above20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptAbove20.txt_EGM2D_UL2018.histo.root"])
        ext.add_weight_sets(["ele_Rereco_below20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptBelow20.txt_EGM2D_UL2018.histo.root"]) 
        ### Muon
        ext.add_weight_sets(["mu_ID NUM_TightID_DEN_TrackerMuons_abseta_pt  data/Muon/UL18/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.histo.root"])
        # ext.add_weight_sets(["mu_Trig NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL18/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.histo.root"])
        ext.add_weight_sets(["mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL18/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.histo.root"])
        ext.add_weight_sets(["* * data/PU/UL18/puwei_UL18.histo.root"])
        ext.add_weight_sets(["* * data/PUJetId/UL18/UL18_jmar.corr.json"])
        with gzip.open("data/JEC_JERSF/UL18/compile_jec.pkl.gz") as fin: jmestuff = pickle.load(fin)
    # ext.add_weight_sets(["* * data/njet.root"])
    ext.finalize()
    evaluator = ext.make_evaluator()
    jet_factory = jmestuff["jet_factory"]
    return evaluator,jet_factory



def add_jec_variables(jets, event_rho):
    jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
    if hasattr(jets, "genJetIdxG"):jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets

## PU weight

## BTag SFs
# deepcsvb_sf = BTagScaleFactor("data/Rereco17_94X/BTag/DeepCSV_94XSF_V5_B_F.csv",BTagScaleFactor.RESHAPE,methods='iterativefit,iterativefit,iterativefit')
# deepcsvc_sf = "data/Rereco17_94X/BTag/DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.histo.root"
# deepjetb_sf = BTagScaleFactor("data/Rereco17_94X/BTag/DeepFlavour_94XSF_V4_B_F.csv",BTagScaleFactor.RESHAPE,methods='iterativefit,iterativefit,iterativefit')
# deepjetc_sf = "data/Rereco17_94X/BTag/DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.histo.root"

### Lepton SFs


def eleSFs(ele,year,evaluator):
    ele_eta = ak.fill_none(ele.eta,-999)
    ele_pt = ak.fill_none(ele.pt,0.)

    if year =="2016":
        weight =  evaluator["ele_ID_pre"](ele_eta,ele_pt)*np.where(evaluator["ele_Rereco_above20_pre"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_pre"](ele_eta,ele_pt))*np.where(evaluator["ele_Rereco_below20_pre"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_pre"](ele_eta,ele_pt))*19.52/(16.81+19.52)+evaluator["ele_ID_post"](ele_eta,ele_pt)*np.where(evaluator["ele_Rereco_above20_post"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_post"](ele_eta,ele_pt))*np.where(evaluator["ele_Rereco_below20_post"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_post"](ele_eta,ele_pt))*16.81/(16.81+19.52)
    else:
        weight =  evaluator["ele_ID"](ele_eta,ele_pt)*np.where(evaluator["ele_Rereco_above20"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20"](ele_eta,ele_pt))*np.where(evaluator["ele_Rereco_below20"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20"](ele_eta,ele_pt))
    return weight

def muSFs(mu,year,evaluator):
    # weight 
    mu_eta=ak.fill_none(mu.eta,-999)
    mu_pt= ak.fill_none(mu.pt,0.)
    if year == "2016":weight = evaluator["mu_ID_pre"](mu_eta,mu_pt)*evaluator["mu_Iso_pre"](mu_eta,mu_pt)*19.52/(16.81+19.52)+evaluator["mu_ID_post"](mu_eta,mu_pt)*evaluator["mu_Iso_post"](mu_eta,mu_pt)*16.81/(16.81+19.52)
    else: weight = evaluator["mu_ID"](mu_eta,mu_pt)*evaluator["mu_Iso"](mu_eta,mu_pt)

    return weight
def puwei(nPU,evaluator):
    return evaluator["PU"](nPU)
def puJetID_SFs(jet,evaluator,syst="nom",WP="L"):
    jet_eta = ak.fill_none(jet.eta,-999)
    jet_pt = ak.fill_none(jet.pt,0)
    _syst = ak.broadcast_arrays(syst,jet_pt)
    _WP = ak.broadcast_arrays(WP,jet_pt)

    return evaluator["PUJetID_eff"](jet_pt,jet_eta,_syst[0],_WP[0])

def ZpT_corr(zpt,year):
    if year =='2016' : weight =1.04713*(-0.055394*np.erf((zpt- 11.21831)/3.87755) + 0.00049300*
zpt+ 0.94418)
    elif year =='2017' : weight = 1.15072*(0.090490*np.erf((zpt-5.50288)/2.28427) + 0.0093880*zpt-3.13579*10^-5*zpt**2+0.74284)
    elif year =='2018' : weight =1.12666*(0.09487*erf((pT,Z-5.47228)/2.21332) + 0.0095931 *zpt -1.67661 *10^-6*zpt**2+ 0.75185)
    return weight

def nJet_corr(njet,evaluator):
    return evaluator["hnj"](njet)
#    return njwei(njet)
                    
