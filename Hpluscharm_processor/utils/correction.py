import numpy as np
import awkward as ak
import gzip
import pickle

from coffea.lookup_tools.lookup_base import lookup_base
from coffea.lookup_tools import extractor,txt_converters,rochester_lookup

import os
from coffea.lumi_tools import LumiMask
from coffea.btag_tools import BTagScaleFactor
from helpers.cTagSFReader import *
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory,JetResolutionScaleFactor,JetResolution

lumiMasks = {
    '2016': LumiMask('data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
    '2017': LumiMask('data/Lumimask/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
    '2018': LumiMask('data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt')
}

jec_stack_names=[]
# def init_corr(year):
ext = extractor()
    # print(os.getcwd())
# if year == "2016":
#         ##Electron
#         ext.add_weight_sets(["ele_Trig TrigSF data/Rereco17_94X/Lepton/Ele32_L1DoubleEG_TrigSF_vhcc.histo.histo.root"])
#         ext.add_weight_sets(["ele_ID_post EGamma_SF2D data/Electron/UL16/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D.histo.root"])
#         ext.add_weight_sets(["ele_Rereco_above20_post EGamma_SF2D data/Electron/UL16/egammaEffi_ptAbove20.txt_EGM2D_UL2016postVFP.histo.root"])
#         ext.add_weight_sets(["ele_Rereco_below20_post EGamma_SF2D data/Electron/UL16/egammaEffi_ptBelow20.txt_EGM2D_UL2016postVFP.histo.root"])
#         ext.add_weight_sets(["ele_ID_pre EGamma_SF2D data/Electron/UL16/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D.histo.root"])
#         ext.add_weight_sets(["ele_Rereco_above20_pre EGamma_SF2D data/Electron/UL16/egammaEffi_ptAbove20.txt_EGM2D_UL2016preVFP.histo.root"])
#         ext.add_weight_sets(["ele_Rereco_below20_pre EGamma_SF2D data/Electron/UL16/egammaEffi_ptBelow20.txt_EGM2D_UL2016preVFP.histo.root"]) 
#         ####Muon 
#         ext.add_weight_sets(["mu_ID_pre NUM_TightID_DEN_TrackerMuons_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.histo.root"])
#         # ext.add_weight_sets(["mu_Trig_pre NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_SingleMuonTriggers.histo.root"])
#         ext.add_weight_sets(["mu_Iso_pre NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.histo.root"])
#         ext.add_weight_sets(["mu_ID_post NUM_TightID_DEN_TrackerMuons_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.histo.root"])
#         # ext.add_weight_sets(["mu_Trig_post NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_SingleMuonTriggers.histo.root"])
#         ext.add_weight_sets(["mu_Iso_post NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL16/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.histo.root"])
#         ### JERC
#         ext.add_weight_sets([
#         "* * data/JEC_JERSF/UL16/Summer20UL16APV_JRV3_MC_SF_AK4PFchs.jersf.txt",
#         "* * data/JEC_JERSF/UL16/Summer20UL16_JRV3_MC_SF_AK4PFchs.jersf.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L1FastJet_AK4PFchs.jec.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs.jec.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L2Relative_AK4PFchs.jec.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L2Relative_AK4PFchs.jec.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L3Absolute_AK4PFchs.jec.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs.jec.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L2L3Residual_AK4PFchs.jec.txt",
#         "* * data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs.jec.txt",
#         ])
#         jec_stack_names = [
#                     "Summer19UL16_V7_MC_L1FastJet_AK4PFchs"
#                     "Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs"
#                     "Summer19UL16_V7_MC_L2Relative_AK4PFchs",
#                     "Summer19UL16APV_V7_MC_L2Relative_AK4PFchs",
#                     "Summer19UL16_V7_MC_L3Absolute_AK4PFchs",
#                     "Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs",
#                     "Summer19UL16_V7_MC_L2L3Residual_AK4PFchs",
#                     "Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs",
#                     ]
#     elif year == "2017":
        ### Electron 
ext.add_weight_sets(["ele_Rereco_above20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptAbove20.txt_EGM2D_UL2017.histo.root"])
ext.add_weight_sets(["ele_ID EGamma_SF2D data/Electron/UL17/egammaEffi.txt_EGM2D_MVA90iso_UL17.histo.root"])
ext.add_weight_sets(["ele_Rereco_below20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptBelow20.txt_EGM2D_UL2017.histo.root"])
### Muon
# ext.add_weight_sets(["mu_Trig NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL17/Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers.histo.root"])
ext.add_weight_sets(["mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL17/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.histo.root"])
ext.add_weight_sets(["mu_ID NUM_TightID_DEN_TrackerMuons_abseta_pt data/Muon/UL17/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.histo.root"])

### JERC
ext.add_weight_sets([
"* * data/JEC_JERSF/UL17/Summer19UL17_JRV2_MC_SF_AK4PFchs.jersf.txt",
"* * data/JEC_JERSF/UL17/Summer19UL17_V5_MC_L1FastJet_AK4PFchs.jec.txt",
"* * data/JEC_JERSF/UL17/Summer19UL17_V5_MC_L2Relative_AK4PFchs.jec.txt",
"* * data/JEC_JERSF/UL17/Summer19UL17_V5_MC_L3Absolute_AK4PFchs.jec.txt",
"* * data/JEC_JERSF/UL17/Summer19UL17_V5_MC_L2L3Residual_AK4PFchs.jec.txt",
])
jec_stack_names = [
    "Summer19UL17_V5_MC_L1FastJet_AK4PFchs",
    "Summer19UL17_V5_MC_L2Relative_AK4PFchs",
    "Summer19UL17_V5_MC_L3Absolute_AK4PFchs",
    "Summer19UL17_V5_MC_L2L3Residual_AK4PFchs",
            ]
    # elif year == "2018":
    #     ### Electron 
    #     ext.add_weight_sets(["ele_ID EGamma_SF2D data/Electron/UL17/egammaEffi.txt_EGM2D_MVA90iso_UL18.histo.root"])
    #     ext.add_weight_sets(["ele_Rereco_above20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptAbove20.txt_EGM2D_UL2018.histo.root"])
    #     ext.add_weight_sets(["ele_Rereco_below20 EGamma_SF2D data/Electron/UL17/egammaEffi_ptBelow20.txt_EGM2D_UL2018.histo.root"]) 
    #     ### Muon
    #     ext.add_weight_sets(["mu_ID NUM_TightID_DEN_TrackerMuons_abseta_pt  data/Muon/UL18/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.histo.root"])
    #     # ext.add_weight_sets(["mu_Trig NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt  data/Muon/UL18/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.histo.root"])
    #     ext.add_weight_sets(["mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt data/Muon/UL18/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.histo.root"])
    #     ### JERC
    #     ext.add_weight_sets([
    #     "* * data/JEC_JERSF/UL18/Summer19UL18_JRV2_MC_SF_AK4PFchs.jersf.txt",
    #     "* * data/JEC_JERSF/UL18/Summer19UL18_V5_MC_L1FastJet_AK4PFchs.jec.txt",
    #     "* * data/JEC_JERSF/UL18/Summer19UL18_V5_MC_L2Relative_AK4PFchs.jec.txt",
    #     "* * data/JEC_JERSF/UL18/Summer19UL18_V5_MC_L3Absolute_AK4PFchs.jec.txt",
    #     "* * data/JEC_JERSF/UL18/Summer19UL18_V5_MC_L2L3Residual_AK4PFchs.jec.txt",
    #     ])
    #     jec_stack_names = [
    #                 "Summer19UL18_V5_MC_L1FastJet_AK4PFchs"
    #                 "Summer19UL18_V5_MC_L2Relative_AK4PFchs",
    #                 "Summer19UL18_V5_MC_L3Absolute_AK4PFchs",
    #                 "Summer19UL18_V5_MC_L2L3Residual_AK4PFchs",
    #                 ]
ext.finalize()
evaluator = ext.make_evaluator()
jec_inputs = {name: evaluator[name] for name in jec_stack_names}
jec_stack = JECStack(jec_inputs)
name_map = jec_stack.blank_name_map
name_map['JetPt'] = 'pt'
name_map['JetMass'] = 'mass'
name_map['JetEta'] = 'eta'
name_map['JetA'] = 'area'


def jec(events,jets,isRealData):
    jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
    jets["event_rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'event_rho'
    if not isRealData:
        jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        name_map['ptGenJet'] = 'pt_gen'
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    events_cache = events.caches[0]
    corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
    return corrected_jets
## PU weight

## BTag SFs
# deepcsvb_sf = BTagScaleFactor("data/Rereco17_94X/BTag/DeepCSV_94XSF_V5_B_F.csv",BTagScaleFactor.RESHAPE,methods='iterativefit,iterativefit,iterativefit')
# deepcsvc_sf = "data/Rereco17_94X/BTag/DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.histo.root"
# deepjetb_sf = BTagScaleFactor("data/Rereco17_94X/BTag/DeepFlavour_94XSF_V4_B_F.csv",BTagScaleFactor.RESHAPE,methods='iterativefit,iterativefit,iterativefit')
# deepjetc_sf = "data/Rereco17_94X/BTag/DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.histo.root"

### Lepton SFs


def eleSFs(ele,year):
    
    ele_eta = ak.fill_none(ele.eta,0.)
    ele_pt = ak.fill_none(ele.pt,0.)
    
    if year =="2016":
        weight =  evaluator["ele_ID_pre"](ele_eta,ele_pt)*np.where(evaluator["ele_Rereco_above20_pre"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_pre"](ele_eta,ele_pt))*np.where(evaluator["ele_Rereco_below20_pre"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_pre"](ele_eta,ele_pt))*19.52/(16.81+19.52)+evaluator["ele_ID_post"](ele_eta,ele_pt)*np.where(evaluator["ele_Rereco_above20_post"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_post"](ele_eta,ele_pt))*np.where(evaluator["ele_Rereco_below20_post"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20_post"](ele_eta,ele_pt))*16.81/(16.81+19.52)
    else:
        weight =  evaluator["ele_ID"](ele_eta,ele_pt)*np.where(evaluator["ele_Rereco_above20"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20"](ele_eta,ele_pt))*np.where(evaluator["ele_Rereco_below20"](ele_eta,ele_pt)==0,1.,evaluator["ele_Rereco_above20"](ele_eta,ele_pt))
    return weight

def muSFs(mu,year):
    # weight 
    mu_eta=ak.fill_none(mu.eta,0.)
    mu_pt= ak.fill_none(mu.pt,0.)
    if year == "2016":weight = evaluator["mu_ID_pre"](mu_eta,mu_pt)*evaluator["mu_Iso_pre"](mu_eta,mu_pt)*19.52/(16.81+19.52)+evaluator["mu_ID_post"](mu_eta,mu_pt)*evaluator["mu_Iso_post"](mu_eta,mu_pt)*16.81/(16.81+19.52)
    else: weight = evaluator["mu_ID"](mu_eta,mu_pt)*evaluator["mu_Iso"](mu_eta,mu_pt)

    return weight
# def rochester(muon,isRealData):

#     rochester_data = lookup_tools.txt_converters.convert_rochester_file("tests/samples/RoccoR2018.txt.gz"loaduncs=True)

#     rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)

#     # to test 1-to-1 agreement with official Rochester requires loading C++ files
#     # instead, preload the correct scales in the sample directory
#     # the script tests/samples/rochester/build_rochester.py produces these

#     official_data_k = np.load("tests/samples/nano_dimuon_rochester.npy")
#     official_data_err = np.load("tests/samples/nano_dimuon_rochester_err.npy")
#     official_mc_k = np.load("tests/samples/nano_dy_rochester.npy")
#     official_mc_err = np.load("tests/samples/nano_dy_rochester_err.npy")
#     mc_rand = np.load("tests/samples/nano_dy_rochester_rand.npy")

#     # test against nanoaod
#     data_k = rochester.kScaleDT(muon.charge, muon.pt, muon.eta, muon.phi)

#     data_k = np.array(ak.flatten(data_k))

#     assert all(np.isclose(data_k, official_data_k))

#     data_err = rochester.kScaleDTerror(muon.charge, muon.pt, muon.eta, muon.phi)

#     data_err = np.array(ak.flatten(data_err), dtype=float)
#     assert all(np.isclose(data_err, official_data_err, atol=1e-8))

#     hasgen = ~np.isnan(ak.fill_none(muon.matched_gen.pt, np.nan))
#     mc_rand = ak.unflatten(mc_rand, ak.num(hasgen))

#     mc_kspread = rochester.kSpreadMC(muon.charge[hasgen],muon.pt[hasgen], muon.eta[hasgen], muon.phi[hasgen],muon.matched_gen.pt[hasgen])

#     mc_ksmear = rochester.kSmearMC(muon.charge[~hasgen],muon.pt[~hasgen],muon.eta[~hasgen],muon.phi[~hasgen],muon.nTrackerLayers[~hasgen],mc_rand[~hasgen],)

#     mc_k = np.array(ak.flatten(ak.ones_like(muon.pt)))
#     hasgen_flat = np.array(ak.flatten(hasgen))
#     mc_k[hasgen_flat] = np.array(ak.flatten(mc_kspread))
#     mc_k[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
#     assert all(np.isclose(mc_k, official_mc_k))

#     mc_errspread = rochester.kSpreadMCerror(muon.charge[hasgen],muon.pt[hasgen],muon.eta[hasgen],muon.phi[hasgen],muon.matched_gen.pt[hasgen],)

#     mc_errsmear = rochester.kSmearMCerror(muon.charge[~hasgen],muon.pt[~hasgen],muon.eta[~hasgen],muon.phi[~hasgen],
# muon.nTrackerLayers[~hasgen],mc_rand[~hasgen],)

#     mc_err = np.array(ak.flatten(ak.ones_like(muon.pt)))

#     mc_err[hasgen_flat] = np.array(ak.flatten(mc_errspread))

#     mc_err[~hasgen_flat] = np.array(ak.flatten(mc_errsmear))

#     assert all(np.isclose(mc_err, official_mc_err, atol=1e-8))
