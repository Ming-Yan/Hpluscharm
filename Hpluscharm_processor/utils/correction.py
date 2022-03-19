import gc
import numpy as np
import awkward as ak
import gzip
import pickle

from coffea.lookup_tools.lookup_base import lookup_base

from coffea.lookup_tools import extractor


from coffea.lumi_tools import LumiMask
from coffea.btag_tools import BTagScaleFactor
from helpers.cTagSFReader import *
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor

lumiMasks = 
{
    '2016': LumiMask('data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
    '2017': LumiMask('data/Lumimask/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
    '2018': LumiMask('data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt')
}
##JEC 
# with gzip.open("data/compiled_jec.pkl.gzt") as fin:
#     jmestuff = cloudpickle.load(fin)
ext = extractor()
ext.add_weight_sets([
    "* * data/Rereco17_94X/JEC_JERSF/MC/Fall17_17Nov2017_V32_MC_L1FastJet_AK4PFchs.jec.txt",
    "* * data/Rereco17_94X/JEC_JERSF/MC/Fall17_17Nov2017_V32_MC_L2Relative_AK4PFchs.jec.txt",
    "* * data/Rereco17_94X/JEC_JERSF/MC/Fall17_17Nov2017_V32_MC_L3Absolute_AK4PFchs.jec.txt",
    "* * data/Rereco17_94X/JEC_JERSF/MC/Fall17_17Nov2017_V32_MC_L2L3Residual_AK4PFchs.jec.txt",
])

ext.finalize()
jec_stack_names = [
                    "Fall17_17Nov2017_V32_MC_L1FastJet_AK4PFchs",
                   "Fall17_17Nov2017_V32_MC_L2Relative_AK4PFchs",
                   "Fall17_17Nov2017_V32_MC_L3Absolute_AK4PFchs",
                   "Fall17_17Nov2017_V32_MC_L2L3Residual_AK4PFchs",
                   ]
evaluator = ext.make_evaluator()
jec_inputs = {name: evaluator[name] for name in jec_stack_names}
jec_stack = JECStack(jec_inputs)
name_map = jec_stack.blank_name_map
name_map['JetPt'] = 'pt'
name_map['JetMass'] = 'mass'
name_map['JetEta'] = 'eta'
name_map['JetA'] = 'area'

def add_jec_variables(events,jets, event_rho):
    jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    events_cache = events.caches[0]
    corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
    gc.collect()
    return corrected_jets
## PU weight
with gzip.open("data/Rereco17_94X/94XPUwei_corrections.pkl.gz") as fin:
    compiled = pickle.load(fin)

## BTag SFs
deepcsvb_sf = BTagScaleFactor("data/Rereco17_94X/BTag/DeepCSV_94XSF_V5_B_F.csv",BTagScaleFactor.RESHAPE,methods='iterativefit,iterativefit,iterativefit')
deepcsvc_sf = "data/Rereco17_94X/BTag/DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root"
deepjetb_sf = BTagScaleFactor("data/Rereco17_94X/BTag/DeepFlavour_94XSF_V4_B_F.csv",BTagScaleFactor.RESHAPE,methods='iterativefit,iterativefit,iterativefit')
deepjetc_sf = "data/Rereco17_94X/BTag/DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2_interp.root"

### Lepton SFs
ext = extractor()
ext.add_weight_sets(["ele_Trig TrigSF data/Rereco17_94X/Lepton/Ele32_L1DoubleEG_TrigSF_vhcc.histo.root"])
ext.add_weight_sets(["ele_ID EGamma_SF2D data/Rereco17_94X/Lepton/ElectronIDSF_94X_MVA80WP.histo.root"])
ext.add_weight_sets(["ele_Rereco EGamma_SF2D data/Rereco17_94X/Lepton/ElectronRecoSF_94X.histo.root"])
ext.add_weight_sets(["mu_ID NUM_TightID_DEN_genTracks_pt_abseta data/Rereco17_94X/Lepton/RunBCDEF_SF_ID.histo.root"])
ext.add_weight_sets(["mu_ID_low NUM_TightID_DEN_genTracks_pt_abseta data/Rereco17_94X/Lepton/RunBCDEF_SF_MuID_lowpT.histo.root"])
ext.add_weight_sets(["mu_Iso NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta data/Rereco17_94X/Lepton/RunBCDEF_SF_ISO.histo.root"])

ext.finalize()
evaluator = ext.make_evaluator()

def eleSFs(ele,year):
    ele_eta = ak.fill_none(ele.eta,0.)
    ele_pt = ak.fill_none(ele.pt,0.)
    print(ak.type(ele_eta))
    #weight = evaluator["ele_Trig"](ele_eta,ele_pt)*evaluator["ele_ID"](ele_eta,ele_pt)*evaluator["ele_Rereco"](ele_eta,ele_pt)
    weight =  evaluator["ele_ID"](ele_eta,ele_pt)*evaluator["ele_Rereco"](ele_eta,ele_pt)  
    return weight

def muSFs(mu,year):
    
    # weight 
    mu_eta=ak.fill_none(mu.eta,0.)
    mu_pt= ak.fill_none(mu.pt,0.)
    weight = evaluator["mu_ID"](mu_eta,mu_pt)*evaluator["mu_ID_low"](mu_eta,mu_pt)*evaluator["mu_Iso"](mu_eta,mu_pt)

    return weight
def rochester():

    rochester_data = lookup_tools.txt_converters.convert_rochester_file(

        "tests/samples/RoccoR2018.txt.gz", loaduncs=True

    )

    rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)



    # to test 1-to-1 agreement with official Rochester requires loading C++ files

    # instead, preload the correct scales in the sample directory

    # the script tests/samples/rochester/build_rochester.py produces these

    official_data_k = np.load("tests/samples/nano_dimuon_rochester.npy")

    official_data_err = np.load("tests/samples/nano_dimuon_rochester_err.npy")

    official_mc_k = np.load("tests/samples/nano_dy_rochester.npy")

    official_mc_err = np.load("tests/samples/nano_dy_rochester_err.npy")

    mc_rand = np.load("tests/samples/nano_dy_rochester_rand.npy")



    # test against nanoaod

    events = NanoEventsFactory.from_root(

        os.path.abspath("tests/samples/nano_dimuon.root")

    ).events()



    data_k = rochester.kScaleDT(

        events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi

    )

    data_k = np.array(ak.flatten(data_k))

    assert all(np.isclose(data_k, official_data_k))

    data_err = rochester.kScaleDTerror(

        events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi

    )

    data_err = np.array(ak.flatten(data_err), dtype=float)

    assert all(np.isclose(data_err, official_data_err, atol=1e-8))



    # test against mc

    events = NanoEventsFactory.from_root(

        os.path.abspath("tests/samples/nano_dy.root")

    ).events()



    hasgen = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))

    mc_rand = ak.unflatten(mc_rand, ak.num(hasgen))

    mc_kspread = rochester.kSpreadMC(

        events.Muon.charge[hasgen],

        events.Muon.pt[hasgen],

        events.Muon.eta[hasgen],

        events.Muon.phi[hasgen],

        events.Muon.matched_gen.pt[hasgen],

    )

    mc_ksmear = rochester.kSmearMC(

        events.Muon.charge[~hasgen],

        events.Muon.pt[~hasgen],

        events.Muon.eta[~hasgen],

        events.Muon.phi[~hasgen],

        events.Muon.nTrackerLayers[~hasgen],

        mc_rand[~hasgen],

    )

    mc_k = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))

    hasgen_flat = np.array(ak.flatten(hasgen))

    mc_k[hasgen_flat] = np.array(ak.flatten(mc_kspread))

    mc_k[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))

    assert all(np.isclose(mc_k, official_mc_k))



    mc_errspread = rochester.kSpreadMCerror(

        events.Muon.charge[hasgen],

        events.Muon.pt[hasgen],

        events.Muon.eta[hasgen],

        events.Muon.phi[hasgen],

        events.Muon.matched_gen.pt[hasgen],

    )

    mc_errsmear = rochester.kSmearMCerror(

        events.Muon.charge[~hasgen],

        events.Muon.pt[~hasgen],

        events.Muon.eta[~hasgen],

        events.Muon.phi[~hasgen],

        events.Muon.nTrackerLayers[~hasgen],

        mc_rand[~hasgen],

    )

    mc_err = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))

    mc_err[hasgen_flat] = np.array(ak.flatten(mc_errspread))

    mc_err[~hasgen_flat] = np.array(ak.flatten(mc_errsmear))

    assert all(np.isclose(mc_err, official_mc_err, atol=1e-8))
