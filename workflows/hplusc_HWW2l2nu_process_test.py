import pickle, os, sys, numpy as np
from coffea import hist, processor
import awkward as ak
import hist as Hist
from coffea.analysis_tools import Weights,PackedSelection
from functools import partial
import xgboost as xgb
import gc
import os,psutil
import coffea
# import tracemalloc
xgb.set_config(verbosity=0)
from BTVNanoCommissioning.utils.correction import (
    lumiMasks,
    add_muSFs,
    add_eleSFs,
    load_pu,
    load_BTV,
    add_jec_variables,
    load_jetfactory,
    load_metfactory,
    met_filters,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_7pt,
    add_scalevar_3pt,
)
from Hpluscharm.utils.util import (
    mT,
    flatten,
    normalize,
    make_p4,
    defaultdict_accumulator,
    update,
)

from BTVNanoCommissioning.helpers.cTagSFReader import getSF

def dphilmet(l1, l2, met):
    return np.where(
        abs(l1.delta_phi(met)) < abs(l2.delta_phi(met)),
        abs(l1.delta_phi(met)),
        abs(l2.delta_phi(met)),
    )
def BDTreader(dmatrix, xgb_model):
    return 1.0 / (1 + np.exp(-xgb_model.predict(dmatrix)))
# code for which memory has to
# be monitored

class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, cfg):
        self.cfg = cfg
        self._year = self.cfg.dataset["year"]
        self._export_array = self.cfg.userconfig["export_array"]
        self.systematics = self.cfg.userconfig["systematics"]
        self._campaign = self.cfg.dataset["campaign"]
        self._mu1hlt = self.cfg.preselections["mu1hlt"]
        self._mu2hlt = self.cfg.preselections["mu2hlt"]
        self._e1hlt = self.cfg.preselections["e1hlt"]
        self._e2hlt = self.cfg.preselections["e2hlt"]
        self._emuhlt = self.cfg.preselections["emuhlt"]
        self._met_filters = met_filters[self._campaign]
        self._lumiMasks = lumiMasks[self._campaign]
        
        
        self._deepjetc_sf = load_BTV(
            self._campaign, self.cfg.weights_config["BTV"], "DeepJetC"
        )
        self._pu = load_pu(self._campaign, self.cfg.weights_config["PU"])
        self._jet_factory = load_jetfactory(
            self._campaign, self.cfg.weights_config["JME"]
        )
        self._met_factory = load_metfactory(
            self._campaign, self.cfg.weights_config["JME"]
        )
        if self._year == "2016":from Hpluscharm.MVA.training_config import config2016 as config
        if self._year == "2017":from Hpluscharm.MVA.training_config import config2017 as config
        if self._year == "2018":from Hpluscharm.MVA.training_config import config2018 as config
        self.xgb_model_ll = xgb.Booster()
        self.xgb_model_ll.load_model(cfg.userconfig["BDT"]["ll"])
        self.xgb_model_emu = xgb.Booster()
        self.xgb_model_emu.load_model(cfg.userconfig["BDT"]["emu"]) 
        
        
        flav_axis = Hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Genflavour")
        lepflav_axis = Hist.axis.StrCategory(
            ["ee", "mumu", "emu"], name="lepflav", label="channel"
        )
        dataset_axis = Hist.axis.StrCategory([], name="dataset", label="Primary dataset",growth=True)

        region_axis = Hist.axis.StrCategory(["SR", "SR2", "top_CR", "DY_CR"], name="region")
        syst_axis = Hist.axis.StrCategory([], name="syst", growth=True)

        pt_axis = Hist.axis.Regular(50,0,300, name="pt", label=" $p_{T}$ [GeV]")
        eta_axis = Hist.axis.Regular(25,-2.5,2.5, name="eta", label=" $\eta$")
        phi_axis = Hist.axis.Regular(30,-3,3, name="phi", label="$\phi$")
        dphi_axis = Hist.axis.Regular(30,-0.5,0.5, name="dphi", label="$D\phi$")
        mass_axis = Hist.axis.Regular(50,0,300, name="mass", label="$m$ [GeV]")
        widemass_axis = Hist.axis.Regular(50,0,1000, name="mass", label="$m$ [GeV]")
        disc_axis = Hist.axis.Regular(50,0, 1, name="disc", label="disc")
        # Events
        n_axis = Hist.axis.IntCategory([0, 1, 2, 3], name="n", label="N obj")
        nsv_axis = Hist.axis.Integer(0,20, name="nsv", label="N secondary vertices")
        npv_axis = Hist.axis.Integer(0,100, name="npvs", label="N primary vertices")
        # kinematic variables
        mt_axis = Hist.axis.Regular(30,0,300, name="mt", label=" $m_{T}$ [GeV]")
        dr_axis = Hist.axis.Regular(20,0,5, name="dr", label="$\Delta$R")
        iso_axis = Hist.axis.Regular(40,0, 0.05, name="pfRelIso03_all", label="Rel. Iso")
        dxy_axis = Hist.axis.Regular(40,-0.05, 0.05, name="dxy",  label="d_{xy}")
        dz_axis = Hist.axis.Regular(40, 0, 0.1, name="dz", label="d_{z}")
        # MET vars
        ratio_axis = Hist.axis.Regular(50,0, 10, name="ratio", label="ratio")
        uparper_axis = Hist.axis.Regular(50,-500, 500, name="uparper", label="$u_\par$")
        

        self.make_output = lambda:{
            "cutflow": processor.defaultdict_accumulator(
                    #         # we don't use a lambda function to avoid pickle issues
                    partial(processor.defaultdict_accumulator, int)),
            "sumw": 0,
            "npvs": Hist.Hist(lepflav_axis,region_axis,flav_axis,npv_axis,syst_axis,Hist.storage.Weight()),    
            "nsv": Hist.Hist(lepflav_axis,region_axis,flav_axis,nsv_axis,Hist.storage.Weight()),    
            "MET_ptdivet": Hist.Hist(lepflav_axis,region_axis,flav_axis,ratio_axis,Hist.storage.Weight()),
            "u_par": Hist.Hist(lepflav_axis,region_axis,flav_axis,uparper_axis,Hist.storage.Weight()),
            "u_per": Hist.Hist(lepflav_axis,region_axis,flav_axis,uparper_axis,Hist.storage.Weight()),       
            "mT1": Hist.Hist(lepflav_axis,region_axis,flav_axis,mt_axis,Hist.storage.Weight()),
            "mT2": Hist.Hist(lepflav_axis,region_axis,flav_axis,mt_axis,Hist.storage.Weight()),
            "mTh": Hist.Hist(lepflav_axis,region_axis,flav_axis,mt_axis,Hist.storage.Weight()),
            "ll_mass":Hist.Hist(lepflav_axis,region_axis,flav_axis,mass_axis,Hist.storage.Weight()),
            "lep1_pfRelIso03_all":Hist.Hist(lepflav_axis,region_axis,flav_axis,iso_axis,Hist.storage.Weight()),
            "lep2_pfRelIso03_all":Hist.Hist(lepflav_axis,region_axis,flav_axis,iso_axis,Hist.storage.Weight()),
            "lep1_dxy":Hist.Hist(lepflav_axis,region_axis,flav_axis,dxy_axis,Hist.storage.Weight()),
            "lep2_dxy":Hist.Hist(lepflav_axis,region_axis,flav_axis,dxy_axis,Hist.storage.Weight()),
            "lep1_dz":Hist.Hist(lepflav_axis,region_axis,flav_axis,dz_axis,Hist.storage.Weight()),
            "lep2_dz":Hist.Hist(lepflav_axis,region_axis,flav_axis,dz_axis,Hist.storage.Weight()),
            "lep1_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
            "lep2_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
            "jetflav_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
            "ll_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
            "ttbar_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
            "topjet1_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
            "topjet2_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
            "lep1_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "lep2_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "jetflav_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "ll_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "ttbar_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "topjet1_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "topjet2_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "lep1_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "lep2_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "jetflav_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "ll_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "ttbar_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "topjet1_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "topjet2_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "nj": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
            "nele": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
            "nmu": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
            "njmet": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
            "h_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "MET_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "TkMET_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "PuppiMET_pt":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "MET_proj":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "TkMET_proj":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "minMET_proj":Hist.Hist(lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "MET_phi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "TkMET_phi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "PuppiMET_phi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "METTkMETdphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l1met_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l2met_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "cmet_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l1W1_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l1W2_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l2W1_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l2W2_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l1W1_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "metW1_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "metW2_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "cW1_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "cW2_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "W1W2_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l1h_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l2h_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "meth_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "ch_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "W1h_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "W2h_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "llmet_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "llW1_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "llW2_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "llh_dphi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
            "l1l2_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
            "l1c_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
            "l2c_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
            "lll1_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
            "lll2_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
            "llc_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
            "jetflav_btagDeepCvL": Hist.Hist(lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
            "jetflav_btagDeepCvB": Hist.Hist(lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
            "jetflav_btagDeepFlavCvL": Hist.Hist(lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
            "jetflav_btagDeepFlavCvB": Hist.Hist(lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
            "template_ll_mass":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mass_axis,Hist.storage.Weight()),
            "template_top1_mass":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mass_axis,Hist.storage.Weight()),
            "template_top2_mass":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mass_axis,Hist.storage.Weight()),
            "template_HT_mass":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,widemass_axis,Hist.storage.Weight()),
            "template_tt_mass":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,widemass_axis,Hist.storage.Weight()),
            "template_llbb_mass":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,widemass_axis,Hist.storage.Weight()),
            "template_mTh":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mt_axis,Hist.storage.Weight()),
            "template_mT1":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mt_axis,Hist.storage.Weight()),
            "template_mT2":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mt_axis,Hist.storage.Weight()),
            "template_MET":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "template_jetpt":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
            "template_BDT":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight())
            }
            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
       
        if isRealData:

            jets =events.Jet
            met = events.MET
        else:
            jetfac_name = "mc"
            jets = self._jet_factory["mc"].build(
            add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), lazy_cache=events.caches[0]
        )
            met = self._met_factory.build(events.MET, jets, {})
            # met = events.MET
        
        shifts = [
            ({"Jet": jets, "MET": met}, None),
        ]
        
        if not isRealData:
            if self.systematics["JERC"]:
                print("jerc")
                shifts += [
                    (
                        {
                            "Jet": jets.JES_jes.up,
                            "MET": met.JES_jes.up,
                        },
                        "JESUp",
                    ),
                    (
                        {
                            "Jet": jets.JES_jes.down,
                            "MET": met.JES_jes.down,
                        },
                        "JESDown",
                    ),
                    (
                        {
                            "Jet": jets,
                            "MET": met.MET_UnclusteredEnergy.up,
                        },
                        "UESUp",
                    ),
                    (
                        {
                            "Jet": jets,
                            "MET": met.MET_UnclusteredEnergy.down,
                        },
                        "UESDown",
                    ),
                    (
                        {
                            "Jet": jets.JER.up,
                            "MET": met.JER.up,
                           
                        },
                        "JERUp",
                    ),
                    (
                        {
                            "Jet": jets.JER.down,
                            "MET": met.JER.down,
                           
                        },
                        "JERDown",
                    ),
                ]
        else:
            # HEM15/16 issue
            if "18" in self._campaign:
                _runid = events.run >= 319077
                j_mask = ak.where(
                    (jets.phi > -1.57)
                    & (jets.phi < -0.87)
                    & (jets.eta > -2.5)
                    & (jets.eta < 1.3),
                    0.8,
                    1,
                )
                shift_jets = copy.deepcopy(jets)
                for collection, mask in zip([shift_jets], [j_mask]):
                    collection["pt"] = mask * collection.pt
                    collection["mass"] = mask * collection.mass
                shifts.extend(
                    [
                        ({"Jet": shift_jets, "MET": met}, "HEM18"),
                    ]
                )

        
        print("load : ",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events,shift_name):
        
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        output = self.make_output()
        # if self._export_array: output_array = ak.Array({shift_name:ak.A({})})
        if shift_name is None and not isRealData:
            output["sumw"] = ak.sum(events.genWeight / abs(events.genWeight))
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self._lumiMasks(events.run, events.luminosityBlock)
        selection.add("lumi", ak.to_numpy(req_lumi))
        del req_lumi

        # #############Selections

        trigger_ee = np.zeros(len(events), dtype="bool")
        trigger_mm = np.zeros(len(events), dtype="bool")
        trigger_em = np.zeros(len(events), dtype="bool")
        trigger_e = np.zeros(len(events), dtype="bool")
        trigger_m = np.zeros(len(events), dtype="bool")
        trigger_ee = np.zeros(len(events), dtype="bool")
        trigger_mm = np.zeros(len(events), dtype="bool")
        trigger_ele = np.zeros(len(events), dtype="bool")
        trigger_mu = np.zeros(len(events), dtype="bool")
        for t in self._mu1hlt:
            if t in events.HLT.fields:
                trigger_m = trigger_m | events.HLT[t]
        for t in self._mu2hlt:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]
        for t in self._e1hlt:
            if t in events.HLT.fields:
                trigger_e = trigger_e | events.HLT[t]
        for t in self._e2hlt:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]
        for t in self._emuhlt:
            if t in events.HLT.fields:
                trigger_em = trigger_em | events.HLT[t]

        if isRealData:
            if "MuonEG" in dataset:
                trigger_em = trigger_em
                trigger_ele = np.zeros(len(events), dtype="bool")
                trigger_mu = np.zeros(len(events), dtype="bool")
            elif "DoubleEG" in dataset:
                trigger_ele = trigger_ee  # & ~trigger_em
                trigger_mu = np.zeros(len(events), dtype="bool")
                trigger_em = np.zeros(len(events), dtype="bool")
            elif "SingleElectron" in dataset:
                trigger_ele = trigger_e & ~trigger_ee & ~trigger_em
                trigger_mu = np.zeros(len(events), dtype="bool")
                trigger_em = np.zeros(len(events), dtype="bool")
            elif "DoubleMuon" in dataset:
                trigger_mu = trigger_mm
                trigger_ele = np.zeros(len(events), dtype="bool")
                trigger_em = np.zeros(len(events), dtype="bool")
            elif "SingleMuon" in dataset:
                trigger_mu = trigger_m & ~trigger_mm & ~trigger_em
                trigger_ele = np.zeros(len(events), dtype="bool")
                trigger_em = np.zeros(len(events), dtype="bool")
        else:
            trigger_mu = trigger_mm | trigger_m
            trigger_ele = trigger_ee | trigger_e
        selection.add("trigger_ee", ak.to_numpy(trigger_ele))
        selection.add("trigger_mumu", ak.to_numpy(trigger_mu))
        selection.add("trigger_emu", ak.to_numpy(trigger_em))
        del trigger_e, trigger_ee, trigger_m, trigger_mm
        metfilter = np.ones(len(events), dtype="bool")
        for flag in self._met_filters["data" if isRealData else "mc"]:
            metfilter &= np.array(events.Flag[flag])
        selection.add("metfilter", metfilter)
        del metfilter
        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon
        musel = (
            (event_mu.pt > 13)
            & (abs(event_mu.eta) < 2.4)
            & (event_mu.mvaId >= 3)
            & (event_mu.pfRelIso04_all < 0.15)
            & (abs(event_mu.dxy) < 0.05)
            & (abs(event_mu.dz) < 0.1)
        )
        event_mu["lep_flav"] = 13 * event_mu.charge
        event_mu = event_mu[ak.argsort(event_mu.pt, axis=1, ascending=False)]
        event_mu = event_mu[musel]
        event_mu = ak.pad_none(event_mu, 2, axis=1)
        nmu = ak.sum(musel, axis=1)
        amu = events.Muon[
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & (events.Muon.mvaId >= 1)
        ]
        namu = ak.count(amu.pt, axis=1)
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron
        event_e["lep_flav"] = 11 * event_e.charge
        elesel = (
            (event_e.pt > 13)
            & (abs(event_e.eta) < 2.5)
            & (event_e.mvaFall17V2Iso_WP90 == 1)
            & (abs(event_e.dxy) < 0.05)
            & (abs(event_e.dz) < 0.1)
        )
        event_e = event_e[elesel]
        event_e = event_e[ak.argsort(event_e.pt, axis=1, ascending=False)]
        event_e = ak.pad_none(event_e, 2, axis=1)
        nele = ak.sum(elesel, axis=1)
        aele = events.Electron[
            (events.Electron.pt > 12)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.mvaFall17V2Iso_WPL == 1)
        ]
        naele = ak.count(aele.pt, axis=1)

        selection.add("lepsel", ak.to_numpy((nele + nmu >= 2)))


        good_leptons = ak.with_name(
            ak.concatenate([event_e, event_mu], axis=1),
            "PtEtaPhiMCandidate",
        )
        # del event_e, event_mu
        good_leptons = good_leptons[
            ak.argsort(good_leptons.pt, axis=1, ascending=False)
        ]
        
        leppair = ak.combinations(
            good_leptons,
            n=2,
            replacement=False,
            axis=-1,
            fields=["lep1", "lep2"],
        )
        
        # del good_leptons
        ll_cand = ak.zip(
            {
                "lep1": leppair.lep1,
                "lep2": leppair.lep2,
                "pt": (leppair.lep1 + leppair.lep2).pt,
                "eta": (leppair.lep1 + leppair.lep2).eta,
                "phi": (leppair.lep1 + leppair.lep2).phi,
                "mass": (leppair.lep1 + leppair.lep2).mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        # del leppair
        ll_cand = ak.pad_none(ll_cand, 1, axis=1)
        ll_cand = ak.packed(ll_cand)
        print("before weight:",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")      
        
        selection.add("ee", ak.to_numpy(nele == 2))
        selection.add("mumu", ak.to_numpy(nmu == 2))
        selection.add("emu", ak.to_numpy((nele == 1) & (nmu == 1)))
        
        # ###########

        met = ak.zip(
            {
                "pt": events.MET.pt,
                "eta": ak.zeros_like(events.MET.pt),
                "phi": events.MET.phi,
                "energy": events.MET.sumEt,
            },
            with_name="PtEtaPhiELorentzVector",
        )
        tkmet = ak.zip(
            {
                "pt": events.TkMET.pt,
                "phi": events.TkMET.phi,
                "eta": ak.zeros_like(events.TkMET.pt),
                "energy": events.TkMET.sumEt,
            },
            with_name="PtEtaPhiELorentzVector",
        )
        
        
        
        jetsel = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) <= 2.4)
            & ((events.Jet.puId > 6) | (events.Jet.pt > 50))
            & (events.Jet.jetId > 5)
            & ak.all(
                (events.Jet.metric_table(ll_cand.lep1) > 0.4)
                & (events.Jet.metric_table(ll_cand.lep2) > 0.4),
                axis=2,
            )
            & ak.all(events.Jet.metric_table(aele) > 0.4, axis=2)
            & ak.all(events.Jet.metric_table(amu) > 0.4, axis=2)
        )
        njet = ak.sum(jetsel, axis=1)
        topjetsel = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) <= 2.4)
            & ((events.Jet.puId > 6) | (events.Jet.pt > 50))
            & (events.Jet.jetId > 5)
            & (events.Jet.btagDeepFlavB > 0.0532)
        )

        cvbcutll = events.Jet.btagDeepFlavCvB >= 0.42
        cvlcutll = events.Jet.btagDeepFlavCvL >= 0.22
        cvbcutem = events.Jet.btagDeepFlavCvB >= 0.5
        cvlcutem = events.Jet.btagDeepFlavCvL >= 0.12
        

        sr_cut = (
            (mT(ll_cand.lep2, met) > 30)
            & (mT(ll_cand, met) > 60)
            & (events.MET.sumEt > 45)
        )
        dy_cr2_cut = (
            (mT(ll_cand.lep2, met) > 30)
            & (events.MET.sumEt > 45)
            & (mT(ll_cand, met) < 60)
        )
        top_cr2_cut = (
            (mT(ll_cand.lep2, met) > 30)
            & (ll_cand.mass > 50)
            & (events.MET.sumEt > 45)
            & (abs(ll_cand.mass - 91.18) > 15)
        )
        global_cut = (
            (ll_cand.lep1.pt > 25)
            & (ll_cand.mass > 12)
            & (ll_cand.pt > 30)
            & (ll_cand.lep1.charge + ll_cand.lep2.charge == 0)
            & (events.MET.pt > 20)
            & (make_p4(ll_cand.lep1).delta_r(make_p4(ll_cand.lep2)) > 0.4)
            & (abs(met.delta_phi(tkmet)) < 0.5)   
        )
        sr_cut = (
            (mT(ll_cand.lep2, met) > 30)
            & (mT(ll_cand, met) > 60)
            & (events.MET.sumEt > 45)
        )
        llmass_cut = abs(ll_cand.mass - 91.18) > 15
        
        if shift_name is None:
            if "DoubleEG" in dataset:
                output["cutflow"][dataset]["trigger"] += ak.sum(trigger_ele)
            elif "DoubleMuon" in dataset:
                output["cutflow"][dataset]["trigger"] += ak.sum(trigger_mu)

            output["cutflow"][dataset]["global selection"] += ak.sum(
            ak.any(global_cut, axis=-1)
        )
            output["cutflow"][dataset]["signal region"] += ak.sum(
                ak.any(global_cut & sr_cut, axis=-1)
            )
            output["cutflow"][dataset]["selected jets"] += ak.sum(
                ak.any(global_cut & sr_cut, axis=-1) & (njet > 0)
            )
            output["cutflow"][dataset]["all ee"] += ak.sum(
                ak.any(global_cut & sr_cut, axis=-1)
                & (njet > 0)
                & (ak.all(llmass_cut) & trigger_ele)
                # & (nele == 2)
                # & (nmu == 0)
            )
            output["cutflow"][dataset]["all mumu"] += ak.sum(
                ak.any(global_cut & sr_cut, axis=-1)
                & (njet > 0)
                & (ak.all(llmass_cut) & trigger_mu)
                # & (nmu == 2)
                # & (nele == 0) 
            )
            output["cutflow"][dataset]["all emu"] += ak.sum(
                ak.any(global_cut & sr_cut, axis=-1)
                & (njet > 0)
                & (nele == 1)
                & (nmu == 1)
                & trigger_em
            )
        selection.add("llmass", ak.to_numpy(ak.all(llmass_cut, axis=-1)))
        selection.add(
            "SR_ee",
            ak.to_numpy(
                ak.any(sr_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) > 1)
            ),
        )
        selection.add(
            "SR2_ee",
            ak.to_numpy(
                ak.any(sr_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) == 1)
            ),
        )
        selection.add(
            "top_CR_ee",
            ak.to_numpy(
                ak.any(top_cr2_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) >= 2)
            ),
        )
        selection.add(
            "DY_CR_ee",
            ak.to_numpy(
                ak.any(dy_cr2_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) >= 1)
            ),
        )
        selection.add(
            "SR_mumu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) > 1)
            ),
        )
        selection.add(
            "SR2_mumu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) == 1)
            ),
        )
        selection.add(
            "top_CR_mumu",
            ak.to_numpy(
                ak.any(top_cr2_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) >= 2)
            ),
        )
        selection.add(
            "DY_CR_mumu",
            ak.to_numpy(
                ak.any(dy_cr2_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutll & cvlcutll, axis=1) >= 1)
            ),
        )
        selection.add(
            "SR_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) > 1)
            ),
        )
        selection.add(
            "SR2_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) == 1)
            ),
        )
        selection.add(
            "top_CR_emu",
            ak.to_numpy(
                ak.any(top_cr2_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) >= 2)
            ),
        )
        selection.add(
            "DY_CR_emu",
            ak.to_numpy(
                ak.any(dy_cr2_cut & global_cut, axis=-1)
                & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) >= 1)
            ),
        )
        # # selection.add('DY_CRb',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&cvlcut&~cvbcut,axis=1)==1)))
        # # selection.add('DY_CRl',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&~cvlcut&cvbcut,axis=1)>=1)))
        # # selection.add('DY_CRc',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&cvlcut&cvbcut,axis=1)==1)))
        
        lepflav = ["ee", "mumu", "emu"]
        reg = ["SR", "SR2", "DY_CR", "top_CR"]
    
        mask_lep = {
            "SR": global_cut & sr_cut,
            "SR2": global_cut & sr_cut,
            "DY_CR": global_cut & dy_cr2_cut,
            "top_CR": global_cut & top_cr2_cut,
        }
        mask_jet = {
            "ee": jetsel & cvbcutll & cvlcutll,
            "mumu": jetsel & cvbcutll & cvlcutll,
            "emu": jetsel & cvbcutem & cvlcutem,
        }
       
        # ### Weights
        weights = Weights(len(events), storeIndividual=True)
        if isRealData:
            weights.add("genweight", np.ones(len(events)))
            # output["cutflow"][dataset]["all"] += len(events)
        else:
            # output["cutflow"][dataset]["all"] += ak.sum(
            #     events.genWeight / abs(events.genWeight)
            # )
            weights.add("genweight", events.genWeight / abs(events.genWeight))
            weights.add(
                "L1prefireweight",
                events.L1PreFiringWeight.Nom,
                events.L1PreFiringWeight.Up,
                events.L1PreFiringWeight.Dn,
            )
            weights.add(
                "puweight",
                self._pu["PU"](events.Pileup.nTrueInt),
                self._pu["PUup"](events.Pileup.nTrueInt),
                self._pu["PUdn"](events.Pileup.nTrueInt),
            )
            if "PSWeight" in events.fields:
                add_ps_weight(weights, events.PSWeight)
            else:
                add_ps_weight(weights, None)
            if "LHEPdfWeight" in events.fields:
                add_pdf_weight(weights, events.LHEPdfWeight)
            else:
                add_pdf_weight(weights, None)
            if "LHEScaleWeight" in events.fields:
                add_scalevar_7pt(weights, events.LHEScaleWeight)
                add_scalevar_3pt(weights, events.LHEScaleWeight)
            else:
                add_scalevar_7pt(weights, [])
                add_scalevar_3pt(weights, [])
        print("weight : ",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
        
        
        for r in reg:
            for ch in lepflav:
                if "SR" in r and (ch == "ee" or ch == "mumu"):
                    cut = selection.all(
                        "lepsel",
                        "metfilter",
                        "lumi",
                        "%s_%s" % (r, ch),
                        ch,
                        "trigger_%s" % (ch),
                        "llmass",
                    )
                else:
                    cut = selection.all(
                        "lepsel",
                        "metfilter",
                        "lumi",
                        "%s_%s" % (r, ch),
                        ch,
                        "trigger_%s" % (ch),
                    )

                ll_cands = ak.mask(ll_cand, mask_lep[r])
                if ak.count(ll_cands.pt) > 0:
                    ll_cands = ll_cands[
                        ak.argsort(ll_cands.pt, axis=1, ascending=False)
                    ]
                sel_jetflav = ak.mask(events.Jet, mask_jet[ch])
                sel_cjet_flav = sel_jetflav
                if ak.count(sel_cjet_flav.pt) > 0:
                    sel_cjet_flav = sel_cjet_flav[
                        ak.argsort(
                            sel_cjet_flav.btagDeepFlavCvL, axis=1, ascending=False
                        )
                    ]
                nseljet = ak.count(sel_cjet_flav.pt, axis=1)
                topjets = ak.mask(events.Jet, topjetsel)
                if ak.count(topjets.pt) > 0:
                    topjets = topjets[
                        ak.argsort(topjets.btagDeepFlavCvB, axis=1)
                    ]
                topjets  = ak.pad_none(topjets,2,axis=1)
                
                sel_cjet_flav = ak.pad_none(sel_cjet_flav, 1, axis=1)
                sel_cjet_flav = sel_cjet_flav[:, 0]
                ll_cands = ak.pad_none(ll_cands, 1, axis=1)

                llcut = ll_cands[:, 0]
                llcut = ak.packed(llcut)
                lep1cut = llcut.lep1
                lep2cut = llcut.lep2
                w1cut = lep1cut + met
                w2cut = lep2cut + met
                hcut = llcut + met
                
                if not isRealData:
                    
                    add_eleSFs(
                        lep1cut,
                        self._campaign,
                        self.cfg.weights_config["LSF"],
                        weights,
                        cut,
                    )
                    add_eleSFs(
                        lep2cut,
                        self._campaign,
                        self.cfg.weights_config["LSF"],
                        weights,
                        cut,
                    )
                    add_muSFs(
                        lep1cut,
                        self._campaign,
                        self.cfg.weights_config["LSF"],
                        weights,
                        cut,
                    )
                    add_muSFs(
                        lep2cut,
                        self._campaign,
                        self.cfg.weights_config["LSF"],
                        weights,
                        cut,
                    )
                    jetsf = np.where(
                        cut,
                        getSF(
                            sel_cjet_flav.hadronFlavour,
                            sel_cjet_flav.btagDeepFlavCvL,
                            sel_cjet_flav.btagDeepFlavCvB,
                            self._deepjetc_sf,
                        ),
                        1.0,
                    )
                    jetsf_up = np.where(
                        cut,
                        getSF(
                            sel_cjet_flav.hadronFlavour,
                            sel_cjet_flav.btagDeepFlavCvL,
                            sel_cjet_flav.btagDeepFlavCvB,
                            self._deepjetc_sf,
                            "TotalUncUp",
                        ),
                        1.0,
                    )
                    jetsf_dn = np.where(
                        cut,
                        getSF(
                            sel_cjet_flav.hadronFlavour,
                            sel_cjet_flav.btagDeepFlavCvL,
                            sel_cjet_flav.btagDeepFlavCvB,
                            self._deepjetc_sf,
                            "TotalUncDown",
                        ),
                        1.0,
                    )
                    weights.add("cjetSFs", jetsf, jetsf+jetsf_up, jetsf-jetsf_dn)
                    if shift_name is None:
                        systematics = ['nominal'] + list(weights.variations)
                    else:
                        systematics = [shift_name]
                else :systematics =['nominal']
                if isRealData:
                    flavor = ak.zeros_like(sel_cjet_flav["pt"])
                else:
                    flavor = sel_cjet_flav.hadronFlavour + 1 * (
                        (sel_cjet_flav.partonFlavour == 0)
                        & (sel_cjet_flav.hadronFlavour == 0)
                    )
                flavor = flavor[cut]
                llcut=llcut[cut]
                lep1cut=lep1cut[cut]
                lep2cut=lep2cut[cut]
                w1cut=w1cut[cut]
                w2cut = w2cut[cut]
                hcut = hcut[cut]
                if "SR" in r:
                    utv = lep1cut+lep2cut+met[cut]
                    utv_p = np.sqrt(utv.px**2 + utv.py**2 + utv.pz**2)
                topjetcut=topjets[cut]
                
                
                # print("nearest before",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
                if ak.count(topjetcut.pt) > 0:
                    topjet1cut = topjetcut[ak.argsort(topjetcut.delta_r(lep1cut), axis=1)]
                    topjet2cut = topjetcut[ak.argsort(topjetcut.delta_r(lep2cut), axis=1)]
                else :
                    topjet1cut = topjetcut
                    topjet2cut = topjetcut
                topjet1cut= topjet1cut[:,0]
                topjet2cut = topjet2cut[:,0]
                
                
                # # lep1cut=ak.packed(lep1cut)
                # # lep2cut=ak.packed(lep2cut)
                # # print(ak.type(topjetcut),ak.type(lep1cut))
                # topjet1cut=llcut.lep1.nearest(events.Jet)
                # topjet2cut=llcut.lep2.nearest(events.Jet)
                # # topjet1cut=events[cut].Muon[:,0].nearest(topjetcut)
                # # topjet2cut=events[cut].Muon[:,0].nearest(topjetcut)
                # # topjet1cut=topjetcut.nearest(events.Muon)
                # # topjet2cut=topjetcut.nearest(events.Muon)
                # print("nearest after")
                ttbarcut= ak.zip(
                    {
                        "x": (topjet1cut.px+topjet2cut.px+lep1cut.px+lep2cut.px+met[cut].px)/np.sqrt(met[cut].pt**2+topjet1cut.pt**2+topjet1cut.pz**2+topjet2cut.pt**2+topjet2cut.pz**2+lep1cut.pt**2+lep1cut.pz**2+lep2cut.pt**2+lep2cut.pz**2),
                        "y": (topjet1cut.py+topjet2cut.py+lep1cut.py+lep2cut.py+met[cut].py)/np.sqrt(met[cut].pt**2+topjet1cut.pt**2+topjet1cut.pz**2+topjet2cut.pt**2+topjet2cut.pz**2+lep1cut.pt**2+lep1cut.pz**2+lep2cut.pt**2+lep2cut.pz**2),
                        "z": (topjet1cut.pz+topjet2cut.pz+lep1cut.pz+lep2cut.pz)/np.sqrt(met[cut].pt**2+topjet1cut.pt**2+topjet1cut.pz**2+topjet2cut.pt**2+topjet2cut.pz**2+lep1cut.pt**2+lep1cut.pz**2+lep2cut.pt**2+lep2cut.pz**2),
                        "t": np.sqrt(met[cut].pt**2+topjet1cut.pt**2+topjet1cut.pz**2+topjet2cut.pt**2+topjet2cut.pz**2+lep1cut.pt**2+lep1cut.pz**2+lep2cut.pt**2+lep2cut.pz**2),
                    },
                    with_name="LorentzVector",
                )
                # ttbarcut=topjet1cut+topjet2cut+lep1cut+lep2cut+met[cut]
                llbbcut = topjet1cut+topjet2cut+lep1cut+lep2cut
                llbbcut = llbbcut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                ttbarcut = ttbarcut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                top1cut = topjet1cut+met[cut]
                top2cut = topjet2cut+met[cut]
                topjet1cut = topjet1cut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                topjet2cut = topjet2cut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                top1cut = top1cut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                top2cut = top2cut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                sel_cjet_flav = sel_cjet_flav[cut]                 
                
                if shift_name is None: 
                    for histname, h in output.items():
                        if "ll_" not in histname and "jetflav_" not in histname and "lep1_" not in histname and "lep2_" not in histname: continue
                        if "jetflav_" in histname:
                            h.fill(ch,r,flavor,normalize(flatten(sel_cjet_flav[histname.replace("jetflav_", "")])),weight=weights.weight()[cut])
                           
                            # if self._export_array:
                                
                                # output[arrayname][dataset][
                                #     histname
                                # ] += processor.column_accumulator(
                                #     ak.to_numpy(
                                #         normalize(
                                #             sel_cjet_flav[histname.replace("jetflav_", "")]
                                #         )
                                #     )
                                # )

                        elif "lep1_" in histname:
                            h.fill(ch,r,flavor,normalize(flatten(lep1cut[histname.replace("lep1_", "")])),weight=weights.weight()[cut])
                            # if self._export_array and arrayname=="array":
                                # output[arrayname][dataset][
                                #     histname
                                # ] += processor.column_accumulator(
                                #     ak.to_numpy(
                                #         normalize(
                                #             flatten(lep1cut[histname.replace("lep1_", "")])
                                #         )
                                #     )
                                # )
                        elif "lep2_" in histname:
                            h.fill(ch,r,flavor,normalize(flatten(lep2cut[histname.replace("lep2_", "")])),weight=weights.weight()[cut])
                            # if self._export_array and arrayname=="array":
                                # output[arrayname][dataset][
                                #     histname
                                # ] += processor.column_accumulator(
                                #     ak.to_numpy(
                                #         normalize(
                                #             flatten(lep2cut[histname.replace("lep2_", "")])
                                #         )
                                #     )
                                # )

                        
                        elif "ll_" in histname and "template" not in histname:
                            h.fill(ch,r,flavor,normalize(flatten(llcut[histname.replace("ll_", "")])),weight=weights.weight()[cut])

                            # if self._export_array and arrayname=="array": 
                                # output[arrayname][dataset][
                                #     histname
                                # ] += processor.column_accumulator(
                                #     ak.to_numpy(
                                #         normalize(
                                #             flatten(llcut[histname.replace("ll_", "")])
                                #         )
                                #     )
                                # )
                        elif 'topjet1_' in histname :
                            h.fill(ch,r,flavor,normalize(flatten(topjet1cut[histname.replace("topjet1_", "")])),weight=weights.weight()[cut])
                            # if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(topjet1cut[histname.replace('topjet1_','')]))))
                        elif 'topjet2_' in histname :
                            h.fill(ch,r,flavor,normalize(flatten(topjet2cut[histname.replace("topjet2_", "")])),weight=weights.weight()[cut])
                            # if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(topjet2cut[histname.replace('topjet2_','')]))))
                        elif 'ttbar_' in histname:
                            h.fill(ch,r,flavor,normalize(flatten(ttbarcut[histname.replace("ttbar_", "")])),weight=weights.weight()[cut])
                            # if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(ttbarcut[histname.replace('ttbar_','')]))))
                        # elif 'top1_' in histname:
                        #     h.fill(ch,r,flavor,normalize(flatten(top1cut[histname.replace("top1_", "")])),weight=weights.weight()[cut])
                        #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(top1cut[histname.replace('top1_','')]))))
                        # elif 'top2_' in histname:
                        #     h.fill(ch,r,flavor,normalize(flatten(top2cut[histname.replace("top2_", "")])),weight=weights.weight()[cut])
                        #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(top2cut[histname.replace('top2_','')]))))
                        # elif 'nw1_' in histname:
                        #     h.fill(ch,r,flavor,normalize(flatten(nw1cut[histname.replace("nw1_", "")])),weight=weights.weight()[cut])
                        #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(nw1cut[histname.replace('nw1_','')]))))
                        # elif 'nw2_' in histname:
                        #     h.fill(ch,r,flavor,normalize(flatten(nw2cut[histname.replace("nw2_", "")])),weight=weights.weight()[cut])
                        #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(nw2cut[histname.replace('nw2_','')]))))
                        # elif 'neu1_' in histname:
                        #     h.fill(ch,r,flavor,normalize(flatten(neu1cut[histname.replace("neu1_", "")])),weight=weights.weight()[cut])
                        #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(neu1cut[histname.replace('neu1_','')]))))
                        # elif 'neu2_' in histname:
                        #     h.fill(ch,r,flavor,normalize(flatten(neu2cut[histname.replace("neu2_", "")])),weight=weights.weight()[cut])
                        #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(neu2cut[histname.replace('neu2_','')]))))

                    
                    output["nj"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        n=normalize(nseljet, cut),
                        weight=weights.weight()[cut],
                    )
                    output["nele"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        n=normalize(naele - nele, cut),
                        weight=weights.weight()[cut],
                    )
                    output["nmu"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        n=normalize(namu - nmu, cut),
                        weight=weights.weight()[cut],
                    )
                    output["njmet"].fill(                    
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        n=normalize(ak.sum((met[cut].delta_phi(sel_jetflav[cut]) < 0.5), axis=1)),
                        weight=weights.weight()[cut],
                    )
                    output["METTkMETdphi"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        phi=flatten(met[cut].delta_phi(tkmet[cut])),
                        weight=weights.weight()[cut],
                    )
                    output["mT1"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        mt=flatten(mT(lep1cut, met[cut])),
                        weight=weights.weight()[cut],
                    )
                    output["mT2"].fill(        
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        mt=flatten(mT(lep2cut, met[cut])),
                        weight=weights.weight()[cut],
                    )
                    output["mTh"].fill(                    
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        mt=flatten(mT(llcut, met[cut])),
                        weight=weights.weight()[cut],
                    )
                    
                    
                    
                    if "SR" in r:
                        for syst in ['nowei','puweight','puweightUp','puweightDown']:
                            if syst=='nowei':weight = weights.partial_weight(exclude=['puweight'])[cut]
                            elif syst=='puweight':weight = weights.weight()[cut]
                            else :weight = weights.weight(modifier=syst)[cut]
                            
                            output["npvs"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            npvs=flatten(events[cut].PV.npvs),
                            syst=syst,
                            weight=weight,
                        )
                        output["nsv"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            nsv=flatten(ak.count(events[cut].SV.ntracks, axis=1)),
                            weight=weights.weight()[cut],
                        )
                        output["u_par"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            uparper=flatten(utv_p * np.cos(utv.delta_phi(lep1cut + lep2cut))),
                            weight=weights.weight()[cut],
                        )
                        output["u_per"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            uparper=flatten(utv_p * np.sin(utv.delta_phi(lep1cut + lep2cut))),
                            weight=weights.weight()[cut],
                        )

                        output["TkMET_pt"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            pt=flatten(tkmet[cut].pt),
                            weight=weights.weight()[cut],
                        )
                        output["TkMET_phi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(tkmet[cut].pt),
                            weight=weights.weight()[cut],
                        )
                        output["MET_pt"].fill(                        
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            pt=flatten(met[cut].pt),
                            weight=weights.weight()[cut],
                        )
                        output["MET_phi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(met[cut].phi),
                            weight=weights.weight()[cut],
                        )
                        output["PuppiMET_pt"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            pt=flatten(events[cut].PuppiMET.pt),
                            weight=weights.weight()[cut],
                        )
                        output["PuppiMET_phi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(events[cut].PuppiMET.phi),
                            weight=weights.weight()[cut],
                        )
                        output["MET_proj"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            pt=flatten(
                                np.where(
                                    dphilmet(lep1cut, lep2cut, met[cut]) < np.pi / 2.0,
                                    np.sin(dphilmet(lep1cut, lep2cut, met[cut]))
                                    * met[cut].pt,
                                    met[cut].pt,
                                )
                            ),
                            weight=weights.weight()[cut],
                        )
                        output["TkMET_proj"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            pt=flatten(
                                np.where(
                                    dphilmet(lep1cut, lep2cut, tkmet[cut]) < np.pi / 2.0,
                                    np.sin(dphilmet(lep1cut, lep2cut, tkmet[cut]))
                                    * tkmet[cut].pt,
                                    tkmet[cut].pt,
                                )
                            ),
                            weight=weights.weight()[cut],
                        )
                        output["minMET_proj"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            pt=flatten(
                                np.where(
                                    np.where(
                                        dphilmet(lep1cut, lep2cut, met[cut]) < np.pi / 2.0,
                                        np.sin(dphilmet(lep1cut, lep2cut, met[cut]))
                                        * met[cut].pt,
                                        met[cut].pt,
                                    )
                                    > np.where(
                                        dphilmet(lep1cut, lep2cut, tkmet[cut])
                                        < np.pi / 2.0,
                                        np.sin(dphilmet(lep1cut, lep2cut, tkmet[cut]))
                                        * tkmet[cut].pt,
                                        tkmet[cut].pt,
                                    ),
                                    np.where(
                                        dphilmet(lep1cut, lep2cut, met[cut]) < np.pi / 2.0,
                                        np.sin(dphilmet(lep1cut, lep2cut, met[cut]))
                                        * met[cut].pt,
                                        met[cut].pt,
                                    ),
                                    np.where(
                                        dphilmet(lep1cut, lep2cut, tkmet[cut])
                                        < np.pi / 2.0,
                                        np.sin(dphilmet(lep1cut, lep2cut, tkmet[cut]))
                                        * tkmet[cut].pt,
                                        tkmet[cut].pt,
                                    ),
                                )
                            ),
                            weight=weights.weight()[cut],
                        )
                        output["MET_ptdivet"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            ratio=flatten(met[cut].pt / np.sqrt(met[cut].energy)),
                            weight=weights.weight()[cut],
                        )
                        output["h_pt"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            pt=flatten(hcut.pt),
                            weight=weights.weight()[cut],
                        )
                        output["l1l2_dr"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            dr=flatten(make_p4(lep1cut).delta_r(make_p4(lep2cut))),
                            weight=weights.weight()[cut],
                        )
                        output["l1met_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(lep1cut.delta_phi(met[cut])),
                            weight=weights.weight()[cut],
                        )
                        output["l2met_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(lep2cut.delta_phi(met[cut])),
                            weight=weights.weight()[cut],
                        )
                        output["llmet_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(met[cut].delta_phi((llcut))),
                            weight=weights.weight()[cut],
                        )
                        output["l1c_dr"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            dr=ak.fill_none(
                                make_p4(lep1cut).delta_r(make_p4(sel_cjet_flav)), np.nan
                            ),
                            weight=weights.weight()[cut],
                        )
                        output["cmet_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=ak.fill_none(met[cut].delta_phi(sel_cjet_flav), np.nan),
                            weight=weights.weight()[cut],
                        )
                        output["l2c_dr"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            dr=ak.fill_none(
                                make_p4(lep2cut).delta_r(make_p4(sel_cjet_flav)), np.nan
                            ),
                            weight=weights.weight()[cut],
                        )
                        output["l1W1_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(lep1cut.delta_phi((w1cut))),
                            weight=weights.weight()[cut],
                        )
                        output["l2W1_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(lep2cut.delta_phi((w1cut))),
                            weight=weights.weight()[cut],
                        )
                        output["metW1_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(met[cut].delta_phi((w1cut))),
                            weight=weights.weight()[cut],
                        )
                        output["cW1_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=ak.fill_none((w1cut).delta_phi(sel_cjet_flav), np.nan),
                            weight=weights.weight()[cut],
                        )
                        output["l1W2_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(lep1cut.delta_phi((w2cut))),
                            weight=weights.weight()[cut],
                        )
                        output["l2W2_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(lep2cut.delta_phi((w2cut))),
                            weight=weights.weight()[cut],
                        )
                        output["metW2_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(met[cut].delta_phi((w2cut))),
                            weight=weights.weight()[cut],
                        )
                        output["cW2_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=ak.fill_none((w2cut).delta_phi(sel_cjet_flav), np.nan),
                            weight=weights.weight()[cut],
                        )
                        output["W1W2_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten((w1cut).delta_phi((w2cut))),
                            weight=weights.weight()[cut],
                        )
                        output["l1h_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten((lep1cut).delta_phi((hcut))),
                            weight=weights.weight()[cut],
                        )
                        output["l2h_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten((lep2cut).delta_phi((hcut))),
                            weight=weights.weight()[cut],
                        )
                        output["meth_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten((met[cut]).delta_phi((hcut))),
                            weight=weights.weight()[cut],
                        )
                        output["ch_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=ak.fill_none(hcut.delta_phi(sel_cjet_flav), np.nan),
                            weight=weights.weight()[cut],
                        )
                        output["W1h_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten((w1cut).delta_phi((hcut))),
                            weight=weights.weight()[cut],
                        )
                        output["W2h_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten((w2cut).delta_phi((hcut))),
                            weight=weights.weight()[cut],
                        )
                        output["lll1_dr"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            dr=flatten(make_p4(lep1cut).delta_r(make_p4(llcut))),
                            weight=weights.weight()[cut],
                        )
                        output["lll2_dr"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            dr=flatten(make_p4(lep2cut).delta_r(make_p4(llcut))),
                            weight=weights.weight()[cut],
                        )
                        output["llc_dr"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            dr=ak.fill_none(
                                make_p4(llcut).delta_r(make_p4(sel_cjet_flav)), np.nan
                            ),
                            weight=weights.weight()[cut],
                        )
                        output["llW1_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(w1cut.delta_phi((llcut))),
                            weight=weights.weight()[cut],
                        )
                        output["llW2_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(w2cut.delta_phi((llcut))),
                            weight=weights.weight()[cut],
                        )
                        output["llh_dphi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=flatten(hcut.delta_phi((llcut))),
                            weight=weights.weight()[cut],
                        )
                        
                    if self._export_array:
                        output_array[shift_name]["weight"]=weights.weight()[cut]
                        # output[arrayname][dataset]["weight"] += processor.column_accumulator(
                            # ak.to_numpy(normalize(weights.weight()[cut]))
                        # )
                        
                    # print(ak.type(output_array),output_array )     
                        
                    #     output[arrayname][dataset]["puwei"] += processor.column_accumulator(
                    #         ak.to_numpy(
                    #             normalize(weights.partial_weight(include=["puweight"])[cut])
                    #         )
                    #     )
                    #     output[arrayname][dataset]["l1wei"] += processor.column_accumulator(
                    #         ak.to_numpy(
                    #             normalize(
                    #                 weights.partial_weight(include=["L1prefireweight"])[cut]
                    #             )
                    #         )
                    #     )
                    #     output[arrayname][dataset]["event"] += processor.column_accumulator(
                    #         ak.to_numpy(normalize(events[cut].event))
                    #     )
                    #     output[arrayname][dataset]["run"] += processor.column_accumulator(
                    #         ak.to_numpy(normalize(events[cut].run))
                    #     )
                    #     output[arrayname][dataset]["lumi"] += processor.column_accumulator(
                    #         ak.to_numpy(normalize(events[cut].luminosityBlock))
                    #     )
                    #     output[arrayname][dataset]["lepflav"] += processor.column_accumulator(
                    #         np.full_like(ak.to_numpy(normalize(nseljet)), ch, dtype="<U6")
                    #     )
                    #     output[arrayname][dataset]["jetflav"] += processor.column_accumulator(
                    #         ak.to_numpy(normalize(flavor))
                    #     )
                    #     output[arrayname][dataset]["region"] += processor.column_accumulator(
                    #         np.full_like(ak.to_numpy(normalize(nseljet)), r, dtype="<U6")
                    #     )
                    #     output[arrayname][dataset]["nj"] += processor.column_accumulator(
                    #         ak.to_numpy(normalize(nseljet))
                    #     )
                    #     output[arrayname][dataset]["nele"] += processor.column_accumulator(
                    #         ak.to_numpy(normalize(naele - nele, cut))
                    #     )
                    #     output[arrayname][dataset]["nmu"] += processor.column_accumulator(
                    #         ak.to_numpy(normalize(namu - nmu, cut))
                    #     )
                    #     output[arrayname][dataset][
                    #         "METTkMETdphi"
                    #     ] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(met[cut].delta_phi(tkmet[cut])))
                    #     )
                    #     output[arrayname][dataset]["njmet"] += processor.column_accumulator(
                    #         ak.to_numpy(
                    #             normalize(
                    #                 ak.sum(
                    #                     (met[cut].delta_phi(sel_jetflav[cut]) < 0.5), axis=1
                    #                 )
                    #             )
                    #         )
                    #     )
                    #     output[arrayname][dataset]["mT1"] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(mT(lep1cut, met[cut])))
                    #     )
                    #     output[arrayname][dataset]["mT2"] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(mT(lep2cut, met[cut])))
                    #     )
                    #     output[arrayname][dataset]["mTh"] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(mT(llcut, met[cut])))
                    #     )
                    #     output[arrayname][dataset]["npvs"] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(events[cut].PV.npvs))
                    #     )

                    #     output[arrayname][dataset]["nsv"] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(ak.count(events[cut].SV.ntracks, axis=1)))
                    #     )

                    #     # output[arrayname][dataset]["u_par"] += processor.column_accumulator(
                    #     #     ak.to_numpy(
                    #     #         flatten(utv_p * np.cos(utv.delta_phi(lep1cut + lep2cut)))
                    #     #     )
                    #     # )
                    #     # output[arrayname][dataset]["u_per"] += processor.column_accumulator(
                    #     #     ak.to_numpy(
                    #     #         flatten(utv_p * np.sin(utv.delta_phi(lep1cut + lep2cut)))
                    #     #     )
                    #     # )
                    #     output[arrayname][dataset][
                    #         "tkMET_pt"
                    #     ] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(tkmet[cut].pt))
                    #     )
                    #     output[arrayname][dataset][
                    #         "PuppiMET_pt"
                    #     ] += processor.column_accumulator(
                    #         ak.to_numpy(flatten(events[cut].PuppiMET.pt))
                    #     )
                    #     output[arrayname][dataset][
                    #         "MET_proj"
                    #     ] += processor.column_accumulator(
                    #         ak.to_numpy(
                    #             flatten(
                    #                 np.where(
                    #                     dphilmet(lep1cut, lep2cut, met[cut]) < np.pi / 2.0,
                    #                     np.sin(dphilmet(lep1cut, lep2cut, met[cut]))
                    #                     * met[cut].pt,
                    #                     met[cut].pt,
                    #                 )
                    #             )
                    #         )
                    #     )
                        # output[arrayname][dataset][
                        #     "TkMET_proj"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(
                        #         flatten(
                        #             np.where(
                        #                 dphilmet(lep1cut, lep2cut, tkmet[cut])
                        #                 < np.pi / 2.0,
                        #                 np.sin(dphilmet(lep1cut, lep2cut, tkmet[cut]))
                        #                 * tkmet[cut].pt,
                        #                 tkmet[cut].pt,
                        #             )
                        #         )
                        #     )
                        # )
                        # output[arrayname][dataset][
                        #     "minMET_proj"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(
                        #         flatten(
                        #             np.where(
                        #                 np.where(
                        #                     dphilmet(lep1cut, lep2cut, met[cut])
                        #                     < np.pi / 2.0,
                        #                     np.sin(dphilmet(lep1cut, lep2cut, met[cut]))
                        #                     * met[cut].pt,
                        #                     met[cut].pt,
                        #                 )
                        #                 > np.where(
                        #                     dphilmet(lep1cut, lep2cut, tkmet[cut])
                        #                     < np.pi / 2.0,
                        #                     np.sin(dphilmet(lep1cut, lep2cut, tkmet[cut]))
                        #                     * tkmet[cut].pt,
                        #                     tkmet[cut].pt,
                        #                 ),
                        #                 np.where(
                        #                     dphilmet(lep1cut, lep2cut, met[cut])
                        #                     < np.pi / 2.0,
                        #                     np.sin(dphilmet(lep1cut, lep2cut, met[cut]))
                        #                     * met[cut].pt,
                        #                     met[cut].pt,
                        #                 ),
                        #                 np.where(
                        #                     dphilmet(lep1cut, lep2cut, tkmet[cut])
                        #                     < np.pi / 2.0,
                        #                     np.sin(dphilmet(lep1cut, lep2cut, tkmet[cut]))
                        #                     * tkmet[cut].pt,
                        #                     tkmet[cut].pt,
                        #                 ),
                        #             )
                        #         )
                        #     )
                        # )
                        # output[arrayname][dataset][
                        #     "MET_ptdivet"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(met[cut].pt / np.sqrt(met[cut].energy)))
                        # )
                        # output[arrayname][dataset]["h_pt"] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(hcut.pt))
                        # )
                        # output[arrayname][dataset]["MET_pt"] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(met[cut].pt))
                        # )
                        # output[arrayname][dataset]["MET_phi"] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(met[cut].pt))
                        # )
                        # output[arrayname][dataset]["l1l2_dr"] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(make_p4(lep1cut).delta_r(make_p4(lep2cut))))
                        # )
                        # output[arrayname][dataset][
                        #     "l1met_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(lep1cut.delta_phi(met[cut])))
                        # )
                        # output[arrayname][dataset][
                        #     "l2met_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(lep2cut.delta_phi(met[cut])))
                        # )
                        # output[arrayname][dataset][
                        #     "llmet_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(llcut.delta_phi(met[cut])))
                        # )
                        # output[arrayname][dataset]["l1c_dr"] += processor.column_accumulator(
                        #     ak.to_numpy(
                        #         normalize(make_p4(lep1cut).delta_r(make_p4(sel_cjet_flav)))
                        #     )
                        # )
                        # output[arrayname][dataset][
                        #     "cmet_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(normalize(met[cut].delta_phi(sel_cjet_flav)))
                        # )
                        # output[arrayname][dataset]["l2c_dr"] += processor.column_accumulator(
                        #     ak.to_numpy(
                        #         normalize(make_p4(lep2cut).delta_r(make_p4(sel_cjet_flav)))
                        #     )
                        # )
                        # output[arrayname][dataset][
                        #     "l2W1_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(lep2cut.delta_phi((w1cut))))
                        # )
                        # output[arrayname][dataset][
                        #     "metW1_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(met[cut].delta_phi((w1cut))))
                        # )
                        # output[arrayname][dataset][
                        #     "cW1_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(normalize((w1cut).delta_phi(sel_cjet_flav)))
                        # )
                        # output[arrayname][dataset][
                        #     "l1W2_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(lep1cut.delta_phi((w2cut))))
                        # )
                        # output[arrayname][dataset]["llc_dr"] += processor.column_accumulator(
                        #     ak.to_numpy(
                        #         normalize(make_p4(llcut).delta_r(make_p4(sel_cjet_flav)))
                        #     )
                        # )
                        # output[arrayname][dataset][
                        #     "llW1_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(w1cut.delta_phi((llcut))))
                        # )
                        # output[arrayname][dataset][
                        #     "llW2_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(w2cut.delta_phi((llcut))))
                        # )
                        # output[arrayname][dataset][
                        #     "llh_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(hcut.delta_phi((llcut))))
                        # )
                        # output[arrayname][dataset][
                        #     "l2W2_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(lep2cut.delta_phi((w2cut))))
                        # )
                        # output[arrayname][dataset][
                        #     "metW2_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(met[cut].delta_phi((w2cut))))
                        # )
                        # output[arrayname][dataset][
                        #     "cW2_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(normalize((w2cut).delta_phi(sel_cjet_flav)))
                        # )
                        # output[arrayname][dataset][
                        #     "l1h_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten((lep1cut).delta_phi((hcut))))
                        # )
                        # output[arrayname][dataset][
                        #     "meth_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten((met[cut]).delta_phi((hcut))))
                        # )
                        # output[arrayname][dataset]["ch_dphi"] += processor.column_accumulator(
                        #     ak.to_numpy(normalize(hcut.delta_phi(sel_cjet_flav)))
                        # )
                        # output[arrayname][dataset][
                        #     "W1h_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten((w1cut).delta_phi((hcut))))
                        # )
                        # output[arrayname][dataset][
                        #     "W2h_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten((w2cut).delta_phi((hcut))))
                        # )
                        # output[arrayname][dataset]["lll1_dr"] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(make_p4(lep1cut).delta_r(make_p4(llcut))))
                        # )
                        # output[arrayname][dataset]["lll2_dr"] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(make_p4(lep2cut).delta_r(make_p4(llcut))))
                        # )
                        # output[arrayname][dataset][
                        #     "l1W1_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten(lep1cut.delta_phi((w1cut))))
                        # )
                        # output[arrayname][dataset][
                        #     "W1W2_dphi"
                        # ] += processor.column_accumulator(
                            
                        #     ak.to_numpy(flatten((w1cut).delta_phi((w2cut))))
                        # )
                        # output[arrayname][dataset][
                        #     "l2h_dphi"
                        # ] += processor.column_accumulator(
                        #     ak.to_numpy(flatten((lep2cut).delta_phi((hcut))))
                        # )
                    
                
                
                for sys in systematics:
                    if not self.systematics["weights"] and sys!='nominal':continue

                    if sys in weights.variations:
                        weight = weights.weight(modifier=sys)[cut]
                    else:
                        weight = weights.weight()[cut]
                    
                    output['template_MET'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,pt=normalize(flatten(met[cut].pt)),weight=weight)
                    output['template_jetpt'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,pt=normalize(flatten(sel_cjet_flav.pt)),weight=weight)
                    output['template_ll_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,mass=normalize(flatten(llcut.mass)),weight=weight)
                    
                    
                    if "top_CR" in r :
                        
                        output['template_top1_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor[(ak.count(topjets[cut].pt,axis=1) > 1)],mass=normalize(flatten(top1cut.mass)),weight=weight[ak.count(topjets[cut].pt,axis=1) > 1])
                        output['template_top2_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor[(ak.count(topjets[cut].pt,axis=1) > 1)],mass=normalize(flatten(top2cut.mass)),weight=weight[ak.count(topjets[cut].pt,axis=1) > 1])
                        output['template_tt_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor[(ak.count(topjets[cut].pt,axis=1) > 1)],mass=normalize(flatten(ttbarcut.mass)),weight=weight[ak.count(topjets[cut].pt,axis=1) > 1])
                        output['template_llbb_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor[(ak.count(topjets[cut].pt,axis=1) > 1)],mass=normalize(flatten(llbbcut.mass)),weight=weight[ak.count(topjets[cut].pt,axis=1) > 1])
                        output['template_HT_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,mass=flatten(ak.sum(events[cut].Jet.pt,axis=1)),weight=weight)

                        output['template_mTh'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,mt=normalize(flatten(mT(llcut, met[cut]))),weight=weight)
                        output['template_mT1'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,mt=normalize(flatten(mT(lep1cut, met[cut]))),weight=weight)
                    if "SR" in r:
                        if r == "emu":
                            x = np.vstack(
                            [normalize(flatten(llcut.mass)),normalize(flatten(met[cut].pt)),normalize(flatten(sel_cjet_flav.pt)),normalize(flatten(lep1cut.pt)),normalize(flatten(lep2cut.pt)),normalize(flatten(llcut.pt)),normalize(flatten(mT(lep1cut, met[cut]))),normalize(flatten(mT(lep2cut, met[cut]))),normalize(flatten(sel_cjet_flav.btagDeepFlavCvL)),normalize(flatten(sel_cjet_flav.btagDeepFlavCvB)),normalize(flatten(lep1cut.delta_r(llcut))),normalize(flatten(lep2cut.delta_r(llcut))),normalize(flatten(llcut.delta_r(sel_cjet_flav))),normalize(flatten(lep1cut.delta_phi(met[cut]))),normalize(flatten(lep2cut.delta_phi(met[cut]))),normalize(flatten(sel_cjet_flav.delta_phi(w1cut)))]
                            ).T
                            xgb_model=self.xgb_model_emu
                        else:
                            x = np.vstack(
                            [normalize(flatten(llcut.pt)),normalize(flatten(lep1cut.delta_r(llcut))),normalize(flatten(lep2cut.delta_r(llcut))),normalize(flatten(llcut.delta_r(sel_cjet_flav))),normalize(flatten(lep1cut.pt)),normalize(flatten(lep2cut.pt)),normalize(flatten(llcut.mass)),normalize(flatten(met[cut].pt)),normalize(flatten(sel_cjet_flav.pt)),normalize(flatten(lep1cut.delta_phi(met[cut]))),normalize(flatten(lep2cut.delta_phi(met[cut]))),normalize(flatten(sel_cjet_flav.delta_phi(w1cut))),normalize(flatten(mT(lep1cut, met[cut]))),normalize(flatten(mT(lep2cut, met[cut]))),normalize(flatten(sel_cjet_flav.btagDeepFlavCvL)),normalize(flatten(sel_cjet_flav.btagDeepFlavCvB))]
                            ).T
                            xgb_model=self.xgb_model_ll
                            
                        dmatrix = xgb.DMatrix(x)
                        bdtscore=BDTreader(dmatrix,xgb_model)
                        output['template_BDT'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,disc=normalize(flatten(bdtscore)),weight=weight)
                        del xgb_model, dmatrix,bdtscore
        
        print("fill : ",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
        del ll_cand, sel_cjet_flav, met, tkmet,cut,ll_cands,lep1cut,lep2cut,llcut,hcut,w1cut,w2cut,ttbarcut,naele,namu,sr_cut,top_cr2_cut,dy_cr2_cut,cvbcutll,cvbcutem,cvlcutem,cvlcutll, selection, weights,mask_jet,mask_lep,global_cut,llmass_cut
        if not isRealData:del jetsf,jetsf_dn,jetsf_up
        if self.systematics["weights"] : del weight
        # del  mask_jet,mask_lep,selection, sr_cut,global_cut,cvbcutll,cvbcutem,cvlcutem,cvlcutll,dy_cr2_cut,top_cr2_cut,llmass_cut
        gc.collect()
        print("clean : ",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")

        return {dataset:output}

    def postprocess(self, accumulator):
        return accumulator
