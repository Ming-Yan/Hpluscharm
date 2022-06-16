import csv
from curses import meta
from dataclasses import dataclass
import gzip
import pickle, os, sys, mplhep as hep, numpy as np
from select import select

from matplotlib.pyplot import jet

import coffea
from coffea import hist, processor
from coffea.nanoevents.methods import vector
import awkward as ak
from utils.correction import jec, muSFs, eleSFs, init_corr, puwei, nJet_corr
from coffea.lumi_tools import LumiMask

from coffea.analysis_tools import Weights
from functools import partial
from helpers.util import reduce_and, reduce_or, nano_mask_or, get_ht, normalize, make_p4


def mT(obj1, obj2):
    return np.sqrt(2.0 * obj1.pt * obj2.pt * (1.0 - np.cos(obj1.phi - obj2.phi)))


def flatten(ar):  # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)


def normalize(val, cut):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, year="2017", version="test"):
        self._year = year
        self._version = version
        self._mu1hlt = {
            "2016": ["IsoTkMu24"],
            "2017": ["IsoMu27"],
            "2018": ["IsoMu24"],
        }
        self._mu2hlt = {
            "2016": [
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",  # runH
                "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",  # runH
            ],
            "2017": [
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
            ],
            "2018": [
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            ],
        }
        self._e1hlt = {
            "2016": ["Ele27_WPTight_Gsf", "Ele25_eta2p1_WPTight_Gsf"],
            "2017": [
                "Ele35_WPTight_Gsf",
            ],
            "2018": [
                "Ele32_WPTight_Gsf",
            ],
        }
        self._e2hlt = {
            "2016": [
                "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            ],
            "2017": [
                "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            ],
            "2018": [
                "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            ],
        }
        self._met_filters = {
            "2016": {
                "data": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "eeBadScFilter",
                ],
                "mc": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "eeBadScFilter",
                ],
            },
            "2017": {
                "data": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "hfNoisyHitsFilter",
                    "eeBadScFilter",
                    "ecalBadCalibFilter",
                ],
                "mc": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "hfNoisyHitsFilter",
                    "eeBadScFilter",
                    "ecalBadCalibFilter",
                ],
            },
            "2018": {
                "data": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "hfNoisyHitsFilter",
                    "eeBadScFilter",
                    "ecalBadCalibFilter",
                ],
                "mc": [
                    "goodVertices",
                    "globalSuperTightHalo2016Filter",
                    "HBHENoiseFilter",
                    "HBHENoiseIsoFilter",
                    "EcalDeadCellTriggerPrimitiveFilter",
                    "BadPFMuonFilter",
                    "BadPFMuonDzFilter",
                    "hfNoisyHitsFilter",
                    "eeBadScFilter",
                    "ecalBadCalibFilter",
                ],
            },
        }
        self._lumiMasks = {
            "2016": LumiMask(
                "data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
            ),
            "2017": LumiMask(
                "data/Lumimask/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
            ),
            "2018": LumiMask(
                "data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
            ),
        }
        self._corr = init_corr(self._year)
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour", [0, 1, 4, 5, 6])
        lepflav_axis = hist.Cat("lepflav", ["ee", "mumu", "emu"])
        # Events
        njet_axis = hist.Bin("nj", r"N jets", [0, 1, 2, 3, 4, 5, 6, 7])
        nalep_axis = hist.Bin("nalep", r"N jets", [0, 1, 2, 3])
        nbjet_axis = hist.Bin("nbj", r"N b-jets", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ncjet_axis = hist.Bin("nbj", r"N b-jets", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # kinematic variables
        pt_axis = hist.Bin("pt", r" $p_{T}$ [GeV]", 50, 0, 300)
        eta_axis = hist.Bin("eta", r" $\eta$", 25, -2.5, 2.5)
        phi_axis = hist.Bin("phi", r" $\phi$", 30, -3, 3)

        mass_axis = hist.Bin("mass", r" $m$ [GeV]", 40, 0, 400)
        llmass_axis = hist.Bin("mass", r" $m$ [GeV]", 40, 60, 120)
        iso_axis = hist.Bin("pfRelIso03_all", r"Rel. Iso", 40, 0, 4)
        dxy_axis = hist.Bin("dxy", r"d_{xy}", 20, -0.2, 0.2)
        dz_axis = hist.Bin("dz", r"d_{z}", 20, 0, 1)
        dr_axis = hist.Bin("dr", "$\Delta$R", 20, 0, 4)
        costheta_axis = hist.Bin("costheta", "cos$\theta$", 20, -1, 1)
        # MET vars

        # axis.StrCategory([], name='region', growth=True),
        disc_list = [
            "btagDeepCvL",
            "btagDeepCvB",
            "btagDeepFlavCvB",
            "btagDeepFlavCvL",
        ]  # ,'particleNetAK4_CvL','particleNetAK4_CvB']
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d, 50, 0, 1))
        _hist_event_dict = {
            "nj": hist.Hist("Counts", dataset_axis, lepflav_axis, njet_axis),
            "njnosel": hist.Hist("Counts", dataset_axis, lepflav_axis, njet_axis),
            "nele": hist.Hist("Counts", dataset_axis, lepflav_axis, nalep_axis),
            "nmu": hist.Hist("Counts", dataset_axis, lepflav_axis, nalep_axis),
            "hc_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "zs_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "jjc_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "cj_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "l1l2_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "lc_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "l1j1_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "l1j2_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "l2j1_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "l2j2_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "j1j2_dr": hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            "costheta_ll": hist.Hist(
                "Counts", dataset_axis, lepflav_axis, costheta_axis
            ),
            "costheta_qq": hist.Hist(
                "Counts", dataset_axis, lepflav_axis, costheta_axis
            ),
            "costheta_pz": hist.Hist(
                "Counts", dataset_axis, lepflav_axis, costheta_axis
            ),
            "phi_zz": hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
            "phi_lq": hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
        }
        objects = ["cjet", "lep1", "lep2", "jet1", "jet2", "jj", "ll", "higgs"]

        for i in objects:
            if "jet" in i:
                _hist_event_dict["%s_pt" % (i)] = hist.Hist(
                    "Counts", dataset_axis, lepflav_axis, flav_axis, pt_axis
                )
                _hist_event_dict["%s_eta" % (i)] = hist.Hist(
                    "Counts", dataset_axis, lepflav_axis, flav_axis, eta_axis
                )
                _hist_event_dict["%s_phi" % (i)] = hist.Hist(
                    "Counts", dataset_axis, lepflav_axis, flav_axis, phi_axis
                )
                _hist_event_dict["%s_mass" % (i)] = hist.Hist(
                    "Counts", dataset_axis, lepflav_axis, flav_axis, mass_axis
                )
            else:
                _hist_event_dict["%s_pt" % (i)] = hist.Hist(
                    "Counts", dataset_axis, lepflav_axis, pt_axis
                )
                _hist_event_dict["%s_eta" % (i)] = hist.Hist(
                    "Counts", dataset_axis, lepflav_axis, eta_axis
                )
                _hist_event_dict["%s_phi" % (i)] = hist.Hist(
                    "Counts", dataset_axis, lepflav_axis, phi_axis
                )
                if i == "ll":
                    _hist_event_dict["%s_mass" % (i)] = hist.Hist(
                        "Counts", dataset_axis, lepflav_axis, llmass_axis
                    )
                else:
                    _hist_event_dict["%s_mass" % (i)] = hist.Hist(
                        "Counts", dataset_axis, lepflav_axis, mass_axis
                    )
                if "lep" in i:
                    _hist_event_dict["%s_pfRelIso03_all" % (i)] = hist.Hist(
                        "Counts", dataset_axis, lepflav_axis, iso_axis
                    )
                    _hist_event_dict["%s_dxy" % (i)] = hist.Hist(
                        "Counts", dataset_axis, lepflav_axis, dxy_axis
                    )
                    _hist_event_dict["%s_dz" % (i)] = hist.Hist(
                        "Counts", dataset_axis, lepflav_axis, dz_axis
                    )

        for disc, axis in zip(disc_list, btag_axes):
            _hist_event_dict["cjet_%s" % (disc)] = hist.Hist(
                "Counts", dataset_axis, lepflav_axis, flav_axis, axis
            )
            _hist_event_dict["jet1_%s" % (disc)] = hist.Hist(
                "Counts", dataset_axis, lepflav_axis, flav_axis, axis
            )
            _hist_event_dict["jet2_%s" % (disc)] = hist.Hist(
                "Counts", dataset_axis, lepflav_axis, flav_axis, axis
            )
        self.event_hists = list(_hist_event_dict.keys())

        self._accumulator = processor.dict_accumulator(
            {
                **_hist_event_dict,
                "cutflow": processor.defaultdict_accumulator(
                    # we don't use a lambda function to avoid pickle issues
                    partial(processor.defaultdict_accumulator, int)
                ),
            }
        )
        self._accumulator["sumw"] = processor.defaultdict_accumulator(float)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = processor.PackedSelection()
        if isRealData:
            output["sumw"][dataset] += len(events)
        else:
            output["sumw"][dataset] += ak.sum(events.genWeight / abs(events.genWeight))
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self._lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add("lumi", ak.to_numpy(req_lumi))
        del req_lumi
        weights = Weights(len(events), storeIndividual=True)
        if isRealData:
            weights.add("genweight", np.ones(len(events)))
        else:
            weights.add("genweight", events.genWeight / abs(events.genWeight))

            if self._year in ("2016", "2017"):
                weights.add(
                    "L1Prefiring",
                    events.L1PreFiringWeight.Nom,
                    events.L1PreFiringWeight.Up,
                    events.L1PreFiringWeight.Dn,
                )
            weights.add("puweight", puwei(events.Pileup.nPU, self._corr))

            if "DY" in dataset:
                genZ = events.GenPart[
                    (events.GenPart.hasFlags(["fromHardProcess"]) == True)
                    & (events.GenPart.hasFlags(["isHardProcess"]) == True)
                    & (events.GenPart.pdgId == 23)
                ]
                # weights.add('zptwei',ZpT_corr(flatten(genZ.pt,self._year))
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
        ##############
        if isRealData:
            output["cutflow"][dataset]["all"] += len(events)
        else:
            output["cutflow"][dataset]["all"] += ak.sum(
                events.genWeight / abs(events.genWeight)
            )
        trigger_e = np.zeros(len(events), dtype="bool")
        trigger_m = np.zeros(len(events), dtype="bool")
        trigger_ee = np.zeros(len(events), dtype="bool")
        trigger_mm = np.zeros(len(events), dtype="bool")
        trigger_ele = np.zeros(len(events), dtype="bool")
        trigger_mu = np.zeros(len(events), dtype="bool")

        for t in self._mu1hlt[self._year]:
            if t in events.HLT.fields:
                trigger_m = trigger_m | events.HLT[t]
        for t in self._mu2hlt[self._year]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]
        for t in self._e1hlt[self._year]:
            if t in events.HLT.fields:
                trigger_e = trigger_e | events.HLT[t]
        for t in self._e2hlt[self._year]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]

        if isRealData:
            if "DoubleEG" in dataset:
                trigger_ele = trigger_ee
                trigger_mu = np.zeros(len(events), dtype="bool")
            elif "SingleElectron" in dataset:
                trigger_ele = ~trigger_ee & trigger_e
                trigger_mu = np.zeros(len(events), dtype="bool")
            elif "DoubleMuon" in dataset:
                trigger_mu = trigger_mm
                trigger_ele = np.zeros(len(events), dtype="bool")
            elif "SingleMuon" in dataset:
                trigger_mu = ~trigger_mm & trigger_m
                trigger_ele = np.zeros(len(events), dtype="bool")

        else:
            trigger_mu = trigger_mm | trigger_m
            trigger_ele = trigger_ee | trigger_e
        selection.add("trigger_ee", ak.to_numpy(trigger_ele))
        selection.add("trigger_mumu", ak.to_numpy(trigger_mu))
        del trigger_ee, trigger_e, trigger_ele, trigger_m, trigger_mm, trigger_mu
        metfilter = np.ones(len(events), dtype="bool")
        for flag in self._met_filters[self._year]["data" if isRealData else "mc"]:
            metfilter &= np.array(events.Flag[flag])
        selection.add("metfilter", metfilter)
        del metfilter

        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1, ascending=False)]
        musel = (
            (event_mu.pt > 13)
            & (abs(event_mu.eta) < 2.4)
            & (event_mu.mvaId >= 3)
            & (event_mu.pfRelIso04_all < 0.15)
            & (abs(event_mu.dxy) < 0.05)
            & (abs(event_mu.dz) < 0.1)
        )
        event_mu["lep_flav"] = 13 * event_mu.charge

        event_mu = event_mu[musel]
        nmu = ak.sum(musel, axis=1)
        namu = ak.sum(
            (event_mu.pt > 10)
            & (abs(event_mu.eta) < 2.4)
            & (event_mu.pfRelIso04_all < 0.2),
            axis=1,
        )
        event_mu = ak.pad_none(event_mu, 2, axis=1)

        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[
            ak.argsort(events.Electron.pt, axis=1, ascending=False)
        ]
        event_e["lep_flav"] = 11 * event_e.charge
        elesel = (
            (event_e.pt > 13)
            & (abs(event_e.eta) < 2.5)
            & (event_e.mvaFall17V2Iso_WP90 == 1)
            & (abs(event_e.dxy) < 0.05)
            & (abs(event_e.dz) < 0.1)
        )
        naele = ak.sum(
            (event_e.pt > 12)
            & (abs(event_e.eta) < 2.5)
            & (event_e.pfRelIso03_all < 0.2),
            axis=1,
        )
        event_e = event_e[elesel]
        nele = ak.sum(elesel, axis=1)

        event_e = ak.pad_none(event_e, 2, axis=1)

        selection.add("lepsel", ak.to_numpy((nele >= 2) | (nmu >= 2)))
        corr_jet = jec(events, events.Jet, dataset, self._year, self._corr)
        ranked_deepJet = corr_jet.btagDeepFlavCvL
        ####TEST
        if self._version == "test":
            corr_jet[corr_jet.btagDeepFlavCvL == 0].btagDeepFlavCvL = 1e-10
            ranked_deepJet = corr_jet.btagDeepFlavCvB / corr_jet.btagDeepFlavCvL
        ######
        event_jet = corr_jet[ak.argsort(ranked_deepJet, axis=1, ascending=False)]
        # event_jet = corr_jet[ak.argsort(corr_jet.btagDeepFlavCvL, axis=1,ascending=False)]
        jet_sel = (
            (event_jet.pt > 20)
            & (abs(event_jet.eta) <= 2.4)
            & ((event_jet.puId > 6) | (event_jet.pt > 50))
            & (event_jet.jetId > 5)
        )  # &(ak.all(event_jet.metric_table(event_e)>0.4,axis=2))&(ak.all(event_jet.metric_table(event_mu)>0.4,axis=2))
        print("dr", (event_jet.metric_table(event_e).tolist()))
        print(ak.all(event_jet.metric_table(event_e) > 0.4, axis=2).tolist())
        event_jet = event_jet[jet_sel]
        njet = ak.sum(jet_sel, axis=1)

        event_jet = ak.pad_none(event_jet, 3, axis=1)
        selection.add("jetsel", ak.to_numpy(njet >= 3))
        if "DY" in dataset:
            weights.add("njwei", nJet_corr(njet, self._corr))
        cjet = event_jet[:, 0]

        rest_jet = event_jet[:, 1:]

        good_leptons = ak.concatenate([event_e, event_mu], axis=1)
        pair_2lep = ak.combinations(
            good_leptons,
            n=2,
            replacement=False,
            axis=-1,
            fields=["lep1", "lep2"],
        )
        ll_cand = ak.zip(
            {
                # "p4": pair_2lep.lep1+pair_2lep.lep2,
                "lep1": pair_2lep.lep1,
                "lep2": pair_2lep.lep2,
                "pt": (pair_2lep.lep1 + pair_2lep.lep2).pt,
                "eta": (pair_2lep.lep1 + pair_2lep.lep2).eta,
                "phi": (pair_2lep.lep1 + pair_2lep.lep2).phi,
                "mass": (pair_2lep.lep1 + pair_2lep.lep2).mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        if ak.count(ll_cand.mass) > 0:
            ll_cand = ll_cand[ak.argsort(abs(ll_cand.mass - 91.18), axis=1)]

        # good_jets = ak.with_name(event_jet,"PtEtaPhiMCandidate")
        pair_2j = ak.combinations(
            rest_jet, n=2, replacement=False, fields=["jet1", "jet2"]
        )
        jj_cand = ak.zip(
            {
                # "p4": pair_2j.jet1+pair_2j.jet2,
                "jet1": pair_2j.jet1,
                "jet2": pair_2j.jet2,
                "pt": (pair_2j.jet1 + pair_2j.jet2).pt,
                "eta": (pair_2j.jet1 + pair_2j.jet2).eta,
                "phi": (pair_2j.jet1 + pair_2j.jet2).phi,
                "mass": (pair_2j.jet1 + pair_2j.jet2).mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        if ak.count(jj_cand.mass) > 0:
            jj_cand = jj_cand[ak.argsort(abs(jj_cand.mass - 91.18), axis=1)]
        higgs_cands, (ll_cands, jj_cands) = ll_cand.metric_table(
            jj_cand,
            axis=1,
            metric=lambda jj_cand, ll_cand: (jj_cand + ll_cand),
            return_combinations=True,
        )
        # print(ak.type(ll_cands.ll_cand))
        #
        higgs_cand = ak.zip(
            {
                "ll_cands": ll_cands,
                "jj_cands": jj_cands,
                "pt": higgs_cands.pt,
                "eta": higgs_cands.eta,
                "phi": higgs_cands.phi,
                "mass": higgs_cands.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        higgs_cand = ak.pad_none(higgs_cand, 1, axis=0)

        req_global = (
            (higgs_cand.ll_cands.lep1.pt > 25)
            & (higgs_cand.ll_cands.lep1.charge + higgs_cand.ll_cands.lep2.charge == 0)
            & (higgs_cand.ll_cands.mass < 120)
            & (higgs_cand.ll_cands.mass > 60)
        )

        req_dr = (
            (
                make_p4(higgs_cand.ll_cands.lep1).delta_r(
                    make_p4(higgs_cand.jj_cands.jet1)
                )
                > 0.4
            )
            & (
                make_p4(higgs_cand.ll_cands.lep1).delta_r(
                    make_p4(higgs_cand.jj_cands.jet2)
                )
                > 0.4
            )
            & (
                make_p4(higgs_cand.ll_cands.lep2).delta_r(
                    make_p4(higgs_cand.jj_cands.jet2)
                )
                > 0.4
            )
            & (
                make_p4(higgs_cand.ll_cands.lep2).delta_r(
                    make_p4(higgs_cand.jj_cands.jet1)
                )
                > 0.4
            )
            & (
                make_p4(higgs_cand.jj_cands.jet1).delta_r(
                    make_p4(higgs_cand.jj_cands.jet2)
                )
                > 0.4
            )
            & (
                make_p4(higgs_cand.ll_cands.lep1).delta_r(
                    make_p4(higgs_cand.ll_cands.lep2)
                )
                > 0.4
            )
        )

        req_zllmass = abs(higgs_cand.ll_cands.mass - 91.0) < 15
        req_zqqmass = higgs_cand.jj_cands.mass < 116
        req_hmass = higgs_cand.mass < 250

        mask2e = (
            req_hmass
            & req_zllmass
            & req_zqqmass
            & req_global
            & (ak.num(event_e) == 2)
            & (event_e[:, 0].pt > 25)
            & (event_e[:, 1].pt > 13)
        )
        mask2mu = (
            req_hmass
            & req_zllmass
            & req_zqqmass
            & req_global
            & (ak.num(event_mu) == 2)
            & (event_mu[:, 0].pt > 25)
            & (event_mu[:, 1].pt > 13)
        )
        mask2lep = [ak.any(tup) for tup in zip(mask2mu, mask2e)]
        # mask2jet = req_hmass & req_zllmass&req_global&req_zqqmass
        # event_jet = ak.mask(event_jet,mask2jet)
        good_leptons = ak.mask(good_leptons, mask2lep)

        # ###########

        seljet = (
            (cjet.pt > 20)
            & (abs(cjet.eta) <= 2.4)
            & ((cjet.puId > 6) | (cjet.pt > 50))
            & (cjet.jetId > 5)
            & (
                ak.all(
                    ak.all(
                        (cjet.delta_r(higgs_cand.ll_cands.lep1) > 0.4)
                        & (cjet.delta_r(higgs_cand.ll_cands.lep2) > 0.4)
                        & (cjet.delta_r(higgs_cand.jj_cands.jet1) > 0.4)
                        & (cjet.delta_r(higgs_cand.jj_cands.jet2) > 0.4),
                        axis=-1,
                    ),
                    axis=-1,
                )
            )
        )
        sel_cjet = ak.mask(cjet, seljet)
        # if(ak.count(sel_cjet.pt)>0):sel_cjet=

        output["cutflow"][dataset]["global selection"] += ak.sum(
            ak.any(ak.any(req_global, axis=-1), axis=-1)
        )
        output["cutflow"][dataset]["dilepton mass"] += ak.sum(
            ak.any(ak.any(req_zllmass & req_global, axis=-1), axis=-1)
        )
        output["cutflow"][dataset]["dijet mass"] += ak.sum(
            ak.any(ak.any(req_zllmass & req_global & req_zqqmass, axis=-1), axis=-1)
        )
        output["cutflow"][dataset]["higgs mass"] += ak.sum(
            ak.any(
                ak.any(req_zllmass & req_global & req_zqqmass & req_hmass, axis=-1),
                axis=-1,
            )
        )
        output["cutflow"][dataset]["dr"] += ak.sum(
            ak.any(
                ak.any(
                    req_zllmass & req_global & req_zqqmass & req_hmass & req_dr, axis=-1
                ),
                axis=-1,
            )
        )

        output["cutflow"][dataset]["tag one jet"] += ak.sum(
            ak.any(
                ak.any(
                    req_zllmass & req_global & req_zqqmass & req_hmass & req_dr, axis=-1
                ),
                axis=-1,
            )
            & seljet
        )
        output["cutflow"][dataset]["jet efficiency"] += ak.sum(
            ak.any(
                ak.any(
                    req_zllmass & req_global & req_zqqmass & req_hmass & req_dr, axis=-1
                ),
                axis=-1,
            )
            & seljet
            & (njet >= 3)
        )
        output["cutflow"][dataset]["electron efficiency"] += ak.sum(
            ak.any(
                ak.any(
                    req_zllmass & req_global & req_zqqmass & req_hmass & req_dr, axis=-1
                ),
                axis=-1,
            )
            & seljet
            & (njet >= 3)
            & (nele == 2)
        )
        output["cutflow"][dataset]["muon efficiency"] += ak.sum(
            ak.any(
                ak.any(
                    req_zllmass & req_global & req_zqqmass & req_hmass & req_dr, axis=-1
                ),
                axis=-1,
            )
            & seljet
            & (njet >= 3)
            & (nmu == 2)
        )
        selection.add(
            "SR",
            ak.to_numpy(
                ak.any(
                    ak.any(
                        req_zllmass & req_global & req_zqqmass & req_hmass & req_dr,
                        axis=-1,
                    ),
                    axis=-1,
                )
            ),
        )

        selection.add("cjetsel", ak.to_numpy(seljet))
        selection.add(
            "ee",
            ak.to_numpy(
                (ak.num(event_e) == 2)
                & (event_e[:, 0].pt > 25)
                & (event_e[:, 1].pt > 13)
            ),
        )
        selection.add(
            "mumu",
            ak.to_numpy(
                (ak.num(event_mu) == 2)
                & (event_mu[:, 0].pt > 25)
                & (event_mu[:, 1].pt > 13)
            ),
        )

        lepflav = ["ee", "mumu"]

        for histname, h in output.items():
            for ch in lepflav:
                cut = selection.all(
                    "jetsel",
                    "lepsel",
                    "SR",
                    "lumi",
                    "metfilter",
                    "cjetsel",
                    ch,
                    "trigger_%s" % (ch),
                )
                sr_mask = ak.mask(
                    higgs_cand,
                    req_zllmass & req_global & req_zqqmass & req_hmass & req_dr,
                )
                if ak.count(sr_mask.mass) > 0:
                    sr_mask = sr_mask[
                        ak.argsort(abs(sr_mask.ll_cands.mass - 91.18), axis=-1)
                    ]

                hcut = sr_mask[cut]
                hcut = hcut[:, :, 0]
                if ak.count(hcut.mass) > 0:
                    hcut = hcut[ak.argsort(abs(hcut.ll_cands.mass - 91.18), axis=-1)]

                hcut = hcut[:, 0]
                llcut = hcut.ll_cands
                jjcut = hcut.jj_cands
                lep1cut = llcut.lep1
                lep2cut = llcut.lep2
                jet1cut = jjcut.jet1
                jet2cut = jjcut.jet2
                charmcut = sel_cjet[cut]

                if not isRealData:
                    if ch == "ee":
                        lepsf = eleSFs(lep1cut, self._year, self._corr) * eleSFs(
                            lep2cut, self._year, self._corr
                        )
                    elif ch == "mumu":
                        lepsf = muSFs(lep1cut, self._year, self._corr) * muSFs(
                            lep2cut, self._year, self._corr
                        )
                    sf = lepsf
                else:
                    sf = weights.weight()[cut]
                if "cjet_" in histname:
                    fields = {
                        l: normalize(sel_cjet[histname.replace("cjet_", "")], cut)
                        for l in h.fields
                        if l in dir(sel_cjet)
                    }
                    if isRealData:
                        flavor = ak.zeros_like((weights.weight()[cut]))
                    else:
                        flavor = normalize(
                            sel_cjet.hadronFlavour
                            + 1
                            * (
                                (sel_cjet.partonFlavour == 0)
                                & (sel_cjet.hadronFlavour == 0)
                            ),
                            cut,
                        )
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        flav=flavor,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                elif "jet1_" in histname:
                    fields = {
                        l: flatten(jet1cut[histname.replace("jet1_", "")])
                        for l in h.fields
                        if l in dir(jet1cut)
                    }
                    if isRealData:
                        flavor = ak.zeros_like(sf)
                    else:
                        flavor = flatten(
                            jet1cut.hadronFlavour
                            + 1
                            * (
                                (jet1cut.partonFlavour == 0)
                                & (jet1cut.hadronFlavour == 0)
                            )
                        )
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        flav=flavor,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                elif "jet2_" in histname:
                    fields = {
                        l: flatten(jet2cut[histname.replace("jet2_", "")])
                        for l in h.fields
                        if l in dir(jet2cut)
                    }
                    if isRealData:
                        flavor = ak.zeros_like((weights.weight()[cut]))
                    else:
                        flavor = flatten(
                            jet2cut.hadronFlavour
                            + 1
                            * (
                                (jet2cut.partonFlavour == 0)
                                & (jet2cut.hadronFlavour == 0)
                            )
                        )
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        flav=flavor,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                elif "lep1_" in histname:
                    fields = {
                        l: flatten(lep1cut[histname.replace("lep1_", "")])
                        for l in h.fields
                        if l in dir(lep1cut)
                    }
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                elif "lep2_" in histname:
                    fields = {
                        l: flatten(lep2cut[histname.replace("lep2_", "")])
                        for l in h.fields
                        if l in dir(lep2cut)
                    }
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                elif "ll_" in histname:
                    fields = {
                        l: flatten(llcut[histname.replace("ll_", "")])
                        for l in h.fields
                        if l in dir(llcut)
                    }
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                elif "jj_" in histname:
                    fields = {
                        l: flatten(jjcut[histname.replace("jj_", "")])
                        for l in h.fields
                        if l in dir(jjcut)
                    }
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                elif "higgs_" in histname:
                    fields = {
                        l: flatten(hcut[histname.replace("higgs_", "")])
                        for l in h.fields
                        if l in dir(hcut)
                    }
                    h.fill(
                        dataset=dataset,
                        lepflav=ch,
                        **fields,
                        weight=weights.weight()[cut] * sf
                    )
                else:

                    output["nj"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        nj=normalize(njet, cut),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["njnosel"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        nj=flatten(ak.count(events[cut].Jet.pt, axis=-1)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["nele"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        nalep=normalize(naele - nele, cut),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["nmu"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        nalep=normalize(namu - nmu, cut),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["hc_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(hcut.delta_r(charmcut)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["zs_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(llcut.delta_r(jjcut)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["j1j2_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(jet1cut).delta_r(make_p4(jet2cut))),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["l1l2_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(lep1cut).delta_r(make_p4(lep2cut))),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["l1j1_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(lep1cut).delta_r(make_p4(jet1cut))),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["l1j2_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(lep1cut).delta_r(make_p4(jet2cut))),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["l2j1_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(jet1cut).delta_r(make_p4(lep2cut))),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["l2j2_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(lep2cut).delta_r(make_p4(jet2cut))),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["jjc_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(jjcut.delta_r(charmcut)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["cj_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(jet1cut).delta_r(charmcut)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["lc_dr"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        dr=flatten(make_p4(charmcut).delta_r(make_p4(lep1cut))),
                        weight=weights.weight()[cut] * sf,
                    )
                    ll_hCM = llcut.boost(-1 * hcut.boostvec)  # boost to higgs frame
                    poslep = make_p4(ak.where(lep1cut.charge > 0, lep1cut, lep2cut))
                    poslep_hCM = poslep.boost(-1 * hcut.boostvec)
                    poslep_ZCM = poslep_hCM.boost(-1 * llcut.boostvec)
                    jet_hCM = jet1cut.boost(-1 * hcut.boostvec)
                    jet_ZCM = jet_hCM.boost(-1 * jjcut.boostvec)

                    output["costheta_pz"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        costheta=flatten(np.cos(ll_hCM.theta)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["costheta_ll"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        costheta=flatten(np.cos(poslep_ZCM.theta)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["costheta_pz"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        costheta=flatten(np.cos(jet_ZCM.theta)),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["phi_zz"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        phi=flatten(ll_hCM.phi),
                        weight=weights.weight()[cut] * sf,
                    )
                    output["phi_lq"].fill(
                        dataset=dataset,
                        lepflav=ch,
                        phi=flatten(jet_ZCM.delta_phi(poslep_ZCM)),
                        weight=weights.weight()[cut] * sf,
                    )

        return output

    def postprocess(self, accumulator):
        return accumulator
