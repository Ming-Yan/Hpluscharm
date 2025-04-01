import pickle, os, sys, numpy as np
from coffea import processor
import awkward as ak
from Hpluscharm.helpers.histogram import histogram
from coffea.analysis_tools import Weights, PackedSelection
from functools import partial
import os, psutil, copy, gc, coffea, hist
from joblib import load
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    JME_shifts,
    Roccor_shifts,
    puwei,
    met_filters,
    eleSFs,
    muSFs,
    btagSFs,
    jmar_sf,
    HLTSFs,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_3pt,
    top_pT_reweighting,
)
from Hpluscharm.utils.util import (
    mT,
    flatten,
    normalize,
    make_p4,
    defaultdict_accumulator,
    update,
)

import xgboost as xgb


def dphilmet(l1, l2, met):
    return np.where(
        abs(l1.delta_phi(met)) < abs(l2.delta_phi(met)),
        abs(l1.delta_phi(met)),
        abs(l2.delta_phi(met)),
    )


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, cfg):
        self.cfg = cfg
        self._year = self.cfg.dataset["year"]
        self._export_array = self.cfg.userconfig["export_array"]
        self.systematics = self.cfg.systematic
        self._met_filters = met_filters[self.cfg.dataset["campaign"]]
        self._lumiMasks = load_lumi(self.cfg.weights_config["lumiMask"])
        self.SF_map = load_SF(
            self.cfg.dataset["campaign"],
            self.cfg.weights_config,
            self.systematics["weights"],
        )

        from Hpluscharm.MVA.training_config import config2017 as config

        self.xgb_model_emu, self.xgb_model_higgs = xgb.Booster(), xgb.Booster()
        self.xgb_model_emu.load_model(cfg.userconfig["BDT"]["jsonbkg"])
        self.xgb_model_higgs.load_model(cfg.userconfig["BDT"]["jsonhiggs"])
        self.cluster_SR2 = load(cfg.userconfig["BDT"]["clusterSR2_LM"])
        self.cluster_SR = load(cfg.userconfig["BDT"]["clusterSR_LM"])

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
        dataset = events.metadata["dataset"]
        shifts = []
        if "JME" in self.SF_map.keys():
            shifts = JME_shifts(
                shifts,
                self.SF_map,
                events,
                self.cfg.dataset["campaign"],
                isRealData,
                self.systematics["JERC"],
                
            )
        else:
            shifts = [
                ({"Jet": events.Jet, "MET": events.MET, "Muon": events.Muon}, None)
            ]
        if "roccor" in self.SF_map.keys():
            shifts = Roccor_shifts(
                shifts, self.SF_map, events, isRealData, self.systematics["roccor"]
            )
        else:
            shifts[0][0]["Muon"] = events.Muon

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        print(events.metadata["dataset"],shift_name)
        dataset = events.metadata["dataset"]
        ## remove W pT < 100GeV

        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        output = histogram(self.cfg)
        if (
            shift_name is None
            and not isRealData
            and dataset != "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr_13TeV-sherpa"
        ):
            output["sumw"] = ak.sum(events.genWeight / abs(events.genWeight))
        elif dataset == "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr_13TeV-sherpa":
            output["sumw"] = ak.sum(events.genWeight) / 1000000.0


        if (
            "WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8" == dataset
            or "WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8" == dataset
            or "WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8" == dataset
        ):
            events = events[events.LHE.Vpt < 100]
        if "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" == dataset:
            events = events[events.LHE.HT < 70]
        if "LHE" in events.fields:
            output["LHE_Vpt"].fill(
                events.LHE.Vpt, weight=events.genWeight / abs(events.genWeight)
            )
            output["LHE_HT"].fill(
                events.LHE.HT, weight=events.genWeight / abs(events.genWeight)
            )
        if "WJetsToLNu_012JetsNLO_34JetsLO_EWNLOcorr_13TeV-sherpa" == dataset:
            output["LHE_Vpt"].fill(events.event, weight=events.genWeight / 1000000.0)
        output["hltlep1_pt"] = hist.Hist(
            hist.axis.StrCategory([], name="cut", growth=True),
            hist.axis.Regular(40, 0, 200, name="pt", label="pt"),
            hist.axis.Integer(0, 15, name="hlt", label="HLT bit"),
            hist.axis.IntCategory([11, 13], name="lepflav", label="lepflav"),
        )
        output["hltlep2_pt"] = hist.Hist(
            hist.axis.StrCategory([], name="cut", growth=True),
            hist.axis.Regular(40, 0, 200, name="pt", label="pt"),
            hist.axis.Integer(0, 15, name="hlt", label="HLT bit"),
            hist.axis.IntCategory([11, 13], name="lepflav", label="lepflav"),
        )
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
        for t in self.cfg.preselections["mu1hlt"]:
            if t in events.HLT.fields:
                trigger_m = trigger_m | events.HLT[t]
        for t in self.cfg.preselections["mu2hlt"]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]
        for t in self.cfg.preselections["e1hlt"]:
            if t in events.HLT.fields:
                trigger_e = trigger_e | events.HLT[t]
        for t in self.cfg.preselections["e2hlt"]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]
        for t in self.cfg.preselections["emuhlt"]:
            if t in events.HLT.fields:
                trigger_em = trigger_em | events.HLT[t]
        trigger = np.zeros(len(events), dtype="bool")
        if isRealData:
            if "MuonEG" in dataset:
                # trigger_em = trigger_em

                trigger = trigger_em

            elif "SingleMuon" in dataset:
                trigger_mu = trigger_m & ~trigger_em 
                trigger = trigger_mu

            elif "SingleElectron" in dataset or "EGamma" in dataset:
                trigger_ele = trigger_e & ~trigger_em &~trigger_m

                trigger = trigger_ele

        else:
            trigger = trigger_em | trigger_m | trigger_e
        mu12leg, mu23leg, singleE, singleMu = (
            np.zeros(len(events), dtype="bool"),
            np.zeros(len(events), dtype="bool"),
            np.zeros(len(events), dtype="bool"),
            np.zeros(len(events), dtype="bool"),
        )
        for t in self.cfg.preselections["emuhlt"]:
            if t in events.HLT.fields and "Ele23" in t:
                mu12leg = trigger_em | events.HLT[t]
            if t in events.HLT.fields and "Ele12" in t:
                mu23leg = trigger_em | events.HLT[t]
        singleE, singleMu = trigger_e, trigger_m
        selection.add("trigger_ee", ak.to_numpy(trigger_ele))
        selection.add("trigger_mumu", ak.to_numpy(trigger_mu))
        selection.add("trigger_emu", ak.to_numpy(trigger))
        del trigger_e, trigger_ee, trigger_m, trigger_mm
        metfilter = np.ones(len(events), dtype="bool")
        for flag in self._met_filters["data" if isRealData else "mc"]:
            metfilter &= np.array(events.Flag[flag])
        selection.add("metfilter", metfilter)
        del metfilter

        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon
        event_e = events.Electron

        if isRealData:
            if (
                "Run2016B" in dataset
                or "Run2016C" in dataset
                or "Run2016D" in dataset
                or "Run2016E" in dataset
                or "Run2016F-HIPM" in dataset
            ):
                mu_pt_matched_hlt = (
                    (event_mu.pt > 15)
                    & (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL == True)
                ) | (
                    (event_mu.pt > 24)
                    & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True)
                )
                ele_pt_matched_hlt = (
                    (event_e.pt > 25)
                    & (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL == True)
                ) | (event_e.pt > 15) & (
                    (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True)
                )
            elif "Run2017B" in dataset or "Run2018":
                mu_pt_matched_hlt = (
                    (event_mu.pt > 15)
                    & (
                        events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
                        == True
                    )
                ) | (
                    (event_mu.pt > 24)
                    & (
                        events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
                        == True
                    )
                )
                ele_pt_matched_hlt = (
                    (event_e.pt > 25)
                    & (
                        events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
                        == True
                    )
                ) | (
                    (event_e.pt > 15)
                    & (
                        events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
                        == True
                    )
                )
            else:
                mu_pt_matched_hlt = (
                    (event_mu.pt > 15)
                    & (
                        events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
                        == True
                    )
                ) | (
                    (event_mu.pt > 24)
                    & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True)
                )
                ele_pt_matched_hlt = (
                    (event_e.pt > 25)
                    & (
                        events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
                        == True
                    )
                ) | (
                    (event_e.pt > 15)
                    & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True)
                )
        else:
            mu_pt_matched_hlt = ((event_mu.pt > 15) & mu12leg) | (
                (event_mu.pt > 25) & mu23leg
            )
            ele_pt_matched_hlt = ((event_e.pt > 25) & mu12leg) | (
                (event_e.pt > 15) & mu23leg
            )

        musel = (
            # mu_pt_matched_hlt &
            (event_mu.pt > 15)
            & (abs(event_mu.eta) < 2.4)
            & (event_mu.tightId > 0)
            & (event_mu.pfRelIso04_all < 0.15)
            # & (abs(event_mu.sip3d)<4)
            & (abs(event_mu.dxy) < 0.2)
            & (abs(event_mu.dz) < 0.5)
        )

        event_mu = event_mu[musel]
        event_mu["lep_flav"] = 13 * event_mu.charge
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

        elesel = (
            # ele_pt_matched_hlt &
            (event_e.pt > 15)
            & (
                (abs(event_e.eta) < 1.4442)
                | ((abs(event_e.eta) > 1.566) & (abs(event_e.eta) < 2.5))
            )
            # & (event_e.mvaFall17V2Iso_WP90 == 1)
            & (event_e.cutBased >= 4)
            # & (
            # ((abs(event_e.dxy) < 0.05)& (abs(event_e.dz) < 0.1) & (abs(event_e.eta) < 1.4442))|
            #     ((abs(event_e.dxy) < 0.1)& (abs(event_e.dz) < 0.2) & (abs(event_e.eta) >1.566))
            # ))
            # & (abs(event_e.dxy) < 0.05)
            # & (abs(event_e.dz) < 0.1)
        )

        event_e = event_e[elesel]
        event_e["lep_flav"] = 11 * event_e.charge
        event_e = ak.pad_none(event_e, 2, axis=1)
        nele = ak.sum(elesel, axis=1)
        aele = events.Electron[
            (events.Electron.pt > 10)
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
        if ak.any(nele + nmu >= 2, axis=-1):
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

        del good_leptons
        ll_cand = ak.zip(
            {
                "lep1": np.where(
                    leppair.lep1.pt > leppair.lep2.pt, leppair.lep1, leppair.lep2
                ),
                "lep2": np.where(
                    leppair.lep1.pt < leppair.lep2.pt, leppair.lep1, leppair.lep2
                ),
                "pt": (leppair.lep1 + leppair.lep2).pt,
                "eta": (leppair.lep1 + leppair.lep2).eta,
                "phi": (leppair.lep1 + leppair.lep2).phi,
                "mass": (leppair.lep1 + leppair.lep2).mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        del leppair
        ll_cand = ak.packed(ak.pad_none(ll_cand, 1, axis=1))

        selection.add("ee", ak.to_numpy(nele == 2))
        selection.add("mumu", ak.to_numpy(nmu == 2))
        selection.add("emu", ak.to_numpy((nele == 1) & (nmu == 1)))
        checkll = ll_cand[:, 0]
        output["hltlep1_pt"].fill(
            "hltlep",
            ak.fill_none(checkll[trigger].lep1.pt, -1),
            (1 * mu12leg + 2 * mu23leg + 4 * singleMu + 8 * singleE)[trigger],
            ak.fill_none(checkll[trigger].lep1.lep_flav, -1),
        )
        output["hltlep2_pt"].fill(
            "hltlep",
            ak.fill_none(checkll[trigger].lep2.pt, -1),
            (1 * mu12leg + 2 * mu23leg + 4 * singleMu + 8 * singleE)[trigger],
            ak.fill_none(checkll[trigger].lep2.lep_flav, -1),
        )
        ############

        met = ak.zip(
            {
                "pt": events.MET.pt,
                "eta": ak.zeros_like(events.MET.pt),
                "phi": events.MET.phi,
                # "energy": events.MET.sumEt,
                "mass": ak.zeros_like(events.TkMET.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        tkmet = ak.zip(
            {
                "pt": events.TkMET.pt,
                "phi": events.TkMET.phi,
                "eta": ak.zeros_like(events.TkMET.pt),
                # "energy": events.TkMET.sumEt,
                "mass": ak.zeros_like(events.TkMET.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        jetpu_bit = 1 if self._year == "2016" else 7

        jet_kin = (events.Jet.pt > 20) & (abs(events.Jet.eta) <= 2.4)
        jet_id = ((events.Jet.puId == jetpu_bit) | (events.Jet.pt > 50)) & (
            events.Jet.jetId > 1
        )

        jet_dr_cand1 = ak.all(
            (events.Jet.metric_table(ll_cand.lep1) > 0.4), axis=2, mask_identity=True
        )
        jet_dr_cand2 = ak.all(
            events.Jet.metric_table(ll_cand.lep2) > 0.4, axis=2, mask_identity=True
        )
        jet_dr_aele = ak.all(
            events.Jet.metric_table(aele) > 0.4, axis=2, mask_identity=True
        )
        jet_dr_amu = ak.all(
            events.Jet.metric_table(amu) > 0.4, axis=2, mask_identity=True
        )
        jetsel = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) <= 2.4)
            & ((events.Jet.puId > 6) | (events.Jet.pt > 50))
            & (events.Jet.jetId > 1)
            & ak.all(
                (events.Jet.metric_table(ll_cand.lep1) > 0.4)
                & (events.Jet.metric_table(ll_cand.lep2) > 0.4),
                axis=2,
                mask_identity=True,
            )
            & ak.all(events.Jet.metric_table(aele) > 0.4, axis=2, mask_identity=True)
            & ak.all(events.Jet.metric_table(amu) > 0.4, axis=2, mask_identity=True)
        )
        jetselout = (
            (events.Jet.pt > 20)
            & (abs(events.Jet.eta) > 2.4)
            & (abs(events.Jet.eta) <= 4.7)
            & ((events.Jet.puId > 6) | (events.Jet.pt > 50))
            & (events.Jet.jetId > 5)
            & ak.all(
                (events.Jet.metric_table(ll_cand.lep1) > 0.4)
                & (events.Jet.metric_table(ll_cand.lep2) > 0.4),
                axis=2,
                mask_identity=True,
            )
            & ak.all(events.Jet.metric_table(aele) > 0.4, axis=2, mask_identity=True)
            & ak.all(events.Jet.metric_table(amu) > 0.4, axis=2, mask_identity=True)
        )
        SV = ak.zip(
            {
                "pt": events.SV.pt,
                "eta": events.SV.eta,
                "phi": events.SV.phi,
                "mass": events.SV.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        svsel = (
            ak.all(
                (SV.metric_table(ll_cand.lep1) > 0.4)
                & (SV.metric_table(ll_cand.lep2) > 0.4),
                axis=2,
                mask_identity=True,
            )
            & ak.all(SV.metric_table(aele) > 0.4, axis=2, mask_identity=True)
            & ak.all(SV.metric_table(amu) > 0.4, axis=2, mask_identity=True)
            & (events.SV.ntracks >= 2)
            & (events.SV.pAngle > 0.98)
            & (np.abs(events.SV.dxy) < 3)
            & (events.SV.dlenSig < 4)
        )
        # njet = ak.sum(jetsel, axis=1)

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
        njet = ak.sum(jetsel & cvbcutem & cvlcutem, axis=1)
        sr_cut = mT(ll_cand, met) > 60

        mass_cut = ll_cand.mass <= 72
        dy_cr2_cut = (
            (mT(ll_cand.lep2, met) > 30)
            & (events.MET.sumEt > 45)
            & (mT(ll_cand, met) < 60)
        )
        top_cr_cut = mT(ll_cand, met) <= 60
        top_cr2_cut = (ll_cand.mass > 50) & (abs(ll_cand.mass - 91.18) > 15)
        lepton_cut = (
            (ll_cand.lep1.pt > 25)
            & (ll_cand.lep2.pt > 15)
            & (ll_cand.lep1.charge + ll_cand.lep2.charge == 0)
            & (make_p4(ll_cand.lep1).delta_r(make_p4(ll_cand.lep2)) > 0.4)
        )

        HLT_pt_threshold = 38 if self._year == "2017" else 35
        HLT_cut = (
            (mu23leg)
            | (mu12leg)
            | (
                (abs(ll_cand.lep1.lep_flav) == 11)
                & (singleE)
                & (ll_cand.lep1.pt > HLT_pt_threshold)
                & (~mu23leg)
                & (~mu12leg)
            )
            | (
                (abs(ll_cand.lep1.lep_flav) == 13)
                & (singleMu)
                & (ll_cand.lep1.pt > 30)
                & (~mu23leg)
                & (~mu12leg)
            )
        )

        fired = ak.fill_none(HLT_cut & trigger, False)[:, 0]

        output["hltlep1_pt"].fill(
            "lep_sel_firedHLT",
            ak.fill_none(checkll[fired].lep1.pt, -1),
            (1 * mu12leg + 2 * mu23leg + 4 * singleMu + 8 * singleE)[fired],
            ak.fill_none(checkll[fired].lep1.lep_flav, -1),
        )
        output["hltlep2_pt"].fill(
            "lep_sel_firedHLT",
            ak.fill_none(checkll[fired].lep2.pt, -1),
            (1 * mu12leg + 2 * mu23leg + 4 * singleMu + 8 * singleE)[fired],
            ak.fill_none(checkll[fired].lep2.lep_flav, -1),
        )
        MET_cut = events.MET.sumEt > 45
        METTkMET_cut = abs(met.delta_phi(tkmet)) < 0.5
        dilep_cut = (ll_cand.mass > 12) & (ll_cand.pt > 30)

        global_cut = (
            HLT_cut
            & (ll_cand.lep1.pt > 25)
            & (ll_cand.mass > 12)
            & (ll_cand.pt > 30)
            & (ll_cand.lep1.charge + ll_cand.lep2.charge == 0)
            & (events.MET.pt > 20)
            & (make_p4(ll_cand.lep1).delta_r(make_p4(ll_cand.lep2)) > 0.4)
            & (abs(met.delta_phi(tkmet)) < 0.5)
            & (events.MET.sumEt > 45)
            & (mT(ll_cand.lep2, met) > 30)
        )

        llmass_cut = abs(ll_cand.mass - 91.18) > 15
        llmass_low = ll_cand.mass < 91.18 - 15
        if shift_name == None:
            output["cutflow"]["raw events"] = len(events)
            if not isRealData:
                output["cutflow"]["all"] += ak.sum(
                    events.genWeight / abs(events.genWeight)
                )
            else:
                output["cutflow"]["all"] += len(events)
            if "DoubleEG" in dataset:
                output["cutflow"]["trigger"] += ak.sum(trigger_ele)
            elif "DoubleMuon" in dataset:
                output["cutflow"]["trigger"] += ak.sum(trigger_mu)
            if isRealData:
                events.genWeight = np.ones_like(events.run)
            output["cutflow"]["lepton selection"] += ak.sum(ak.any(lepton_cut, axis=-1))
            output["cutflow"]["MET selection"] += ak.sum(
                ak.any(lepton_cut & MET_cut, axis=-1)
            )
            output["cutflow"]["METTkMET selection"] += ak.sum(
                ak.any(lepton_cut & MET_cut & METTkMET_cut, axis=-1)
            )
            output["cutflow"]["dilepton selection"] += ak.sum(
                ak.any(lepton_cut & MET_cut & METTkMET_cut & dilep_cut, axis=-1)
            )
            output["cutflow"]["mT1 selection"] += ak.sum(
                ak.any(
                    lepton_cut
                    & MET_cut
                    & METTkMET_cut
                    & dilep_cut
                    & (mT(ll_cand.lep2, met) > 30),
                    axis=-1,
                )
            )
            output["cutflow"]["mT1 selection"] += ak.sum(
                ak.any(
                    lepton_cut
                    & MET_cut
                    & METTkMET_cut
                    & dilep_cut
                    & (mT(ll_cand, met) > 60),
                    axis=-1,
                )
            )
            # output["cutflow"]["global selection"] += ak.sum(
            # (ak.any(glvoms-proxy-init --voms cms --vomses ~/.grid-security/vomses --valid 192:00obal_cut, axis=-1))*(events.genWeight))
            # output["cutflow"]["signal region"]+= ak.sum(
            #     (ak.any(global_cut & sr_cut, axis=-1))*(events.genWeight)
            # )
            output["cutflow"]["jet kinematics"] += ak.sum(
                (ak.any(global_cut & sr_cut, axis=-1) & (ak.sum(jet_kin, axis=1) > 0))
            )
            cuts = ak.any(global_cut & sr_cut, axis=-1) & (
                ak.sum(jet_kin & jet_id, axis=1) > 0
            )
            sortjet = events[cuts].Jet
            ll_cand_cut = ak.mask(ll_cand, global_cut & sr_cut)
            ll_cand_cut = ll_cand_cut[cuts]
            if ak.count(ll_cand_cut.pt) > 0:
                ll_cand_cut = ll_cand_cut[
                    ak.argsort(ll_cand_cut.pt, axis=1, ascending=False)
                ]
            ll_cand_cut = ll_cand_cut[:, 0]
            sortjet = sortjet[
                ak.argsort(sortjet.btagDeepFlavCvL, axis=1, ascending=False)
            ]
            sortjet = sortjet[:, 0]

            output["jet_etaphi"] = hist.Hist(
                hist.axis.Regular(40, -2.5, -1.5, name="eta", label="jet $\eta$"),
                hist.axis.Regular(50, -3.5, -1.0, name="phi", label="jet $\phi$"),
                hist.storage.Weight(),
            )

            # output['dr_zoom'].fill(flatten(sortjet.pt),ll_cand_cut.lep2.pt,flatten(sortjet.delta_r(ll_cand_cut.lep1)),flatten(sortjet.btagDeepFlavCvL),flatten(np.where(matched[:,0].delta_r(sortjet)<0.1,abs(matched[:,0].pdgId),-1)),weight=flatten(ak.broadcast_arrays(events[cuts].genWeight, sortjet.delta_r(ll_cand_cut.lep1))[0]))

            output["cutflow"]["jet dr_cand1"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (ak.sum(jet_kin & jet_dr_cand1, axis=1) > 0)
                )
            )
            output["cutflow"]["jet dr_cand2"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (ak.sum(jet_kin & jet_dr_cand1 & jet_dr_cand2, axis=1) > 0)
                )
            )
            output["cutflow"]["jet dr_e"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (
                        ak.sum(
                            jet_kin & jet_dr_cand1 & jet_dr_cand2 & jet_dr_aele, axis=1
                        )
                        > 0
                    )
                )
            )
            output["cutflow"]["jet dr_mu"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (
                        ak.sum(
                            jet_kin
                            & jet_dr_cand1
                            & jet_dr_cand2
                            & jet_dr_amu
                            & jet_dr_aele,
                            axis=1,
                        )
                        > 0
                    )
                )
            )
            output["cutflow"]["jet id"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (
                        ak.sum(
                            jet_kin
                            & jet_dr_cand1
                            & jet_dr_cand2
                            & jet_dr_amu
                            & jet_dr_aele
                            & jet_id,
                            axis=1,
                        )
                        > 0
                    )
                )
            )
            output["cutflow"]["jet id"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (
                        ak.sum(
                            jet_kin
                            & jet_dr_cand1
                            & jet_dr_cand2
                            & jet_dr_amu
                            & jet_dr_aele
                            & jet_id,
                            axis=1,
                        )
                        > 0
                    )
                )[cuts]
                # * (events[cuts].genWeight)
            )

            output["cutflow"]["jet CvB"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (ak.sum((jetsel & cvbcutem), axis=1) > 0)
                )
            )
            output["cutflow"]["jet CvL"] += ak.sum(
                (
                    ak.any(global_cut & sr_cut, axis=-1)
                    & (ak.sum((jetsel & cvbcutem & cvlcutem), axis=1) > 0)
                )
            )
            output["cutflow"]["all emu"] += ak.sum(
                (ak.any(global_cut & sr_cut, axis=-1) & (njet > 0) & trigger_em)
            )

        selection.add(
            "noc_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut & llmass_low, axis=-1)
                & (ak.sum(jetsel, axis=1) >= 1)
            ),
        )
        selection.add(
            "SR_LM_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut & llmass_low, axis=-1)
                & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) > 1)
            ),
        )
        selection.add(
            "SR2_LM_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut & llmass_low, axis=-1)
                & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) == 1)
            ),
        )

        selection.add(
            "HM_CR_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut & ~llmass_low, axis=-1)
                & ((ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) >= 1))
            ),
        )

        nsv = ak.sum(svsel, axis=1)

        selection.add(
            "top_CR_emu",
            ak.to_numpy(
                ak.any(top_cr_cut & global_cut & ~llmass_low, axis=-1)
                & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) >= 1)
            ),
        )
        # selection.add(
        #    "top_CR_nj_emu",
        #    ak.to_numpy(
        #        ak.any(top_cr_cut & global_cut & ~llmass_low, axis=-1)
        #        & (ak.sum(jetsel & cvbcutem & cvlcutem, axis=1) > 1)
        #    ),
        # )
        selection.add(
            "bSR_LM_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut & llmass_low, axis=-1)
                & (ak.sum(jetsel & (~cvbcutem & cvlcutem), axis=1) > 1)
            ),
        )
        selection.add(
            "bSR2_LM_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut & llmass_low, axis=-1)
                & (ak.sum(jetsel & (~cvbcutem  & cvlcutem), axis=1) == 1)
            ),
        )

        selection.add(
            "bHM_CR_emu",
            ak.to_numpy(
                ak.any(sr_cut & global_cut & ~llmass_low, axis=-1)
                & ((ak.sum(jetsel & (~cvbcutem & cvlcutem), axis=1) >= 1))
            ),
        )
        selection.add(
            "btop_CR_emu",
            ak.to_numpy(
                ak.any(top_cr_cut & global_cut & ~llmass_low, axis=-1)
                & (ak.sum(jetsel & (~cvbcutem & cvlcutem), axis=1) >= 1)
            ),
        )
        reg = [
            "SR_LM",
            "SR2_LM",
            "HM_CR",
            "top_CR",
            "bSR_LM",
            "bSR2_LM",
            "bHM_CR",
            "btop_CR"
            # "HM_CR_1j",
            # "HM_CR_nj",
            # "top_CR_1j",
            # "top_CR_nj",
        ]  # ,"DY_CR"]
        lepflav = ["emu"]
        mask_lep = {
            "SR_LM": global_cut & sr_cut & llmass_low,
            "SR2_LM": global_cut & sr_cut & llmass_low,
            "HM_CR": global_cut & sr_cut & ~llmass_low,
            "top_CR": global_cut & top_cr_cut & ~llmass_low,
        }
        mask_jet = {
            "ee": jetsel & cvbcutll & cvlcutll,
            "mumu": jetsel & cvbcutll & cvlcutll,
            "emu": jetsel & cvbcutem & cvlcutem,
        }
        ### Weights

        for r in reg:
            for ch in ["emu"]:
                cut = selection.all(
                    "lepsel",
                    "metfilter",
                    "lumi",
                    "%s_%s" % (r, ch),
                    ch,
                    "trigger_%s" % (ch),
                )


                if len(events[cut].event) == 0:
                    continue

                ll_cands = ak.mask(ll_cand[cut], mask_lep[r.replace("b", "")][cut])
                # if ak.count(ll_cands.pt) > 0:
                #     ll_cands = ll_cands[
                #         ak.argsort(ll_cands.pt, axis=1, ascending=False)
                #     ]

                maskjet = (
                    (jetsel & cvbcutem & cvlcutem)
                    if "b" not in r
                    else (jetsel & (~cvbcutem  & cvlcutem))
                )
                sel_cjet_flav = events[cut].Jet[maskjet[cut]]
                if ak.count_nonzero(sel_cjet_flav.pt) > 0:
                    sel_cjet_flav = sel_cjet_flav[
                        ak.argsort(
                            sel_cjet_flav.btagDeepFlavCvL, axis=1, ascending=False
                        )
                    ]
                nseljet = ak.count(sel_cjet_flav.pt, axis=1)
                selsv = events[cut].SV[svsel[cut]]
                nselsv = ak.count(selsv.pt, axis=1)
                selsv = ak.pad_none(selsv, 2, axis=1)
                selsv = selsv[:, 0]
                # sel_cjet_flav = ak.pad_none(sel_cjet_flav, 2, axis=1)
                # sel_cjet_flav = sel_cjet_flav[:, :2]
                sel_cjet_flav = sel_cjet_flav[:, 0]
                ll_cands = ak.pad_none(ll_cands, 1, axis=1)

                llcut = ll_cands[:, 0]
                lep1cut = llcut.lep1
                lep2cut = llcut.lep2
                w1cut = lep1cut + met[cut]
                w2cut = lep2cut + met[cut]
                hcut = llcut + met[cut]

                if isRealData:
                    flavor = ak.zeros_like(sel_cjet_flav["pt"], dtype=np.int)
                else:
                    flavor = ak.values_astype(
                        sel_cjet_flav.hadronFlavour
                        + 1
                        * (
                            (sel_cjet_flav.partonFlavour == 0)
                            & (sel_cjet_flav.hadronFlavour == 0)
                        ),
                        np.int8,
                    )

                ele = np.where(lep1cut.lep_flav == 11, lep1cut, lep2cut)
                mu = np.where(lep1cut.lep_flav == 13, lep1cut, lep2cut)
                weights = Weights(len(events[cut]), storeIndividual=True)

                output["hltlep1_pt"].fill(
                    "event_sel",
                    lep1cut.pt,
                    (1 * mu12leg + 2 * mu23leg + 4 * singleMu + 8 * singleE)[cut],
                    lep1cut.lep_flav,
                )
                output["hltlep2_pt"].fill(
                    "event_sel",
                    lep2cut.pt,
                    (1 * mu12leg + 2 * mu23leg + 4 * singleMu + 8 * singleE)[cut],
                    lep2cut.lep_flav,
                )
                if isRealData:
                    weights.add("genweight", np.ones(len(events[cut])))
                else:
                    if "sherpa" in dataset:
                        weights.add("genweight", events[cut].genWeight / 1000000.0)

                    else:
                        weights.add(
                            "genweight",
                            events[cut].genWeight / abs(events[cut].genWeight),
                        )
                    weights.add(
                        "L1prefireweight",
                        events[cut].L1PreFiringWeight.Nom,
                        events[cut].L1PreFiringWeight.Up,
                        events[cut].L1PreFiringWeight.Dn,
                    )
                    weights.add(
                        "puweight",
                        puwei(self.SF_map, events[cut].Pileup.nTrueInt),
                        puwei(self.SF_map, events[cut].Pileup.nTrueInt, "up"),
                        puwei(self.SF_map, events[cut].Pileup.nTrueInt, "down"),
                    )

                    if "PSWeight" in events.fields:
                        add_ps_weight(weights, events[cut].PSWeight)
                    else:
                        add_ps_weight(weights, None)
                    if "LHEPdfWeight" in events.fields:
                        add_pdf_weight(
                            weights, events[cut].LHEPdfWeight
                        )  # ,self.systematics["LHE"])
                    else:
                        add_pdf_weight(weights, None)

                    if "LHEScaleWeight" in events.fields and ak.all(
                        ak.num(events[cut].LHEScaleWeight) > 0, mask_identity=True
                    ):
                        add_scalevar_3pt(
                            weights, events[cut].LHEScaleWeight, self.systematics["LHE"]
                        )
                    else:
                        add_scalevar_3pt(weights, [])
                    if "TTTo" in dataset:
                        weights.add(
                            "ttbar_weight",
                            top_pT_reweighting(events[cut].GenPart),
                            (
                                top_pT_reweighting(events[cut].GenPart)
                                - ak.ones_like(top_pT_reweighting(events[cut].GenPart))
                            )
                            * 2.0
                            + ak.ones_like(top_pT_reweighting(events[cut].GenPart)),
                            ak.ones_like(top_pT_reweighting(events[cut].GenPart)),
                        )
                    HLTSFs(
                        ele,
                        mu,
                        (1 * mu12leg + 2 * mu23leg + 4 * singleMu + 8 * singleE)[cut],
                        self.SF_map,
                        weights,
                        self.cfg.dataset["campaign"],
                        syst=self.systematics["weights"],
                    )
                    eleSFs(ele, self.SF_map, weights, syst=self.systematics["weights"])
                    muSFs(mu, self.SF_map, weights, syst=self.systematics["weights"])
                    btagSFs(
                        sel_cjet_flav,
                        self.SF_map,
                        weights,
                        "DeepJetC",
                        syst=self.systematics["weights"],
                    )
                    jmar_sf(
                        sel_cjet_flav,
                        self.SF_map,
                        weights,
                        syst=self.systematics["weights"],
                    )

                if shift_name is None:
                    systematics = ["nominal"] + list(weights.variations) + ["nocSF"]
                else:
                    systematics = [shift_name]
                if self._export_array:
                    region_name = r
                    if isRealData:
                        output["array"][region_name][ch][
                            "event"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(events[cut].event))
                        )
                        output["array"][region_name][ch][
                            "run"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(events[cut].run))
                        )
                        output["array"][region_name][ch][
                            "lumi"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(events[cut].luminosityBlock))
                        )
                    else:
                        for w in weights.weightStatistics.keys():
                            output["array"][region_name][ch][
                                w
                            ] = processor.column_accumulator(
                                ak.to_numpy(
                                    normalize(weights.partial_weight(include=[w]))
                                )
                            )
                        output["array"][region_name][ch][
                            "event"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(events[cut].event))
                        )
                        output["array"][region_name][ch][
                            "run"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(events[cut].run))
                        )
                        output["array"][region_name][ch][
                            "lumi"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(events[cut].luminosityBlock))
                        )
                        output["array"][region_name][ch][
                            "weight"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(weights.weight()))
                        )
                        output["array"][region_name][ch][
                            "genwei"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(events[cut].genWeight))
                        )
                        output["array"][region_name][ch][
                            "jetflav_flav"
                        ] = processor.column_accumulator(
                            ak.to_numpy(normalize(sel_cjet_flav.hadronFlavour))
                        )

                        if "LHEPart" in events.fields:
                            output["array"][region_name][ch][
                                "LHEPart_status"
                            ] = processor.column_accumulator(
                                ak.to_numpy(flatten(events[cut].LHEPart.status))
                            )
                            output["array"][region_name][ch][
                                "LHEPart_pdgId"
                            ] = processor.column_accumulator(
                                ak.to_numpy(flatten(events[cut].LHEPart.pdgId))
                            )
                            output["array"][region_name][ch][
                                "LHEPart_pt"
                            ] = processor.column_accumulator(
                                ak.to_numpy(flatten(events[cut].LHEPart.pt))
                            )
                            output["array"][region_name][ch][
                                "LHEPart_eta"
                            ] = processor.column_accumulator(
                                ak.to_numpy(flatten(events[cut].LHEPart.eta))
                            )
                            output["array"][region_name][ch][
                                "LHEPart_phi"
                            ] = processor.column_accumulator(
                                ak.to_numpy(flatten(events[cut].LHEPart.phi))
                            )
                            output["array"][region_name][ch][
                                "LHEPart_mass"
                            ] = processor.column_accumulator(
                                ak.to_numpy(flatten(events[cut].LHEPart.mass))
                            )
                            output["array"][region_name][ch][
                                "LHE_Nc"
                            ] = processor.column_accumulator(
                                ak.to_numpy(
                                    flatten(
                                        ak.broadcast_arrays(
                                            events[cut].LHE.Nc,
                                            events[cut].LHEPart.pdgId,
                                        )[0]
                                    )
                                )
                            )
                            output["array"][region_name][ch][
                                "LHE_Njets"
                            ] = processor.column_accumulator(
                                ak.to_numpy(
                                    flatten(
                                        ak.broadcast_arrays(
                                            events[cut].LHE.Njets,
                                            events[cut].LHEPart.pdgId,
                                        )[0]
                                    )
                                )
                            )
                    output["array"][region_name][ch][
                        "nselj"
                    ] = processor.column_accumulator(ak.to_numpy(normalize(nseljet)))
                    output["array"][region_name][ch][
                        "njout"
                    ] = processor.column_accumulator(
                        ak.to_numpy(
                            flatten(
                                ak.count(events[cut].Jet[jetselout[cut]].pt, axis=1)
                            )
                        )
                    )

                    output["array"][region_name][ch][
                        "jetflav_btagDeepFlavCvL"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(sel_cjet_flav.btagDeepFlavCvL))
                    )
                    output["array"][region_name][ch][
                        "jetflav_btagDeepFlavCvB"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(sel_cjet_flav.btagDeepFlavCvB))
                    )
                    obj_dict = {
                        "h": hcut,
                        "ll": llcut,
                        "lep1": lep1cut,
                        "lep2": lep2cut,
                        "jetflav": sel_cjet_flav,
                    }
                    output["array"][region_name][ch][
                        "lep1flav"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(lep1cut.lep_flav))
                    )
                    output["array"][region_name][ch][
                        "lep2flav"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(lep2cut.lep_flav))
                    )
                    for obj in obj_dict.keys():
                        output["array"][region_name][ch][
                            f"{obj}_pt"
                        ] = processor.column_accumulator(
                            ak.to_numpy(flatten(obj_dict[obj].pt))
                        )
                        output["array"][region_name][ch][
                            f"{obj}_eta"
                        ] = processor.column_accumulator(
                            ak.to_numpy(flatten(obj_dict[obj].eta))
                        )
                        output["array"][region_name][ch][
                            f"{obj}_phi"
                        ] = processor.column_accumulator(
                            ak.to_numpy(flatten(obj_dict[obj].phi))
                        )
                        output["array"][region_name][ch][
                            f"{obj}_mass"
                        ] = processor.column_accumulator(
                            ak.to_numpy(flatten(obj_dict[obj].mass))
                        )

                    output["array"][region_name][ch][
                        "mT1"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(mT(lep1cut, met[cut])))
                    )
                    output["array"][region_name][ch][
                        "mT2"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(mT(lep2cut, met[cut])))
                    )
                    output["array"][region_name][ch][
                        "mTh"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(mT(llcut, met[cut])))
                    )
                    output["array"][region_name][ch][
                        "ll_mass"
                    ] = processor.column_accumulator(ak.to_numpy(flatten(llcut.mass)))
                    output["array"][region_name][ch][
                        "nselsv"
                    ] = processor.column_accumulator(ak.to_numpy(normalize(nselsv)))
                    output["array"][region_name][ch][
                        "nsv"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(ak.count(events[cut].SV.pt, axis=1)))
                    )
                    output["array"][region_name][ch][
                        "nele"
                    ] = processor.column_accumulator(
                        ak.to_numpy(normalize(naele - nele, cut))
                    )
                    output["array"][region_name][ch][
                        "nmu"
                    ] = processor.column_accumulator(
                        ak.to_numpy(normalize(namu - nmu, cut))
                    )
                    output["array"][region_name][ch][
                        "npvs"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(events[cut].PV.npvs))
                    )

                    output["array"][region_name][ch][
                        "MET_pt"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(events.MET[cut].pt))
                    )
                    output["array"][region_name][ch][
                        "MET_significance"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(events[cut].MET.significance))
                    )
                    output["array"][region_name][ch][
                        "MET_sumEt"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(events[cut].MET.sumEt))
                    )
                    output["array"][region_name][ch][
                        "MET_phi"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(events[cut].MET.phi))
                    )

                    output["array"][region_name][ch][
                        "lll1_dr"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(lep1cut.delta_r(llcut)))
                    )
                    output["array"][region_name][ch][
                        "llc_dr"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(sel_cjet_flav.delta_r(llcut)))
                    )
                    output["array"][region_name][ch][
                        "lll2_dr"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(lep2cut.delta_r(llcut)))
                    )
                    output["array"][region_name][ch][
                        "l1met_dphi"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(lep1cut.delta_phi(met[cut])))
                    )
                    output["array"][region_name][ch][
                        "l2met_dphi"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(lep2cut.delta_phi(met[cut])))
                    )
                    output["array"][region_name][ch][
                        "cW1_dphi"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(sel_cjet_flav.delta_phi(w1cut)))
                    )

                    output["array"][region_name][ch][
                        "METTkMETdphi"
                    ] = processor.column_accumulator(
                        ak.to_numpy(flatten(met[cut].delta_phi(tkmet[cut])))
                    )

                # if "SV" in region_name:
                #     for SVs in [
                #         "pt",
                #         "eta",
                #         "phi",
                #         "ndof",
                #         "mass",
                #         "chi2",
                #         "charge",
                #         "pAngle",
                #         "dxySig",
                #         "dxy",
                #         "dlenSig",
                #         "dlen",
                #         "x",
                #         "y",
                #         "z",
                #         "ntracks",
                #     ]:
                #         output["array"][region_name][ch][
                #             f"sv1_{SVs}"
                #         ] = processor.column_accumulator(
                #             ak.to_numpy(flatten(selsv[SVs]))
                #         )
                #         output["array"][region_name][ch][
                #             f"sv2_{SVs}"
                #         ] = processor.column_accumulator(
                #             ak.to_numpy(flatten(ak.fill_none(selsv2[SVs], -99)))
                #         )

                elif shift_name is None or shift_name == "HEM18":
                    '''
                    output["jet_etaphi"].fill(
                        eta=sel_cjet_flav.eta,
                        phi=sel_cjet_flav.phi,
                        weight=weights.weight(),
                    )
                    
                    output["npv"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        npv=ak.values_astype(events[cut].PV.npvs, np.int8),
                        weight=weights.weight(),
                    )

                    output["MET_phi"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        phi=events[cut].MET.phi,
                        weight=weights.weight(),
                    )

                    output["nj"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        n=nseljet,
                        weight=weights.weight(),
                    )
                    output["l1c_dr"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        dr=sel_cjet_flav.delta_r(lep1cut),
                        weight=weights.weight(),
                    )
                    output["l2c_dr"].fill(
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        dr=sel_cjet_flav.delta_r(lep2cut),
                        weight=weights.weight(),
                    )

                    output["weight"].fill(
                        lepflav=ch, region=r, flav=flavor, wei=weights.weight()
                    )
                    '''

                    obj_dict = {
                        "ll": llcut,
                        "lep1": lep1cut,
                        "lep2": lep2cut,
                        "jetflav": sel_cjet_flav,
                    }
                    '''
                    for obj in obj_dict.keys():
                        output[f"{obj}_phi"].fill(
                            lepflav=ch,
                            region=r,
                            flav=flavor,
                            phi=normalize(flatten(obj_dict[obj].phi)),
                            weight=weights.weight(),
                        )
                        if "lep" in obj:
                            output[f"{obj}_eta"].fill(
                                lepflav="ee",
                                region=r,
                                flav=flavor[abs(obj_dict[obj].lep_flav) == 11],
                                eta=normalize(
                                    flatten(
                                        obj_dict[obj][
                                            abs(obj_dict[obj].lep_flav) == 11
                                        ].eta
                                    )
                                ),
                                weight=weights.weight()[
                                    abs(obj_dict[obj].lep_flav) == 11
                                ],
                            )
                            output[f"{obj}_eta"].fill(
                                lepflav="mumu",
                                region=r,
                                flav=flavor[abs(obj_dict[obj].lep_flav) == 13],
                                eta=normalize(
                                    flatten(
                                        obj_dict[obj][
                                            abs(obj_dict[obj].lep_flav) == 13
                                        ].eta
                                    )
                                ),
                                weight=weights.weight()[
                                    abs(obj_dict[obj].lep_flav) == 13
                                ],
                            )
                            output[f"{obj}_eta"].fill(
                                lepflav=ch,
                                region=r,
                                flav=flavor,
                                eta=normalize(flatten(obj_dict[obj].eta)),
                                weight=weights.weight(),
                            )
                            # output[f"{obj}_dxy"].fill(
                            #     lepflav=ch,
                            #     region=r,
                            #     flav=flavor,
                            #     dxy=normalize(flatten(obj_dict[obj].dxy)),
                            #     weight=weights.weight(),
                            # )
                            # output[f"{obj}_dz"].fill(
                            #     lepflav=ch,
                            #     region=r,
                            #     flav=flavor,
                            #     dz=normalize(flatten(obj_dict[obj].dz)),
                            #     weight=weights.weight(),
                            # )
                    '''

                cshape_syst = {
                    "scalevar_muF": "LHEScaleWeight_muF",
                    "scalevar_muR": "LHEScaleWeight_muR",
                    "UEPS_FSR": "PSWeightFSR",
                    "UEPS_ISR": "PSWeightISR",
                    "puweight": "PUWeight",
                    "JER": "jer",
                    "JES": "jesTotal",
                }
                
                for sys in systematics:
                    
                    #if not self.systematics["weights"] and sys != "nominal":
                    #    continue
                    #if 'JES' not in sys and 'nominal'!=sys: continue
                    #print(sys)
                    # if "DeepJetC" in sys and (
                    #     "LHEScaleWeight" in sys
                    #     or "PSWeight" in sys
                    #     or "PUWeight" in sys
                    #     or "jer" in sys
                    #     or "jesTotal" in sys
                    # ):
                    #     continue
                    syst_type = sys.replace("Up", "").replace("Down", "")
                    if syst_type in cshape_syst.keys():
                        if "JER" in sys or "JES" in sys:
                            weight = weights.weight(
                                modifier="DeepJetC_"
                                + cshape_syst[syst_type]
                                + sys.replace(syst_type, "")
                            )
                        else:
                            weight = weights.partial_weight(
                                exclude=["DeepJetC"], modifier=sys
                            ) * weights.partial_weight(
                                include=["DeepJetC"],
                                modifier="DeepJetC_"
                                + cshape_syst[syst_type]
                                + sys.replace(syst_type, ""),
                            )

                    elif sys in weights.variations:
                        weight = weights.weight(modifier=sys)
                    elif sys == "nocSF":
                        weight = weights.partial_weight(exclude=["DeepJetC"])

                    else:
                        weight = weights.weight()

                        ## variables
                        #if 'nominal' not in sys:continue
                    if 'nominal' in sys:
                        output["template_nsv"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        nsv=ak.values_astype(
                            normalize(flatten(ak.count(events[cut].SV.pt, axis=1))),
                            np.int8,
                        ),
                        weight=weights.weight(),
                    )
                        output[f"template_lep1_pt"].fill(
                        syst=sys,
                        lepflav="ee",
                        region=r,
                        flav=flavor[abs(lep1cut.lep_flav) == 11],
                        pt=normalize(
                            flatten(lep1cut[abs(lep1cut.lep_flav) == 11].pt)
                        ),
                        weight=weights.weight()[abs(lep1cut.lep_flav) == 11],
                    )

                        output[f"template_lep1_pt"].fill(
                        syst=sys,
                        lepflav="ee",
                        region=r,
                        flav=flavor[abs(lep1cut.lep_flav) == 11],
                        pt=normalize(
                            flatten(lep1cut[abs(lep1cut.lep_flav) == 11].pt)
                        ),
                        weight=weights.weight()[abs(lep1cut.lep_flav) == 11],
                    )
                        output[f"template_lep1_pt"].fill(
                        syst=sys,
                        lepflav="mumu",
                        region=r,
                        flav=flavor[abs(lep1cut.lep_flav) == 13],
                        pt=normalize(
                            flatten(lep1cut[abs(lep1cut.lep_flav) == 13].pt)
                        ),
                        weight=weights.weight()[abs(lep1cut.lep_flav) == 13],
                    )
                        output[f"template_lep2_pt"].fill(
                        syst=sys,
                        lepflav="ee",
                        region=r,
                        flav=flavor[abs(lep2cut.lep_flav) == 11],
                        pt=normalize(
                            flatten(lep1cut[abs(lep2cut.lep_flav) == 11].pt)
                        ),
                        weight=weights.weight()[abs(lep2cut.lep_flav) == 11],
                        )
                        output[f"template_lep2_pt"].fill(
                        syst=sys,
                        lepflav="mumu",
                        region=r,
                        flav=flavor[abs(lep2cut.lep_flav) == 13],
                        pt=normalize(
                            flatten(lep2cut[abs(lep2cut.lep_flav) == 13].pt)
                        ),
                        weight=weights.weight()[abs(lep2cut.lep_flav) == 13],
                    )
                        output[f"template_lep1_pt"].fill(
                        syst=sys,
                        lepflav="emu",
                        region=r,
                        flav=flavor,
                        pt=normalize(flatten(lep1cut.pt)),
                        weight=weight,
                    )
                        output[f"template_lep2_pt"].fill(
                        syst=sys,
                        lepflav="emu",
                        region=r,
                        flav=flavor,
                        pt=normalize(flatten(lep2cut.pt)),
                        weight=weight,
                    )

                        output["template_MET"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        pt=normalize(flatten(met[cut].pt)),
                        weight=weight,
                    )
                        output["template_jetpt"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        pt=normalize(flatten(sel_cjet_flav.pt)),
                        weight=weight,
                    )

                        output["template_ll_pt"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        pt=normalize(flatten(llcut.pt)),
                        weight=weight,
                    )
                        output["template_l1met_dphi"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        phi=lep1cut.delta_phi(met[cut]),
                        weight=weight,
                    )
                        output["template_l2met_dphi"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        phi=lep2cut.delta_phi(met[cut]),
                        weight=weights.weight(),
                    )
                        output["template_lll1_dr"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        dr=normalize(flatten(llcut.delta_r(lep1cut))),
                        weight=weight,
                    )
                        output["template_lll2_dr"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        dr=normalize(flatten(llcut.delta_r(lep2cut))),
                        weight=weight,
                        )
                        output["template_llc_dr"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        dr=normalize(flatten(llcut.delta_r(sel_cjet_flav))),
                        weight=weight,
                    )
                        output["template_ll_mass"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        mass=normalize(flatten(llcut.mass)),
                        weight=weight,
                    )
                        output["template_mTh"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        mt=normalize(flatten(mT(llcut, met[cut]))),
                        weight=weight,
                    )
                        output["template_mT1"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        mt=normalize(flatten(mT(lep1cut, met[cut]))),
                        weight=weight,
                    )
                        output["template_mT2"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        mt=normalize(flatten(mT(lep1cut, met[cut]))),
                        weight=weight,
                    )
                        output["template_cW1_dphi"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        phi=sel_cjet_flav.delta_phi(w1cut),
                        weight=weights.weight(),
                    )
                    output["template_jetflav_btagDeepFlavCvL"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        discr=sel_cjet_flav.btagDeepFlavCvL,
                        weight=weight,
                    )
                    output["template_jetflav_btagDeepFlavCvB"].fill(
                        syst=sys,
                        lepflav=ch,
                        region=r,
                        flav=flavor,
                        discr=sel_cjet_flav.btagDeepFlavCvB,
                        weight=weight,
                    )

                    if ch == "emu":
                        x = np.vstack(
                            [
                                normalize(flatten(llcut.pt)),
                                normalize(flatten(lep1cut.delta_r(llcut))),
                                normalize(flatten(lep2cut.delta_r(llcut))),
                                normalize(flatten(llcut.delta_r(sel_cjet_flav))),
                                normalize(flatten(lep1cut.pt)),
                                normalize(flatten(lep2cut.pt)),
                                normalize(flatten(llcut.mass)),
                                normalize(flatten(met[cut].pt)),
                                normalize(flatten(sel_cjet_flav.pt)),
                                normalize(flatten(lep1cut.delta_phi(met[cut]))),
                                normalize(flatten(lep2cut.delta_phi(met[cut]))),
                                normalize(flatten(sel_cjet_flav.delta_phi(w1cut))),
                                normalize(flatten(mT(lep1cut, met[cut]))),
                                normalize(flatten(mT(lep2cut, met[cut]))),
                                normalize(flatten(sel_cjet_flav.btagDeepFlavCvL)),
                                normalize(flatten(sel_cjet_flav.btagDeepFlavCvB)),
                                normalize(flatten(ak.count(events[cut].SV.pt, axis=1))),
                            ]
                        ).T
                        dmatrix_nosv = xgb.DMatrix(x[:, :-1])
                        dmatrix = xgb.DMatrix(x)
                        bdtscore = self.xgb_model_emu.predict(dmatrix)
                        bdtscore_higgs = self.xgb_model_higgs.predict(dmatrix)


                        if 'nominal'==sys:output["BDT_2D"].fill(
                            region=r,
                            syst=sys,
                            BDT_bkg=bdtscore,
                            BDT_H=bdtscore_higgs,
                            weight=weight,
                        )
                

                        # if self._export_array:
                        # output['array'][r]={}
                        '''

                        if 'b' not in r and 'SR2' in r :
                            output['array'][sys]={}
                            output["array"][sys][r]={}
                            

                            output["array"][r]["run"]=processor.column_accumulator(ak.to_numpy(normalize(events[cut].run)))
                            output["array"][r]["event"]=processor.column_accumulator(ak.to_numpy(normalize(events[cut].event)))
                            output["array"][r]["lumi"]=processor.column_accumulator(ak.to_numpy(normalize(events[cut].luminosityBlock)))
                            
                            output["array"][r]["lep1_charge"]=processor.column_accumulator(ak.to_numpy(normalize(lep1cut.charge)))
                            output["array"][r]["lep2_charge"]=processor.column_accumulator(ak.to_numpy(normalize(lep2cut.charge)))
                            if "GenPart" in events.fields:
                                output["array"][r]["electron_pdg"]=processor.column_accumulator(ak.to_numpy(normalize(events[cut].GenPart[ak.singletons(ele.genPartIdx)].pdgId)))
                                output["array"][r]["muon_pdg"]=processor.column_accumulator(ak.to_numpy(normalize(events[cut].GenPart[ak.singletons(mu.genPartIdx)].pdgId)))
                                output["array"][r]["electron_pdg_mom"]=processor.column_accumulator(ak.to_numpy(normalize(events[cut].GenPart[events[cut].GenPart[ak.singletons(ele.genPartIdx)].genPartIdxMother].pdgId)))
                                output["array"][r]["muon_pdg_mom"]=processor.column_accumulator(ak.to_numpy(normalize(events[cut].GenPart[events[cut].GenPart[ak.singletons(mu.genPartIdx)].genPartIdxMother].pdgId)))

                            
                            output['array'][sys][r]['jetflav_pt']=normalize(sel_cjet_flav.pt)
                            output['array'][sys][r]['MET_pt']=normalize(flatten(met[cut].pt))
                            output["array"][sys][r][
                                "BDT_bkg"
                            ] =                                 bdtscore
                            
                            output["array"][sys][r][
                                "BDT_higgs"
                            ] = bdtscore_higgs
                                
                            
                            output["array"][sys][r][
                                "weight"
                            ] = weight 
                               
                            output['array'][sys][r]['MET_phi']=normalize(flatten(met[cut].phi))
                            output['array'][sys][r]['jetflav_phi']=normalize(flatten(sel_cjet_flav.phi))
                            output['array'][sys][r]['jetflav_eta']=normalize(flatten(sel_cjet_flav.eta))
                            
                            output['array'][sys][r]['event']=ak.to_numpy(normalize(events[cut].run))
                        '''
                        if "HM" in r:
                            output[f"template_BDT_SR_HM"].fill(
                                syst=sys,
                                lepflav=ch,
                                region=r,
                                flav=flavor,
                                discr=normalize(flatten(bdtscore)),
                                weight=weight,
                            )

                        elif "SR_LM" == r:
                            score = self.cluster_SR.predict(
                                np.vstack(
                                    [np.array(bdtscore), np.array(bdtscore_higgs)]
                                ).T
                            )
                            output[f"template_BDT_SR_LM"].fill(
                                syst=sys,
                                lepflav=ch,
                                region=r,
                                flav=flavor,
                                discr=normalize(flatten(score)),
                                weight=weight,
                            )

                        # output[f"template_BDT_SR_LM_1D"].fill(
                        #     syst=sys,
                        #     lepflav=ch,
                        #     region=r,
                        #     flav=flavor,
                        #     discr=normalize(flatten(bdtscore)),
                        #     weight=weight,
                        # )

                        elif "SR2_LM" == r:
                            score = self.cluster_SR2.predict(
                                np.vstack(
                                    [np.array(bdtscore), np.array(bdtscore_higgs)]
                                ).T
                            )
                            output[f"template_BDT_SR2_LM"].fill(
                                syst=sys,
                                lepflav=ch,
                                region=r,
                                flav=flavor,
                                discr=normalize(flatten(score)),
                                weight=weight,
                            )
                        # output[f"template_BDT_SR2_LM_1D"].fill(
                        #     syst=sys,
                        #     lepflav=ch,
                        #     region=r,
                        #     flav=flavor,
                        #     discr=normalize(flatten(bdtscore)),
                        #     weight=weight,
                        # )
                        del dmatrix, bdtscore, bdtscore_higgs
        gc.collect()
        #np.save(f'{dataset.split("_")[0]}_{events.metadata["filename"][-10:-5]}_{shift_name}.npy')
        return {dataset: output}

    # @profile
    def postprocess(self, accumulator):
        return accumulator
