import pickle, os, sys, numpy as np
from coffea import hist, processor
import awkward as ak
import hist as Hist
from coffea.analysis_tools import Weights
from functools import partial
import xgboost as xgb
import gc

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

from Hpluscharm.config.HWW2l2nu_config import (
    HLTmenu,
    correction_config,
)
from Hpluscharm.config.histogrammer import create_hist
# from Hpluscharm.utils.top_mass import get_nu4vec

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
    def __init__(
        self, year="2017", campaign="UL17", BDTversion="dymore", export_array=False, systematics= True
    ):
        self._year = year
        self._export_array = export_array
        self.systematics = systematics
        self._mu1hlt = HLTmenu["mu1hlt"][campaign]
        self._mu2hlt = HLTmenu["mu2hlt"][campaign]
        self._e1hlt = HLTmenu["e1hlt"][campaign]
        self._e2hlt = HLTmenu["e2hlt"][campaign]
        self._emuhlt = HLTmenu["emuhlt"][campaign]
        self._met_filters = met_filters[campaign]
        self._lumiMasks = lumiMasks[campaign]
        self._campaign = campaign
        self._deepjetc_sf = load_BTV(
            self._campaign, correction_config[self._campaign]["BTV"], "DeepJetC"
        )
        self._pu = load_pu(self._campaign, correction_config[self._campaign]["PU"])
        self._jet_factory = load_jetfactory(
            self._campaign, correction_config[self._campaign]["JME"]
        )
        self._met_factory = load_metfactory(
            self._campaign, correction_config[self._campaign]["JME"]
         )
        if self._year == "2016":from Hpluscharm.MVA.training_config import config2016 as config
        if self._year == "2017":from Hpluscharm.MVA.training_config import config2017 as config
        if self._year == "2018":from Hpluscharm.MVA.training_config import config2018 as config
        self.xgb_model_ll = xgb.Booster()
        self.xgb_model_ll.load_model(f"src/Hpluscharm/MVA/{config['input_json']['ll'][BDTversion]}")
        self.xgb_model_emu = xgb.Booster()
        self.xgb_model_emu.load_model(f"src/Hpluscharm/MVA/{config['input_json']['emu'][BDTversion]}") 
        _hist_event_dict = create_hist()
        
        if self._export_array:
            _hist_event_dict[arrayname] = processor.defaultdict_accumulator(
                defaultdict_accumulator
            )
            if self.systematics:
                _hist_event_dict["array_JESUp"] = processor.defaultdict_accumulator(
                defaultdict_accumulator
            )
                _hist_event_dict["array_JESDown"] = processor.defaultdict_accumulator(
                defaultdict_accumulator
            )
                _hist_event_dict["array_JERUp"] = processor.defaultdict_accumulator(
                defaultdict_accumulator
            )
                _hist_event_dict["array_JERDown"] = processor.defaultdict_accumulator(
                defaultdict_accumulator
            )
                _hist_event_dict["array_UESUp"] = processor.defaultdict_accumulator(
                defaultdict_accumulator
            )
                _hist_event_dict["array_UESDown"] = processor.defaultdict_accumulator(
                defaultdict_accumulator
            )
               
        
        # self._accumulator = processor.dict_accumulator({
        #         **_hist_event_dict,
        #         "cutflow": processor.defaultdict_accumulator(
        #             #         # we don't use a lambda function to avoid pickle issues
        #             partial(processor.defaultdict_accumulator, int)
        #         ),
        #         'sumw': processor.defaultdict_accumulator(float),
        #     })
        self.make_output = lambda:{
              **_hist_event_dict,
                "cutflow": processor.defaultdict_accumulator(
                    #         # we don't use a lambda function to avoid pickle issues
                    partial(processor.defaultdict_accumulator, int)
                ),
                'sumw': processor.defaultdict_accumulator(float),
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
            jets = self._jet_factory[jetfac_name].build(
            add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), lazy_cache=events.caches[0]
        )
            met = self._met_factory.build(events.MET, jets, {})
        # jets = events.Jet
        # met = events.MET
        
        shifts = [
            ({"Jet": jets, "MET": met}, None),
        ]
        
        if not isRealData:
            if self.systematics:
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
        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events,shift_name):
        print(shift_name)
        # shift_name=None
        output = self.make_output()
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = processor.PackedSelection()
        if isRealData:
            output["sumw"][dataset] += len(events)
        else:
            output["sumw"][dataset] += ak.sum(events.genWeight / abs(events.genWeight))
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self._lumiMasks(events.run, events.luminosityBlock)
        selection.add("lumi", ak.to_numpy(req_lumi))
        del req_lumi

        #############Selections

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
        del event_e, event_mu
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
                "lep1": leppair.lep1,
                "lep2": leppair.lep2,
                "pt": (leppair.lep1 + leppair.lep2).pt,
                "eta": (leppair.lep1 + leppair.lep2).eta,
                "phi": (leppair.lep1 + leppair.lep2).phi,
                "mass": (leppair.lep1 + leppair.lep2).mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        ll_cand = ak.pad_none(ll_cand, 1, axis=1)
        if ak.count(ll_cand.pt) > 0:
            ll_cand = ll_cand[ak.argsort(ll_cand.pt, axis=1, ascending=False)]

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
        
                
        
        eventflav_jet = events.Jet[ak.argsort(events.Jet.btagDeepFlavCvL, axis=1, ascending=False)]
        jetsel = (
            (eventflav_jet.pt > 20)
            & (abs(eventflav_jet.eta) <= 2.4)
            & ((eventflav_jet.puId > 6) | (eventflav_jet.pt > 50))
            & (eventflav_jet.jetId > 5)
            & ak.all(
                (eventflav_jet.metric_table(ll_cand.lep1) > 0.4)
                & (eventflav_jet.metric_table(ll_cand.lep2) > 0.4),
                axis=2,
            )
            & ak.all(eventflav_jet.metric_table(aele) > 0.4, axis=2)
            & ak.all(eventflav_jet.metric_table(amu) > 0.4, axis=2)
        )
        njet = ak.sum(jetsel, axis=1)
        topjetsel = (
            (eventflav_jet.pt > 20)
            & (abs(eventflav_jet.eta) <= 2.4)
            & ((eventflav_jet.puId > 6) | (eventflav_jet.pt > 50))
            & (eventflav_jet.jetId > 5)
            & (eventflav_jet.btagDeepFlavB > 0.0532)
        )

        cvbcutll = eventflav_jet.btagDeepFlavCvB >= 0.42
        cvlcutll = eventflav_jet.btagDeepFlavCvL >= 0.22
        cvbcutem = eventflav_jet.btagDeepFlavCvB >= 0.5
        cvlcutem = eventflav_jet.btagDeepFlavCvL >= 0.12

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
        # req_WW_cr = ak.any((mT(ll_cand.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) & (ll_cand.mass),axis=-1)
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
        # shift_name = None
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
                & (nele == 2)
                & (nmu == 0)
            )
            output["cutflow"][dataset]["all mumu"] += ak.sum(
                ak.any(global_cut & sr_cut, axis=-1)
                & (njet > 0)
                & (ak.all(llmass_cut) & trigger_mu)
                & (nmu == 2)
                & (nele == 0) 
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

        # selection.add('DY_CRb',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&cvlcut&~cvbcut,axis=1)==1)))
        # selection.add('DY_CRl',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&~cvlcut&cvbcut,axis=1)>=1)))
        # selection.add('DY_CRc',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&cvlcut&cvbcut,axis=1)==1)))

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
        ### Weights
        weights = Weights(len(events), storeIndividual=True)
        if isRealData:
            weights.add("genweight", np.ones(len(events)))
            output["cutflow"][dataset]["all"] += len(events)
        else:
            output["cutflow"][dataset]["all"] += ak.sum(
                events.genWeight / abs(events.genWeight)
            )
            weights.add("genweight", events.genWeight / abs(events.genWeight))
            weights.add(
                "L1prefireweight",
                events.L1PreFiringWeight.Nom,
                events.L1PreFiringWeight.Up,
                events.L1PreFiringWeight.Dn,
            )
            weights.add(
                "puweight",
                self._pu["PU"](events.Pileup.nPU),
                self._pu["PUup"](events.Pileup.nPU),
                self._pu["PUdn"](events.Pileup.nPU),
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
                sel_jetflav = ak.mask(eventflav_jet, mask_jet[ch])
                sel_cjet_flav = sel_jetflav
                if ak.count(sel_cjet_flav.pt) > 0:
                    sel_cjet_flav = sel_cjet_flav[
                        ak.argsort(
                            sel_cjet_flav.btagDeepFlavCvB, axis=1, ascending=False
                        )
                    ]
                nseljet = ak.count(sel_cjet_flav.pt, axis=1)
                topjets = ak.mask(eventflav_jet, topjetsel)
                if ak.count(topjets.pt) > 0:
                    topjets = topjets[
                        ak.argsort(topjets.btagDeepFlavCvB, axis=1)
                    ]
                topjets  = ak.pad_none(topjets,2,axis=1)
                
                sel_cjet_flav = ak.pad_none(sel_cjet_flav, 1, axis=1)
                sel_cjet_flav = sel_cjet_flav[:, 0]
                # llcut = ll_cands
                ll_cands = ak.pad_none(ll_cands, 1, axis=1)

                llcut = ll_cands[:, 0]
                lep1cut = llcut.lep1
                lep2cut = llcut.lep2
                w1cut = lep1cut + met
                w2cut = lep2cut + met
                hcut = llcut + met
                utv = -met - (lep1cut + lep2cut)
                utv_p = np.sqrt(utv.px**2 + utv.py**2 + utv.pz**2)
                topjet1 = lep1cut.nearest(topjets)
                topjet2 = lep2cut.nearest(topjets)
                
                if not isRealData:
                    add_eleSFs(
                        lep1cut,
                        self._campaign,
                        correction_config[self._campaign]["LSF"],
                        weights,
                        cut,
                    )
                    add_eleSFs(
                        lep2cut,
                        self._campaign,
                        correction_config[self._campaign]["LSF"],
                        weights,
                        cut,
                    )
                    add_muSFs(
                        lep1cut,
                        self._campaign,
                        correction_config[self._campaign]["LSF"],
                        weights,
                        cut,
                    )
                    add_muSFs(
                        lep2cut,
                        self._campaign,
                        correction_config[self._campaign]["LSF"],
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
                        systematics = [None] + list(weights.variations)
                    else:
                        systematics = [shift_name]
                
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
                utv = utv[cut]
                utv_p = utv_p[cut]
                
                # topjet1 = topjet1[]
                topjet1cut = topjet1[cut]
                topjet2cut = topjet2[cut]
                # ttbarcut= ak.zip(
                #     {
                #         "pt": topjet1cut.pt+topjet2cut.pt+lep1cut.pt+lep2cut.pt+met[cut].pt,
                #         "eta": topjet1cut.eta+topjet2cut.eta+lep1cut.eta+lep2cut.eta,
                #         "phi": topjet1cut.phi+topjet2cut.phi+lep1cut.phi+lep2cut.phi+met[cut].phi,
                #         "energy": topjet1cut.energy+topjet2cut.energy+lep1cut.energy+lep2cut.energy+met[cut].energy,
                #     },
                #     with_name="PtEtaPhiELorentzVector",
                # )
                ttbarcut=topjet1cut+topjet2cut+lep1cut+lep2cut+met[cut]
                ttbarcut = ttbarcut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                topjet1cut = topjet1cut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                topjet2cut = topjet2cut[(ak.count(topjets[cut].pt,axis=1) > 1)]
                # ttbarcut = topjet1cut+topjet2cut+lep1cut+lep2cut+met[cut]

                # neu1 = get_nu4vec(lep1cut,met[cut])
                # neu2 = get_nu4vec(lep2cut,met[cut])
                # top1cut=topjet1+lep1cut+neu1
                # top2cut=topjet2+lep2cut+neu2
                # nw1cut=lep1cut+neu1
                # nw2cut=lep2cut+neu2
                sel_cjet_flav = sel_cjet_flav[cut]

                if shift_name is None: arrayname="array"
                else : arrayname="array_"+shift_name
                for histname, h in output.items():
                    if "ll_" not in histname and "jetflav_" not in histname and "lep1_" not in histname and "lep2_" not in histname: continue
                    if "jetflav_" in histname:
                        haxis = histname.replace("jetflav_", "")
                        h.fill(ch,r,flavor,normalize(flatten(sel_cjet_flav[histname.replace("jetflav_", "")])),weight=weights.weight()[cut])
                        if self._export_array and arrayname=="array":
                            output[arrayname][dataset][
                                histname
                            ] += processor.column_accumulator(
                                ak.to_numpy(
                                    normalize(
                                        sel_cjet_flav[histname.replace("jetflav_", "")]
                                    )
                                )
                            )
                    elif "lep1_" in histname:
                        h.fill(ch,r,flavor,normalize(flatten(lep1cut[histname.replace("lep1_", "")])),weight=weights.weight()[cut])
                        if self._export_array and arrayname=="array":
                            output[arrayname][dataset][
                                histname
                            ] += processor.column_accumulator(
                                ak.to_numpy(
                                    normalize(
                                        flatten(lep1cut[histname.replace("lep1_", "")])
                                    )
                                )
                            )
                    elif "lep2_" in histname:
                        h.fill(ch,r,flavor,normalize(flatten(lep2cut[histname.replace("lep2_", "")])),weight=weights.weight()[cut])
                        if self._export_array and arrayname=="array":
                            output[arrayname][dataset][
                                histname
                            ] += processor.column_accumulator(
                                ak.to_numpy(
                                    normalize(
                                        flatten(lep2cut[histname.replace("lep2_", "")])
                                    )
                                )
                            )

                    
                    elif "ll_" in histname and "template" not in histname:
                        h.fill(ch,r,flavor,normalize(flatten(llcut[histname.replace("ll_", "")])),weight=weights.weight()[cut])
                        if self._export_array and arrayname=="array": 
                            output[arrayname][dataset][
                                histname
                            ] += processor.column_accumulator(
                                ak.to_numpy(
                                    normalize(
                                        flatten(llcut[histname.replace("ll_", "")])
                                    )
                                )
                            )
                    elif 'topjet1_' in histname :
                        
                        h.fill(ch,r,flavor,normalize(flatten(topjet1cut[histname.replace("topjet1_", "")])),weight=weights.weight()[cut])
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(topjet1cut[histname.replace('topjet1_','')]))))
                    elif 'topjet2_' in histname :
                        h.fill(ch,r,flavor,normalize(flatten(topjet2cut[histname.replace("topjet2_", "")])),weight=weights.weight()[cut])
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(topjet2cut[histname.replace('topjet2_','')]))))
                    elif 'ttbar_' in histname:
                        h.fill(ch,r,flavor,normalize(flatten(ttbarcut[histname.replace("ttbar_", "")])),weight=weights.weight()[cut])
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(ttbarcut[histname.replace('ttbar_','')]))))
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
                
                if self.systematics:
                    for sys in systematics:
                        if sys in weights.variations:
                            weight = weights.weight(modifier=sys)[cut]
                        else:
                            weight = weights.weight()[cut]
                        if sys is None: sys="nominal"
                        if "DY_CR" in r:output['template_ll_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,mass=normalize(flatten(llcut.mass)),weight=weight)
                        
                        
                        if "top_CR" in r :
                            # if ak.count(topjets[cut].pt) > 1:
                                # print(ttbarcut.mass,weight,(ak.count(topjets[cut].pt) > 1))
                            #tweight = weight[ak.count(topjets[cut].pt,axis=1) > 1]
                            output['template_tt_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor[(ak.count(topjets[cut].pt,axis=1) > 1)],mass=normalize(flatten(ttbarcut.mass)),weight=weight[ak.count(topjets[cut].pt,axis=1) > 1])
                                

                            output['template_mTh'].fill(syst=sys,
lepflav=ch,region=r,flav=flavor,mt=normalize(flatten(mT(llcut, met[cut]))),weight=weight)
                            output['template_mT1'].fill(syst=sys,
lepflav=ch,region=r,flav=flavor,mt=normalize(flatten(mT(lep1cut, met[cut]))),weight=weight)
                        #output['template_top1_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,mass=normalize(flatten(top1cut.mass)),weight=weight)
                        #output['template_top2_mass'].fill(syst=sys,lepflav=ch,region=r,flav=flavor,mass=normalize(flatten(top2cut.mass)),weight=weight)
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
                            output['template_BDT'].fill(syst=sys,
lepflav=ch,region=r,flav=flavor,BDT=normalize(flatten(bdtscore)),weight=weight)
                if "SR" in r:
                    
                    output["npvs"].fill(
                        
                        
lepflav=ch,
                        region=r,
                        flav=flavor,
                        npvs=flatten(events[cut].PV.npvs),
                        weight=weights.weight()[cut],
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
                    output[arrayname][dataset]["weight"] += processor.column_accumulator(
                        ak.to_numpy(normalize(weights.weight()[cut]))
                    )
                    
                    output[arrayname][dataset]["lepwei"] += processor.column_accumulator(
                        ak.to_numpy(normalize(sf))
                    )
                    output[arrayname][dataset]["puwei"] += processor.column_accumulator(
                        ak.to_numpy(
                            normalize(weights.partial_weight(include=["puweight"])[cut])
                        )
                    )
                    output[arrayname][dataset]["l1wei"] += processor.column_accumulator(
                        ak.to_numpy(
                            normalize(
                                weights.partial_weight(include=["L1prefireweight"])[cut]
                            )
                        )
                    )
                    output[arrayname][dataset]["event"] += processor.column_accumulator(
                        ak.to_numpy(normalize(events[cut].event))
                    )
                    output[arrayname][dataset]["run"] += processor.column_accumulator(
                        ak.to_numpy(normalize(events[cut].run))
                    )
                    output[arrayname][dataset]["lumi"] += processor.column_accumulator(
                        ak.to_numpy(normalize(events[cut].luminosityBlock))
                    )
                    output[arrayname][dataset]["lepflav"] += processor.column_accumulator(
                        np.full_like(ak.to_numpy(normalize(nseljet)), ch, dtype="<U6")
                    )
                    output[arrayname][dataset]["jetflav"] += processor.column_accumulator(
                        ak.to_numpy(normalize(flavor))
                    )
                    output[arrayname][dataset]["region"] += processor.column_accumulator(
                        np.full_like(ak.to_numpy(normalize(nseljet)), r, dtype="<U6")
                    )
                    output[arrayname][dataset]["nj"] += processor.column_accumulator(
                        ak.to_numpy(normalize(nseljet))
                    )
                    output[arrayname][dataset]["nele"] += processor.column_accumulator(
                        ak.to_numpy(normalize(naele - nele, cut))
                    )
                    output[arrayname][dataset]["nmu"] += processor.column_accumulator(
                        ak.to_numpy(normalize(namu - nmu, cut))
                    )
                    output[arrayname][dataset][
                        "METTkMETdphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(met[cut].delta_phi(tkmet[cut])))
                    )
                    output[arrayname][dataset]["njmet"] += processor.column_accumulator(
                        ak.to_numpy(
                            normalize(
                                ak.sum(
                                    (met[cut].delta_phi(sel_jetflav[cut]) < 0.5), axis=1
                                )
                            )
                        )
                    )
                    output[arrayname][dataset]["mT1"] += processor.column_accumulator(
                        ak.to_numpy(flatten(mT(lep1cut, met[cut])))
                    )
                    output[arrayname][dataset]["mT2"] += processor.column_accumulator(
                        ak.to_numpy(flatten(mT(lep2cut, met[cut])))
                    )
                    output[arrayname][dataset]["mTh"] += processor.column_accumulator(
                        ak.to_numpy(flatten(mT(llcut, met[cut])))
                    )
                    output[arrayname][dataset]["npvs"] += processor.column_accumulator(
                        ak.to_numpy(flatten(events[cut].PV.npvs))
                    )

                    output[arrayname][dataset]["nsv"] += processor.column_accumulator(
                        ak.to_numpy(flatten(ak.count(events[cut].SV.ntracks, axis=1)))
                    )

                    output[arrayname][dataset]["u_par"] += processor.column_accumulator(
                        ak.to_numpy(
                            flatten(utv_p * np.cos(utv.delta_phi(lep1cut + lep2cut)))
                        )
                    )
                    output[arrayname][dataset]["u_per"] += processor.column_accumulator(
                        ak.to_numpy(
                            flatten(utv_p * np.sin(utv.delta_phi(lep1cut + lep2cut)))
                        )
                    )
                    output[arrayname][dataset][
                        "tkMET_pt"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(tkmet[cut].pt))
                    )
                    output[arrayname][dataset][
                        "PuppiMET_pt"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(events[cut].PuppiMET.pt))
                    )
                    output[arrayname][dataset][
                        "MET_proj"
                    ] += processor.column_accumulator(
                        ak.to_numpy(
                            flatten(
                                np.where(
                                    dphilmet(lep1cut, lep2cut, met[cut]) < np.pi / 2.0,
                                    np.sin(dphilmet(lep1cut, lep2cut, met[cut]))
                                    * met[cut].pt,
                                    met[cut].pt,
                                )
                            )
                        )
                    )
                    output[arrayname][dataset][
                        "TkMET_proj"
                    ] += processor.column_accumulator(
                        ak.to_numpy(
                            flatten(
                                np.where(
                                    dphilmet(lep1cut, lep2cut, tkmet[cut])
                                    < np.pi / 2.0,
                                    np.sin(dphilmet(lep1cut, lep2cut, tkmet[cut]))
                                    * tkmet[cut].pt,
                                    tkmet[cut].pt,
                                )
                            )
                        )
                    )
                    output[arrayname][dataset][
                        "minMET_proj"
                    ] += processor.column_accumulator(
                        ak.to_numpy(
                            flatten(
                                np.where(
                                    np.where(
                                        dphilmet(lep1cut, lep2cut, met[cut])
                                        < np.pi / 2.0,
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
                                        dphilmet(lep1cut, lep2cut, met[cut])
                                        < np.pi / 2.0,
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
                            )
                        )
                    )
                    output[arrayname][dataset][
                        "MET_ptdivet"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(met[cut].pt / np.sqrt(met[cut].energy)))
                    )
                    output[arrayname][dataset]["h_pt"] += processor.column_accumulator(
                        ak.to_numpy(flatten(hcut.pt))
                    )
                    output[arrayname][dataset]["MET_pt"] += processor.column_accumulator(
                        ak.to_numpy(flatten(met[cut].pt))
                    )
                    output[arrayname][dataset]["MET_phi"] += processor.column_accumulator(
                        ak.to_numpy(flatten(met[cut].pt))
                    )
                    output[arrayname][dataset]["l1l2_dr"] += processor.column_accumulator(
                        ak.to_numpy(flatten(make_p4(lep1cut).delta_r(make_p4(lep2cut))))
                    )
                    output[arrayname][dataset][
                        "l1met_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(lep1cut.delta_phi(met[cut])))
                    )
                    output[arrayname][dataset][
                        "l2met_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(lep2cut.delta_phi(met[cut])))
                    )
                    output[arrayname][dataset][
                        "llmet_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(llcut.delta_phi(met[cut])))
                    )
                    output[arrayname][dataset]["l1c_dr"] += processor.column_accumulator(
                        ak.to_numpy(
                            normalize(make_p4(lep1cut).delta_r(make_p4(sel_cjet_flav)))
                        )
                    )
                    output[arrayname][dataset][
                        "cmet_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(normalize(met[cut].delta_phi(sel_cjet_flav)))
                    )
                    output[arrayname][dataset]["l2c_dr"] += processor.column_accumulator(
                        ak.to_numpy(
                            normalize(make_p4(lep2cut).delta_r(make_p4(sel_cjet_flav)))
                        )
                    )
                    output[arrayname][dataset][
                        "l2W1_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(lep2cut.delta_phi((w1cut))))
                    )
                    output[arrayname][dataset][
                        "metW1_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(met[cut].delta_phi((w1cut))))
                    )
                    output[arrayname][dataset][
                        "cW1_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(normalize((w1cut).delta_phi(sel_cjet_flav)))
                    )
                    output[arrayname][dataset][
                        "l1W2_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(lep1cut.delta_phi((w2cut))))
                    )
                    output[arrayname][dataset]["llc_dr"] += processor.column_accumulator(
                        ak.to_numpy(
                            normalize(make_p4(llcut).delta_r(make_p4(sel_cjet_flav)))
                        )
                    )
                    output[arrayname][dataset][
                        "llW1_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(w1cut.delta_phi((llcut))))
                    )
                    output[arrayname][dataset][
                        "llW2_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(w2cut.delta_phi((llcut))))
                    )
                    output[arrayname][dataset][
                        "llh_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(hcut.delta_phi((llcut))))
                    )
                    output[arrayname][dataset][
                        "l2W2_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(lep2cut.delta_phi((w2cut))))
                    )
                    output[arrayname][dataset][
                        "metW2_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(met[cut].delta_phi((w2cut))))
                    )
                    output[arrayname][dataset][
                        "cW2_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(normalize((w2cut).delta_phi(sel_cjet_flav)))
                    )
                    output[arrayname][dataset][
                        "l1h_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten((lep1cut).delta_phi((hcut))))
                    )
                    output[arrayname][dataset][
                        "meth_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten((met[cut]).delta_phi((hcut))))
                    )
                    output[arrayname][dataset]["ch_dphi"] += processor.column_accumulator(
                        ak.to_numpy(normalize(hcut.delta_phi(sel_cjet_flav)))
                    )
                    output[arrayname][dataset][
                        "W1h_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten((w1cut).delta_phi((hcut))))
                    )
                    output[arrayname][dataset][
                        "W2h_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten((w2cut).delta_phi((hcut))))
                    )
                    output[arrayname][dataset]["lll1_dr"] += processor.column_accumulator(
                        ak.to_numpy(flatten(make_p4(lep1cut).delta_r(make_p4(llcut))))
                    )
                    output[arrayname][dataset]["lll2_dr"] += processor.column_accumulator(
                        ak.to_numpy(flatten(make_p4(lep2cut).delta_r(make_p4(llcut))))
                    )
                    output[arrayname][dataset][
                        "l1W1_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten(lep1cut.delta_phi((w1cut))))
                    )
                    output[arrayname][dataset][
                        "W1W2_dphi"
                    ] += processor.column_accumulator(
                        
                        ak.to_numpy(flatten((w1cut).delta_phi((w2cut))))
                    )
                    output[arrayname][dataset][
                        "l2h_dphi"
                    ] += processor.column_accumulator(
                        ak.to_numpy(flatten((lep2cut).delta_phi((hcut))))
                    )

        del leppair, ll_cand, sel_cjet_flav, met, tkmet,cut,ll_cands,lep1cut,lep2cut,llcut,hcut,w1cut,w2cut,ttbarcut,naele,namu,topjet1,topjet2,utv,utv_p,sr_cut,top_cr2_cut,dy_cr2_cut,cvbcutll,cvbcutem,cvlcutem,cvlcutll
        if not isRealData:del jetsf,jetsf_dn,jetsf_up,weight
        gc.collect()
        
        return {dataset:output}

    def postprocess(self, accumulator):
        return accumulator
