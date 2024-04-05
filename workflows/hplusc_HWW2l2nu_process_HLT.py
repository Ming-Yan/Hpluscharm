import pickle, os, sys, numpy as np
from coffea import processor
import awkward as ak
from  Hpluscharm.helpers.histogram import histogram
from coffea.analysis_tools import Weights,PackedSelection
from functools import partial
import os,psutil, copy, gc,coffea, hist
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
    # 
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
        # self._export_array = self.cfg.userconfig["export_array"]
        self.systematics = self.cfg.systematic        
        self._met_filters = met_filters[self.cfg.dataset["campaign"]]
        self._lumiMasks =  load_lumi(self.cfg.weights_config["lumiMask"])
        self.SF_map= load_SF(self.cfg.dataset["campaign"],self.cfg.weights_config,self.systematics["weights"])
        
        
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
    
    def process_shift(self, events,shift_name):
        print(shift_name)
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        output = {
            "hist_no":hist.Hist(
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_e", label=" $p_{T}^{e}$ [GeV]"),
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_mu", label=" $p_{T}^{\\mu}$ [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_e", label=" $|\\eta^{e}|$ [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_mu", label=" $|\\eta^{\\mu}|$ [GeV]"),
                hist.axis.Variable([0,10,15,20,25,30,40],name='nvtx',label="nVtx"),
               hist.axis.Integer(0,6,name='nj',label='# of jet'),
                hist.axis.StrCategory([], name="cut", growth=True),
                hist.storage.Weight()),
            "hist_MET":hist.Hist(
                
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_e", label=" $p_{T}^{e}$ [GeV]"),
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_mu", label=" $p_{T}^{\\mu}$ [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_e", label=" e [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_mu", label=" $|\\eta^{\\mu}|$ [GeV]"),
                hist.axis.Variable([0,10,15,20,25,30,40],name='nvtx',label="nVtx"),
                hist.axis.Integer(0,6,name='nj',label='# of jet'),
                hist.axis.StrCategory([], name="cut", growth=True),
                hist.storage.Weight()),
            "hist_all":hist.Hist(
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_e", label=" $p_{T}^{e}$ [GeV]"),
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_mu", label=" $p_{T}^{\\mu}$ [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_e", label=" $|\\eta^{e}|$ [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_mu", label=" $|\\eta^{\\mu}|$ [GeV]"),
                hist.axis.Variable([0,10,15,20,25,30,40],name='nvtx',label="nVtx"),
               hist.axis.Integer(0,6,name='nj',label='# of jet'),
                hist.axis.StrCategory([], name="cut", growth=True),
                hist.storage.Weight()),
            "hist_lep":hist.Hist(
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_e", label=" $p_{T}^{e}$ [GeV]"),
                hist.axis.Variable([12,16,20,25,30,35,45,60,80,100,200], name="pt_mu", label=" $p_{T}^{\\mu}$ [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_e", label=" $|\\eta^{e}|$ [GeV]"),
                hist.axis.Variable([0,0.5,1.0,1.442,1.566,2.5], name="abseta_mu", label=" $|\\eta^{\\mu}|$ [GeV]"),
                hist.axis.Variable([0,10,15,20,25,30,40],name='nvtx',label="nVtx"),
                hist.axis.Integer(0,6,name='nj',label='# of jet'),
                hist.axis.StrCategory([], name="cut", growth=True),
                hist.storage.Weight())
        }
        print("histo",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
        if shift_name is None:
            if isRealData:output["sumw"]=len(events)
            else:output["sumw"] = ak.sum(events.genWeight / abs(events.genWeight))
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
           req_lumi = self._lumiMasks(events.run, events.luminosityBlock)
        # selection.add("lumi", ak.to_numpy(req_lumi))
        # del req_lumi

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
        req_MET_HLT = np.zeros(len(events), dtype="bool")
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
        for t in self.cfg.preselections["METhlt"]:
            if t in events.HLT.fields:
                req_MET_HLT = req_MET_HLT | events.HLT[t]
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
        # selection.add("trigger_ee", ak.to_numpy(trigger_ele))
        # selection.add("trigger_mumu", ak.to_numpy(trigger_mu))
        # selection.add("trigger_emu", ak.to_numpy(trigger_em))
        
        del trigger_e, trigger_ee, trigger_m, trigger_mm
        metfilter = np.ones(len(events), dtype="bool")
        for flag in self._met_filters["data" if isRealData else "mc"]:
            metfilter &= np.array(events.Flag[flag])
        # selection.add("metfilter", metfilter)
        # del metfilter
        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon
        event_e = events.Electron
        if isRealData:
            if "Run2016B" in dataset or "Run2016C" in dataset or "Run2016D" in dataset or "Run2016E" in dataset or "Run2016F-HIPM" in dataset:
                mu_pt_matched_hlt= ((event_mu.pt > 13) & (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL == True ))|((event_mu.pt > 24) & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True))
                ele_pt_matched_hlt= ((event_e.pt > 25) & (events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL == True ))|(event_e.pt > 13) &((events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True))
            elif "Run2017B" in dataset or "Run2018":
                mu_pt_matched_hlt=((event_mu.pt > 13) & (events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ == True ))|((event_mu.pt > 24) & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ == True))
                ele_pt_matched_hlt=((event_e.pt > 25) & (events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ == True ))|((event_e.pt > 13) & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ == True))
            else:
                mu_pt_matched_hlt=((event_mu.pt > 13) & (events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
                == True ))|((event_mu.pt > 24) & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True))
                ele_pt_matched_hlt=((event_e.pt > 25) & (events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ == True ))|((event_e.pt > 13) & (events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL == True))
        else:
            mu12leg=((events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
                == True )|(events.HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ
                == True )|(events.HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL
                == True))
            mu23leg=((events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ
                == True )|(events.HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL
                == True ))
            mu_pt_matched_hlt=((event_mu.pt > 13)&mu12leg)|((event_mu.pt > 24) & mu23leg)
            ele_pt_matched_hlt=((event_e.pt > 25) & mu12leg)|((event_e.pt > 13) & mu23leg)

        
        musel = (
            # mu_pt_matched_hlt& 
            # e&
            (event_mu.pt>13) &
             (abs(event_mu.eta) < 2.4)
            & (event_mu.tightId==1)
            & (event_mu.pfRelIso04_all < 0.15)
            # & (abs(event_mu.sip3d)<4)
            & (abs(event_mu.dxy) < 0.05)
            & (abs(event_mu.dz) < 0.1)
        )
        
        event_mu = event_mu[musel]
        event_mu["lep_flav"] = 13 * event_mu.charge
        
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        
        elesel =  (event_e.pt>13)&((abs(event_e.eta) < 1.4442)|((abs(event_e.eta) > 1.566)&(abs(event_e.eta)<2.5))) & (event_e.mvaFall17V2Iso_WP90 == 1) & (abs(event_e.dxy) < 0.05) & (abs(event_e.dz) < 0.1)
        
        event_e = event_e[elesel]
        event_e["lep_flav"] = 11 * event_e.charge
        
        aele = events.Electron[
            (events.Electron.pt > 12)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.mvaFall17V2Iso_WPL == 1)
        ]
        amu = events.Muon[
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & (events.Muon.mvaId >= 1)
        ]
        

        good_leptons = ak.with_name(
            ak.concatenate([event_e, event_mu], axis=1),
            "PtEtaPhiMCandidate",
        )
        # del event_e, event_mu
        if ak.any((ak.sum(elesel,axis=-1)+(ak.sum(musel,axis=-1)))>=2,axis=-1):
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
                "lep1": np.where(leppair.lep1.pt>leppair.lep2.pt, leppair.lep1,leppair.lep2),
                "lep2": np.where(leppair.lep1.pt<leppair.lep2.pt, leppair.lep1,leppair.lep2),
                "pt": (leppair.lep1 + leppair.lep2).pt,
                "eta": (leppair.lep1 + leppair.lep2).eta,
                "phi": (leppair.lep1 + leppair.lep2).phi,
                "mass": (leppair.lep1 + leppair.lep2).mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        del leppair
        # print(ak.any(abs(ll_cand.lep1.lep_flav+ll_cand.lep2.lep_flav)!=2))
        
        
        ############

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
        print("obj",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
        
        lepton_cut=(ll_cand.lep1.pt > 25) &(ll_cand.lep2.pt > 13)& (ll_cand.lep1.charge + ll_cand.lep2.charge == 0)& (make_p4(ll_cand.lep1).delta_r(make_p4(ll_cand.lep2)) > 0.4) & (abs(ll_cand.lep1.lep_flav+ll_cand.lep2.lep_flav)==2)
        dilep_cut=(ll_cand.mass > 12)

        llpt_cut= (ll_cand.pt > 30)
       
        jetpu_bit= 1 if self._year=="2016" else 7
        jet_kin =(events.Jet.pt > 20) & (abs(events.Jet.eta) <= 2.4)
        jet_id =  ((events.Jet.puId ==jetpu_bit) | (events.Jet.pt > 50)) & (events.Jet.jetId > 1)
        jet_dr_cand1 = ak.all((events.Jet.metric_table(ll_cand.lep1) > 0.4),axis=2,mask_identity=True)
        jet_dr_cand2 =  ak.all(events.Jet.metric_table(ll_cand.lep2) > 0.4,axis=2,mask_identity=True)
        jet_dr_aele = ak.all(events.Jet.metric_table(aele) > 0.4, axis=2,mask_identity=True)
        jet_dr_amu = ak.all(events.Jet.metric_table(amu) > 0.4, axis=2,mask_identity=True)
        jet_sel = jet_kin & jet_id & jet_dr_cand1 & jet_dr_cand2 
        mtsel = ak.sum((mT(ll_cand.lep2, met) > 30),axis=-1)>0
        metsel =(events.MET.sumEt > 45)& (met.pt >20)
        metdphisel= abs(met.delta_phi(tkmet)) < 0.5
        
        
        cvbcutem = events.Jet.btagDeepFlavCvB >= 0.5
        cvlcutem = events.Jet.btagDeepFlavCvL >= 0.12
        cjetsel=cvbcutem&cvlcutem
        met_cut = 120 if self._year=="2016" else 100
        baseline =  req_lumi & metfilter & (met.pt>met_cut) & (ak.sum(lepton_cut & dilep_cut,axis=-1)>0)
        print(met_cut,ak.any(met[baseline].pt<120))
        # print(ak.flatten(ll_cand[lepton_cut & dilep_cut].lep1.lep_flav+ll_cand[baseline].lep2.lep_flav).tolist())
        # print(event_e.metric_table(event_mu)>0.4,ak.type())
        cuts = {
            "base" : baseline, 
            #"llptsel" : baseline & (ak.sum(llpt_cut,axis=-1)>0), 
            #"mtsel" : baseline & mtsel,
            #"metsel" : baseline & mtsel& metsel,
            #"metdphisel" : baseline & mtsel& metsel & metdphisel,
            #"jet" : baseline & mtsel& metsel & metdphisel & (ak.sum(jet_sel,axis=-1)>=1),
            #"cjet" : baseline & mtsel& metsel & metdphisel & (ak.sum(jet_sel&cjetsel,axis=-1)>=1),
            #"bljet" : baseline & mtsel& metsel & metdphisel & (ak.sum(jet_sel&~cjetsel,axis=-1)>=1)
        }
        ### Weights   
        print("before cut",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
        ll_cand=ll_cand[lepton_cut & dilep_cut]
        for r in cuts:
        
            cut = cuts[r]
            MET_HLT=req_MET_HLT[cut]
            lep_HLT=trigger_em[cut]
            if len(events[cut].event)==0 : continue
            ll_cands = ll_cand[cut]
            llcut = ll_cands[:, 0]
            lep1cut = llcut.lep1
            lep2cut = llcut.lep2
           
    
            
            ele = np.where(lep1cut.lep_flav==11,lep1cut,lep2cut)
            mu = np.where(lep1cut.lep_flav==13,lep1cut,lep2cut)
            weights = Weights(len(events[cut]), storeIndividual=True)
            if isRealData:weights.add("genweight", np.ones(len(events[cut])))
            else: 
                weights.add("genweight", events[cut].genWeight / abs(events[cut].genWeight))
                weights.add(
                    "L1prefireweight",
                    events[cut].L1PreFiringWeight.Nom,
                    events[cut].L1PreFiringWeight.Up,
                    events[cut].L1PreFiringWeight.Dn,
                )
                weights.add(
                    "puweight",
                    puwei(self.SF_map, events[cut].Pileup.nTrueInt),
                    puwei(self.SF_map, events[cut].Pileup.nTrueInt,"up"),
                    puwei(self.SF_map, events[cut].Pileup.nTrueInt,"down"),
                )
                
                if "PSWeight" in events.fields:
                    add_ps_weight(weights, events[cut].PSWeight)
                else:
                    add_ps_weight(weights, None)
                if "LHEPdfWeight" in events.fields:
                    add_pdf_weight(weights, events[cut].LHEPdfWeight)
                else:
                    add_pdf_weight(weights, None)

                # if "LHEScaleWeight" in events.fields and ak.all(ak.num(events[cut].LHEScaleWeight)>0,mask_identity=True):
                #     # add_scalevar_7pt(weights, events[cut].LHEScaleWeight)
                #     add_scalevar_3pt(weights, events[cut].LHEScaleWeight)
                # else:
                #     # add_scalevar_7pt(weights, [])
                #     add_scalevar_3pt(weights, [])
                if "TTTo" in dataset:
                    weights.add("ttbar_weight",
                    top_pT_reweighting(events[cut].GenPart),
                    (top_pT_reweighting(events[cut].GenPart)-ak.ones_like(top_pT_reweighting(events[cut].GenPart)))*2.+ak.ones_like(top_pT_reweighting(events[cut].GenPart)),
                    ak.ones_like(top_pT_reweighting(events[cut].GenPart)))
                # HLTSFs(ele,mu,self.SF_map,weights,syst=self.systematics["weights"])
                eleSFs(ele,self.SF_map,weights,syst=self.systematics["weights"])
                muSFs(mu,self.SF_map,weights,syst=self.systematics["weights"])                    
                # btagSFs(sel_cjet_flav,self.SF_map,weights,"DeepJetC",syst=self.systematics["weights"])
                # jmar_sf(sel_cjet_flav,self.SF_map,weights,syst=self.systematics["weights"])

            

           
            weight = weights.weight()
            
            
            output['hist_MET'].fill(
            np.minimum(ele.pt[MET_HLT],200),
            np.minimum(mu.pt[MET_HLT],200),
            abs(ele.eta[MET_HLT]),
            abs(mu.eta[MET_HLT]),
            np.clip(0,40,events[cut&req_MET_HLT].PV.npvs),
            np.minimum(ak.sum(jet_sel[cut&req_MET_HLT],axis=-1),6),
            r,
            weight=weight[MET_HLT]
            )
            output['hist_no'].fill(
            np.minimum(ele.pt,200),
            np.minimum(mu.pt,200),
            abs(ele.eta),
            abs(mu.eta),
            np.clip(0,40,events[cut].PV.npvs),
            np.minimum(ak.sum(jet_sel[cut],axis=-1),6),
            r,
            weight=weight
            )
            output['hist_lep'].fill(
            np.minimum(ele[lep_HLT].pt,200),
            np.minimum(mu[lep_HLT].pt,200),
            abs(ele.eta[lep_HLT]),
            abs(mu.eta[lep_HLT]),
            np.clip(0,40,events[cut&trigger_em].PV.npvs),
            np.minimum(ak.sum(jet_sel[cut&trigger_em],axis=-1),6),
            r,
            weight=weight[lep_HLT]
        )
            
            output['hist_all'].fill(
            np.minimum(ele[lep_HLT&MET_HLT].pt,200),
            np.minimum(mu[lep_HLT&MET_HLT].pt,200),
            abs(ele.eta[lep_HLT&MET_HLT]),
            abs(mu.eta[lep_HLT&MET_HLT]),
            np.clip(0,40,events[cut&trigger_em&req_MET_HLT].PV.npvs),
            np.minimum(ak.sum(jet_sel[cut&trigger_em&req_MET_HLT],axis=-1),6),
            r,
            weight=weight[lep_HLT&MET_HLT]
        )
        
        # del weight, ele,mu, weight, weights    
        gc.collect()

        return {dataset:output}
    # @profile
    def postprocess(self, accumulator):
        return accumulator
