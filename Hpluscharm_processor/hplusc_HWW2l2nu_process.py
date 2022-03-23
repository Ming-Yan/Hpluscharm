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
from utils.correction import jec,muSFs,eleSFs,init_corr
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights
from functools import partial
# import numba
from helpers.util import reduce_and, reduce_or, nano_mask_or, get_ht, normalize, make_p4

def mT(obj1,obj2):
    return np.sqrt(2.*obj1.pt*obj2.pt*(1.-np.cos(obj1.phi-obj2.phi)))
def flatten(ar): # flatten awkward into a 1d array to hist
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
    def __init__(self, year="2017"):    
        self._year=year
       
        self._mu1hlt = {
            '2016': [
                'IsoTkMu24'
            ],
            '2017': [
                'IsoMu27'
            ],
            '2018': [
                'IsoMu24'
            ],
        }   
        self._mu2hlt = {
            '2016': [
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',#runH
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ',#runH
            ],
            '2017': [
                
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8',
            ],
            '2018': [
                
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
            ],
        }   
        self._e1hlt = {
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele25_eta2p1_WPTight_Gsf'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
            ],
        }   
        self._e2hlt = {
            '2016': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
            ],
            '2017': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
            ],
            '2018': [
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
            ],
        }    
        self._emuhlt =  {
            '2016': [
                'Mu8_TrkIsoVVL_Elle23_CaloIdL_TrackIdL_IsoVL',
                'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                'Mu8_TrkIsoVVL_Elle23_CaloIdL_TrackIdL_IsoVL_DZ',
                'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
            ],
            '2017': [
                'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                'Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL',
                'Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',    
            ],
            '2018': [
               'Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',  
            ],
        }   
        self._met_filters = {
            '2016': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'BadPFMuonDzFilter',
                    'eeBadScFilter',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'BadPFMuonDzFilter',
                    'eeBadScFilter',
                ],
            },
            '2017': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'BadPFMuonDzFilter',
                    'hfNoisyHitsFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilter',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'BadPFMuonDzFilter',
                    'hfNoisyHitsFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilter',
                ],
            },
            '2018': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'BadPFMuonDzFilter',
                    'hfNoisyHitsFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilter',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'BadPFMuonDzFilter',
                    'hfNoisyHitsFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilter',
                ],
            },
        }
        self._lumiMasks = {
    '2016': LumiMask('data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
    '2017': LumiMask('data/Lumimask/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
    '2018': LumiMask('data/Lumimask/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt')
}
        
        self._corr = init_corr(year)
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        lepflav_axis = hist.Cat("lepflav",['ee','mumu','emu'])
        region_axis = hist.Cat("region",['SR','top_CR','DY_CR'])
        # Events
        njet_axis  = hist.Bin("nj",  r"N jets",      [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_axis = hist.Bin("nbj", r"N b-jets",    [0,1,2,3,4,5,6,7,8,9,10])            
        ncjet_axis = hist.Bin("nbj", r"N b-jets",    [0,1,2,3,4,5,6,7,8,9,10])  
        # kinematic variables       
        pt_axis   = hist.Bin("pt",   r" $p_{T}$ [GeV]", 50, 0, 300)
        eta_axis  = hist.Bin("eta",  r" $\eta$", 25, -2.5, 2.5)
        phi_axis  = hist.Bin("phi",  r" $\phi$", 30, -3, 3)
        mass_axis = hist.Bin("mass", r" $m$ [GeV]", 50, 0, 300)
        mt_axis =  hist.Bin("mt", r" $m_{T}$ [GeV]", 30, 0, 300)
        dr_axis = hist.Bin("dr","$\Delta$R",20,0,5)
        # MET vars
        signi_axis = hist.Bin("significance", r"MET $\sigma$",20,0,10)
        covXX_axis = hist.Bin("covXX",r"MET covXX",20,0,10)
        covXY_axis = hist.Bin("covXY",r"MET covXY",20,0,10)
        covYY_axis = hist.Bin("covYY",r"MET covYY",20,0,10)
        sumEt_axis = hist.Bin("sumEt", r" MET sumEt", 50, 0, 300)
        
        # axis.StrCategory([], name='region', growth=True),
        disc_list = [ 'btagDeepCvL', 'btagDeepCvB','btagDeepFlavCvB','btagDeepFlavCvL']#,'particleNetAK4_CvL','particleNetAK4_CvB']
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d , 50, 0, 1))  
        _hist_event_dict = {
                'nj'  : hist.Hist("Counts", dataset_axis,  lepflav_axis,region_axis, njet_axis),
                'nbj' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, nbjet_axis),
                'ncj' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, ncjet_axis),
                'hj_dr'  : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, dr_axis),
                'MET_sumEt' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, sumEt_axis),
                'MET_significance' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, signi_axis),
                'MET_covXX' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, covXX_axis),
                'MET_covXY' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, covXY_axis),
                'MET_covYY' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, covYY_axis),
                'MET_phi' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, phi_axis),
                'MET_pt' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, pt_axis),
                'mT1' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mt_axis),
                'mT2' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mt_axis),
                'mTh':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mt_axis),
                'dphi_lep1':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, phi_axis),
                'dphi_lep2':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, phi_axis),
                'dphi_ll':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, phi_axis),

            }
        objects=['jetflav','jetpn','lep1','lep2','ll']
        
        for i in objects:
            if  'jet' in i: 
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis, pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, mass_axis)
            else:
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis,  lepflav_axis,region_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mass_axis)
            
        
        for disc, axis in zip(disc_list,btag_axes):
            _hist_event_dict["jetflav_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis,flav_axis, axis)
            _hist_event_dict["jetpn_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis,flav_axis, axis)
        self.event_hists = list(_hist_event_dict.keys())
    
        self._accumulator = processor.dict_accumulator(
            {**_hist_event_dict,   
        'cutflow': processor.defaultdict_accumulator(
                # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int))})
        self._accumulator['sumw'] = processor.defaultdict_accumulator(float)


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = processor.PackedSelection()
        if isRealData :output['sumw'][dataset] += 1.
        else:output['sumw'][dataset] += ak.sum(events.genWeight/abs(events.genWeight))
        req_lumi=np.ones(len(events), dtype='bool')
        if(isRealData): req_lumi=self._lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumi',ak.to_numpy(req_lumi))
        del req_lumi
        weights = Weights(len(events), storeIndividual=True)
        if isRealData:
            weights.add('genweight',np.ones(len(events)))
        else:
            weights.add('genweight',events.genWeight/abs(events.genWeight))
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
        ##############
        if(isRealData):output['cutflow'][dataset]['all']  += 1.
        else:output['cutflow'][dataset]['all']  += ak.sum(events.genWeight/abs(events.genWeight))
        trigger_ee = np.zeros(len(events), dtype='bool')
        trigger_mm = np.zeros(len(events), dtype='bool')
        trigger_em = np.zeros(len(events), dtype='bool')
        trigger_e = np.zeros(len(events), dtype='bool')
        trigger_m = np.zeros(len(events), dtype='bool')
        trigger_ee = np.zeros(len(events), dtype='bool')
        trigger_mm = np.zeros(len(events), dtype='bool')
        trigger_ele = np.zeros(len(events), dtype='bool')
        trigger_mu = np.zeros(len(events), dtype='bool')
        
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
        for t in self._emuhlt[self._year]:
            if t in events.HLT.fields:
                trigger_em = trigger_em | events.HLT[t]       
        
        if isRealData:
            if "MuonEG" in dataset : 
                trigger_em = trigger_em 
            elif "DoubleElectron" in dataset:trigger_ele = trigger_ee & ~trigger_em
            elif "SingleElectron" in dataset:trigger_ele =  trigger_e & ~trigger_ee & ~trigger_em 
            elif "DoubleMuon" in dataset:trigger_mu = trigger_mm 
            elif "SingleMuon" in dataset:trigger_mu = trigger_m & ~trigger_mm & ~trigger_em

        else : 
            trigger_mu = trigger_mm|trigger_m
            trigger_ele = trigger_ee|trigger_e
        selection.add('trigger_ee', ak.to_numpy(trigger_ele))
        selection.add('trigger_mumu', ak.to_numpy(trigger_mu))
        selection.add('trigger_emu', ak.to_numpy(trigger_em))
        # del trigger_ele,trigger_e,trigger_ee,trigger_em,trigger_m,trigger_mm,trigger_mu
        metfilter = np.ones(len(events), dtype='bool')
        for flag in self._met_filters[self._year]['data' if isRealData else 'mc']:
            metfilter &= np.array(events.Flag[flag])
        selection.add('metfilter', metfilter)
        del metfilter
        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1,ascending=False)]
        musel = ((event_mu.pt > 13) & (abs(event_mu.eta) < 2.4)&(event_mu.mvaId>=3) &(event_mu.pfRelIso04_all<0.15)&(abs(event_mu.dxy)<0.5)&(abs(event_mu.dz)<1))
        event_mu["lep_flav"] = 13*event_mu.charge
        
        event_mu = event_mu[musel]
        event_mu= ak.pad_none(event_mu,2,axis=1)
        nmu = ak.sum(musel,axis=1)
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        event_e["lep_flav"] = 11*event_e.charge
        elesel = ((event_e.pt > 13) & (abs(event_e.eta) < 2.5)&(event_e.mvaFall17V2Iso_WP90==1)& (abs(event_e.dxy)<0.5)&(abs(event_e.dz)<1))
        event_e = event_e[elesel]
        event_e = ak.pad_none(event_e,2,axis=1)
        nele = ak.sum(elesel,axis=1)

        selection.add('lepsel',ak.to_numpy((nele+nmu>=2)))
        
        good_leptons = ak.with_name(
                ak.concatenate([ event_e, event_mu], axis=1),
                "PtEtaPhiMCandidate", )
        good_leptons = good_leptons[ak.argsort(good_leptons.pt, axis=1,ascending=False)]
        leppair = ak.combinations(
                good_leptons,
                n=2,
                replacement=False,
                axis=-1,
                fields=["lep1", "lep2"],
            )
        # print(leppair.tolist())
        ll_cand = ak.zip({
                    "lep1" : leppair.lep1,
                    "lep2" : leppair.lep2,
                    "pt": (leppair.lep1+leppair.lep2).pt,
                    "eta": (leppair.lep1+leppair.lep2).eta,
                    "phi": (leppair.lep1+leppair.lep2).phi,
                    "mass": (leppair.lep1+leppair.lep2).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        ll_cand = ak.pad_none(ll_cand,1,axis=1)
        if(ak.count(ll_cand.pt)>0):ll_cand  = ll_cand[ak.argsort(ll_cand.pt, axis=1,ascending=False)]
        met = ak.zip({
                    "pt":  events.MET.pt,
                    "phi": events.MET.phi,
                    "energy":events.MET.sumEt,
                },with_name="PtEtaPhiMLorentzVector",)
        
        req_global = ak.any((leppair.lep1.pt>25) & (ll_cand.mass>12) & (ll_cand.pt>30) & (leppair.lep1.charge+leppair.lep2.charge==0) & (events.MET.pt>20) & (make_p4(leppair.lep1).delta_r(make_p4(leppair.lep2))>0.02),axis=-1)
        req_sr = ak.any((mT(leppair.lep2,met)>30) & (mT(ll_cand,met)>60) & (abs(ll_cand.mass-91.18)>15) & (events.MET.sumEt>45),axis=-1) 
        
        selection.add('global_selection',ak.to_numpy(req_global))
        selection.add('SR',ak.to_numpy(req_sr))
        
        mask2e =  req_sr&req_global & (ak.num(event_e)==2)& (event_e[:,0].pt>25) & (event_e[:,1].pt>13)
        mask2mu =  req_sr&req_global & (ak.num(event_mu)==2)& (event_mu[:,0].pt>25) &(event_mu[:,1].pt>13)
        maskemu = req_sr&req_global & (ak.num(event_e)==1)& (ak.num(event_mu) ==1 )& (((event_mu[:,0].pt>25)&(event_mu[:,1].pt>13))|((event_e[:,0].pt>25)&(event_e[:,1].pt>10)))
        
        mask2lep = [ak.any(tup) for tup in zip(maskemu, mask2mu, mask2e)]
        
        good_leptons = ak.mask(good_leptons,mask2lep)
       
        
        # output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(good_leptons)>0)
        
        selection.add('ee',ak.to_numpy(nele==2))
        selection.add('mumu',ak.to_numpy(nmu==2))
        selection.add('emu',ak.to_numpy((nele==1)&(nmu==1)))
             
        # ###########
        corr_jet =  jec(events,events.Jet,dataset,self._year,self._corr)
        seljet = (corr_jet.pt > 20) & (abs(corr_jet.eta) <= 2.4)&((corr_jet.puId > 0)|(corr_jet.pt>50)) &(corr_jet.jetId>5)&ak.all(corr_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(corr_jet.metric_table(leppair.lep2)>0.4,axis=2) & (corr_jet.btagDeepFlavCvB>0.4) & (corr_jet.btagDeepFlavCvL>0.225)
        selection.add('jetsel',ak.to_numpy(ak.sum(seljet,axis=1)>0))
        eventflav_jet = corr_jet[ak.argsort(corr_jet.btagDeepFlavCvL,axis=1,ascending=False)]
        # eventpn_jet = corr_jet[ak.argsort(corr_jet.particleNetAK4_CvL,axis=1,ascending=False)]
        req_dy_cr =ak.any((mT(leppair.lep2,met)>30)& (abs(ll_cand.mass-91.18)<15) & (events.MET.sumEt>45)& (mT(ll_cand,met)<60) ,axis=-1) 
        req_top_cr =ak.any((mT(leppair.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) & (ll_cand.mass>50),axis=-1) 
        # req_WW_cr = ak.any((mT(leppair.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) & (ll_cand.mass),axis=-1) 
        selection.add('top_CR',ak.to_numpy(req_top_cr))
        selection.add('DY_CR',ak.to_numpy(req_dy_cr))
        # selection.add('WW_CR',ak.to_numpy(req_WW_cr))
        sel_jet =  eventflav_jet[(eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 0)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all(eventflav_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(eventflav_jet.metric_table(leppair.lep2)>0.4,axis=2)& (eventflav_jet.btagDeepFlavCvB>0.4) & (eventflav_jet.btagDeepFlavCvL>0.225)]

        sel_jet = ak.mask(sel_jet,ak.num(good_leptons)>0)
        good_leptons = ak.mask(good_leptons,ak.num(sel_jet)>0)

        sel_jetflav =  eventflav_jet[(eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 0)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all(eventflav_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(eventflav_jet.metric_table(leppair.lep2)>0.4,axis=2)& (eventflav_jet.btagDeepFlavCvB>0.4) & (eventflav_jet.btagDeepFlavCvL>0.225)]
        sel_jetflav = ak.mask(sel_jetflav,ak.num(good_leptons)>0)
        sel_cjet_flav = ak.pad_none(sel_jetflav,1,axis=1)
        sel_cjet_flav = sel_cjet_flav[:,0]

        # sel_jetpn =  eventpn_jet[(eventpn_jet.pt > 20) & (abs(eventpn_jet.eta) <= 2.4)&((eventpn_jet.puId > 0)|(eventpn_jet.pt>50)) &(eventpn_jet.jetId>5)&ak.all(eventpn_jet.metric_table(leppair.lep1)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(leppair.lep2)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        # sel_jetpn = ak.mask(sel_jetpn,ak.num(pair_4lep)>0)
        # sel_cjet_pn = ak.pad_none(sel_jetpn,1,axis=1)
        # sel_cjet_pn = sel_cjet_pn[:,0]

        
        output['cutflow'][dataset]['global selection'] += ak.sum(req_global)
        output['cutflow'][dataset]['signal region'] += ak.sum(req_sr&req_global)  
        output['cutflow'][dataset]['selected jets'] +=ak.sum(req_sr&req_global&(ak.sum(seljet,axis=1)>0))
        output['cutflow'][dataset]['all ee'] +=ak.sum(req_sr&req_global&(ak.sum(seljet,axis=1)>0)
        &(nele==2))
        output['cutflow'][dataset]['all mumu'] +=ak.sum(req_sr&req_global&(ak.sum(seljet,axis=1)>0)&(nmu==2))
        output['cutflow'][dataset]['all emu'] +=ak.sum(req_sr&req_global&(ak.sum(seljet,axis=1)>0)&(nele==1)&(nmu==1))
        # output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.num(sel_jet) > 0)

        lepflav = ['ee','mumu','emu']
        reg = ['SR','top_CR','DY_CR']
            
        for histname, h in output.items():
            for ch in lepflav:
                for r in reg:
                    cut = selection.all('jetsel','lepsel','global_selection','metfilter','lumi',r,ch, 'trigger_%s'%(ch))
                    print(dataset,ch,'ele',ak.num(trigger_ele==True),trigger_ele[:,:20])
                    print(dataset,ch,'mu',ak.num(trigger_mu==True),trigger_mu[:,20])
                    print(dataset,ch,'EMu',ak.num(trigger_em==True),trigger_em[:,20])
                    llcut = ll_cand[cut]
                    llcut = llcut[:,0]
                    lep1cut = llcut.lep1
                    lep2cut = llcut.lep2
                    if not isRealData:
                        if ch=='ee':lepsf=eleSFs(lep1cut,self._year,self._corr)*eleSFs(lep2cut,self._year,self._corr)
                        elif ch=='mumu':lepsf=muSFs(lep1cut,self._year,self._corr)*muSFs(lep2cut,self._year,self._corr)
                        else:
                            lepsf=np.where(lep1cut.lep_flav==11,eleSFs(lep1cut,self._year,self._corr)*muSFs(lep2cut,self._year,self._corr),1.)*np.where(lep1cut.lep_flav==13,eleSFs(lep2cut,self._year,self._corr)*muSFs(lep1cut,self._year,self._corr),1.)
                    else : lepsf =weights.weight()[cut]
                    if 'jetflav_' in histname:
                        fields = {l: normalize(sel_cjet_flav[histname.replace('jetflav_','')],cut) for l in h.fields if l in dir(sel_cjet_flav)}
                        if isRealData:flavor= ak.zeros_like(normalize(sel_cjet_flav['pt'],cut))
                        else :flavor= normalize(sel_cjet_flav.hadronFlavour+1*((sel_cjet_flav.partonFlavour == 0 ) & (sel_cjet_flav.hadronFlavour==0)),cut)
                        h.fill(dataset=dataset, lepflav =ch, region = r, flav=flavor, **fields,weight=weights.weight()[cut]*lepsf)  
                    # elif 'jetpn_' in histname:
                    #     fields = {l: normalize(sel_cjet_pn[histname.replace('jetpn_','')],cut) for l in h.fields if l in dir(sel_cjet_pn)}
                    #     h.fill(dataset=dataset,lepflav =ch, flav=normalize(sel_cjet_pn.hadronFlavour+1*((sel_cjet_pn.partonFlavour == 0 ) & (sel_cjet_pn.hadronFlavour==0)),cut), **fields,weight=weights.weight()[cut]*lepsf)    
                    elif 'lep1_' in histname:
                        fields = {l: ak.fill_none(flatten(lep1cut[histname.replace('lep1_','')]),np.nan) for l in h.fields if l in dir(lep1cut)}
                        h.fill(dataset=dataset,lepflav=ch,region = r, **fields,weight=weights.weight()[cut]*lepsf)
                    elif 'lep2_' in histname:
                        fields = {l: ak.fill_none(flatten(lep2cut[histname.replace('lep2_','')]),np.nan) for l in h.fields if l in dir(lep2cut)}
                        h.fill(dataset=dataset,lepflav=ch,region = r, **fields,weight=weights.weight()[cut]*lepsf)
                    elif 'MET_' in histname:
                        fields = {l: normalize(events.MET[histname.replace('MET_','')],cut) for l in h.fields if l in dir(events.MET)}
                        h.fill(dataset=dataset, lepflav =ch, region = r,**fields,weight=weights.weight()[cut]*lepsf) 
                    elif 'll_' in histname:
                        fields = {l: ak.fill_none(flatten(llcut[histname.replace('ll_','')]),np.nan) for l in h.fields if l in dir(llcut)}
                        h.fill(dataset=dataset,lepflav=ch, region = r,**fields,weight=weights.weight()[cut]*lepsf) 
                    else :
                        output['nj'].fill(dataset=dataset,lepflav=ch,region = r,nj=normalize(ak.num(sel_jet),cut),weight=weights.weight()[cut]*lepsf)                            
                        # print(ak.type(ak.flatten(mT(lep1cut,met[cut]))),ak.type(weights.weight()[cut]*lepsf))            
                        output['mT1'].fill(dataset=dataset,lepflav=ch,region = r,mt=flatten(mT(lep1cut,met[cut])),weight=weights.weight()[cut]*lepsf)
                        output['mT2'].fill(dataset=dataset,lepflav=ch,region = r,mt=flatten(mT(lep2cut,met[cut])),weight=weights.weight()[cut]*lepsf)
                        output['mTh'].fill(dataset=dataset,lepflav=ch,region = r,mt=flatten(mT(llcut,met[cut])),weight=weights.weight()[cut]*lepsf)
                        output['dphi_ll'].fill(dataset=dataset,lepflav=ch,region = r,phi=flatten(met[cut].delta_phi(llcut)),weight=weights.weight()[cut]*lepsf)
                        output['dphi_lep1'].fill(dataset=dataset,lepflav=ch,region = r,phi=flatten(met[cut].delta_phi(lep1cut)),weight=weights.weight()[cut]*lepsf)
                        output['dphi_lep2'].fill(dataset=dataset,lepflav=ch,region = r,phi=flatten(met[cut].delta_phi(lep2cut)),weight=weights.weight()[cut]*lepsf)
                    
        return output

    def postprocess(self, accumulator):
        print(accumulator)
        return accumulator
