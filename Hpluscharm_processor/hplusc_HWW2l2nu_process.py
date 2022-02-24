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
# from utils.correction import *
from coffea.analysis_tools import Weights
from functools import partial
import numba
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
       
        self._muhlt = {
            '2016': [
                'IsoMu2',
                'IsoTkMu24',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ',#runH
                'Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ',#runH
            ],
            '2017': [
                'IsoMu27',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8',
            ],
            '2018': [
                'IsoMu24',
                'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
            ],
        }   
        self._ehlt = {
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele25_eta2p1_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
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
        print(self._muhlt)
        # print(self._muhlt[self._year])
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        lepflav_axis = hist.Cat("lepflav",['ee','mumu','emu'])
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
                'nj'  : hist.Hist("Counts", dataset_axis,  lepflav_axis, njet_axis),
                'nbj' : hist.Hist("Counts", dataset_axis, lepflav_axis, nbjet_axis),
                'ncj' : hist.Hist("Counts", dataset_axis, lepflav_axis, ncjet_axis),
                'hj_dr'  : hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
                'MET_sumEt' : hist.Hist("Counts", dataset_axis, lepflav_axis, sumEt_axis),
                'MET_significance' : hist.Hist("Counts", dataset_axis, lepflav_axis, signi_axis),
                'MET_covXX' : hist.Hist("Counts", dataset_axis, lepflav_axis, covXX_axis),
                'MET_covXY' : hist.Hist("Counts", dataset_axis, lepflav_axis, covXY_axis),
                'MET_covYY' : hist.Hist("Counts", dataset_axis, lepflav_axis, covYY_axis),
                'MET_phi' : hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
                'MET_pt' : hist.Hist("Counts", dataset_axis, lepflav_axis, pt_axis),
                'mT1' : hist.Hist("Counts", dataset_axis, lepflav_axis, mt_axis),
                'mT2' : hist.Hist("Counts", dataset_axis, lepflav_axis, mt_axis),
                'mTh':hist.Hist("Counts", dataset_axis, lepflav_axis, mt_axis),
                'dphi_lep1':hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
                'dphi_lep2':hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
                'dphi_ll':hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),

            }
        objects=['jetcsv','jetflav','jetpn','jetpt','lep1','lep2']
        
        for i in objects:
            if  'jet' in i: 
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, flav_axis, pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,flav_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,flav_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,flav_axis, mass_axis)
            else:
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis ,pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis,lepflav_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis,  lepflav_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, mass_axis)
            
        
        for disc, axis in zip(disc_list,btag_axes):
            _hist_event_dict["jetcsv_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,flav_axis, axis)
            _hist_event_dict["jetflav_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,flav_axis, axis)
            _hist_event_dict["jetpn_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,flav_axis, axis)
            _hist_event_dict["jetpt_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,flav_axis, axis)
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
        if(isRealData):output['sumw'][dataset] += 1.
        else:output['sumw'][dataset] += ak.sum(events.genWeight)
        # req_lumi=np.ones(len(events), dtype='bool')
        # if(isRealData): req_lumi=lumiMasks['2017'](events.run, events.luminosityBlock)
        weights = Weights(len(events), storeIndividual=True)
        if isRealData:weights.add('genweight',len(events)*1.)
        else:
            weights.add('genweight',events.genWeight)
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
        ##############
        output['cutflow'][dataset]['all'] += len(events.Muon)
        trigger_ee = np.zeros(len(events), dtype='bool')
        trigger_mm = np.zeros(len(events), dtype='bool')
        trigger_em = np.zeros(len(events), dtype='bool')
        for t in self._muhlt[self._year]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]
        for t in self._ehlt[self._year]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]
        for t in self._emuhlt[self._year]:
            if t in events.HLT.fields:
                trigger_em = trigger_em | events.HLT[t]
        selection.add('trigger_ee', ak.to_numpy(trigger_ee))
        selection.add('trigger_mm', ak.to_numpy(trigger_mm))
        selection.add('trigger_em', ak.to_numpy(trigger_em))
            

        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1,ascending=False)]
        musel = ((event_mu.pt > 13) & (abs(event_mu.eta) < 2.4)&(event_mu.mvaId>=3) &(event_mu.pfRelIso03_all<0.35)&(event_mu.sip3d<10)&(abs(event_mu.dxy)<0.5)&(abs(event_mu.dz)<1))
        event_mu["lep_flav"] = 13*event_mu.charge
        
        event_mu = event_mu[musel]
        event_mu= ak.pad_none(event_mu,2,axis=1)
        nmu = ak.sum(musel,axis=1)
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        event_e["lep_flav"] = 11*event_e.charge
        elesel = ((event_e.pt > 13) & (abs(event_e.eta) < 2.5)&(event_e.mvaFall17V2Iso_WP90==1)&(event_e.sip3d<10)& (abs(event_e.dxy)<0.5)&(abs(event_e.dz)<1))
        event_e = event_e[elesel]
        event_e = ak.pad_none(event_e,2,axis=1)
        nele = ak.sum(elesel,axis=1)
        selection.add('lepsel',ak.to_numpy((nele+nmu==2)))
        
        good_leptons = ak.with_name(
                ak.concatenate([event_e, event_mu], axis=1),
                "PtEtaPhiMCandidate", )
        ll_cand = ak.zip({
                    "p4" : good_leptons[:,0]+good_leptons[:,1],
                    "pt": (good_leptons[:,0]+good_leptons[:,1]).pt,
                    "eta": (good_leptons[:,0]+good_leptons[:,1]).eta,
                    "phi": (good_leptons[:,0]+good_leptons[:,1]).phi,
                    "mass": (good_leptons[:,0]+good_leptons[:,1]).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        met = ak.zip({
                    "pt":  events.MET.pt,
                    "phi": events.MET.phi,
                    "energy":events.MET.sumEt,
                },with_name="PtEtaPhiMLorentzVector",)
       
        req_global = (good_leptons[:,0].pt>25) & (ll_cand.mass>12) & (ll_cand.pt>30) & (good_leptons[:,0].charge+good_leptons[:,1].charge==0) & (events.MET.pt>20) & (good_leptons[:,0].delta_r(good_leptons[:,1]>0.02))
        req_sr = (mT(good_leptons[:,1],met)>30) & (mT(ll_cand,met)>60) 
        
        selection.add('global_selection',ak.to_numpy(req_global))
        selection.add('SR',ak.to_numpy(req_sr))
        mask2e =  req_global & (ak.num(event_e)==2)& (event_e[:,0].pt>25) & (event_e[:,1].pt>13)
        mask2mu =  req_global & (ak.num(event_mu)==2)& (event_mu[:,0].pt>25) &(event_mu[:,1].pt>13)
        maskemu = req_global & (ak.num(event_e)==1)& (ak.num(event_mu) ==2 )& (((event_mu[:,0].pt>25)&(event_mu[:,1].pt>13))|((event_e[:,0].pt>25)&(event_e[:,1].pt>10)))
        
        mask2lep = [ak.any(tup) for tup in zip(maskemu, mask2mu, mask2e)]
        good_leptons = ak.mask(good_leptons,mask2lep)
       
        
        output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(good_leptons)>0)
        output['cutflow'][dataset]['selected Z pairs'] += ak.sum((ak.num(good_leptons)>0) & (ak.num(req_sr)>0))
        selection.add('ee',ak.to_numpy((ak.num(event_e)==2)& (event_e[:,0].pt>25) & (event_e[:,1].pt>13)))
        selection.add('mumu',ak.to_numpy((ak.num(event_mu)==2)& (event_mu[:,0].pt>25) &(event_mu[:,1].pt>13)))
        selection.add('emu',ak.to_numpy((ak.num(event_e)==1)& (ak.num(event_mu) ==1 )& (((event_mu[:,0].pt>25)&(event_mu[:,1].pt>13))|((event_e[:,0].pt>25)&(event_e[:,1].pt>13)))))
        
               
        # ###########
        seljet = (events.Jet.pt > 20) & (abs(events.Jet.eta) <= 2.4)&((events.Jet.puId > 0)|(events.Jet.pt>50)) &(events.Jet.jetId>5)&ak.all(events.Jet.metric_table(good_leptons[:,0])>0.4,axis=2)&ak.all(events.Jet.metric_table(good_leptons[:,1])>0.4,axis=2)
        selection.add('jetsel',ak.to_numpy(ak.sum(seljet,axis=1)>0))
        eventcsv_jet = events.Jet[ak.argsort(events.Jet.btagDeepCvL,axis=1,ascending=False)]
        eventflav_jet = events.Jet[ak.argsort(events.Jet.btagDeepFlavCvL,axis=1,ascending=False)]
        # eventpn_jet = events.Jet[ak.argsort(events.Jet.particleNetAK4_CvL,axis=1,ascending=False)]
        eventpt_jet = events.Jet[ak.argsort(events.Jet.pt,axis=1,ascending=False)]

        sel_jet = eventcsv_jet[(eventcsv_jet.pt > 20) & (abs(eventcsv_jet.eta) <= 2.4)&((eventcsv_jet.puId > 0)|(eventcsv_jet.pt>50)) &(eventcsv_jet.jetId>5)&ak.all(eventcsv_jet.metric_table(good_leptons[:,0])>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(good_leptons[:,1])>0.4,axis=2)]
        # sel_jet = eventcsv_jet[ak.all(eventcsv_jet.metric_table(good_leptons[:,1])>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]

        sel_jet = ak.mask(sel_jet,ak.num(good_leptons)>0)
        good_leptons = ak.mask(good_leptons,ak.num(sel_jet)>0)
        sel_cjet_csv = ak.pad_none(sel_jet,1,axis=1)
        sel_cjet_csv= sel_cjet_csv[:,0]

        sel_jetflav =  eventflav_jet[(eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 0)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all(eventflav_jet.metric_table(good_leptons[:,0])>0.4,axis=2)&ak.all(eventflav_jet.metric_table(good_leptons[:,1])>0.4,axis=2)]
        sel_jetflav = ak.mask(sel_jetflav,ak.num(good_leptons)>0)
        sel_cjet_flav = ak.pad_none(sel_jetflav,1,axis=1)
        sel_cjet_flav = sel_cjet_flav[:,0]

        # sel_jetpn =  eventpn_jet[(eventpn_jet.pt > 20) & (abs(eventpn_jet.eta) <= 2.4)&((eventpn_jet.puId > 0)|(eventpn_jet.pt>50)) &(eventpn_jet.jetId>5)&ak.all(eventpn_jet.metric_table(good_leptons[:,0])>0.4,axis=2)&ak.all(eventpn_jet.metric_table(good_leptons[:,1])>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        # sel_jetpn = ak.mask(sel_jetpn,ak.num(pair_4lep)>0)
        # sel_cjet_pn = ak.pad_none(sel_jetpn,1,axis=1)
        # sel_cjet_pn = sel_cjet_pn[:,0]

        sel_jetpt =  eventpt_jet[(eventpt_jet.pt > 20) & (abs(eventpt_jet.eta) <= 2.4)&((eventpt_jet.puId > 0)|(eventpt_jet.pt>50)) &(eventpt_jet.jetId>5)&ak.all(eventpt_jet.metric_table(good_leptons[:,0])>0.4,axis=2)]
        
        sel_jetpt = ak.mask(sel_jetpt,ak.num(good_leptons)>0)
        sel_cjet_pt = ak.pad_none(sel_jetpt,1,axis=1)
        sel_cjet_pt = sel_cjet_pt[:,0]
        
          
        
        output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.num(sel_jet) > 0)
        # output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.num(sel_jet) > 0)

        lepflav = ['ee','mumu','emu']
        lep1=good_leptons[:,0] 
        lep2=good_leptons[:,1]
        for histname, h in output.items():
            for ch in lepflav:
                cut = selection.all('jetsel','lepsel','global_selection','SR',ch)
                if 'jetcsv_' in histname:
                    fields = {l: normalize(sel_cjet_csv[histname.replace('jetcsv_','')],cut) for l in h.fields if l in dir(sel_cjet_csv)}
                    h.fill(dataset=dataset, lepflav =ch,flav=normalize(sel_cjet_csv.hadronFlavour+1*((sel_cjet_csv.partonFlavour == 0 ) & (sel_cjet_csv.hadronFlavour==0)),cut), **fields)    
                elif 'jetflav_' in histname:
                    fields = {l: normalize(sel_cjet_flav[histname.replace('jetflav_','')],cut) for l in h.fields if l in dir(sel_cjet_flav)}
                    h.fill(dataset=dataset, lepflav =ch,flav=normalize(sel_cjet_flav.hadronFlavour+1*((sel_cjet_flav.partonFlavour == 0 ) & (sel_cjet_flav.hadronFlavour==0)),cut), **fields)  
                # elif 'jetpn_' in histname:
                #     fields = {l: normalize(sel_cjet_pn[histname.replace('jetpn_','')],cut) for l in h.fields if l in dir(sel_cjet_pn)}
                #     h.fill(dataset=dataset,lepflav =ch, flav=normalize(sel_cjet_pn.hadronFlavour+1*((sel_cjet_pn.partonFlavour == 0 ) & (sel_cjet_pn.hadronFlavour==0)),cut), **fields)    
                elif 'jetpt_' in histname:
                    fields = {l: normalize(sel_cjet_pt[histname.replace('jetpt_','')],cut) for l in h.fields if l in dir(sel_cjet_pt)}
                    h.fill(dataset=dataset, lepflav =ch,flav=normalize(sel_cjet_pt.hadronFlavour+1*((sel_cjet_pt.partonFlavour == 0 ) & (sel_cjet_pt.hadronFlavour==0)),cut), **fields)  
                   
                elif 'lep1_' in histname:
                    
                    fields = {l: normalize(lep1[histname.replace('lep1_','')],cut) for l in h.fields if l in dir(lep1)}

                    h.fill(dataset=dataset,lepflav=ch, **fields)
                elif 'lep2_' in histname:
                    
                    fields = {l: normalize(lep2[histname.replace('lep2_','')],cut)  for l in h.fields if l in dir(lep2)}
                    h.fill(dataset=dataset,lepflav=ch, **fields)
                elif 'MET_' in histname:
                    fields = {l: normalize(events.MET[histname.replace('MET_','')],cut) for l in h.fields if l in dir(events.MET)}
                    h.fill(dataset=dataset, lepflav =ch, **fields)  
                
                else :
                    output['nj'].fill(dataset=dataset,lepflav=ch,nj=normalize(ak.num(sel_jet),cut))                    
                    output['mT1'].fill(dataset=dataset,lepflav=ch,mt=normalize(mT(good_leptons[:,0],met),cut))
                    output['mT2'].fill(dataset=dataset,lepflav=ch,mt=normalize(mT(good_leptons[:,1],met),cut))
                    output['mTh'].fill(dataset=dataset,lepflav=ch,mt=normalize(mT(ll_cand,met),cut))
                    output['dphi_ll'].fill(dataset=dataset,lepflav=ch,phi=normalize(met.delta_phi(ll_cand),cut))
                    output['dphi_lep1'].fill(dataset=dataset,lepflav=ch,phi=normalize(met.delta_phi(lep1),cut))
                    output['dphi_lep2'].fill(dataset=dataset,lepflav=ch,phi=normalize(met.delta_phi(lep2),cut))
                    
        return output

    def postprocess(self, accumulator):
        return accumulator