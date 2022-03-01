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
                'IsoTkMu24',
            ],
            '2017': [
                'IsoMu27',
            ],
            '2018': [
                'IsoMu24',
            ],
        }   
        self._ehlt = {
            '2016': [
                'Ele27_WPTight_Gsf',
                'Ele25_eta2p1_WPTight_Gsf',
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
            ],
        }   
        
        # print(self._muhlt[self._year])
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        lepflav_axis = hist.Cat("lepflav",['ee','mumu'])
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
                'mT' : hist.Hist("Counts", dataset_axis, lepflav_axis, mt_axis),
                'mTh' : hist.Hist("Counts", dataset_axis, lepflav_axis, mt_axis),
                'mjjl' : hist.Hist("Counts", dataset_axis, lepflav_axis, mt_axis),
                'dphi_lep':hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
                'dphi_ww':hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
            }
        objects=['cjet','lep','jet1','jet2','jj']
        
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
            _hist_event_dict["cjet_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,flav_axis, axis)
            _hist_event_dict["jet1_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,flav_axis, axis)
            _hist_event_dict["jet2_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,flav_axis, axis)
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
        for t in self._muhlt[self._year]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]
        for t in self._ehlt[self._year]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]
        
        selection.add('trigger_ee', ak.to_numpy(trigger_ee))
        selection.add('trigger_mm', ak.to_numpy(trigger_mm))
        

        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1,ascending=False)]
        musel = ((event_mu.pt > 30) & (abs(event_mu.eta) < 2.4)&(event_mu.mvaId>=3) &(event_mu.pfRelIso03_all<0.35)&(abs(event_mu.dxy)<0.5)&(abs(event_mu.dz)<1))
        
        event_mu = event_mu[musel]
        event_mu= ak.pad_none(event_mu,1,axis=1)
        nmu = ak.sum(musel,axis=1)
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        elesel = ((event_e.pt > 30) & (abs(event_e.eta) < 2.5)&(event_e.mvaFall17V2Iso_WP90==1)& (abs(event_e.dxy)<0.5)&(abs(event_e.dz)<1))
        event_e = event_e[elesel]
        event_e = ak.pad_none(event_e,1,axis=1)
        nele = ak.sum(elesel,axis=1)
        selection.add('lepsel',ak.to_numpy((nele+nmu>=1)))
        
        event_jet = events.Jet[ak.argsort(events.Jet.btagDeepFlavCvL, axis=1,ascending=False)]
        jet_sel = (event_jet.pt > 20) & (abs(event_jet.eta) <= 2.4)&((event_jet.puId > 0)|(event_jet.pt>50)) &(event_jet.jetId>5) 
        event_jet =event_jet[jet_sel]
        njet = ak.sum(jet_sel,axis=1)
        event_jet = ak.pad_none(event_jet,3,axis=1)
        selection.add('jetsel',ak.to_numpy(njet>=3))
        sel_cjet = event_jet[:,0]
        rest_jet = event_jet[:,1:]

        good_leptons = ak.with_name(
                ak.concatenate([event_e, event_mu], axis=1),
                "PtEtaPhiMCandidate", )
        good_leptons = good_leptons[ak.argsort(good_leptons.pt, axis=1,ascending=False)]
        good_leptons = good_leptons[:,0]
        pair_2j = ak.combinations(
                rest_jet,
                n=2,
                replacement=False,
                fields = ['jet1','jet2']
            )
        jj_cand = ak.zip({
                    "jet1" : pair_2j.jet1,
                    "jet2" : pair_2j.jet2,
                    "pt": (pair_2j.jet1+pair_2j.jet2).pt,
                    "eta": (pair_2j.jet1+pair_2j.jet2).eta,
                    "phi": (pair_2j.jet1+pair_2j.jet2).phi,
                    "mass": (pair_2j.jet1+pair_2j.jet2).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        met = ak.zip({
                    "pt":  events.MET.pt,
                    "phi": events.MET.phi,
                    "eta": ak.zeros_like(events.MET.pt),
                    "mass": ak.zeros_like(events.MET.pt),
                    "energy":events.MET.sumEt,
                },with_name="PtEtaPhiMLorentzVector",)
        
        req_global = (good_leptons.pt>30)& (events.MET.pt>20) & ak.any((make_p4(jj_cand.jet1).delta_r(good_leptons)>0.4),axis=-1)& ak.any((make_p4(jj_cand.jet2).delta_r(good_leptons)>0.4),axis=-1)
        req_wqqmass = ak.any(jj_cand.mass<116,axis=-1)
       
        
        req_sr = mT(make_p4(good_leptons),met)>60 & ak.any(met.delta_phi(jj_cand)<np.pi/2.,axis=-1)
        
        selection.add('global_selection',ak.to_numpy(req_global))
        selection.add('wqq',ak.to_numpy(req_wqqmass))
        selection.add('mT_deltaphi',ak.to_numpy(req_sr))

        mask2e =  req_sr&req_global & (ak.num(event_e)==1)& (event_e[:,0].pt>30) 
        mask2mu =  req_sr&req_global & (ak.num(event_mu)==1)& (event_mu[:,0].pt>30)
        mask2lep = [ak.any(tup) for tup in zip(mask2mu, mask2e)]
        good_leptons = ak.mask(good_leptons,mask2lep)
        
        # output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(good_leptons)>0)
        
        selection.add('ee',ak.to_numpy(nele==1))
        selection.add('mumu',ak.to_numpy(nmu==1))
        
               
        # ###########
        seljet = sel_cjet.delta_r(good_leptons)>0.4

        # print(ak.type(seljet))
        # sel_cjet = ak.mask(cjet,seljet)
        # sel_cjet
        # selection.add('cjetsel',ak.to_numpy(seljet))
        
        output['cutflow'][dataset]['global selection'] += ak.sum(req_global)
        output['cutflow'][dataset]['mjj mass'] += ak.sum(req_wqqmass&req_global)  
        output['cutflow'][dataset]['dphi mT'] += ak.sum(req_wqqmass&req_sr&req_global)  
        output['cutflow'][dataset]['tag one jets'] +=ak.sum(req_wqqmass&req_sr&req_global&seljet)
        output['cutflow'][dataset]['jet eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nmu+nele>=1)&(njet>=3)&seljet)
        output['cutflow'][dataset]['electron eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nele==1)&seljet)
        output['cutflow'][dataset]['muon eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nmu==1)&seljet)

        lepflav = ['ee','mumu']
        
        for histname, h in output.items():
            for ch in lepflav:
                cut = selection.all('jetsel','lepsel','global_selection','wqq','mT_deltaphi',ch)
                lepcut=good_leptons[cut]
                jjcut = jj_cand[cut]
                jet1cut=jj_cand.jet1[cut]
                jet2cut=jj_cand.jet2[cut]

                if 'cjet_' in histname:
                    fields = {l: normalize(sel_cjet[histname.replace('cjet_','')],cut) for l in h.fields if l in dir(sel_cjet)}
                    h.fill(dataset=dataset, lepflav =ch,flav=normalize(sel_cjet.hadronFlavour+1*((sel_cjet.partonFlavour == 0 ) & (sel_cjet.hadronFlavour==0)),cut), **fields)    
                elif 'jet1_' in histname:
                    fields = {l: flatten(jet1cut[histname.replace('jet1_','')]) for l in h.fields if l in dir(jet1cut)}
                    h.fill(dataset=dataset, lepflav =ch,flav=flatten((jet1cut.hadronFlavour+1*((jet1cut.partonFlavour == 0 ) & (jet1cut.hadronFlavour==0)))), **fields) 
                elif 'jet2_' in histname:
                    fields = {l: flatten(jet2cut[histname.replace('jet2_','')]) for l in h.fields if l in dir(jet2cut)}
                    h.fill(dataset=dataset, lepflav =ch,flav=flatten((jet2cut.hadronFlavour+1*((jet2cut.partonFlavour == 0 ) & (jet2cut.hadronFlavour==0)))), **fields) 
                elif 'jj_' in histname:
                    fields = {l:  flatten(jjcut[histname.replace('jj_','')]) for l in h.fields if l in dir(jjcut)}
                    h.fill(dataset=dataset, lepflav =ch, **fields)
                elif 'lep_' in histname:
                    fields = {l: ak.fill_none(lepcut[histname.replace('lep_','')],np.nan) for l in h.fields if l in dir(lepcut)}
                    h.fill(dataset=dataset,lepflav=ch, **fields)
                elif 'MET_' in histname:
                    fields = {l: normalize(events.MET[histname.replace('MET_','')],cut) for l in h.fields if l in dir(events.MET)}
                    h.fill(dataset=dataset, lepflav =ch, **fields)  
                else :
                    # output['nj'].fill(dataset=dataset,lepflav=ch,nj=normalize(ak.num(sel_jet),cut))                                        
                    output['mT'].fill(dataset=dataset,lepflav=ch,mt=flatten(mT(lepcut,met[cut])))
                    output['mTh'].fill(dataset=dataset,lepflav=ch,mt=flatten((met[cut]+lepcut+jjcut).mass))
                    output['mjjl'].fill(dataset=dataset,lepflav=ch,mt=flatten((lepcut+jjcut).mass))
                    output['dphi_ww'].fill(dataset=dataset,lepflav=ch,phi=flatten(met[cut].delta_phi(jjcut)))
                    output['dphi_lep'].fill(dataset=dataset,lepflav=ch,phi=flatten(met[cut].delta_phi(lepcut)))
                    
        return output

    def postprocess(self, accumulator):
        return accumulator
