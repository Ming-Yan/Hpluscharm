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
        dr_axis = hist.Bin("dr","$\Delta$R",20,0,5)
        costheta_axis = hist.Bin("costheta", "cos$\theta$",20,-1,1)
        # MET vars
        
        # axis.StrCategory([], name='region', growth=True),
        disc_list = [ 'btagDeepCvL', 'btagDeepCvB','btagDeepFlavCvB','btagDeepFlavCvL']#,'particleNetAK4_CvL','particleNetAK4_CvB']
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d , 50, 0, 1))  
        _hist_event_dict = {
                'nj'  : hist.Hist("Counts", dataset_axis,  lepflav_axis, njet_axis),
                'hc_dr'  : hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
                'zs_dr' : hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
                'cj_dr': hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
                'costheta_ll'  : hist.Hist("Counts", dataset_axis, lepflav_axis, costheta_axis),
                'costheta_qq'  : hist.Hist("Counts", dataset_axis, lepflav_axis, costheta_axis),
                'costheta_pz'  : hist.Hist("Counts", dataset_axis, lepflav_axis, costheta_axis),
                'phi_zz'  : hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
                'phi_lq'  : hist.Hist("Counts", dataset_axis, lepflav_axis, phi_axis),
            }
        objects=['cjet','lep1','lep2','jet1','jet2','jj','ll','higgs']
        
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
        if isRealData:weights.add('genweight',np.ones(len(events)))
        else:
            weights.add('genweight',events.genWeight)
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
        ##############
        if(isRealData):output['cutflow'][dataset]['all']  += 1.
        else:output['cutflow'][dataset]['all']  += ak.sum(events.genWeight/abs(events.genWeight))
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
        
        if isRealData:
            if "DoubleElectron" in dataset:trigger_ele = trigger_ee
            elif "SingleElectron" in dataset:trigger_ele = ~trigger_ee & trigger_e
            elif "DoubleMuon" in dataset:trigger_mu = trigger_mm
            elif "SingleMuon" in dataset:trigger_mu = ~trigger_mm & trigger_e

        else : 
            trigger_mu = trigger_mm|trigger_m
            trigger_ele = trigger_ee|trigger_e
        selection.add('trigger_ee', ak.to_numpy(trigger_ele))
        selection.add('trigger_mumu', ak.to_numpy(trigger_mu))
            

        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1,ascending=False)]
        musel = ((event_mu.pt > 13) & (abs(event_mu.eta) < 2.4)&(event_mu.mvaId>=3) &(event_mu.pfRelIso03_all<0.35)&(abs(event_mu.dxy)<0.5)&(abs(event_mu.dz)<1))
        event_mu["lep_flav"] = 13*event_mu.charge
        
        event_mu = event_mu[musel]
        nmu = ak.sum(musel,axis=1)
        event_mu= ak.pad_none(event_mu,2,axis=1)
        
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        event_e["lep_flav"] = 11*event_e.charge
        elesel = ((event_e.pt > 13) & (abs(event_e.eta) < 2.5)&(event_e.mvaFall17V2Iso_WP90==1)& (abs(event_e.dxy)<0.5)&(abs(event_e.dz)<1))
        event_e = event_e[elesel]
        nele = ak.sum(elesel,axis=1)
        event_e = ak.pad_none(event_e,2,axis=1)
        
        selection.add('lepsel',ak.to_numpy((nele==2)|(nmu==2)))
        event_jet = events.Jet[ak.argsort(events.Jet.btagDeepFlavCvL, axis=1,ascending=False)]
        jet_sel = (event_jet.pt > 20) & (abs(event_jet.eta) <= 2.4)&((event_jet.puId > 0)|(event_jet.pt>50)) &(event_jet.jetId>5)
        event_jet =event_jet[jet_sel]
        njet = ak.sum(jet_sel,axis=1)
        event_jet = ak.pad_none(event_jet,3,axis=1)
        selection.add('jetsel',ak.to_numpy(njet>=3))
        cjet = event_jet[:,0]
     
        rest_jet = event_jet[:,1:]
        # good_leptons = ak.with_name(
                # ak.concatenate([event_e, event_mu], axis=1),
                # "PtEtaPhiMCandidate", )
        good_leptons = ak.concatenate([event_e, event_mu], axis=1)
        pair_2lep = ak.combinations(
                good_leptons,
                n=2,
                replacement=False,
                axis=-1,
                fields=["lep1", "lep2"],
            )
        ll_cand = ak.zip({
                    # "p4": pair_2lep.lep1+pair_2lep.lep2,
                    "lep1": pair_2lep.lep1,
                    "lep2": pair_2lep.lep2,
                    "pt": (pair_2lep.lep1+pair_2lep.lep2).pt,
                    "eta": (pair_2lep.lep1+pair_2lep.lep2).eta,
                    "phi": (pair_2lep.lep1+pair_2lep.lep2).phi,
                    "mass": (pair_2lep.lep1+pair_2lep.lep2).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        
        if(ak.count(ll_cand.mass)>0):ll_cand = ll_cand[ak.argsort(abs(ll_cand.mass-91.18), axis=1)]

        # good_jets = ak.with_name(event_jet,"PtEtaPhiMCandidate")
        pair_2j = ak.combinations(
                rest_jet,
                n=2,
                replacement=False,
                fields = ['jet1','jet2']
            )
        jj_cand =ak.zip({
                    # "p4": pair_2j.jet1+pair_2j.jet2,
                    "jet1" : pair_2j.jet1,
                    "jet2" : pair_2j.jet2,
                    "pt": (pair_2j.jet1+pair_2j.jet2).pt,
                    "eta": (pair_2j.jet1+pair_2j.jet2).eta,
                    "phi": (pair_2j.jet1+pair_2j.jet2).phi,
                    "mass": (pair_2j.jet1+pair_2j.jet2).mass,
                },with_name="PtEtaPhiMLorentzVector",)
        jj_cand = jj_cand[ak.argsort(jj_cand.pt, axis=1,ascending=False)]

        higgs_cands, (ll_cands,jj_cands)= ll_cand.metric_table(jj_cand,axis=1,metric=lambda jj_cand, ll_cand: (jj_cand+ll_cand),return_combinations=True)
        # print(ak.type(ll_cands.ll_cand))
        # 
        higgs_cand = ak.zip(
            {
                "ll_cands"  :ll_cands,
                "jj_cands"  :jj_cands,
                "pt": higgs_cands.pt,
                "eta": higgs_cands.eta,
                "phi": higgs_cands.phi,
                "mass": higgs_cands.mass
            },with_name="PtEtaPhiMLorentzVector",
        )
        higgs_cand = ak.pad_none(higgs_cand,1,axis=0)
        #print(higgs_cand)
        #higgs_cand = higgs_cand[ak.argsort(higgs_cand.pt, axis=-1,ascending=False)]
        req_global = ak.any(ak.any((ll_cands.lep1.pt>25) & (ll_cands.lep1.charge+ll_cands.lep2.charge==0)  & (ll_cands.mass<120) & (ll_cands.mass>60),axis=-1),axis=-1) & ak.any(ak.any(make_p4(ll_cands.lep1).delta_r(jj_cands.jet1)>0.4,axis=-1),axis=-1) & ak.any(ak.any(make_p4(ll_cands.lep1).delta_r(jj_cands.jet2)>0.4,axis=-1),axis=-1)& ak.any(ak.any(make_p4(ll_cands.lep2).delta_r(jj_cands.jet1)>0.4,axis=-1),axis=-1)& ak.any(ak.any(make_p4(ll_cands.lep2).delta_r(jj_cands.jet2)>0.4,axis=-1),axis=-1) #& (make_p4(ll_cands.lep1).delta_r(make_p4(ll_cands.lep2))>0.02),axis=-1),axis=-1)
        req_zllmass =  ak.any(ak.any((abs(ll_cands.mass-91.)<15),axis=-1),axis=-1)
        req_zqqmass = ak.any(ak.any(jj_cands.mass<116,axis=-1),axis=-1)
        req_hmass =  ak.any(ak.any(higgs_cand.mass<250,axis=-1),axis=-1)

        selection.add('global_selection',ak.to_numpy(req_global))
        selection.add('Z_selection',ak.to_numpy(req_zllmass&req_zqqmass))
        selection.add('H_selection',ak.to_numpy(req_hmass))

        mask2e =  req_hmass & req_zllmass&req_zqqmass&req_global & (ak.num(event_e)==2)& (event_e[:,0].pt>25) & (event_e[:,1].pt>13)
        mask2mu =  req_hmass & req_zllmass&req_zqqmass&req_global & (ak.num(event_mu)==2)& (event_mu[:,0].pt>25) &(event_mu[:,1].pt>13)
        mask2lep = [ak.any(tup) for tup in zip(mask2mu, mask2e)]
        # mask2jet = req_hmass & req_zllmass&req_global&req_zqqmass
        # event_jet = ak.mask(event_jet,mask2jet)
        good_leptons = ak.mask(good_leptons,mask2lep)
        
               
        # ###########

        seljet = (cjet.pt > 20) & (abs(cjet.eta) <= 2.4)&((cjet.puId > 0)|(cjet.pt>50)) &(cjet.jetId>5)&(ak.all(ak.all(cjet.delta_r(ll_cands.lep1)>0.4,axis=-1),axis=-1))&(ak.all(ak.all(cjet.delta_r(ll_cands.lep2)>0.4,axis=-1),axis=-1))
        sel_cjet = ak.mask(cjet,seljet)
        selection.add('cjetsel',ak.to_numpy(seljet))
        
        output['cutflow'][dataset]['global selection'] += ak.sum(req_global)
        output['cutflow'][dataset]['dilepton mass'] += ak.sum(req_zllmass&req_global)
        output['cutflow'][dataset]['dijet mass'] +=ak.sum(req_zllmass&req_global&req_zqqmass)
        output['cutflow'][dataset]['higgs mass'] +=ak.sum(req_zllmass&req_global&req_zqqmass&req_hmass)
        output['cutflow'][dataset]['tag one jet'] +=ak.sum(req_zllmass&req_global&req_zqqmass&seljet&req_hmass)
        output['cutflow'][dataset]['jet efficiency'] +=ak.sum(req_zllmass&req_global&req_zqqmass&seljet&(njet>=3)&req_hmass)
        output['cutflow'][dataset]['electron efficiency'] +=ak.sum(req_zllmass&req_global&req_zqqmass&seljet&(njet>=3)&(nele==2)&req_hmass)
        output['cutflow'][dataset]['muon efficiency'] +=ak.sum(req_zllmass&req_global&req_zqqmass&seljet&(njet>=3)&(nmu==2)&req_hmass)
        
        selection.add('ee',ak.to_numpy((ak.num(event_e)==2)& (event_e[:,0].pt>25) & (event_e[:,1].pt>13)))
        selection.add('mumu',ak.to_numpy((ak.num(event_mu)==2)& (event_mu[:,0].pt>25) &(event_mu[:,1].pt>13)))


        lepflav = ['ee','mumu']

        for histname, h in output.items():
            for ch in lepflav:
                cut = selection.all('jetsel','lepsel','global_selection','Z_selection','H_selection','cjetsel',ch,'trigger_%s'%(ch))
                
                
                hcut = higgs_cand[cut]
                hcut = hcut[:,0,0] 
                llcut = hcut.ll_cands
                jjcut = hcut.jj_cands
                lep1cut=llcut.lep1
                lep2cut=llcut.lep2
                jet1cut=jjcut.jet1
                jet2cut=jjcut.jet2   
                charmcut = sel_cjet[cut]
                if 'cjet_' in histname:
                    fields = {l: normalize(sel_cjet[histname.replace('cjet_','')],cut) for l in h.fields if l in dir(sel_cjet)}
                    if isRealData:flavor= ak.zeros_like(normalize(sel_cjet['pt'],cut))
                    else :flavor= normalize(sel_cjet.hadronFlavour+1*((sel_cjet.partonFlavour == 0 ) & (sel_cjet.hadronFlavour==0)),cut)
                    h.fill(dataset=dataset, lepflav =ch,flav=flavor, **fields,weight=weights.weight()[cut])    
                
                elif 'jet1_' in histname:
                    fields = {l: flatten(jet1cut[histname.replace('jet1_','')]) for l in h.fields if l in dir(jet1cut)}
                    if isRealData:flavor= ak.zeros_like(normalize(jet1cut['pt'],cut))
                    else :flavor= flatten(jet1cut.hadronFlavour+1*((jet1cut.partonFlavour == 0 ) & (jet1cut.hadronFlavour==0)))
                    h.fill(dataset=dataset, lepflav =ch,flav=flavor, **fields,weight=weights.weight()[cut]) 
                elif 'jet2_' in histname:
                    fields = {l: flatten(jet2cut[histname.replace('jet2_','')]) for l in h.fields if l in dir(jet2cut)}
                    if isRealData:flavor= ak.zeros_like(normalize(jet2cut['pt'],cut))
                    else :flavor= flatten(jet2cut.hadronFlavour+1*((jet2cut.partonFlavour == 0 ) & (jet2cut.hadronFlavour==0)))
                    h.fill(dataset=dataset, lepflav =ch,flav=flavor, **fields,weight=weights.weight()[cut])    
                elif 'lep1_' in histname:
                    fields = {l: flatten(lep1cut[histname.replace('lep1_','')]) for l in h.fields if l in dir(lep1cut)}
                    h.fill(dataset=dataset,lepflav=ch, **fields,weight=weights.weight()[cut])
                elif 'lep2_' in histname:
                    fields = {l: flatten(lep2cut[histname.replace('lep2_','')]) for l in h.fields if l in dir(lep2cut)}
                    h.fill(dataset=dataset,lepflav=ch, **fields,weight=weights.weight()[cut])
                elif 'll_' in histname:
                    fields = {l: flatten(llcut[histname.replace('ll_','')]) for l in h.fields if l in dir(llcut)}
                    h.fill(dataset=dataset, lepflav =ch, **fields,weight=weights.weight()[cut])  
                elif 'jj_' in histname:
                    fields = {l:  flatten(jjcut[histname.replace('jj_','')]) for l in h.fields if l in dir(jjcut)}
                    h.fill(dataset=dataset, lepflav =ch, **fields,weight=weights.weight()[cut])
                elif 'higgs_' in histname:
                    fields = {l:  flatten(hcut[histname.replace('higgs_','')]) for l in h.fields if l in dir(hcut)}
                    h.fill(dataset=dataset, lepflav =ch, **fields,weight=weights.weight()[cut])  
                else :
                    # print("h",hcut.pt.tolist())
                    # print("c",charmcut.pt.tolist())
                    output['hc_dr'].fill(dataset=dataset,lepflav=ch,dr=hcut.delta_r(charmcut),weight=weights.weight()[cut])                    
                    output['zs_dr'].fill(dataset=dataset,lepflav=ch,dr=flatten(llcut.delta_r(jjcut)),weight=weights.weight()[cut]) 
                    output['cj_dr'].fill(dataset=dataset,lepflav=ch,dr=flatten(jjcut.delta_r(charmcut)),weight=weights.weight()[cut])
                    
                    ll_hCM = llcut.boost(-1*hcut.boostvec)#boost to higgs frame
                    poslep = make_p4(ak.where(lep1cut.charge>0,lep1cut,lep2cut))
                    poslep_hCM = poslep.boost(-1*hcut.boostvec)
                    poslep_ZCM = poslep_hCM.boost(-1*llcut.boostvec)
                    jet_hCM =jet1cut.boost(-1*hcut.boostvec)
                    jet_ZCM = jet_hCM.boost(-1*jjcut.boostvec)
                    
                    output['costheta_pz'].fill(dataset=dataset,lepflav=ch,costheta=flatten(np.cos(ll_hCM.theta)),weight=weights.weight()[cut]) 
                    output['costheta_ll'].fill(dataset=dataset,lepflav=ch,costheta=flatten(np.cos(poslep_ZCM.theta)),weight=weights.weight()[cut]) 
                    output['costheta_pz'].fill(dataset=dataset,lepflav=ch,costheta=flatten(np.cos(jet_ZCM.theta)),weight=weights.weight()[cut]) 
                    output['phi_zz'].fill(dataset=dataset,lepflav=ch,phi=flatten(ll_hCM.phi),weight=weights.weight()[cut]) 
                    output['phi_lq'].fill(dataset=dataset,lepflav=ch,phi=flatten(jet_ZCM.delta_phi(poslep_ZCM)),weight=weights.weight()[cut])
                    
        return output

    def postprocess(self, accumulator):
        return accumulator
