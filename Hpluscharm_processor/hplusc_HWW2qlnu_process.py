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
from coffea.analysis_tools import Weights
from coffea.lumi_tools import LumiMask
from functools import partial

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
        self._corr = init_corr(self._year)
        
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        lepflav_axis = hist.Cat("lepflav",['ee','mumu'])
        region_axis = hist.Cat("region",['SR','WW_CR','Vjets_CR'])
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
                'mT' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mt_axis),
                'mTh' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mt_axis),
                'mjjl' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mt_axis),
                'dphi_lep':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, phi_axis),
                'dphi_ww':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, phi_axis),
            }
        objects=['cjet','lep','jet1','jet2','jj','lep',]
        
        
        for i in objects:
            if  'jet' in i: 
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis, pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, mass_axis)
            else:
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis ,pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis,  lepflav_axis,region_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, mass_axis)
            
        
        for disc, axis in zip(disc_list,btag_axes):
            _hist_event_dict["cjet_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis,flav_axis, axis)
            _hist_event_dict["jet1_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis,flav_axis, axis)
            _hist_event_dict["jet2_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis,flav_axis, axis)
        self.event_hists = list(_hist_event_dict.keys())
    
        self._accumulator = processor.dict_accumulator(
            {**_hist_event_dict,   
        'cutflow': processor.defaultdict_accumulator(
                # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int)),
                'cuts': processor.defaultdict_accumulator(
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
        else:output['sumw'][dataset] += ak.sum(events.genWeight/abs(events.genWeight))
        req_lumi=np.ones(len(events), dtype='bool')
        if(isRealData): req_lumi=self._lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumi',ak.to_numpy(req_lumi))
        del req_lumi
        
        weights = Weights(len(events), storeIndividual=True)
        if isRealData:weights.add('genweight',np.ones(len(events)))
        else:
            weights.add('genweight',events.genWeight/abs(events.genWeight))
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
        ##############
        if(isRealData):output['cutflow'][dataset]['all']  += 1.
        else:output['cutflow'][dataset]['all']  += ak.sum(abs(events.genWeight)/abs(events.genWeight))
        # output['cutflow'][dataset]['all'] +=1.
        trigger_ee = np.zeros(len(events), dtype='bool')
        trigger_mm = np.zeros(len(events), dtype='bool')
        for t in self._muhlt[self._year]:
            if t in events.HLT.fields:
                trigger_mm = trigger_mm | events.HLT[t]
        for t in self._ehlt[self._year]:
            if t in events.HLT.fields:
                trigger_ee = trigger_ee | events.HLT[t]
        
        selection.add('trigger_ee', ak.to_numpy(trigger_ee))
        selection.add('trigger_mumu', ak.to_numpy(trigger_mm))
        del trigger_ee, trigger_mm
        metfilter = np.ones(len(events), dtype='bool')
        for flag in self._met_filters[self._year]['data' if isRealData else 'mc']:
            metfilter &= np.array(events.Flag[flag])
        selection.add('metfilter', metfilter)
        del metfilter

        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon
        
        # reco_mu = gen_mu.nearest(event_mu)
        #[ak.argsort(events.Muon.pfRelIso04_all, axis=1)]
        musel = (event_mu.pt > 30) & (abs(event_mu.eta) < 2.4)&(event_mu.mvaId>=3) &(event_mu.pfRelIso04_all<0.35)&(abs(event_mu.dxy)<0.5)&(abs(event_mu.dz)<1)
        event_mu = event_mu[musel]

        event_mu= ak.pad_none(event_mu,1,axis=1)
        
        nmu = ak.sum(musel,axis=1)
        # print(dataset,ak.sum(nmu))
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        elesel = ((event_e.pt > 30) & (abs(event_e.eta) < 2.5))&(event_e.mvaFall17V2Iso_WP90==1)& (abs(event_e.dxy)<0.5)&(abs(event_e.dz)<1)
        event_e = event_e[elesel]
        event_e = ak.pad_none(event_e,1,axis=1)
        nele = ak.sum(elesel,axis=1)
        selection.add('lepsel',ak.to_numpy(((nele+nmu)>=1)))
        corr_jet =  jec(events,events.Jet,dataset,self._year,self._corr)
        event_jet = corr_jet[ak.argsort(corr_jet.btagDeepFlavCvL, axis=1,ascending=False)]
        
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
        # jj_cand = 
        jj_cand = jj_cand[ak.argsort(abs(jj_cand.mass-80.4), axis=1)]
        # print(jj_cand.mass)
        met = ak.zip({
                    "pt":  events.MET.pt,
                    "phi": events.MET.phi,
                    "eta": ak.zeros_like(events.MET.pt),
                    "mass": ak.zeros_like(events.MET.pt),
                    "energy":events.MET.sumEt,
                },with_name="PtEtaPhiMLorentzVector",)
        
        req_global =  (events.MET.pt>20) 
        req_dr = ak.any((make_p4(jj_cand.jet1).delta_r(good_leptons)>0.4),axis=-1)& ak.any((make_p4(jj_cand.jet2).delta_r(good_leptons)>0.4),axis=-1)
        req_wqqmass = ak.any((jj_cand.mass<116)&(jj_cand.mass>46),axis=-1)
       
        
        req_mT = mT(make_p4(good_leptons),met)>60 
        req_dphi =  ak.any(met.delta_phi(jj_cand)<np.pi/2.,axis=-1) 
        req_sr =  req_mT & req_dphi
        selection.add('global_selection',ak.to_numpy(req_global&req_dr))
        # selection.add('dr',ak.to_numpy(req_dr))
        # selection.add('wqq',ak.to_numpy())
        selection.add('SR',ak.to_numpy(req_sr&req_wqqmass))
        req_cr_WW = req_dr & req_mT &  (ak.any((jj_cand.mass>=116),axis=-1))
        req_cr_Vjets = req_dr &(mT(make_p4(good_leptons),met)<=60) & req_wqqmass
        selection.add('WW_CR',ak.to_numpy(req_cr_WW))
        selection.add('Vjets_CR',ak.to_numpy(req_cr_Vjets))
        mask2e =  req_sr&req_global & (ak.num(event_e)==1)& (event_e[:,0].pt>30) & req_wqqmass
        mask2mu =  req_sr&req_global & (ak.num(event_mu)==1)& (event_mu[:,0].pt>30)& req_wqqmass
        mask2lep = [ak.any(tup) for tup in zip(mask2mu, mask2e)]
        good_leptons = ak.mask(good_leptons,mask2lep)
        
        # output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(good_leptons)>0)
        
        selection.add('ee',ak.to_numpy(nele==1))
        selection.add('mumu',ak.to_numpy(nmu==1))
        
        nlep = 1
        if "2l2nu" in dataset : nlep=2       
        # ###########
        seljet = sel_cjet.delta_r(good_leptons)>0.4

        # print(ak.type(seljet))
        # sel_cjet = ak.mask(cjet,seljet)
        # sel_cjet
        # selection.add('cjetsel',ak.to_numpy(seljet))
        # output['cutflow'][dataset]['e candidates'] += ak.sum((nele>=1))
        # output['cutflow'][dataset]['mu candidates'] += ak.sum((nmu>=1))
        # output['cutflow'][dataset]['match e selection'] += ak.sum(ak.count(matche.pt,axis=-1)==nlep)
        # output['cutflow'][dataset]['match  mu selection'] += ak.sum(ak.count(matchmu.pt,axis=-1)==nlep)
        output['cutflow'][dataset]['njet candidates']+= ak.sum((njet>=3))
        output['cutflow'][dataset]['wqq mass'] += ak.sum((njet>=3)&req_wqqmass)

        output['cutflow'][dataset]['lep candidates'] += ak.sum((njet>=3)&req_wqqmass&(nmu+nele>=1))
        output['cutflow'][dataset]['MET'] += ak.sum((njet>=3)&req_wqqmass&(nmu+nele>=1)&req_global)
        output['cutflow'][dataset]['mT'] += ak.sum((njet>=3)&req_wqqmass&(nmu+nele>=1)&req_global&req_mT)
        output['cutflow'][dataset]['dr'] += ak.sum((njet>=3)&req_wqqmass&(nmu+nele>=1)&req_global&req_mT&req_dr)        
        output['cutflow'][dataset]['dphi'] += ak.sum((njet>=3)&req_wqqmass&(nmu+nele>=1)&req_global&req_mT&req_dr&req_dphi)  
        output['cutflow'][dataset]['c jets'] +=ak.sum((njet>=3)&req_wqqmass&(nmu+nele>=1)&req_global&req_mT&req_dr&seljet)
        # output['cutflow'][dataset]['jet eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nmu+nele>=1)&(njet>=3)&seljet&req_dr&(nmu+nele>=1)&(njet>=3))
        output['cutflow'][dataset]['electron eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nele==1)&seljet&req_dr&(nele==1)&(njet>=3))
        output['cutflow'][dataset]['muon eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nmu==1)&seljet&req_dr&(nmu==1)&(njet>=3))
        
        output['cuts'][dataset]['lep candidates'] += ak.sum(nmu+nele>=1)
        output['cuts'][dataset]['jet candidates'] += ak.sum((njet>=3))
        output['cuts'][dataset]['MET'] += ak.sum(req_global)
        output['cuts'][dataset]['dr'] += ak.sum(req_dr)
        output['cuts'][dataset]['wqq mass'] += ak.sum(req_wqqmass)  
        output['cuts'][dataset]['mt'] += ak.sum(req_mT)  
        output['cuts'][dataset]['dphi'] += ak.sum(req_dphi)  
        output['cuts'][dataset]['tag one jets'] +=ak.sum(seljet)
        # output['cuts'][dataset]['electron eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nele==1)&seljet&req_dr&(nmu+nele>=1)&(njet>=3))
        # output['cuts'][dataset]['muon eff'] +=ak.sum(req_wqqmass&req_sr&req_global&(nmu==1)&seljet&req_dr&(nmu+nele>=1)&(njet>=3))

        lepflav = ['ee','mumu']
        region = ['SR','Vjets_CR','WW_CR']
        for histname, h in output.items():
            for ch in lepflav:
                for r in region : 
                
                    cut = selection.all('jetsel','lepsel','global_selection',r,'lumi','metfilter',ch, 'trigger_%s'%(ch))
                    lepcut=good_leptons[cut]
                    jjcut = jj_cand[cut]
                    jjcut = jjcut[:,0]
                    jet1cut=jjcut.jet1
                    jet2cut=jjcut.jet2
                    if not isRealData:
                        if ch=='ee':lepsf=eleSFs(lepcut,self._year,self._corr)
                        elif ch=='mumu':lepsf=muSFs(lepcut,self._year,self._corr)
                    else : lepsf =weights.weight()[cut]
                    if 'cjet_' in histname:
                        fields = {l: normalize(sel_cjet[histname.replace('cjet_','')],cut) for l in h.fields if l in dir(sel_cjet)}
                        if isRealData:flavor= ak.zeros_like(normalize(sel_cjet['pt'],cut))
                        else :flavor= normalize(sel_cjet.hadronFlavour+1*((sel_cjet.partonFlavour == 0 ) & (sel_cjet.hadronFlavour==0)),cut)
                        
                        h.fill(dataset=dataset, lepflav =ch,flav=flavor,region=r, **fields,weight=weights.weight()[cut]*lepsf)    
                    elif 'jet1_' in histname:
                        fields = {l: flatten(jet1cut[histname.replace('jet1_','')]) for l in h.fields if l in dir(jet1cut)}
                        if isRealData:flavor= ak.zeros_like(flatten(jet1cut['pt']) )
                        else :flavor= flatten((jet1cut.hadronFlavour+1*((jet1cut.partonFlavour == 0 ) & (jet1cut.hadronFlavour==0))))
                        h.fill(dataset=dataset, lepflav =ch,flav=flavor,region=r, **fields,weight=weights.weight()[cut]*lepsf) 
                    elif 'jet2_' in histname:
                        fields = {l: flatten(jet2cut[histname.replace('jet2_','')]) for l in h.fields if l in dir(jet2cut)}
                        if isRealData:flavor= ak.zeros_like(flatten(jet2cut['pt']) )
                        else :flavor= flatten((jet2cut.hadronFlavour+1*((jet2cut.partonFlavour == 0 ) & (jet2cut.hadronFlavour==0))))
                        h.fill(dataset=dataset, lepflav =ch,flav=flavor,region=r, **fields,weight=weights.weight()[cut]*lepsf) 
                    elif 'jj_' in histname:
                        fields = {l:  flatten(jjcut[histname.replace('jj_','')]) for l in h.fields if l in dir(jjcut)}
                        h.fill(dataset=dataset, lepflav =ch,region=r, **fields,weight=weights.weight()[cut]*lepsf)
                    elif 'lep_' in histname:
                        fields = {l: ak.fill_none(lepcut[histname.replace('lep_','')],np.nan) for l in h.fields if l in dir(lepcut)}
                        h.fill(dataset=dataset,lepflav=ch,region=r, **fields,weight=weights.weight()[cut]*lepsf)
                    elif 'MET_' in histname:
                        fields = {l: normalize(events.MET[histname.replace('MET_','')],cut) for l in h.fields if l in dir(events.MET)}
                        h.fill(dataset=dataset, lepflav =ch,region=r, **fields,weight=weights.weight()[cut]*lepsf)  
                    else :
                        # output['nj'].fill(dataset=dataset,lepflav=ch,nj=normalize(ak.num(sel_jet),cut))               
                        # jjcut = ak.mask(jjcut,lepcut.pt>0)
                        # metcut =met[cut]
                        # print(mT(lepcut,met[cut]),lepcut.pt,met[cut].pt)
                        output['mT'].fill(dataset=dataset,lepflav=ch,region=r,mt=ak.fill_none(mT(lepcut,met[cut]),np.nan),weight=weights.weight()[cut]*lepsf)
                        output['mTh'].fill(dataset=dataset,lepflav=ch,region=r,mt=ak.fill_none((met[cut]+lepcut+jjcut).mass,np.nan),weight=weights.weight()[cut]*lepsf)
                        output['mjjl'].fill(dataset=dataset,lepflav=ch,region=r,mt=ak.fill_none((lepcut+jjcut).mass,np.nan),weight=weights.weight()[cut]*lepsf)
                        output['dphi_ww'].fill(dataset=dataset,lepflav=ch,region=r,phi=ak.fill_none(met[cut].delta_phi(jjcut),np.nan),weight=weights.weight()[cut]*lepsf)
                        output['dphi_lep'].fill(dataset=dataset,lepflav=ch,region=r,phi=ak.fill_none(met[cut].delta_phi(lepcut),np.nan),weight=weights.weight()[cut]*lepsf)
                    
        return output

    def postprocess(self, accumulator):
        return accumulator
