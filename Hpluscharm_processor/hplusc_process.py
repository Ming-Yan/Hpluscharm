from dataclasses import dataclass
import gzip
import pickle, os, sys, mplhep as hep, numpy as np

from matplotlib.pyplot import jet

import coffea
from coffea import hist, processor
from coffea.nanoevents.methods import vector
import awkward as ak
# from utils.correction import *
from coffea.analysis_tools import Weights
from functools import partial
import numba
def flatten(ar): # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)

@numba.njit
def find_4lep(events_leptons, builder):
    """Search for valid 4-lepton combinations from an array of events * leptons {charge, ...}

    A valid candidate has two pairs of leptons that each have balanced charge
    Outputs an array of events * candidates {indices 0..3} corresponding to all valid
    permutations of all valid combinations of unique leptons in each event
    (omitting permutations of the pairs)
    """
    for leptons in events_leptons:
        builder.begin_list()
        nlep = len(leptons)
        for i0 in range(nlep):
            for i1 in range(i0 + 1, nlep):
                if leptons[i0].charge + leptons[i1].charge != 0:
                    continue
                for i2 in range(nlep):
                    for i3 in range(i2 + 1, nlep):
                        if len({i0, i1, i2, i3}) < 4:
                            continue
                        if leptons[i2].charge + leptons[i3].charge != 0:
                            continue
                        builder.begin_tuple(4)
                        builder.index(0).integer(i0)
                        builder.index(1).integer(i1)
                        builder.index(2).integer(i2)
                        builder.index(3).integer(i3)
                        builder.end_tuple()
        builder.end_list()
    return builder
@numba.njit
def find_2lep(events_leptons, builder):
    """Search for valid 2-lepton combinations from an array of events * leptons {charge, ...}

    A valid candidate has two pairs of leptons that each have balanced charge
    Outputs an array of events * candidates {indices 0 1} corresponding to all valid
    permutations of all valid combinations of unique leptons in each event
    (omitting permutations of the pairs)
    """
    for leptons in events_leptons:
        builder.begin_list()
        nlep = len(leptons)
        for i0 in range(nlep):
            for i1 in range(i0 + 1, nlep):
                if leptons[i0].charge + leptons[i1].charge != 0:
                    continue
                if len({i0, i1}) < 2:
                    continue
                builder.begin_tuple(2)
                builder.index(0).integer(i0)
                builder.index(1).integer(i1)
                builder.end_tuple()
        builder.end_list()
    return builder
class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self):        
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        # cutflow_axis   = hist.Cat("cut",   "Cut")

        # Events
        njet_axis  = hist.Bin("nj",  r"N jets",      [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_axis = hist.Bin("nbj", r"N b-jets",    [0,1,2,3,4,5,6,7,8,9,10])            
        ncjet_axis = hist.Bin("nbj", r"N b-jets",    [0,1,2,3,4,5,6,7,8,9,10])  


       
        pt_axis   = hist.Bin("pt",   r" $p_{T}$ [GeV]", 50, 0, 200)
        eta_axis  = hist.Bin("eta",  r" $\eta$", 25, -2.5, 2.5)
        phi_axis  = hist.Bin("phi",  r" $\phi$", 30, -3, 3)
        mass_axis = hist.Bin("mass", r" $m$ [GeV]", 50, 0, 200)
        
        dr_axis = hist.Bin("dr","$\Delta$R",20,0,5)
    
       
        disc_list = [ 'btagDeepCvL', 'btagDeepCvB','btagDeepFlavCvB','btagDeepFlavCvL']
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d , 50, 0, 1))  
        _hist_event_dict = {
                'nj'  : hist.Hist("Counts", dataset_axis,  njet_axis),
                'nbj' : hist.Hist("Counts", dataset_axis,  nbjet_axis),
                'ncj' : hist.Hist("Counts", dataset_axis, ncjet_axis),
                'zs_dr'  : hist.Hist("Counts", dataset_axis,  dr_axis),
                'hj_dr'  : hist.Hist("Counts", dataset_axis, dr_axis),
            }
        objects=['jet','higgs','z1','z2','lep1','lep2','lep3','lep4']
        
        for i in objects:
            if i == 'jet' : 
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, flav_axis, pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, flav_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis, flav_axis, phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, flav_axis, mass_axis)
            else:
                _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis,  pt_axis)
                _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, eta_axis)
                _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis,  phi_axis)
                _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, mass_axis)
        
        for disc, axis in zip(disc_list,btag_axes):
            _hist_event_dict["%s" %(disc)] = hist.Hist("Counts", dataset_axis,flav_axis, axis)

        self.event_hists = list(_hist_event_dict.keys())
        _hist_dict = {}
        #}
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
        # Trigger level
        # triggers = [
        # # "HLT_IsoMu24",
        # "HLT_IsoMu24",
        # ]
        
        # trig_arrs = [events.HLT[_trig.strip("HLT_")] for _trig in triggers]
        # req_trig = np.zeros(len(events), dtype='bool')
        # for t in trig_arrs:
        #     req_trig = req_trig | t

        ############
        # Event level
        
        output['cutflow'][dataset]['all'] += len(events.Muon)
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1)]
        event_mu = events.Muon[(events.Muon.pt > 5) & (abs(events.Muon.eta < 2.4))& (events.Muon.tightId>=1)&(events.Muon.pfRelIso03_all<0.35)&(events.Muon.sip3d>4)&(events.Muon.dxy<0.5)&(events.Muon.dz<1)]
        
                  
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1)]
        event_e = events.Electron[(events.Electron.pt > 7) & (abs(events.Electron.eta) < 2.5)&(events.Electron.cutBased>3)&(events.Electron.sip3d<4)& (events.Electron.dxy<0.5)&(events.Electron.dz<1) ]
        #
        
        
        req_lep = (ak.count(event_e.pt,axis=1)>=4)|(ak.count(event_mu.pt,axis=1)>=4)| ((ak.count(event_mu.pt,axis=1)>=2)&(ak.count(event_e.pt,axis=1)>=2))
       

        # output['cutflow'][dataset]['selected leptons'] += ak.count(events[req_lep])
        #########
        # event_e = event_e[ak.num(event_e)>=2]
        # event_mu = event_mu[ak.num(event_mu)>=2]
       
        # event_4e = event_e[ak.num(event_e)>=4]
        # event_4mu = event_mu[ak.num(event_mu)>=4] 
        pair_2e = find_2lep(event_e, ak.ArrayBuilder()).snapshot()
        pair_2mu = find_2lep(event_mu, ak.ArrayBuilder()).snapshot()           
        pair_4mu = find_4lep(event_mu, ak.ArrayBuilder()).snapshot()
        pair_4e = find_4lep(event_e, ak.ArrayBuilder()).snapshot()
   
        ak.behavior.update(vector.behavior)
      
        masked2e2mu=(ak.num(pair_2mu)>0)&(ak.num(pair_2e)>0)
        masked4e = ak.num(pair_4e)>0
        masked4mu = ak.num(pair_4mu)>0
        pair_emu_2e = ak.mask(pair_2e,masked2e2mu)
        pair_emu_2mu = ak.mask(pair_2mu,masked2e2mu)
        pair_4e = ak.mask(pair_4e,masked4e)
        pair_4mu = ak.mask(pair_4mu,masked4mu)
        if(ak.all(ak.num(pair_2e)+ak.num(pair_2mu)<2)&ak.all(ak.num(pair_4e)==0)&ak.all(ak.num(pair_4mu)==0)):return output
        # print(ak.type(ak.num(pair_emu_2e)))
        # print(ak.type(pair_4e))
        print(ak.sum(ak.num(pair_emu_2e)>0),ak.sum(ak.num(pair_emu_2mu)>0))
        check_2e = pair_2e[ak.num(pair)]
        check_2mu = pair_emu_2mu[ak.num(pair_emu_2mu>0)]
        print(check_2e,check_2mu)
        if (ak.any(ak.num(pair_emu_2e)>0)&ak.any(ak.num(pair_emu_2mu))):
        # if (ak.any(ak.num(pair_2e)>0)&ak.any(ak.num(pair_2mu)>0)):
            # print(ak.type(pair_2e2mu),ak.type(pair_2e),ak.type(pair_2mu))
            pair_emu_2e = [event_e[pair_emu_2e[idx]] for idx in "01"]
            pair_emu_2mu = [event_mu[pair_emu_2mu[idx]] for idx in "01"]
        
            print(pair_emu_2e[0])
            print(pair_emu_2e[1])
            print(pair_emu_2mu[0])
            print(pair_emu_2mu[1])
            _hcand= pair_emu_2e[0] + pair_emu_2e[1] + pair_emu_2mu[0] + pair_emu_2mu[1]
            ak.behavior.update(vector.behavior)
            pair_4lep = ak.zip({
                    "z1": ak.zip({
                        "lep1": pair_emu_2mu[0],
                        "lep2": pair_emu_2mu[1],
                        "p4": pair_emu_2mu[0] + pair_emu_2mu[1],
                        "pt": (pair_emu_2mu[0] + pair_emu_2mu[1]).pt,
                        "eta": (pair_emu_2mu[0] + pair_emu_2mu[1]).eta,
                        "phi": (pair_emu_2mu[0] + pair_emu_2mu[1]).phi,
                        "mass": (pair_emu_2mu[0] + pair_emu_2mu[1]).mass,
                    }),
                    "z2": ak.zip({
                        "lep1": pair_emu_2e[0],
                        "lep2": pair_emu_2e[1],
                        "p4": pair_emu_2e[0] + pair_emu_2e[1],
                        "pt": (pair_emu_2e[0] + pair_emu_2e[1]).pt,
                        "eta": (pair_emu_2e[0] + pair_emu_2e[1]).eta,
                        "phi": (pair_emu_2e[0] + pair_emu_2e[1]).phi,
                        "mass": (pair_emu_2e[0] + pair_emu_2e[1]).mass,
                    }),
                    "cand" : ak.zip({
                        "pt": _hcand.pt,
                        "eta": _hcand.eta,
                        "phi": _hcand.phi,
                        "mass": _hcand.mass,
                    },with_name="PtEtaPhiMLorentzVector",)
                })
                
        elif ak.any(ak.num(pair_4mu)>0):
            pair_4mu = [event_mu[pair_4mu[idx]] for idx in "0123"]
            _hcand= pair_4mu[0] + pair_4mu[1] + pair_4mu[2] + pair_4mu[3]
            ak.behavior.update(vector.behavior)
            pair_4lep = ak.zip({
                "z1": ak.zip({
                    "lep1": pair_4mu[0],
                    "lep2": pair_4mu[1],
                    "p4": pair_4mu[0] + pair_4mu[1],
                    "pt": (pair_4mu[0] + pair_4mu[1]).pt,
                    "eta": (pair_4mu[0] + pair_4mu[1]).eta,
                    "phi": (pair_4mu[0] + pair_4mu[1]).phi,
                    "mass": (pair_4mu[0] + pair_4mu[1]).mass,
                }),
                "z2": ak.zip({
                    "lep1": pair_4mu[2],
                    "lep2": pair_4mu[3],
                    "p4": pair_4mu[2] + pair_4mu[3],
                    "pt": (pair_4mu[2] + pair_4mu[3]).pt,
                    "eta": (pair_4mu[2] + pair_4mu[3]).eta,
                    "phi": (pair_4mu[2] + pair_4mu[3]).phi,
                    "mass": (pair_4mu[2] + pair_4mu[3]).mass,
                }),
                "cand" : ak.zip({
                    "pt": _hcand.pt,
                    "eta": _hcand.eta,
                    "phi": _hcand.phi,
                    "mass": _hcand.mass,
                },with_name="PtEtaPhiMLorentzVector",)
            })

        elif ak.any(ak.num(pair_4e)>0):
            pair_4e = [event_e[pair_4e[idx]] for idx in "0123"]
            _hcand= pair_4e[0] + pair_4e[1] + pair_4e[2] + pair_4e[3]
            ak.behavior.update(vector.behavior)
            pair_4lep = ak.zip({
                "z1": ak.zip({
                    "lep1": pair_4e[0],
                    "lep2": pair_4e[1],
                    "p4": pair_4e[0] + pair_4e[1],
                    "pt": (pair_4e[0] + pair_4e[1]).pt,
                    "eta": (pair_4e[0] + pair_4e[1]).eta,
                    "phi": (pair_4e[0] + pair_4e[1]).phi,
                    "mass": (pair_4e[0] + pair_4e[1]).mass,
                }),
                "z2": ak.zip({
                    "lep1": pair_4e[2],
                    "lep2": pair_4e[3],
                    "p4": pair_4e[2] + pair_4e[3],
                    "pt": (pair_4e[2] + pair_4e[3]).pt,
                    "eta": (pair_4e[2] + pair_4e[3]).eta,
                    "phi": (pair_4e[2] + pair_4e[3]).phi,
                    "mass": (pair_4e[2] + pair_4e[3]).mass,
                }),
                "cand" : ak.zip({
                    "pt": _hcand.pt,
                    "eta": _hcand.eta,
                    "phi": _hcand.phi,
                    "mass": _hcand.mass,
                },with_name="PtEtaPhiMLorentzVector",)
            })
        else : return output
        req_zmass=(pair_4lep.z1.p4.mass>12)&(pair_4lep.z1.p4.mass<120)&(pair_4lep.z2.p4.mass>12)&(pair_4lep.z2.p4.mass<120)&((pair_4lep.z1.p4.mass>40)|(pair_4lep.z2.p4.mass>40))
        req_ghost_removal = (pair_4lep.z1.lep1.delta_r(pair_4lep.z1.lep2)>0.02) & (pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep1)>0.02)&(pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep2)>0.02)&(pair_4lep.z1.lep2.delta_r(pair_4lep.z2.lep1)>0.02)&(pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep2)>0.02)&(pair_4lep.z2.lep1.delta_r(pair_4lep.z2.lep2)>0.02)
        req_leppt = (((pair_4lep.z1.lep1.pt>20)&(pair_4lep.z1.lep2.pt>10))|((pair_4lep.z2.lep1.pt>20)&(pair_4lep.z2.lep2.pt>10)))
        req_hmass = (pair_4lep.z2.p4+pair_4lep.z1.p4).mass>70
        pair_4lep = pair_4lep[req_zmass&req_ghost_removal&req_leppt&req_hmass]
            
        best_z1 =  ak.singletons(ak.argmin(abs(pair_4lep.z1.p4.mass - 91.1876), axis=1))
           
        pair_4lep = pair_4lep[best_z1]
        output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(pair_4lep)>0)
        print(ak.count(pair_4lep),ak.num(pair_4lep)>0,ak.sum(ak.num(pair_4lep) > 0))
        ###########
        
        sel_jet = events.Jet[(events.Jet.pt > 25) & (abs(events.Jet.eta) <= 2.4)&((events.Jet.puId > 0)|(events.Jet.pt>50)) &(events.Jet.jetId>5)&ak.all(events.Jet.metric_table(pair_4lep.z1.lep1)>0.4,axis=2)&ak.all(events.Jet.metric_table(pair_4lep.z1.lep2)>0.4,axis=2)&ak.all(events.Jet.metric_table(pair_4lep.z2.lep1)>0.4,axis=2)&ak.all(events.Jet.metric_table(pair_4lep.z2.lep2)>0.4,axis=2)]
        sel_jet = ak.mask(sel_jet,ak.num(pair_4lep)>0)
        print(ak.count(sel_jet),ak.num(sel_jet)>0,ak.sum(ak.num(sel_jet) > 0))
        output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.num(sel_jet) > 0)
       
        output['nj'].fill(dataset=dataset,nj=flatten(ak.num(sel_jet)))
        output['zs_dr'].fill(dataset=dataset,dr=flatten(pair_4lep.z1.p4.delta_r(pair_4lep.z2.p4)))
        output['hj_dr'].fill(dataset=dataset,dr=flatten(pair_4lep.cand.metric_table(sel_jet)))
        for histname, h in output.items():
            if 'jet' in histname or 'btag' in histname:
                fields = {l: flatten(sel_jet[histname.replace('jet_','')]) for l in h.fields if l in dir(sel_jet)}
                h.fill(dataset=dataset,flav=flatten(sel_jet.hadronFlavour), **fields)
            elif 'lep1_' in histname:
                fields = {l: flatten(pair_4lep.z1.lep1[histname.replace('lep1_','')]) for l in h.fields if l in dir(pair_4lep.z1.lep1)}
                h.fill(dataset=dataset, **fields)
            elif 'lep2_' in histname:
                fields = {l: flatten(pair_4lep.z1.lep2[histname.replace('lep2_','')]) for l in h.fields if l in dir(pair_4lep.z1.lep2)}
                h.fill(dataset=dataset, **fields)
            elif 'lep3_' in histname:
                fields = {l: flatten(pair_4lep.z2.lep1[histname.replace('lep3_','')]) for l in h.fields if l in dir(pair_4lep.z2.lep2)}
                h.fill(dataset=dataset, **fields)
            elif 'lep4_' in histname:
                fields = {l: flatten(pair_4lep.z2.lep2[histname.replace('lep4_','')]) for l in h.fields if l in dir(pair_4lep.z2.lep2)}
                h.fill(dataset=dataset, **fields)  
            elif 'h_' in histname:
                fields = {l: flatten(pair_4lep.cand[histname.replace('h_','')]) for l in h.fields if l in dir(pair_4lep.cand)}
                h.fill(dataset=dataset, **fields)
            elif 'z1_' in histname:
                fields = {l: flatten(pair_4lep.z1[histname.replace('z1_','')]) for l in h.fields if l in dir(pair_4lep.z1)}
                h.fill(dataset=dataset, **fields)  
            elif 'z2_' in histname:
                fields = {l: flatten(pair_4lep.z2[histname.replace('z2_','')]) for l in h.fields if l in dir(pair_4lep.z2)}
                h.fill(dataset=dataset, **fields)  
        
        

        # #########
        

            
       

        return output

    def postprocess(self, accumulator):
        return accumulator
