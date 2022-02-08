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
        event_mu = events.Muon[(events.Muon.pt > 5) & (abs(events.Muon.eta < 2.4))& (events.Muon.tightId>=1)&(events.Muon.pfRelIso03_all<0.35)&(events.Muon.sip3d>4)&(events.Muon.dxy<0.5)&(events.Muon.dz<1)]
        
                  
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[(events.Electron.pt > 7) & (abs(events.Electron.eta) < 2.5)&(events.Electron.cutBased>3)&(events.Electron.sip3d<4)& (events.Electron.dxy<0.5)&(events.Electron.dz<1) ]
        
        
        req_lep = (ak.count(event_e.pt,axis=1)>=4)|(ak.count(event_mu.pt,axis=1)>=4)| ((ak.count(event_mu.pt,axis=1)>=2)&(ak.count(event_e.pt,axis=1)>=2))
       

        output['cutflow'][dataset]['selected leptons'] += ak.sum(ak.num(events[req_lep].Muon))
        #########
        event_e = event_e[ak.num(event_e)>=2]
        event_mu = event_mu[ak.num(event_mu)>=2]
        event_e = event_e[ak.argsort(event_e.pt,axis=1)]
        event_mu = event_mu[ak.argsort(event_mu.pt,axis=1)]
        event_4e = event_e[ak.num(event_e)>=4]
        event_4mu = event_mu[ak.num(event_mu)>=4] 
        pair_2e = find_2lep(event_e, ak.ArrayBuilder()).snapshot()
        pair_2mu = find_2lep(event_mu, ak.ArrayBuilder()).snapshot()           
        pair_4mu = find_4lep(event_4mu, ak.ArrayBuilder()).snapshot()
        pair_4e = find_4lep(event_4e, ak.ArrayBuilder()).snapshot()
        ak.behavior.update(vector.behavior)
        if (ak.all(ak.num(pair_2e))&ak.all(ak.num(pair_2mu))):
            pair_2mu  = [event_e[pair_2mu[idx]] for idx in "01"]
            pair_2e  = [event_mu[pair_2e[idx]] for idx in "01"]
            _hcand= pair_2e[0] + pair_2e[1] + pair_2mu[0] + pair_2mu[1]
            ak.behavior.update(vector.behavior)
            pair_4lep = ak.zip({
                "z1": ak.zip({
                    "lep1": pair_2mu[0],
                    "lep2": pair_2mu[1],
                    "p4": pair_2mu[0] + pair_2mu[1],
                    "pt": (pair_2mu[0] + pair_2mu[1]).pt,
                    "eta": (pair_2mu[0] + pair_2mu[1]).eta,
                    "phi": (pair_2mu[0] + pair_2mu[1]).phi,
                    "mass": (pair_2mu[0] + pair_2mu[1]).mass,
                }),
                "z2": ak.zip({
                    "lep1": pair_2e[0],
                    "lep2": pair_2e[1],
                    "p4": pair_2e[0] + pair_2e[1],
                    "pt": (pair_2e[0] + pair_2e[1]).pt,
                    "eta": (pair_2e[0] + pair_2e[1]).eta,
                    "phi": (pair_2e[0] + pair_2e[1]).phi,
                    "mass": (pair_2e[0] + pair_2e[1]).mass,
                }),
                 "cand" : ak.zip({
                    "pt": _hcand.pt,
                    "eta": _hcand.eta,
                    "phi": _hcand.phi,
                    "mass": _hcand.mass,
                },with_name="PtEtaPhiMLorentzVector",)
            })
                
        elif ak.num(pair_4mu,axis=0) > 0:
            pair_4mu = [event_4mu[pair_4mu[idx]] for idx in "0123"]
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

        elif ak.num(pair_4e,axis=0) > 0:
            pair_4e = [event_4e[pair_4e[idx]] for idx in "0123"]
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
        else: return output
        req_zmass=(pair_4lep.z1.p4.mass>12)&(pair_4lep.z1.p4.mass<120)&(pair_4lep.z2.p4.mass>12)&(pair_4lep.z2.p4.mass<120)&((pair_4lep.z1.p4.mass>40)|(pair_4lep.z2.p4.mass>40))
        req_ghost_removal = (pair_4lep.z1.lep1.delta_r(pair_4lep.z1.lep2)>0.02) & (pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep1)>0.02)&(pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep2)>0.02)&(pair_4lep.z1.lep2.delta_r(pair_4lep.z2.lep1)>0.02)&(pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep2)>0.02)&(pair_4lep.z2.lep1.delta_r(pair_4lep.z2.lep2)>0.02)
        req_leppt = (((pair_4lep.z1.lep1.pt>20)&(pair_4lep.z1.lep2.pt>10))|((pair_4lep.z2.lep1.pt>20)&(pair_4lep.z2.lep2.pt>10)))
        req_hmass = (pair_4lep.z2.p4+pair_4lep.z1.p4).mass>70
        pair_4lep = pair_4lep[req_zmass&req_ghost_removal&req_leppt&req_hmass]
            
        best_z1 =  ak.singletons(ak.argmin(abs(pair_4lep.z1.p4.mass - 91.1876), axis=1))
           
        pair_4lep = pair_4lep[best_z1]
        print(ak.to_list(pair_4lep))
        # print(ak.to_list(events.Jet))
        output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(pair_4lep))
        ###########
        
        # print(pair_4lep.z1.lep1.metric_table(events.Jet,axis=2))
        # print(ak.to_list(pair_4lep))
        # print(ak.to_list(events.Jet))
        # print(events.Jet.nearest(pair_4lep.z1.lep1,axis=2))
        print(pair_4lep.z1.lep1.metric_table(events.Jet))
        sel_jet = events.Jet[(events.Jet.pt > 25) & (abs(events.Jet.eta) <= 2.4)&((events.Jet.puId > 0)|(events.Jet.pt>50)) &(events.Jet.jetId>5)]
        
        # req_jets = (ak.count(event_jet.puId,axis=1) >= 1) 
        # &(events.Jet.nearest(pair_4lep.z1.lep2,axis=2,threshold=0.4))&(events.Jet.nearest(pair_4lep.z2.lep1,axis=2,threshold=0.4))&(events.Jet.nearest(pair_4lep.z2.lep2,axis=2,threshold=0.4))
        output['cutflow'][dataset]['selected jets'] += ak.sum(ak.num(sel_jet))
        
        print(ak.type(sel_jet))        
       
         # output['nj'].fill(dataset=dataset,nj=ak.flatten(ak.num(maskjet),axis=None))
            # output['zs_dr'].fill(dataset=dataset,dr=ak.flatten(pair_4lep.z1.p4.delta_r(pair_4lep.z2.p4)))
            # output['z1_pt'].fill(dataset=dataset,pt=ak.flatten(pair_4lep.z1.p4.pt))
            # output['z1_eta'].fill(dataset=dataset,eta=ak.flatten(pair_4lep.z1.p4.eta))
            # output['z1_phi'].fill(dataset=dataset,phi=ak.flatten(pair_4lep.z1.p4.phi))
            # output['z1_mass'].fill(dataset=dataset,mass=ak.flatten(pair_4lep.z1.p4.mass))
            # output['z2_pt'].fill(dataset=dataset,pt=ak.flatten(pair_4lep.z2.p4.pt))
            # output['z2_eta'].fill(dataset=dataset,eta=ak.flatten(pair_4lep.z2.p4.eta))
            # output['z2_phi'].fill(dataset=dataset,phi=ak.flatten(pair_4lep.z2.p4.phi))
            # output['z2_mass'].fill(dataset=dataset,mass=ak.flatten(pair_4lep.z2.p4.mass))
            # output['higgs_pt'].fill(dataset=dataset,pt=ak.flatten(Higgs.pt))
            # output['higgs_eta'].fill(dataset=dataset,eta=ak.flatten(Higgs.eta))
            # output['higgs_phi'].fill(dataset=dataset,phi=ak.flatten(Higgs.phi))
            # output['higgs_mass'].fill(dataset=dataset,mass=ak.flatten(Higgs.mass))
            # genweiev=ak.flatten(ak.broadcast_arrays(weights.weight()[event_level],maskjet.metric_table(maskH))[0])
            # print("jhdrL",maskjet.metric_table(maskH))
            # output['hj_dr'].fill(dataset=dataset,dr=ak.flatten(maskjet.metric_table(maskH),axis=None))
            # for histname, h in output.items():
            #     if 'jet' in histname or 'btag' in histname:
            #         if(isRealData):
            #             fields = {l: ak.flatten(sel_jet[l], axis=None) for l in h.fields if l in dir(sel_jet)}
            #             h.fill(dataset=dataset,flav=5, **fields)
                        
            #         else:
            #             fields = {l: ak.flatten(sel_jet[histname.replace('jet_','')]) for l in h.fields if l in dir(sel_jet)}
            #             genweiev=ak.flatten(ak.broadcast_arrays(weights.weight()[event_level],sel_jet['pt'])[0])
            #             h.fill(dataset=dataset,flav=ak.flatten(sel_jet.hadronFlavour), **fields)
            #     elif 'lep1_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z1.lep1[histname.replace('lep1_','')]) for l in h.fields if l in dir(pair_4lep.z1.lep1)}
            #         genweiev=ak.flatten(ak.broadcast_arrays(weights.weight()[event_level],pair_4lep.z1.lep1['pt'])[0])
            #         h.fill(dataset=dataset, **fields)
            #     elif 'lep2_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z1.lep2[histname.replace('lep2_','')]) for l in h.fields if l in dir(pair_4lep.z1.lep2)}
            #         h.fill(dataset=dataset, **fields)
            #     elif 'lep3_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z2.lep1[histname.replace('lep3_','')]) for l in h.fields if l in dir(pair_4lep.z2.lep2)}
            #         h.fill(dataset=dataset, **fields)
            #     elif 'lep4_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z2.lep2[histname.replace('lep4_','')]) for l in h.fields if l in dir(pair_4lep.z2.lep2)}
            #         h.fill(dataset=dataset, **fields)  
        
        # # output['cutflow'][dataset]['selected lepton&jets'] += ak.sum(ak.num(selev,axis=0))

        # #########
        
        # # Per muon
        # smu = selev.Muon[(selev.Muon.pt > 5) & (abs(selev.Muon.eta < 2.4))& (selev.Muon.tightId==1)&(selev.Muon.pfRelIso03_all<0.35)&(selev.Muon.sip3d<4)&(selev.Muon.dxy<0.5)&(selev.Muon.dz<1)]
        
        
                  
        # # ## Electron cuts
        # # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        # sele =  selev.Electron[(selev.Electron.pt > 7) & (abs(selev.Electron.eta) < 2.5)&(selev.Electron.mvaFall17V2Iso_WP90==1)&(selev.Electron.sip3d<4)& (selev.Electron.dxy<0.5)&(selev.Electron.dz<1)]
        
        
        
        
        

        # # Per jet : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetID
        
        # sel_jet = selev.Jet[(selev.Jet.pt > 25) & (abs(selev.Jet.eta) <= 2.4)&((selev.Jet.puId > 0)|(selev.Jet.pt>50)) &(selev.Jet.jetId>5)]
        
        # if(ak.num(selev,axis=0)>0):
            
        #     sele = sele[ak.argsort(sele.pt,axis=1)]
        #     smu = smu[ak.argsort(smu.pt,axis=1)]
        #     print(ak.type(selev),ak.type(sele),ak.type(smu))
        #     pair_2e = find_2lep(sele, ak.ArrayBuilder()).snapshot()
        #     pair_2mu = find_2lep(smu, ak.ArrayBuilder()).snapshot()
        #     s4ele = sele[ak.num(sele)>=4]
        #     s4mu = smu[ak.num(smu)>=4]            
        #     pair_4mu = find_4lep(s4mu, ak.ArrayBuilder()).snapshot()
        #     pair_4e = find_4lep(s4ele, ak.ArrayBuilder()).snapshot()
        #     if (ak.all(ak.num(pair_2e))&ak.all(ak.num(pair_2mu))):
        #         pair_2mu  = [sele[pair_2mu[idx]] for idx in "01"]
        #         pair_2e  = [smu[pair_2e[idx]] for idx in "01"]
        #         pair_4lep = ak.zip({
        #         "z1": ak.zip({
        #             "lep1": pair_2mu[0],
        #             "lep2": pair_2mu[1],
        #             "p4": pair_2mu[0] + pair_2mu[1],
        #         }),
        #         "z2": ak.zip({
        #            "lep1": pair_2e[0],
        #             "lep2": pair_2e[1],
        #             "p4": pair_2e[0] + pair_2e[1],
        #         }),
        #     })
                
        #     elif ak.num(pair_4mu,axis=0) > 0:
        #         pair_4mu = [s4mu[pair_4mu[idx]] for idx in "0123"]
        #         pair_4lep = ak.zip({
        #         "z1": ak.zip({
        #             "lep1": pair_4mu[0],
        #             "lep2": pair_4mu[1],
        #             "p4": pair_4mu[0] + pair_4mu[1],
        #         }),
        #         "z2": ak.zip({
        #             "lep1": pair_4mu[2],
        #             "lep2": pair_4mu[3],
        #             "p4": pair_4mu[2] + pair_4mu[3],
        #         }),
        #     })

        #     elif ak.num(pair_4e,axis=0) > 0:
        #         pair_4e = [s4ele[pair_4e[idx]] for idx in "0123"]
        #         pair_4lep = ak.zip({
        #         "z1": ak.zip({
        #             "lep1": pair_4e[0],
        #             "lep2": pair_4e[1],
        #             "p4": pair_4e[0] + pair_4e[1],
        #         }),
        #         "z2": ak.zip({
        #             "lep1": pair_4e[2],
        #             "lep2": pair_4e[3],
        #             "p4": pair_4e[2] + pair_4e[3],
        #         }),
        #     })
        #     else:
        #         print("???") 
        #         return output
        #     print(ak.type(pair_4lep))
        #     print("pair_4lep",ak.num(pair_4lep),pair_4lep)
        #     zmass_cut=(pair_4lep.z1.p4.mass>12)&(pair_4lep.z1.p4.mass<120)&(pair_4lep.z2.p4.mass>12)&(pair_4lep.z2.p4.mass<120)&((pair_4lep.z1.p4.mass>40)|(pair_4lep.z2.p4.mass>40))
        #     ghost_removal = (pair_4lep.z1.lep1.delta_r(pair_4lep.z1.lep2)>0.02) & (pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep1)>0.02)&(pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep2)>0.02)&(pair_4lep.z1.lep2.delta_r(pair_4lep.z2.lep1)>0.02)&(pair_4lep.z1.lep1.delta_r(pair_4lep.z2.lep2)>0.02)&(pair_4lep.z2.lep1.delta_r(pair_4lep.z2.lep2)>0.02)
        #     leppt = (((pair_4lep.z1.lep1.pt>20)&(pair_4lep.z1.lep2.pt>10))|((pair_4lep.z2.lep1.pt>20)&(pair_4lep.z2.lep2.pt>10)))
        #     hmass = (pair_4lep.z2.p4+pair_4lep.z1.p4).mass>70
        #     pair_4lep = pair_4lep[zmass_cut&ghost_removal&leppt&hmass]
             
        #     best_z1 =  ak.singletons(ak.argmin(abs(pair_4lep.z1.p4.mass - 91.1876), axis=1))
           
        #     pair_4lep = pair_4lep[best_z1]

        #     Higgs=pair_4lep.z1.p4+pair_4lep.z2.p4
        #     print("Higgs",ak.num(Higgs),Higgs)
        #     print("seljet",ak.num(sel_jet),sel_jet)
        #     maskjet=ak.mask(sel_jet,ak.num(Higgs)>0)

        #     # genweiev=ak.flatten(ak.broadcast_ar rays(weights.weight()[event_level],pair_4lep.z2.p4.pt)[0])
        #     # Higgs= ak.pad_none(Higgs,len(selev),axis=-1)
        #     maskjet = maskjet[ak.all((maskjet.metric_table(pair_4lep.z1.lep1)>0.4),axis=2)&ak.all((maskjet.metric_table(pair_4lep.z1.lep2)>0.4),axis=2)&ak.all((maskjet.metric_table(pair_4lep.z2.lep1)>0.4),axis=2)&ak.all((maskjet.metric_table(pair_4lep.z2.lep2)>0.4),axis=2)]
        #     mask4lep = ak.mask(pair_4lep,ak.num(maskjet)>0)
        #     print("mask4lep",mask4lep)
            
            
            

            # output['nj'].fill(dataset=dataset,nj=ak.flatten(ak.num(maskjet),axis=None))
            # output['zs_dr'].fill(dataset=dataset,dr=ak.flatten(pair_4lep.z1.p4.delta_r(pair_4lep.z2.p4)))
            # output['z1_pt'].fill(dataset=dataset,pt=ak.flatten(pair_4lep.z1.p4.pt))
            # output['z1_eta'].fill(dataset=dataset,eta=ak.flatten(pair_4lep.z1.p4.eta))
            # output['z1_phi'].fill(dataset=dataset,phi=ak.flatten(pair_4lep.z1.p4.phi))
            # output['z1_mass'].fill(dataset=dataset,mass=ak.flatten(pair_4lep.z1.p4.mass))
            # output['z2_pt'].fill(dataset=dataset,pt=ak.flatten(pair_4lep.z2.p4.pt))
            # output['z2_eta'].fill(dataset=dataset,eta=ak.flatten(pair_4lep.z2.p4.eta))
            # output['z2_phi'].fill(dataset=dataset,phi=ak.flatten(pair_4lep.z2.p4.phi))
            # output['z2_mass'].fill(dataset=dataset,mass=ak.flatten(pair_4lep.z2.p4.mass))
            # output['higgs_pt'].fill(dataset=dataset,pt=ak.flatten(Higgs.pt))
            # output['higgs_eta'].fill(dataset=dataset,eta=ak.flatten(Higgs.eta))
            # output['higgs_phi'].fill(dataset=dataset,phi=ak.flatten(Higgs.phi))
            # output['higgs_mass'].fill(dataset=dataset,mass=ak.flatten(Higgs.mass))
            # genweiev=ak.flatten(ak.broadcast_arrays(weights.weight()[event_level],maskjet.metric_table(maskH))[0])
            # print("jhdrL",maskjet.metric_table(maskH))
            # output['hj_dr'].fill(dataset=dataset,dr=ak.flatten(maskjet.metric_table(maskH),axis=None))
            # for histname, h in output.items():
            #     if 'jet' in histname or 'btag' in histname:
            #         if(isRealData):
            #             fields = {l: ak.flatten(sel_jet[l], axis=None) for l in h.fields if l in dir(sel_jet)}
            #             h.fill(dataset=dataset,flav=5, **fields)
                        
            #         else:
            #             fields = {l: ak.flatten(sel_jet[histname.replace('jet_','')]) for l in h.fields if l in dir(sel_jet)}
            #             genweiev=ak.flatten(ak.broadcast_arrays(weights.weight()[event_level],sel_jet['pt'])[0])
            #             h.fill(dataset=dataset,flav=ak.flatten(sel_jet.hadronFlavour), **fields)
            #     elif 'lep1_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z1.lep1[histname.replace('lep1_','')]) for l in h.fields if l in dir(pair_4lep.z1.lep1)}
            #         genweiev=ak.flatten(ak.broadcast_arrays(weights.weight()[event_level],pair_4lep.z1.lep1['pt'])[0])
            #         h.fill(dataset=dataset, **fields)
            #     elif 'lep2_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z1.lep2[histname.replace('lep2_','')]) for l in h.fields if l in dir(pair_4lep.z1.lep2)}
            #         h.fill(dataset=dataset, **fields)
            #     elif 'lep3_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z2.lep1[histname.replace('lep3_','')]) for l in h.fields if l in dir(pair_4lep.z2.lep2)}
            #         h.fill(dataset=dataset, **fields)
            #     elif 'lep4_' in histname:
            #         fields = {l: ak.flatten(pair_4lep.z2.lep2[histname.replace('lep4_','')]) for l in h.fields if l in dir(pair_4lep.z2.lep2)}
            #         h.fill(dataset=dataset, **fields)  
            
       

        return output

    def postprocess(self, accumulator):
        return accumulator
