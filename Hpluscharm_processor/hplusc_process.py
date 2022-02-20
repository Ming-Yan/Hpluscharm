import csv
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

# from helpers.util import reduce_and, reduce_or, nano_mask_or, get_ht, normalize, make_p4

import numba

@numba.njit
def find_4lep(events_leptons, builder):
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
    def __init__(self):     
        self._muhlt = {
            '2016': [
                'HLT_IsoMu20',
                'HLT_IsoTkMu20',
                'HLT_IsoMu22',
                'HLT_IsoTkMu22',
                'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL',
                'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL',
                'HLT_TripleMu_12_10_5',
            ],
            '2017': [
                'HLT_IsoMu27',
                'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
                'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8',
                'HLT_TripleMu_10_5_5_DZ',  # redundant
                'HLT_TripleMu_12_10_5',
            ],
            '2018': [
                'HLT_IsoMu24',
                'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8',
                'HLT_TripleMu_10_5_5_DZ',  # redundant
                'HLT_TripleMu_12_10_5',
            ],
        }   
        self._ehlt = {
            '2016': [
                'HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL',
                'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL',
                'HLT_Ele25_eta2p1_WPTight',
                'HLT_Ele27_eta2p1_WPLoose_Gsf',
                'HLT_Ele27_WPTight',
            ],
            '2017': [
                'HLT_Ele35_WPTight_Gsf',
                'HLT_Ele38_WPTight_Gsf',
                'HLT_Ele40_WPTight_Gsf',
                'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'HLT_DoubleEle33_CaloIdL_MW',
                'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL',
            ],
            '2018': [
                'HLT_Ele32_WPTight_Gsf',
                'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'HLT_DoubleEle25_CaloIdL_MW',
            ],
        }   
        self._emuhlt =  {
            '2016': [
                'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL',
                'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL',
                'HLT_Mu17_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL',
                'HLT_Mu8_DiEle12_CaloIdL_TrackIdL',
                'HLT_DiMu9_Ele9_CaloIdL_TrackIdL',
            ],
            '2017': [
                'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',    
                'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ',            
                'HLT_Mu8_DiEle12_CaloIdL_TrackIdL',
                'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ',
            ],
            '2018': [
                'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',    
                'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ',            
                'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ',
            ],
        }   
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        # lepflav_axis = hist.Bin("lepflav", r"lepton flavor",[1,2,3])
        # cutflow_axis   = hist.Cat("cut",   "Cut")
        lepflav_axis = hist.Cat("lepflav",['ch4e','ch4mu',''])
        # Events
        njet_axis  = hist.Bin("nj",  r"N jets",      [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_axis = hist.Bin("nbj", r"N b-jets",    [0,1,2,3,4,5,6,7,8,9,10])            
        ncjet_axis = hist.Bin("nbj", r"N b-jets",    [0,1,2,3,4,5,6,7,8,9,10])  


       
        pt_axis   = hist.Bin("pt",   r" $p_{T}$ [GeV]", 50, 0, 200)
        eta_axis  = hist.Bin("eta",  r" $\eta$", 25, -2.5, 2.5)
        phi_axis  = hist.Bin("phi",  r" $\phi$", 30, -3, 3)
        mass_axis = hist.Bin("mass", r" $m$ [GeV]", 50, 0, 200)
        
        dr_axis = hist.Bin("dr","$\Delta$R",20,0,5)
    
        # axis.StrCategory([], name='region', growth=True),
        disc_list = [ 'btagDeepCvL', 'btagDeepCvB','btagDeepFlavCvB','btagDeepFlavCvL']#,'particleNetAK4_CvL','particleNetAK4_CvB']
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d , 50, 0, 1))  
        _hist_event_dict = {
                'nj'  : hist.Hist("Counts", dataset_axis,  lepflav_axis, njet_axis),
                # 'nbj' : hist.Hist("Counts", dataset_axis, lepflav_axis, nbjet_axis),
                # 'ncj' : hist.Hist("Counts", dataset_axis, lepflav_axis, ncjet_axis),
                'zs_dr'  : hist.Hist("Counts", dataset_axis,lepflav_axis, dr_axis),
                'hj_dr'  : hist.Hist("Counts", dataset_axis, lepflav_axis, dr_axis),
            }
        objects=['jetcsv','jetflav','jetpn','jetpt','higgs','z1','z2','lep1','lep2','lep3','lep4','jetcharm']
        
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

        # trig_arrs = [events.HLT[_trig.strip("HLT_")] for _trig in triggers]
        # req_trig = np.zeros(len(events), dtype='bool')
        # for t in trig_arrs:
        #     req_trig = req_trig | t

        
        ############
        ## Trigger cuts

        # trig_arrs = [events.HLT[_trig.strip("HLT_")] for _trig in triggers]
        # req_trig = np.zeros(len(events), dtype='bool')
        # for t in trig_arrs:
        #     req_trig = req_trig | t


        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon[ak.argsort(events.Muon.pt, axis=1,ascending=False)]
        musel = ((event_mu.pt > 5) & (abs(event_mu.eta) < 2.4)& ((event_mu.isGlobal==1)|(event_mu.isTracker==1))&(event_mu.pfRelIso03_all<0.35)&(event_mu.sip3d<10)&(abs(event_mu.dxy)<0.5)&(abs(event_mu.dz)<1))
        event_mu["lep_flav"] = 13*event_mu.charge
        
        event_mu = event_mu[musel]
        event_mu= ak.pad_none(event_mu,2,axis=1)
        nmu = ak.sum(musel,axis=1)
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron[ak.argsort(events.Electron.pt, axis=1,ascending=False)]
        event_e["lep_flav"] = 11*event_e.charge
        elesel = ((event_e.pt > 7) & (abs(event_e.eta) < 2.5)&(event_e.mvaFall17V2Iso_WPL==1)&(event_e.sip3d<10)& (abs(event_e.dxy)<0.5)&(abs(event_e.dz)<1))
        event_e = event_e[elesel]
        event_e = ak.pad_none(event_e,2,axis=1)
        nele = ak.sum(elesel,axis=1)
        # print(ak.type(ak.sum(nmu)==0))
        selection.add('lepsel',ak.to_numpy((nele>=4)|(nmu>=4)|((nele>=2)&(nmu>=2))))
        good_leptons = ak.with_name(
                ak.concatenate([event_e, event_mu], axis=1),
                "PtEtaPhiMCandidate",
            )
        # print(ak.type(event_e))

        # good_leptons = ak.concatenate([event_e, event_mu], axis=1)
        # print(ak.type(good_leptons))       
        # pair_4lep = ak.combinations(
        #         good_leptons,
        #         n=4,
        #         replacement=False,
        #         axis=-1,
        #         fields=["lep1", "lep2","lep3","lep4"]
        #     )
        # print((pair_4lep[0].lep_flav+pair_4lep[1].lep_flav==0).tolist())
        # print(ak.any(pair_4lep[0].lep_flav+pair_4lep[1].lep_flav==0,axis=-1))
        # print(ak.type(pair_4lep.fields))
        # print(ak.any(pair_4lep[0].lep_flav+pair_4lep[1].lep_flav==0).tolist())
        # if ak.any(pair_4lep[0].lep_flav+pair_4lep[1].lep_flav==0):
        #     if ak.any(abs((pair_4lep[0]+pair_4lep[1]).mass-91.18) < abs((pair_4lep.lep3+pair_4lep.lep4).mass-91.18)):
        #         tmpz1 = pair_4lep[0]+pair_4lep[1]
        #         tmpz2 = pair_4lep.lep3+pair_4lep.lep4
        #     else :
        #         tmpz2 = pair_4lep[0]+pair_4lep[1]
        #         tmpz1 = pair_4lep.lep3+pair_4lep.lep4
        # elif ak.any(pair_4lep[0].lep_flav+pair_4lep.lep3.lep_flav)==0:
        #     if ak.any(abs((pair_4lep[0]+pair_4lep.lep3).mass-91.18) <abs((pair_4lep[1]+pair_4lep.lep4).mass-91.18)):
        #         tmpz1 = pair_4lep[0]+pair_4lep.lep3
        #         tmpz2 = pair_4lep[1]+pair_4lep.lep4
        #     else :
        #         tmpz2 = pair_4lep[0]+pair_4lep.lep3
        #         tmpz1 = pair_4lep[1]+pair_4lep.lep4
        # else :
        #     if ak.any(abs((pair_4lep[0]+pair_4lep.lep4).mass-91.18) <abs((pair_4lep[1]+pair_4lep.lep3).mass-91.18)):
        #         tmpz1 = pair_4lep[0]+pair_4lep.lep4
        #         tmpz2 = pair_4lep[1]+pair_4lep.lep3
        #     else :
        #         tmpz2 = pair_4lep[0]+pair_4lep.lep4
        #         tmpz1 = pair_4lep[1]+pair_4lep.lep3
        # z1 = ak.zip({
        #             "pt": tmpz1.pt,
        #             "eta": tmpz1.eta,
        #             "phi": tmpz1.phi,
        #             "mass": tmpz1.mass,
        #         },with_name="PtEtaPhiMLorentzVector",)
        # z2 = ak.zip({
        #             "pt": tmpz2.pt,
        #             "eta": tmpz2.eta,
        #             "phi": tmpz2.phi,
        #             "mass": tmpz2.mass,
        #         },with_name="PtEtaPhiMLorentzVector",)
        fourmuon = find_4lep(event_e, ak.ArrayBuilder()).snapshot()

        fourmuon = [event_e[fourmuon[idx]] for idx in "0123"]
        # pair_4lep = find_4lep(event_e, ak.ArrayBuilder()).snapshot()
        # pair_4lep = [good_leptons[pair_4lep[idx]] for idx in "0123"]
        # ak.behavior.update(vector.behavior)
        # pair_4lep_comb = ak.zip({
        #     "z1" : ak.zip({
        #             "lep1": pair_4lep[0],
        #             "lep2": pair_4lep[1],
        #             "pt": (pair_4lep[0]+pair_4lep[1]).pt,
        #             "eta": (pair_4lep[0]+pair_4lep[1]).eta,
        #             "phi": (pair_4lep[0]+pair_4lep[1]).phi,
        #             "mass": (pair_4lep[0]+pair_4lep[1]).mass,
        #         },with_name="PtEtaPhiMLorentzVector",),
        #     "z2" :ak.zip({
        #                 "lep3": pair_4lep[2],
        #                 "lep4": pair_4lep[3],
        #                 "pt": (pair_4lep.lep3+pair_4lep.lep4).pt,
        #                 "eta": (pair_4lep.lep3+pair_4lep.lep4).eta,
        #                 "phi": (pair_4lep.lep3+pair_4lep.lep4).phi,
        #                 "mass": (pair_4lep.lep3+pair_4lep.lep4).mass,
        #             },with_name="PtEtaPhiMLorentzVector",),  
        #     "higgs" :  ak.zip({
        #                 "pt": (pair_4lep[0]+pair_4lep[1]+pair_4lep.lep3+pair_4lep.lep4).pt,
        #                 "eta": (pair_4lep[0]+pair_4lep[1]+pair_4lep.lep3+pair_4lep.lep4).eta,
        #                 "phi": (pair_4lep[0]+pair_4lep[1]+pair_4lep.lep3+pair_4lep.lep4).phi,
        #                 "mass": (pair_4lep[0]+pair_4lep[1]+pair_4lep.lep3+pair_4lep.lep4).mass,
        #             },with_name="PtEtaPhiMLorentzVector",)
        # })
        '''
        bestz1 = ak.singletons(ak.argmin(abs(pair_4lep_comb.z1.mass - 91.1876), axis=1))
        pair_4lep_comb = pair_4lep_comb[bestz1]
        req_opp_charge = ak.any((pair_4lep[0].lep_flav+pair_4lep[1].lep_flav+pair_4lep.lep3.lep_flav+pair_4lep.lep4.lep_flav==0),axis=-1,mask_identity=False)
        # req_zmass = ak.any((((z1.mass>12)&(z1.mass<120))&((z2.mass>12)&(z2.mass<120))&(z1.mass>40)),axis=-1,mask_identity=False)
        req_zmass = ak.any((((pair_4lep[0]+pair_4lep[1]).mass>12) & ((pair_4lep[0]+pair_4lep[1]).mass<120) &((pair_4lep.lep3+pair_4lep.lep4).mass>12)&((pair_4lep.lep3+pair_4lep.lep4).mass<120)&(((pair_4lep[0]+pair_4lep[1]).mass>40)|((pair_4lep.lep3+pair_4lep.lep4).mass>40)))|(((pair_4lep[0]+pair_4lep.lep3).mass>12) & ((pair_4lep[0]+pair_4lep.lep3).mass<120) &((pair_4lep.lep4+pair_4lep[1]).mass>12)&((pair_4lep.lep4+pair_4lep[1]).mass<120)&(((pair_4lep.lep4+pair_4lep[1]).mass>40)|((pair_4lep.lep3+pair_4lep[0]).mass>40)))|(((pair_4lep[0]+pair_4lep.lep4).mass>12) & ((pair_4lep[0]+pair_4lep.lep4).mass<120) &((pair_4lep.lep3+pair_4lep[1]).mass>12)&((pair_4lep.lep3+pair_4lep[1]).mass<120)&(((pair_4lep.lep3+pair_4lep[1]).mass>40)|((pair_4lep.lep4+pair_4lep[0]).mass>40))),axis=-1)

        # req_ghost_removal = ak.any((make_p4(pair_4lep[0]).delta_r(make_p4(pair_4lep[1]))>0.02)&(make_p4(pair_4lep[1]).delta_r(make_p4(pair_4lep.lep3))>0.02)&(make_p4(pair_4lep[0]).delta_r(make_p4(pair_4lep.lep4))>0.02)&(make_p4(pair_4lep.lep3).delta_r(make_p4(pair_4lep.lep4))>0.02),axis=-1,mask_identity=False)
        req_ghost_removal = ak.any(((pair_4lep[0]).delta_r((pair_4lep[1]))>0.02)&((pair_4lep[1]).delta_r((pair_4lep.lep3))>0.02)&((pair_4lep[0]).delta_r((pair_4lep.lep4))>0.02)&((pair_4lep.lep3).delta_r((pair_4lep.lep4))>0.02),axis=-1,mask_identity=False)

        req_hmass = ak.any((pair_4lep[0]+pair_4lep[1]+pair_4lep.lep3+pair_4lep.lep4).mass>70,axis=-1,mask_identity=False)
        

        # print((req_ghost_removal&req_opp_charge&req_zmass&req_hmass).tolist())
        selection.add('fourpair',ak.to_numpy(req_ghost_removal&req_opp_charge&req_zmass&req_hmass))
        mask4e =  req_ghost_removal&req_opp_charge&req_zmass&req_hmass & (ak.num(event_e)>=4)& (event_e[:,0].pt>20) & (event_e[:,1].pt>10)
        mask4mu =  req_ghost_removal&req_opp_charge&req_zmass&req_hmass & (ak.num(event_mu)>=4)& (event_mu[:,0].pt>20) &(event_mu[:,1].pt>10)
        mask2e2mu = req_ghost_removal&req_opp_charge&req_zmass&req_hmass & (ak.num(event_e)>=2)& (ak.num(event_mu) >=2 )& (((event_mu[:,0].pt>20)&(event_mu[:,1].pt>10))|((event_e[:,0].pt>20)&(event_e[:,1].pt>10)))
        
        mask4lep = [ak.any(tup) for tup in zip(mask2e2mu, mask4mu, mask4e)]
        pair_4lep = ak.mask(pair_4lep,mask4lep)
       
        
        output['cutflow'][dataset]['selected Z pairs'] += ak.sum(ak.num(pair_4lep)>0)
        selection.add('ch4e',ak.to_numpy((ak.num(event_e)>=4)& (event_e[:,0].pt>20) & (event_e[:,1].pt>10)))
        selection.add('ch4mu',ak.to_numpy((ak.num(event_mu)>=4)& (event_mu[:,0].pt>20) &(event_mu[:,1].pt>10)))
        selection.add('ch2e2mu',ak.to_numpy((ak.num(event_e)>=2)& (ak.num(event_mu) >=2 )& (((event_mu[:,0].pt>20)&(event_mu[:,1].pt>10))|((event_e[:,0].pt>20)&(event_e[:,1].pt>10)))))
        
               
        # ###########
        seljet = (events.Jet.pt > 20) & (abs(events.Jet.eta) <= 2.4)&((events.Jet.puId > 0)|(events.Jet.pt>50)) &(events.Jet.jetId>5)&ak.all(events.Jet.metric_table(pair_4lep[0])>0.4,axis=2)&ak.all(events.Jet.metric_table(pair_4lep[1])>0.4,axis=2)&ak.all(events.Jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(events.Jet.metric_table(pair_4lep.lep4)>0.4,axis=2)
        selection.add('jetsel',ak.to_numpy(ak.sum(seljet,axis=1)>0))
        eventcsv_jet = events.Jet[ak.argsort(events.Jet.btagDeepCvL,axis=1,ascending=False)]
        eventflav_jet = events.Jet[ak.argsort(events.Jet.btagDeepFlavCvL,axis=1,ascending=False)]
        # eventpn_jet = events.Jet[ak.argsort(events.Jet.particleNetAK4_CvL,axis=1,ascending=False)]
        eventpt_jet = events.Jet[ak.argsort(events.Jet.pt,axis=1,ascending=False)]

        sel_jet = eventcsv_jet[(eventcsv_jet.pt > 20) & (abs(eventcsv_jet.eta) <= 2.4)&((eventcsv_jet.puId > 0)|(eventcsv_jet.pt>50)) &(eventcsv_jet.jetId>5)&ak.all(eventcsv_jet.metric_table(pair_4lep[0])>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(pair_4lep[1])>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        sel_jet = ak.mask(sel_jet,ak.num(pair_4lep)>0)
        pair_4lep = ak.mask(pair_4lep,ak.num(eventcsv_jet)>0)
        sel_cjet_csv = ak.pad_none(sel_jet,1,axis=1)
        sel_cjet_csv= sel_cjet_csv[:,0]

        sel_jetflav =  eventflav_jet[(eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 0)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all(eventflav_jet.metric_table(pair_4lep[0])>0.4,axis=2)&ak.all(eventflav_jet.metric_table(pair_4lep[1])>0.4,axis=2)&ak.all(eventflav_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventflav_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        sel_jetflav = ak.mask(sel_jetflav,ak.num(pair_4lep)>0)
        sel_cjet_flav = ak.pad_none(sel_jetflav,1,axis=1)
        sel_cjet_flav = sel_cjet_flav[:,0]

        # sel_jetpn =  eventpn_jet[(eventpn_jet.pt > 20) & (abs(eventpn_jet.eta) <= 2.4)&((eventpn_jet.puId > 0)|(eventpn_jet.pt>50)) &(eventpn_jet.jetId>5)&ak.all(eventpn_jet.metric_table(pair_4lep[0])>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep[1])>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        # sel_jetpn = ak.mask(sel_jetpn,ak.num(pair_4lep)>0)
        # sel_cjet_pn = ak.pad_none(sel_jetpn,1,axis=1)
        # sel_cjet_pn = sel_cjet_pn[:,0]

        sel_jetpt =  eventpt_jet[(eventpt_jet.pt > 20) & (abs(eventpt_jet.eta) <= 2.4)&((eventpt_jet.puId > 0)|(eventpt_jet.pt>50)) &(eventpt_jet.jetId>5)&ak.all(eventpt_jet.metric_table(pair_4lep[0])>0.4,axis=2)&ak.all(eventpt_jet.metric_table(pair_4lep[1])>0.4,axis=2)&ak.all(eventpt_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventpt_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        
        sel_jetpt = ak.mask(sel_jetpt,ak.num(pair_4lep)>0)
        sel_cjet_pt = ak.pad_none(sel_jetpt,1,axis=1)
        sel_cjet_pt = sel_cjet_pt[:,0]
        
        
        output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.num(sel_jet) > 0)
        # output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.num(sel_jet) > 0)

        lepflav = ['ch4e','ch4mu','ch2e2mu']
        
        for histname, h in output.items():
            for ch in lepflav:
                cut = selection.all('jetsel','lepsel','fourpair',ch)
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
                    lep1cut=pair_4lep[0][cut]           
                    fields = {l: ak.fill_none(ak.flatten(lep1cut[histname.replace('lep1_','')]),np.nan) for l in h.fields if l in dir(lep1cut)}

                    h.fill(dataset=dataset,lepflav=ch, **fields)
                elif 'lep2_' in histname:
                    lep2cut=pair_4lep[1][cut]
                    fields = {l: ak.fill_none(ak.flatten(lep2cut[histname.replace('lep2_','')]),np.nan) for l in h.fields if l in dir(lep2cut)}
                    h.fill(dataset=dataset,lepflav=ch, **fields)
                elif 'lep3_' in histname:
                    lep3cut=pair_4lep.lep3[cut]
                    fields = {l: ak.fill_none(ak.flatten(lep3cut[histname.replace('lep3_','')]),np.nan) for l in h.fields if l in dir(lep3cut)}
                    h.fill(dataset=dataset,lepflav=ch, **fields)
                elif 'lep4_' in histname:
                    lep4cut=pair_4lep.lep4[cut]
                    fields = {l: ak.fill_none(ak.flatten(lep4cut[histname.replace('lep4_','')]),np.nan) for l in h.fields if l in dir(lep4cut)}
                    h.fill(dataset=dataset,lepflav=ch, **fields)  
                elif 'higgs_' in histname:
                    higgscut=higgs[cut]
                    fields = {l: flatten(higgscut[histname.replace('higgs_','')]) for l in h.fields if l in dir(higgs)}
                    h.fill(dataset=dataset, lepflav=ch,**fields)
                elif 'z1_' in histname:
                    z1cut = z1[cut]
                    fields = {l: flatten(z1cut[histname.replace('z1_','')]) for l in h.fields if l in dir(z1cut) }
                    h.fill(dataset=dataset,lepflav=ch, **fields)  
                elif 'z2_' in histname:
                    z2cut= z2[cut]
                    fields = {l: flatten(z2cut[histname.replace('z2_','')]) for l in h.fields if l in dir(z2cut)}
                    h.fill(dataset=dataset,lepflav=ch, **fields) 
                else :
                    output['nj'].fill(dataset=dataset,lepflav=ch,nj=normalize(ak.num(sel_jet),cut))
                    output['zs_dr'].fill(dataset=dataset,lepflav =ch, dr=flatten(z1[cut].delta_r(z2[cut])))
                    output['hj_dr'].fill(dataset=dataset,lepflav =ch,dr=flatten(higgs[cut].delta_r(sel_cjet_flav[cut])))
        '''
        return output

    def postprocess(self, accumulator):
        return accumulator
