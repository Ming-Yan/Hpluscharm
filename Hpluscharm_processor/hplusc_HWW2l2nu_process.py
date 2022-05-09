import pickle, os, sys, mplhep as hep, numpy as np
from select import select
import coffea
from coffea import hist, processor
import awkward as ak
from utils.correction import add_jec_variables,muSFs,eleSFs,init_corr,puwei
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights
from functools import partial
# import numba
from helpers.util import make_p4

from utils.util import mT, flatten, normalize
# from utils.topmass import getnu4vec
# from config.config import *

class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, year="2017",version="test",export_array=False):    
        self._year=year
        self._version=version
        self._export_array=export_array
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
                'Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL',
                'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                'Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
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
        
        self._corr,self._jetfactory = init_corr(year)
        # Define axes
        # Should read axes from NanoAOD config

        dataset_axis = hist.Cat("dataset", "Primary dataset")
        flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
        lepflav_axis = hist.Cat("lepflav",['ee','mumu','emu'])
        region_axis = hist.Cat("region",['SR','SR2','top_CR','DY_CR'])
        # Events
        njet_axis  = hist.Bin("nj",  r"N jets",      [0,1,2,3,4,5,6,7])
        nalep_axis = hist.Bin("nalep",  r"N jets",      [0,1,2,3])
        # kinematic variables       
        pt_axis   = hist.Bin("pt",   r" $p_{T}$ [GeV]", 50, 0, 300)
        eta_axis  = hist.Bin("eta",  r" $\eta$", 25, -2.5, 2.5)
        phi_axis  = hist.Bin("phi",  r" $\phi$", 30, -3, 3)
        mass_axis = hist.Bin("mass", r" $m$ [GeV]", 50, 0, 300)
        mt_axis =  hist.Bin("mt", r" $m_{T}$ [GeV]", 30, 0, 300)
        dr_axis = hist.Bin("dr","$\Delta$R",20,0,5)
        iso_axis = hist.Bin("pfRelIso03_all", r"Rel. Iso", 40,0,4)
        dxy_axis = hist.Bin("dxy", r"d_{xy}", 40,-2,2)
        dz_axis = hist.Bin("dz", r"d_{z}", 40,0,10)
        # MET vars
        signi_axis = hist.Bin("significance", r"MET $\sigma$",20,0,10)
        
        # sumEt_axis = hist.Bin("sumEt", r" MET sumEt", 50, 0, 300)
        
        # axis.StrCategory([], name='region', growth=True),
        disc_list = [ 'btagDeepCvL', 'btagDeepCvB','btagDeepFlavCvB','btagDeepFlavCvL']#,'particleNetAK4_CvL','particleNetAK4_CvB']
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d , 50, 0, 1))  
        _hist_event_dict = {
                'nj'  : hist.Hist("Counts", dataset_axis,  lepflav_axis,region_axis, flav_axis,njet_axis),
                'nele'  : hist.Hist("Counts", dataset_axis,  lepflav_axis,region_axis, flav_axis,nalep_axis),
                'nmu'  : hist.Hist("Counts", dataset_axis,  lepflav_axis,region_axis, flav_axis,nalep_axis),
                'njmet'  : hist.Hist("Counts", dataset_axis,  lepflav_axis,region_axis, flav_axis,njet_axis),
                # 'MET_sumEt' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,sumEt_axis),
                'MET_significance' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,flav_axis,signi_axis),
                # 'MET_covXX' : 
                # st.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,covYY_axis),
                'MET_phi' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'MET_pt' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,pt_axis),
                'METTkMETdphi': hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'mT1' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,mt_axis),
                'mT2' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,mt_axis),
                'mTh':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,mt_axis),
                # 'W1_phi' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                # 'W2_phi' : hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                # 'H_phi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l1l2_dr': hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,dr_axis),
                'l1met_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l2met_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l1c_dr': hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis,flav_axis,dr_axis),
                'l2c_dr': hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis,flav_axis,dr_axis),
                'cmet_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l1W1_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l2W1_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'metW1_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'cW1_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),                
                'l1W2_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l2W2_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'metW2_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'cW2_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),                
                'W1W2_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l1h_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'l2h_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'meth_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'ch_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'W1h_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'W2h_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'meth_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'lll1_dr': hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis,flav_axis,dr_axis),
                'lll2_dr': hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis,flav_axis,dr_axis),
                'llmet_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'llc_dr': hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis,flav_axis,dr_axis),
                'llW1_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'llW2_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                'llh_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis,phi_axis),
                # 'jetmet_dphi':hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, phi_axis),

            }
        objects=['jetflav','lep1','lep2','ll','top1','top2','nw1','nw2']
        
        for i in objects:
            
            _hist_event_dict["%s_pt" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis, flav_axis, pt_axis)
            _hist_event_dict["%s_eta" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, eta_axis)
            _hist_event_dict["%s_phi" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, phi_axis)
                # _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, mass_axis)
            
            if i =='ll': _hist_event_dict["%s_mass" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis,region_axis,flav_axis, mass_axis)
            if 'lep' in i: 
                    _hist_event_dict["%s_pfRelIso03_all" %(i)]=hist.Hist("Counts", dataset_axis,region_axis, lepflav_axis,flav_axis, iso_axis)
                    _hist_event_dict["%s_dxy" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis,flav_axis,dxy_axis)
                    _hist_event_dict["%s_dz" %(i)]=hist.Hist("Counts", dataset_axis, lepflav_axis, region_axis,flav_axis,dz_axis)
            
        
        for disc, axis in zip(disc_list,btag_axes):
            _hist_event_dict["jetflav_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis,flav_axis, axis)
            # _hist_event_dict["jetpn_%s" %(disc)] = hist.Hist("Counts", dataset_axis,lepflav_axis,region_axis,flav_axis, axis)
        self.event_hists = list(_hist_event_dict.keys())
    
        self._accumulator = processor.dict_accumulator(
            {**_hist_event_dict,   
            'cutflow': processor.defaultdict_accumulator(
                # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int))}
            
                )
        self._accumulator['sumw'] = processor.defaultdict_accumulator(float)
        
        array_dict = {}
        if self._export_array:
            for d in _hist_event_dict.keys():
                if d=='nj' or d=='nele' or d=='nmu' or d=='njmet':
                    array_dict[d]=processor.column_accumulator(np.empty(shape=(0,),dtype='int'))
                else: array_dict[d] = processor.column_accumulator(np.zeros(shape=(0,)))
            general_array = ['lumi','run','event','jetflav', 'region','lepflav','weight','lepwei','puwei','l1wei']
            for d in general_array:
                if d=='region' or d=='lepflav': array_dict[d]= processor.column_accumulator(np.empty(shape=(0,),dtype='<U6'))
                elif 'wei' in d:array_dict[d]=processor.column_accumulator(np.ones(shape=(0,)))
                elif d=='run' or d=='lumi' or d=='jetflav': array_dict[d]=processor.column_accumulator(np.empty(shape=(0,),dtype='int'))
                elif d=='event':array_dict[d]=processor.column_accumulator(np.empty(shape=(0,),dtype='long'))
                else:array_dict[d]= processor.column_accumulator(np.zeros(shape=(0,)))
            self._accumulator['array'] = processor.defaultdict_accumulator(partial(processor.defaultdict_accumulator, **array_dict))
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        # print(hasattr(events, "nJet"))
        selection = processor.PackedSelection()
        if isRealData :output['sumw'][dataset] += len(events)
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
            weights.add("L1prefireweight", events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
            weights.add('puweight',puwei(events.Pileup.nPU,self._corr))
        ##############
        if(isRealData):output['cutflow'][dataset]['all']  +=  len(events)
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
                trigger_ele = np.zeros(len(events), dtype='bool')
                trigger_mu = np.zeros(len(events), dtype='bool')
            elif "DoubleEG" in dataset:
                trigger_ele = trigger_ee #& ~trigger_em                                                            
                trigger_mu = np.zeros(len(events), dtype='bool')
                trigger_em = np.zeros(len(events), dtype='bool')
            elif "SingleElectron" in dataset:
                trigger_ele =  trigger_e & ~trigger_ee & ~trigger_em
                trigger_mu = np.zeros(len(events), dtype='bool')
                trigger_em= np.zeros(len(events), dtype='bool')
            elif "DoubleMuon" in dataset:
                trigger_mu = trigger_mm
                trigger_ele= np.zeros(len(events), dtype='bool')
                trigger_em= np.zeros(len(events), dtype='bool')
            elif "SingleMuon" in dataset:
                trigger_mu = trigger_m & ~trigger_mm & ~trigger_em
                trigger_ele = np.zeros(len(events), dtype='bool')
                trigger_em = np.zeros(len(events), dtype='bool')
        else : 
            trigger_mu = trigger_mm|trigger_m
            trigger_ele = trigger_ee|trigger_e
        selection.add('trigger_ee', ak.to_numpy(trigger_ele))
        selection.add('trigger_mumu', ak.to_numpy(trigger_mu))
        selection.add('trigger_emu', ak.to_numpy(trigger_em))
        del trigger_e,trigger_ee,trigger_m,trigger_mm
        metfilter = np.ones(len(events), dtype='bool')
        for flag in self._met_filters[self._year]['data' if isRealData else 'mc']:
            metfilter &= np.array(events.Flag[flag])
        selection.add('metfilter', metfilter)
        del metfilter
        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon
        musel = ((event_mu.pt > 13) & (abs(event_mu.eta) < 2.4)&(event_mu.mvaId>=3) &(event_mu.pfRelIso04_all<0.15)&(abs(event_mu.dxy)<0.05)&(abs(event_mu.dz)<0.1))
        event_mu["lep_flav"] = 13*event_mu.charge
        event_mu = event_mu[ak.argsort(event_mu.pt, axis=1,ascending=False)]
        event_mu = event_mu[musel]
        event_mu= ak.pad_none(event_mu,2,axis=1)
        nmu = ak.sum(musel,axis=1)
        amu = events.Muon[(events.Muon.pt>10)&(abs(events.Muon.eta) < 2.4)&(events.Muon.pfRelIso04_all<0.25)&(events.Muon.mvaId>=1)]
        namu = ak.count(amu.pt,axis=1)
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron
        event_e["lep_flav"] = 11*event_e.charge
        elesel = ((event_e.pt > 13) & (abs(event_e.eta) < 2.5)&(event_e.mvaFall17V2Iso_WP90==1)& (abs(event_e.dxy)<0.05)&(abs(event_e.dz)<0.1))
        event_e = event_e[elesel]
        event_e = event_e[ak.argsort(event_e.pt, axis=1,ascending=False)]
        event_e = ak.pad_none(event_e,2,axis=1)
        nele = ak.sum(elesel,axis=1)
        aele = events.Electron[(events.Electron.pt>12)&(abs(events.Electron.eta) < 2.5)&(events.Electron.mvaFall17V2Iso_WPL==1)]
        naele = ak.count(aele.pt,axis=1)

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
                    "eta": ak.zeros_like(events.MET.pt),
                    "phi": events.MET.phi,
                    "energy":events.MET.sumEt,
                },with_name="PtEtaPhiELorentzVector",)
        tkmet = ak.zip({
                    "pt":  events.TkMET.pt,
                    "phi": events.TkMET.phi,
                    "eta": ak.zeros_like(events.TkMET.pt),
                    "energy":events.TkMET.sumEt,
                },with_name="PtEtaPhiELorentzVector",)
        
        selection.add('ee',ak.to_numpy(nele==2))
        selection.add('mumu',ak.to_numpy(nmu==2))
        selection.add('emu',ak.to_numpy((nele==1)&(nmu==1)))
             
        # ###########
        if not isRealData: corr_jet=self._jetfactory["UL17_MC"].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll),lazy_cache=events.caches[0])
        else :corr_jet=self._jetfactory[dataset[dataset.find('Run'):dataset.find('Run')+8]].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll),lazy_cache=events.caches[0])
        
        # selection.add('jetsel',ak.to_numpy(ak.sum(seljet,axis=-1)==1))
        ranked_deepJet = corr_jet.btagDeepFlavCvL
        # ranked_deepCSV = corr_jet.btagDeepCvL
        ####TEST
        if self._version =='test':
            corr_jet[corr_jet.btagDeepCvL==0].btagDeepCvL=1e-10
            # corr_jet[corr_jet.btagDeepFlavCvL==0].btagDeepFlavCvL=1e-10
            ranked_deepJet = corr_jet.btagDeepFlavCvB/corr_jet.btagDeepFlavCvL
            # ranked_deepCSV = corr_jet.btagDeepCvL/corr_jet.btagDeepCvL
        ######
        eventflav_jet = corr_jet[ak.argsort(ranked_deepJet,axis=1,ascending=False)]
        # eventcsv_jet = corr_jet[ak.argsort(ranked_deepCSV,axis=1,ascending=False)]
        jetsel = (eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 6)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all((eventflav_jet.metric_table(ll_cand.lep1)>0.4)&(eventflav_jet.metric_table(ll_cand.lep2)>0.4),axis=2)&ak.all(eventflav_jet.metric_table(aele)>0.4,axis=2)&ak.all(eventflav_jet.metric_table(amu)>0.4,axis=2)
        njet = ak.sum(jetsel,axis=1)
        topjetsel= (eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 6)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&(eventflav_jet.btagDeepFlavB>0.0532)
        

        cvbcutll = (eventflav_jet.btagDeepFlavCvB>=0.42)
        cvlcutll = (eventflav_jet.btagDeepFlavCvL>=0.22)
        cvbcutem = (eventflav_jet.btagDeepFlavCvB>=0.5)
        cvlcutem = (eventflav_jet.btagDeepFlavCvL>=0.12)

        
        sr_cut = (mT(ll_cand.lep2,met)>30) & (mT(ll_cand,met)>60)  & (events.MET.sumEt>45)
        dy_cr2_cut=(mT(ll_cand.lep2,met)>30)& (events.MET.sumEt>45)& (mT(ll_cand,met)<60)
        top_cr2_cut = (mT(ll_cand.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) 
        # req_WW_cr = ak.any((mT(ll_cand.lep2,met)>30)& (ll_cand.mass>50) & (events.MET.sumEt>45)& (abs(ll_cand.mass-91.18)>15) & (ll_cand.mass),axis=-1) 
        global_cut = (ll_cand.lep1.pt>25) & (ll_cand.mass>12) & (ll_cand.pt>30) & (ll_cand.lep1.charge+ll_cand.lep2.charge==0) & (events.MET.pt>20) & (make_p4(ll_cand.lep1).delta_r(make_p4(ll_cand.lep2))>0.4) & (abs(met.delta_phi(tkmet))<0.5)
        sr_cut = (mT(ll_cand.lep2,met)>30) & (mT(ll_cand,met)>60)  & (events.MET.sumEt>45) 
        llmass_cut = (abs(ll_cand.mass-91.18)> 15)
        
        # sel_jet =  eventflav_jet[(eventflav_jet.pt > 20) & (abs(eventflav_jet.eta) <= 2.4)&((eventflav_jet.puId > 0)|(eventflav_jet.pt>50)) &(eventflav_jet.jetId>5)&ak.all((eventflav_jet.metric_table(ll_cand.lep1)>0.4)&(eventflav_jet.metric_table(ll_cand.lep2)>0.4),axis=2)]
        # selection.add('jetsel',ak.to_numpy(ak.count(sel_jet.pt,axis=-1)>0))

        # sel_jetcsv = eventcsv_jet[(eventcsv_jet.pt > 20) & (abs(eventcsv_jet.eta) <= 2.4)&((eventcsv_jet.puId > 6)|(eventcsv_jet.pt>50)) &(eventcsv_jet.jetId>5)&ak.all(eventcsv_jet.metric_table(ll_cand.lep1)>0.4,axis=2)&ak.all(eventcsv_jet.metric_table(ll_cand.lep2)>0.4,axis=2)]
        # sel_cjet_csv = ak.pad_none(sel_jetcsv,1,axis=1)
        # sel_cjet_csv = sel_cjet_csv[:,0]
        # sel_jetpn =  eventpn_jet[(eventpn_jet.pt > 20) & (abs(eventpn_jet.eta) <= 2.4)&((eventpn_jet.puId > 6)|(eventpn_jet.pt>50)) &(eventpn_jet.jetId>5)&ak.all(eventpn_jet.metric_table(ll_cand.lep1)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(ll_cand.lep2)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep3)>0.4,axis=2)&ak.all(eventpn_jet.metric_table(pair_4lep.lep4)>0.4,axis=2)]
        # sel_jetpn = ak.mask(sel_jetpn,ak.num(pair_4lep)>0)
        # sel_cjet_pn = ak.pad_none(sel_jetpn,1,axis=1)
        # sel_cjet_pn = sel_cjet_pn[:,0]
        
        if 'DoubleEG' in dataset :output['cutflow'][dataset]['trigger'] += ak.sum(trigger_ele)
        elif 'DoubleMuon' in dataset :output['cutflow'][dataset]['trigger'] += ak.sum(trigger_mu)
         
        output['cutflow'][dataset]['global selection'] += ak.sum(ak.any(global_cut,axis=-1))
        output['cutflow'][dataset]['signal region'] += ak.sum(ak.any(global_cut&sr_cut,axis=-1))  
        output['cutflow'][dataset]['selected jets'] +=ak.sum(ak.any(global_cut&sr_cut,axis=-1)&(njet>0))
        output['cutflow'][dataset]['all ee'] +=ak.sum(ak.any(global_cut&sr_cut,axis=-1)&(njet>0)&(ak.all(llmass_cut)&trigger_ele)&(nele==2)&(nmu==0))
        output['cutflow'][dataset]['all mumu'] +=ak.sum(ak.any(global_cut&sr_cut,axis=-1)&(njet>0)&(ak.all(llmass_cut)&trigger_mu)&(nmu==2)&(nele==0))
        output['cutflow'][dataset]['all emu'] +=ak.sum(ak.any(global_cut&sr_cut,axis=-1)&(njet>0)&(nele==1)&(nmu==1)&trigger_em)
        selection.add('llmass',ak.to_numpy(ak.all(llmass_cut,axis=-1)))
        selection.add('SR_ee',ak.to_numpy(ak.any(sr_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)>1)))
        selection.add('SR2_ee',ak.to_numpy(ak.any(sr_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)==1)))
        selection.add('top_CR_ee',ak.to_numpy(ak.any(top_cr2_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)>=2)))
        selection.add('DY_CR_ee',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)>=1)))
        selection.add('SR_mumu',ak.to_numpy(ak.any(sr_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)>1)))
        selection.add('SR2_mumu',ak.to_numpy(ak.any(sr_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)==1)))
        selection.add('top_CR_mumu',ak.to_numpy(ak.any(top_cr2_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)>=2)))
        selection.add('DY_CR_mumu',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutll&cvlcutll,axis=1)>=1)))
        selection.add('SR_emu',ak.to_numpy(ak.any(sr_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutem&cvlcutem,axis=1)>1)))
        selection.add('SR2_emu',ak.to_numpy(ak.any(sr_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutem&cvlcutem,axis=1)==1)))
        selection.add('top_CR_emu',ak.to_numpy(ak.any(top_cr2_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutem&cvlcutem,axis=1)>=2)))
        selection.add('DY_CR_emu',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(jetsel&cvbcutem&cvlcutem,axis=1)>=1)))
        
        # selection.add('DY_CRb',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&cvlcut&~cvbcut,axis=1)==1)))
        # selection.add('DY_CRl',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&~cvlcut&cvbcut,axis=1)>=1)))
        # selection.add('DY_CRc',ak.to_numpy(ak.any(dy_cr2_cut&global_cut,axis=-1)&(ak.sum(seljet&cvlcut&cvbcut,axis=1)==1)))
        
        lepflav = ['ee','mumu','emu']
        reg = ['SR','SR2','DY_CR','top_CR']
        mask_lep = {'SR':global_cut&sr_cut,'SR2':global_cut&sr_cut,'DY_CR':global_cut&dy_cr2_cut,'top_CR':global_cut&top_cr2_cut} 
        mask_jet = {'ee':jetsel&cvbcutll&cvlcutll,'mumu':jetsel&cvbcutll&cvlcutll,'emu':jetsel&cvbcutem&cvlcutem}
        
        
        for r in reg:
            for ch in lepflav:                
                if 'SR' in r and (ch=='ee' or ch=='mumu') :cut = selection.all('lepsel','metfilter','lumi','%s_%s'%(r,ch),ch, 'trigger_%s'%(ch),'llmass')
                else :cut = selection.all('lepsel','metfilter','lumi','%s_%s'%(r,ch),ch, 'trigger_%s'%(ch))
                
                ll_cands = ak.mask(ll_cand,mask_lep[r])
                if(ak.count(ll_cands.pt)>0):ll_cands  = ll_cands[ak.argsort(ll_cands.pt, axis=1,ascending=False)]
                sel_jetflav = ak.mask(eventflav_jet,mask_jet[ch])
                sel_cjet_flav = sel_jetflav[cut]    
                if(ak.count(sel_cjet_flav.pt)>0):sel_cjet_flav  = sel_cjet_flav[ak.argsort(sel_cjet_flav.btagDeepFlavCvB,axis=1,ascending=False)]
                nseljet = ak.count(sel_cjet_flav.pt,axis=1)
                topjets = ak.mask(eventflav_jet,topjetsel)[cut]
                if (ak.count(topjets.pt)>0):topjets  = topjets[ak.argsort(topjets.btagDeepFlavCvB,axis=1,ascending=False)]
                
                sel_cjet_flav = ak.pad_none(sel_cjet_flav,1,axis=1)    
                sel_cjet_flav = sel_cjet_flav[:,0]               
                 
                llcut = ll_cands[cut]
                llcut = llcut[:,0]
                lep1cut = llcut.lep1
                lep2cut = llcut.lep2
                w1cut = lep1cut+met[cut]
                w2cut = lep2cut+met[cut]
                hcut = llcut+met[cut]

                # topjet1 = lep1cut.nearest(topjets)
                # topjet2 = lep2cut.nearest(topjets)
                # neu1 = getnu4vec(lep1cut,met[cut])
                # neu2 = getnu4vec(lep2cut,met[cut])
                # top1cut=topjet1+lep1cut+neu1
                # top2cut=topjet2+lep2cut+neu2
                # nw1cut=lep1cut+neu1
                # nw2cut=lep2cut+neu2
                if isRealData:flavor= ak.zeros_like(sel_cjet_flav['pt'])
                else :flavor=sel_cjet_flav.hadronFlavour+1*((sel_cjet_flav.partonFlavour == 0 ) & (sel_cjet_flav.hadronFlavour==0))
                if not isRealData:
                        if ch=='ee':lepsf=eleSFs(lep1cut,self._year,self._corr)*eleSFs(lep2cut,self._year,self._corr)
                        elif ch=='mumu':lepsf=muSFs(lep1cut,self._year,self._corr)*muSFs(lep2cut,self._year,self._corr)
                        else:
                            lepsf=np.where(lep1cut.lep_flav==11,eleSFs(lep1cut,self._year,self._corr)*muSFs(lep2cut,self._year,self._corr),1.)*np.where(lep1cut.lep_flav==13,eleSFs(lep2cut,self._year,self._corr)*muSFs(lep1cut,self._year,self._corr),1.)
                        # jetsf = np.where(ak.count(sel_cjet_flav.pt,axis=1)!=0,puJetID_SFs(sel_cjet_flav),1.)
                        # jetsf = puJetID_SFs(sel_cjet_flav,self._corr)
                        sf = lepsf
                        
                else : 
                    sf =weights.weight()[cut]
                for histname, h in output.items():
                    if 'jetflav_' in histname:
                        fields = {l: normalize(sel_cjet_flav[histname.replace('jetflav_','')]) for l in h.fields if l in dir(sel_cjet_flav)}
                        h.fill(dataset=dataset, lepflav =ch, region = r, flav=flavor, **fields,weight=weights.weight()[cut]*sf) 
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(sel_cjet_flav[histname.replace('jetflav_','')]))) 
                    # # elif 'jetcsv_' in histname:
                    #     fields = {l: normalize(sel_cjet_csv[histname.replace('jetcsv_','')],cut) for l in h.fields if l in dir(sel_cjet_csv)}
                    #     h.fill(dataset=dataset,lepflav =ch, flav=normalize(sel_cjet_csv.hadronFlavour+1*((sel_cjet_csv.partonFlavour == 0 ) & (sel_cjet_csv.hadronFlavour==0)),cut), **fields,weight=weights.weight()[cut]*sf)    
                    elif 'lep1_' in histname:
                        fields = {l: normalize(flatten(lep1cut[histname.replace('lep1_','')])) for l in h.fields if l in dir(lep1cut)}
                        h.fill(dataset=dataset,lepflav=ch,region = r, flav=flavor, **fields,weight=weights.weight()[cut]*sf)
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(lep1cut[histname.replace('lep1_','')])))) 
                    elif 'lep2_' in histname:
                        fields = {l: normalize(flatten(lep2cut[histname.replace('lep2_','')])) for l in h.fields if l in dir(lep2cut)}
                        h.fill(dataset=dataset,lepflav=ch,region = r, flav=flavor, **fields,weight=weights.weight()[cut]*sf)
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(lep2cut[histname.replace('lep2_','')])))) 
                    elif 'MET_' in histname:
                        fields = {l: normalize(events.MET[histname.replace('MET_','')],cut) for l in h.fields if l in dir(events.MET)}
                        h.fill(dataset=dataset, lepflav =ch, region = r, flav=flavor,**fields,weight=weights.weight()[cut]*sf) 
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(events.MET[histname.replace('MET_','')],cut))) 
                    elif 'll_' in histname:
                        fields = {l: normalize(flatten(llcut[histname.replace('ll_','')])) for l in h.fields if l in dir(llcut)}
                        h.fill(dataset=dataset,lepflav=ch, region = r, flav=flavor,**fields,weight=weights.weight()[cut]*sf) 
                        if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(llcut[histname.replace('ll_','')])))) 
                    # elif 'top1_' in histname:
                    #     fields = {l: normalize(flatten(top1cut[histname.replace('top1_','')])) for l in h.fields if l in dir(top1cut)}
                    #     h.fill(dataset=dataset,lepflav=ch, region = r, flav=flavor,**fields,weight=weights.weight()[cut]*sf) 
                    #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(top1cut[histname.replace('top1_','')])))) 
                    # elif 'top2_' in histname:
                    #     fields = {l: normalize(flatten(top2cut[histname.replace('top2_','')])) for l in h.fields if l in dir(top2cut)}
                    #     h.fill(dataset=dataset,lepflav=ch, region = r, flav=flavor,**fields,weight=weights.weight()[cut]*sf) 
                    #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(top2cut[histname.replace('top2_','')]))))
                    # elif 'nw1_' in histname:
                    #     fields = {l: normalize(flatten(nw1cut[histname.replace('nw1_','')])) for l in h.fields if l in dir(nw1cut)}
                    #     h.fill(dataset=dataset,lepflav=ch, region = r, flav=flavor,**fields,weight=weights.weight()[cut]*sf) 
                    #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(nw1cut[histname.replace('nw1_','')])))) 
                    # elif 'nw2_' in histname:
                    #     fields = {l: normalize(flatten(nw2cut[histname.replace('nw2_','')])) for l in h.fields if l in dir(nw2cut)}
                    #     h.fill(dataset=dataset,lepflav=ch, region = r, flav=flavor,**fields,weight=weights.weight()[cut]*sf) 
                    #     if self._export_array:output['array'][dataset][histname]+=processor.column_accumulator(ak.to_numpy(normalize(flatten(nw2cut[histname.replace('nw2_','')])))) 
                if self._export_array:
                    output['array'][dataset]['weight']+=processor.column_accumulator(ak.to_numpy(normalize(weights.weight()[cut]*sf)))
                    output['array'][dataset]['lepwei']+=processor.column_accumulator(ak.to_numpy(normalize(sf)))
                    output['array'][dataset]['puwei']+=processor.column_accumulator(ak.to_numpy(normalize(weights.partial_weight(include=['puweight'])[cut])))
                    output['array'][dataset]['l1wei']+=processor.column_accumulator(ak.to_numpy(normalize(weights.partial_weight(include=['L1prefireweight'])[cut])))
                    output['array'][dataset]['event']+=processor.column_accumulator(ak.to_numpy(normalize(events[cut].event)))
                    output['array'][dataset]['run']+=processor.column_accumulator(ak.to_numpy(normalize(events[cut].run)))
                    output['array'][dataset]['lumi']+=processor.column_accumulator(ak.to_numpy(normalize(events[cut].luminosityBlock)))
                    output['array'][dataset]['lepflav']+=processor.column_accumulator(np.full_like(ak.to_numpy(normalize(nseljet)),ch,dtype='<U6'))
                    output['array'][dataset]['jetflav']+=processor.column_accumulator(ak.to_numpy(normalize(flavor)))
                    output['array'][dataset]['region']+=processor.column_accumulator(np.full_like(ak.to_numpy(normalize(nseljet)),r,dtype='<U6'))
                    output['array'][dataset]['nj']+=processor.column_accumulator(ak.to_numpy(normalize(nseljet)))
                    output['array'][dataset]['nele']+=processor.column_accumulator(ak.to_numpy(normalize(naele-nele,cut)))
                    output['array'][dataset]['nmu']+=processor.column_accumulator(ak.to_numpy(normalize(namu-nmu,cut)))
                    output['array'][dataset]['METTkMETdphi']+=processor.column_accumulator(ak.to_numpy(flatten(met[cut].delta_phi(tkmet[cut]))))
                    output['array'][dataset]['njmet']+=processor.column_accumulator(ak.to_numpy(normalize(ak.sum((met[cut].delta_phi(sel_jetflav[cut])<0.5),axis=1))))
                    output['array'][dataset]['mT1']+=processor.column_accumulator(ak.to_numpy(flatten(mT(lep1cut,met[cut]))))
                    output['array'][dataset]['mT2']+=processor.column_accumulator(ak.to_numpy(flatten(mT(lep2cut,met[cut]))))
                    output['array'][dataset]['mTh']+=processor.column_accumulator(ak.to_numpy(flatten(mT(llcut,met[cut]))))
                    # if 'SR' in r:
                    output['array'][dataset]['l1l2_dr']+=processor.column_accumulator(ak.to_numpy(flatten(make_p4(lep1cut).delta_r(make_p4(lep2cut)))))
                    output['array'][dataset]['l1met_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(lep1cut.delta_phi(met[cut]))))
                    output['array'][dataset]['l2met_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(lep2cut.delta_phi(met[cut]))))
                    output['array'][dataset]['llmet_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(llcut.delta_phi(met[cut]))))
                    output['array'][dataset]['l1c_dr']+=processor.column_accumulator(ak.to_numpy(normalize(make_p4(lep1cut).delta_r(make_p4(sel_cjet_flav)))))
                    output['array'][dataset]['cmet_dphi']+=processor.column_accumulator(ak.to_numpy(normalize(met[cut].delta_phi(sel_cjet_flav))))
                    output['array'][dataset]['l2c_dr']+=processor.column_accumulator(ak.to_numpy(normalize(make_p4(lep2cut).delta_r(make_p4(sel_cjet_flav)))))
                    output['array'][dataset]['l2W1_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(lep2cut.delta_phi((w1cut)))))
                    output['array'][dataset]['metW1_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(met[cut].delta_phi((w1cut)))))
                    output['array'][dataset]['cW1_dphi']+=processor.column_accumulator(ak.to_numpy(normalize((w1cut).delta_phi(sel_cjet_flav))))
                    output['array'][dataset]['l1W2_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(lep1cut.delta_phi((w2cut)))))
                    output['array'][dataset]['llc_dr']+=processor.column_accumulator(ak.to_numpy(normalize(make_p4(llcut).delta_r(make_p4(sel_cjet_flav)))))  
                    output['array'][dataset]['llW1_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(w1cut.delta_phi((llcut)))))
                    output['array'][dataset]['llW2_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(w2cut.delta_phi((llcut)))))
                    output['array'][dataset]['llh_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(hcut.delta_phi((llcut)))))
                    output['array'][dataset]['l2W2_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(lep2cut.delta_phi((w2cut)))))
                    output['array'][dataset]['metW2_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(met[cut].delta_phi((w2cut)))))
                    output['array'][dataset]['cW2_dphi']+=processor.column_accumulator(ak.to_numpy(normalize((w2cut).delta_phi(sel_cjet_flav))))            
                    output['array'][dataset]['l1h_dphi']+=processor.column_accumulator(ak.to_numpy(flatten((lep1cut).delta_phi((hcut)))))     
                    output['array'][dataset]['meth_dphi']+=processor.column_accumulator(ak.to_numpy(flatten((met[cut]).delta_phi((hcut)))))  
                    output['array'][dataset]['ch_dphi']+=processor.column_accumulator(ak.to_numpy(normalize(hcut.delta_phi(sel_cjet_flav))))  
                    output['array'][dataset]['W1h_dphi']+=processor.column_accumulator(ak.to_numpy(flatten((w1cut).delta_phi((hcut)))))  
                    output['array'][dataset]['W2h_dphi']+=processor.column_accumulator(ak.to_numpy(flatten((w2cut).delta_phi((hcut)))))  
                    output['array'][dataset]['lll1_dr']+=processor.column_accumulator(ak.to_numpy(flatten(make_p4(lep1cut).delta_r(make_p4(llcut)))))  
                    output['array'][dataset]['lll2_dr']+=processor.column_accumulator(ak.to_numpy(flatten(make_p4(lep2cut).delta_r(make_p4(llcut)))))  
                    output['array'][dataset]['l1W1_dphi']+=processor.column_accumulator(ak.to_numpy(flatten(lep1cut.delta_phi((w1cut)))))
                    output['array'][dataset]['W1W2_dphi']+=processor.column_accumulator(ak.to_numpy(flatten((w1cut).delta_phi((w2cut)))))     
                    output['array'][dataset]['l2h_dphi']+=processor.column_accumulator(ak.to_numpy(flatten((lep2cut).delta_phi((hcut)))))  
                      
                output['nj'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,nj=ak.fill_none(nseljet,np.nan),weight=weights.weight()[cut]*sf)                            
                output['nele'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,nalep=normalize(naele-nele,cut),weight=weights.weight()[cut]*sf)
                output['nmu'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,nalep=normalize(namu-nmu,cut),weight=weights.weight()[cut]*sf)        
                output['njmet'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,nj=ak.fill_none(ak.sum((met[cut].delta_phi(sel_jetflav[cut])<0.5),axis=1),np.nan),weight=weights.weight()[cut]*sf)                    
                output['METTkMETdphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(met[cut].delta_phi(tkmet[cut])),weight=weights.weight()[cut]*sf)
                    
                output['mT1'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,mt=flatten(mT(lep1cut,met[cut])),weight=weights.weight()[cut]*sf)
                output['mT2'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,mt=flatten(mT(lep2cut,met[cut])),weight=weights.weight()[cut]*sf)
                output['mTh'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,mt=flatten(mT(llcut,met[cut])),weight=weights.weight()[cut]*sf)
                    

                if 'SR' in r :
                    output['l1l2_dr'].fill(dataset=dataset,lepflav=ch,region=r,flav=flavor,dr=flatten(make_p4(lep1cut).delta_r(make_p4(lep2cut))),weight=weights.weight()[cut]*sf)
                    output['l1met_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(lep1cut.delta_phi(met[cut])),weight=weights.weight()[cut]*sf)
                    output['l2met_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(lep2cut.delta_phi(met[cut])),weight=weights.weight()[cut]*sf)
                    output['llmet_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(met[cut].delta_phi((llcut))),weight=weights.weight()[cut]*sf)
                    output['l1c_dr'].fill(dataset=dataset,lepflav=ch,region=r,flav=flavor,dr=ak.fill_none(make_p4(lep1cut).delta_r(make_p4(sel_cjet_flav)),np.nan),weight=weights.weight()[cut]*sf)
                    output['cmet_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=ak.fill_none(met[cut].delta_phi(sel_cjet_flav),np.nan),weight=weights.weight()[cut]*sf)
                    output['l2c_dr'].fill(dataset=dataset,lepflav=ch,region=r,flav=flavor,dr=ak.fill_none(make_p4(lep2cut).delta_r(make_p4(sel_cjet_flav)),np.nan),weight=weights.weight()[cut]*sf)
                    output['l1W1_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(lep1cut.delta_phi((w1cut))),weight=weights.weight()[cut]*sf)
                    output['l2W1_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(lep2cut.delta_phi((w1cut))),weight=weights.weight()[cut]*sf)
                    output['metW1_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(met[cut].delta_phi((w1cut))),weight=weights.weight()[cut]*sf)
                    output['cW1_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=ak.fill_none((w1cut).delta_phi(sel_cjet_flav),np.nan),weight=weights.weight()[cut]*sf)                
                    output['l1W2_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(lep1cut.delta_phi((w2cut))),weight=weights.weight()[cut]*sf)
                    output['l2W2_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(lep2cut.delta_phi((w2cut))),weight=weights.weight()[cut]*sf)
                    output['metW2_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(met[cut].delta_phi((w2cut))),weight=weights.weight()[cut]*sf)
                    output['cW2_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=ak.fill_none((w2cut).delta_phi(sel_cjet_flav),np.nan),weight=weights.weight()[cut]*sf)                
                    output['W1W2_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten((w1cut).delta_phi((w2cut))),weight=weights.weight()[cut]*sf)
                    output['l1h_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten((lep1cut).delta_phi((hcut))),weight=weights.weight()[cut]*sf)
                    output['l2h_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten((lep2cut).delta_phi((hcut))),weight=weights.weight()[cut]*sf)
                    output['meth_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten((met[cut]).delta_phi((hcut))),weight=weights.weight()[cut]*sf)
                    output['ch_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=ak.fill_none(hcut.delta_phi(sel_cjet_flav),np.nan),weight=weights.weight()[cut]*sf)
                    output['W1h_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten((w1cut).delta_phi((hcut))),weight=weights.weight()[cut]*sf)
                    output['W2h_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten((w2cut).delta_phi((hcut))),weight=weights.weight()[cut]*sf)
                    output['lll1_dr'].fill(dataset=dataset,lepflav=ch,region=r,flav=flavor,dr=flatten(make_p4(lep1cut).delta_r(make_p4(llcut))),weight=weights.weight()[cut]*sf)
                    output['lll2_dr'].fill(dataset=dataset,lepflav=ch,region=r,flav=flavor,dr=flatten(make_p4(lep2cut).delta_r(make_p4(llcut))),weight=weights.weight()[cut]*sf)
                    output['llc_dr'].fill(dataset=dataset,lepflav=ch,region=r,flav=flavor,dr=ak.fill_none(make_p4(llcut).delta_r(make_p4(sel_cjet_flav)),np.nan),weight=weights.weight()[cut]*sf)
                    output['llW1_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(w1cut.delta_phi((llcut))),weight=weights.weight()[cut]*sf)
                    output['llW2_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(w2cut.delta_phi((llcut))),weight=weights.weight()[cut]*sf)
                    output['llh_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(hcut.delta_phi((llcut))),weight=weights.weight()[cut]*sf)
                        # output['jetmet_dphi'].fill(dataset=dataset,lepflav=ch,region = r, flav=flavor,phi=flatten(jet.delta_phi((llcut))),weight=weights.weight()[cut]*sf)
                
        del leppair, ll_cand, sel_cjet_flav,met,tkmet    
                    
        return output

    def postprocess(self, accumulator):
        print(accumulator)
        return accumulator
