import hist as Hist
from collections import defaultdict
import numpy as np
def histogram(cfg):
    #categories = ["SR_LM","SR2_LM","HM_CR_1j","HM_CR_nj","top_CR_1j","top_CR_nj"]
    categories = ["SR_LM","SR2_LM","HM_CR","top_CR","bSR_LM","bSR2_LM","bHM_CR","btop_CR"]
    region_axis = Hist.axis.StrCategory(categories, name="region",growth=True)
    syst_axis = Hist.axis.StrCategory([], name="syst", growth=True)
    flav_axis = Hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Genflavour")
    lepflav_axis = Hist.axis.StrCategory(
        ["emu"], name="lepflav", label="channel"
    )
    npv_axis= Hist.axis.Integer(0,100,name="npv",label='# of PVs')
    nsv_axis= Hist.axis.Integer(0,20,name="nsv",label='# of SVs')
    n_axis= Hist.axis.Integer(0,10,name="n",label='# of SVs')
    pt_axis = Hist.axis.Regular(50,0,300, name="pt", label=" $p_{T}$ [GeV]")
    eta_axis = Hist.axis.Regular(25,-2.5,2.5, name="eta", label=" $\eta$")
    phi_axis = Hist.axis.Regular(30,-np.pi,np.pi, name="phi", label="$\phi$")
    dphi_axis = Hist.axis.Regular(30,-0.5,0.5, name="dphi", label="$D\phi$")
    mass_axis = Hist.axis.Regular(50,0,300, name="mass", label="$m$ [GeV]")
    iso_axis = Hist.axis.Regular(30,0,0.15, name="iso", label="Rel Iso")
    dxy_axis = Hist.axis.Regular(50,-0.05,0.05, name="dxy", label="$d_{xy}$")
    dz_axis = Hist.axis.Regular(50,0,0.1, name="dz", label="$d_{z}$")
    dr_axis = Hist.axis.Regular(40,0,4, name="dr", label="$\\Delta R$")
    disc_axis=Hist.axis.Regular(50,0.,1, name="discr", label="Discr")
    # kinematic variables
    mt_axis = Hist.axis.Regular(30,0,300, name="mt", label=" $m_{T}$ [GeV]")
    output = {
        "cutflow": defaultdict(float),
        "sumw": 0,
        "npv": Hist.Hist(lepflav_axis,region_axis,flav_axis,npv_axis,Hist.storage.Weight()),    
        "LHE_Vpt" : Hist.Hist(Hist.axis.Regular(10,0,1000, name="pt", label="LHE V$p_{T}$ [GeV]"),Hist.storage.Weight()),
        "LHE_HT" : Hist.Hist(Hist.axis.Regular(50,0,2500, name="pt", label="LHE $H_{T}$ [GeV]"),Hist.storage.Weight()),    
        "template_nsv": Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,nsv_axis,Hist.storage.Weight()),    
        "template_ll_pt":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
        "template_lll1_dr" : Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
        "template_llc_dr" : Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
        "template_lll2_dr" : Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
        "template_lep1_pt":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis, pt_axis,Hist.storage.Weight()),
        "template_lep2_pt":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,Hist.axis.Regular(50,0,150, name="pt", label=" $p_{T}$ [GeV]"),Hist.storage.Weight()),
        "template_l1met_dphi" : Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "template_l2met_dphi" : Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "template_cW1_dphi" : Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "template_jetflav_btagDeepFlavCvL": Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
        "template_jetflav_btagDeepFlavCvB": Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
        # "lep1_dxy":Hist.Hist(lepflav_axis,region_axis,flav_axis,dxy_axis,Hist.storage.Weight()),
        # "lep2_dxy":Hist.Hist(lepflav_axis,region_axis,flav_axis,dxy_axis,Hist.storage.Weight()),
        # "lep1_dz":Hist.Hist(lepflav_axis,region_axis,flav_axis,dz_axis,Hist.storage.Weight()),
        # "lep2_dz":Hist.Hist(lepflav_axis,region_axis,flav_axis,dz_axis,Hist.storage.Weight()),
        "lep1_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
        "lep2_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
        # "ele_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
        # "mu_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
        "jetflav_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
        "ll_eta":Hist.Hist(lepflav_axis,region_axis,flav_axis,eta_axis,Hist.storage.Weight()),
        "lep1_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "lep2_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "ele_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "mu_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "jetflav_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "ll_phi":Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "nj": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
        "nele": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
        "nmu": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
        "njmet": Hist.Hist(lepflav_axis,region_axis,flav_axis,n_axis,Hist.storage.Weight()),
        "MET_phi" : Hist.Hist(lepflav_axis,region_axis,flav_axis,phi_axis,Hist.storage.Weight()),
        "l1c_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
        "l2c_dr" : Hist.Hist(lepflav_axis,region_axis,flav_axis,dr_axis,Hist.storage.Weight()),
        "l1c_dr_nosel" : Hist.Hist(dr_axis,Hist.storage.Weight()),
        "l2c_dr_nosel" : Hist.Hist(dr_axis,Hist.storage.Weight()),
        "jetflav_btagDeepFlav": Hist.Hist(lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
      
        "template_ll_mass":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mass_axis,Hist.storage.Weight()),
        "template_mTh":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,Hist.axis.Regular(60,0,300, name="mt"),Hist.storage.Weight()),
        "template_mT1":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,mt_axis,Hist.storage.Weight()),
        "template_mT2":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,Hist.axis.Regular(30,30,150, name="mt"),Hist.storage.Weight()),
        "BDT_2D":Hist.Hist(region_axis,syst_axis,Hist.axis.Regular(10,0.,1, name="BDT_bkg", label="BDT(H+c,Other Bkg)"),Hist.axis.Regular(10,0.,1, name="BDT_H", label="BDT(H+c,Bkg-H)"),Hist.storage.Weight()),
        "template_MET":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
        "template_jetpt":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,pt_axis,Hist.storage.Weight()),
        "template_BDT_SR2_LM":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,Hist.axis.Variable(cfg.userconfig["BDT"]["binning"]["SR2_LM"], name="discr", label="BDT"),Hist.storage.Weight()),
        "template_BDT_SR_LM":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,Hist.axis.Variable(cfg.userconfig["BDT"]["binning"]["SR_LM"], name="discr", label="BDT"),Hist.storage.Weight()),
        # "template_BDT_SR2_LM_1D":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,Hist.axis.Variable(cfg.userconfig["BDT"]["binning"]["SR2_LM_1D"], name="discr", label="BDT"),Hist.storage.Weight()),
        # "template_BDT_SR_LM_1D":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,Hist.axis.Variable(cfg.userconfig["BDT"]["binning"]["SR_LM_1D"], name="discr", label="BDT"),Hist.storage.Weight()),
        "template_BDT_SR_HM":Hist.Hist(syst_axis,lepflav_axis,region_axis,flav_axis,disc_axis,Hist.storage.Weight()),
    "weight":Hist.Hist(lepflav_axis,region_axis,flav_axis,Hist.axis.Regular(50,-2,2, name="wei", label="BDT")),
        #"array":{ch:{flav :{} for flav in ['emu']}for ch in categories}}
        "array":{}}
    return output


