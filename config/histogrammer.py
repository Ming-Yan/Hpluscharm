import hist as Hist
import numpy as np

def create_hist():

    ## Common variables
    flav_axis = Hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Genflavour")
    lepflav_axis = Hist.axis.StrCategory(
        ["ee", "mumu", "emu"], name="lepflav", label="channel"
    )
    dataset_axis = Hist.axis.StrCategory([], name="dataset", label="Primary dataset",growth=True)

    region_axis = Hist.axis.StrCategory(["SR", "SR2", "top_CR", "DY_CR"], name="region")
    syst_axis = Hist.axis.StrCategory([], name="syst", growth=True)

    pt_axis = Hist.axis.Regular(50,0,300, name="pt", label=" $p_{T}$ [GeV]")
    eta_axis = Hist.axis.Regular(25,-2.5,2.5, name="eta", label=" $\eta$")
    phi_axis = Hist.axis.Regular(30,-3,3, name="phi", label="$\phi$")
    dphi_axis = Hist.axis.Regular(30,-0.5,0.5, name="dphi", label="$D\phi$")
    mass_axis = Hist.axis.Regular(50,0,300, name="mass", label="$m$ [GeV]")
    # Events
    n_axis = Hist.axis.IntCategory([0, 1, 2, 3], name="n", label="N obj")
    nsv_axis = Hist.axis.Integer(0,20, name="nsv", label="N secondary vertices")
    npv_axis = Hist.axis.Integer(0,100, name="npvs", label="N primary vertices")
    # kinematic variables
    mt_axis = Hist.axis.Regular(30,0,300, name="mt", label=" $m_{T}$ [GeV]")
    dr_axis = Hist.axis.Regular(20,0,5, name="dr", label="$\Delta$R")
    iso_axis = Hist.axis.Regular(
        40,0, 0.05, name="pfRelIso03_all", label="Rel. Iso"
    )
    dxy_axis = Hist.axis.Regular(40,-0.05, 0.05, name="dxy", label="d_{xy}")
    dz_axis = Hist.axis.Regular(40, 0, 0.1, name="dz", label="d_{z}")
    # MET vars
   
    ratio_axis = Hist.axis.Regular(50,0, 10, name="ratio", label="ratio")
    
    uparper_axis = Hist.axis.Regular(50,-500, 500, name="uparper", label="$u_\par$")
    

    
    _hist_event_dict = {
        "npvs": Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            npv_axis,
            Hist.storage.Weight(),
        ),    
        "nsv": Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            nsv_axis,
            Hist.storage.Weight(),
        ),    
        "MET_ptdivet": Hist.Hist(
            
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            ratio_axis,
            Hist.storage.Weight(),
        ),
        "u_par": Hist.Hist(
            
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            uparper_axis,
            Hist.storage.Weight(),
        ),
        "u_per": Hist.Hist(
            
            
            
            lepflav_axis,
            region_axis,
            flav_axis,
            uparper_axis,
            Hist.storage.Weight(),
        ),       
        "mT1": Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            mt_axis,
            Hist.storage.Weight(),
        ),
        "mT2": Hist.Hist(
            
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            mt_axis,
            Hist.storage.Weight(),
        ),
        "mTh": Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            mt_axis,
            Hist.storage.Weight(),
        )
    }
    for n in ['nj','nele','nmu','njmet']:
        _hist_event_dict[n]=  Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            n_axis,
            Hist.storage.Weight(),
        )
    for phi in ['MET_phi','TkMET_phi','PuppiMET_phi','METTkMETdphi','l1met_dphi','l2met_dphi','cmet_dphi','l1W1_dphi','l1W2_dphi','l2W2_dphi','l2W1_dphi','metW1_dphi','metW2_dphi','cW1_dphi','cW2_dphi','W1W2_dphi','l1h_dphi','l2h_dphi','meth_dphi','ch_dphi','W1h_dphi','W2h_dphi','llmet_dphi','llW1_dphi','llW2_dphi','llh_dphi']:
        _hist_event_dict[phi]=Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            phi_axis,
            Hist.storage.Weight(),
        )
    for pt in ['MET_pt','TkMET_pt','PuppiMET_pt','MET_proj','TkMET_proj','minMET_proj','h_pt']:
        _hist_event_dict[pt]=Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            pt_axis,
            Hist.storage.Weight(),
        )
    for dr in ['l1l2_dr','l1c_dr','l2c_dr','lll1_dr','lll2_dr','llc_dr']:
        _hist_event_dict[dr]=Hist.Hist(
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            dr_axis,
            Hist.storage.Weight(),
        )
    objects = ["jetflav", "lep1", "lep2", "ll","topjet1","topjet2","ttbar"]# "top1", "top2", "nw1", "nw2","neu2","neu2"]

    for i in objects:

        _hist_event_dict["%s_pt" % (i)] = Hist.Hist(
            
lepflav_axis,
            region_axis,
            flav_axis,
            pt_axis,
            Hist.storage.Weight(),
        )
        _hist_event_dict["%s_eta" % (i)] = Hist.Hist(
            
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            eta_axis,
            Hist.storage.Weight(),
        )
        _hist_event_dict["%s_phi" % (i)] = Hist.Hist(
            
            
            
lepflav_axis,
            region_axis,
            flav_axis,
            phi_axis,
            Hist.storage.Weight(),
        )

        if i == "ll":
            _hist_event_dict["%s_mass" % (i)] = Hist.Hist(
                
                
                
lepflav_axis,
                region_axis,
                flav_axis,
                mass_axis,
                Hist.storage.Weight(),
            )
        if "lep" in i:
            _hist_event_dict["%s_pfRelIso03_all" % (i)] = Hist.Hist(
                
                
                region_axis,
                
lepflav_axis,
                flav_axis,
                iso_axis,
                Hist.storage.Weight(),
            )
            _hist_event_dict["%s_dxy" % (i)] = Hist.Hist(
                
                
                
lepflav_axis,
                region_axis,
                flav_axis,
                dxy_axis,
                Hist.storage.Weight(),
            )
            _hist_event_dict["%s_dz" % (i)] = Hist.Hist(
                
                
                
lepflav_axis,
                region_axis,
                flav_axis,
                dz_axis,
                Hist.storage.Weight(),
            )
    disc_list = [
            "btagDeepCvL",
            "btagDeepCvB",
            "btagDeepFlavCvB",
            "btagDeepFlavCvL",
        ]
    for disc in disc_list:
        _hist_event_dict["jetflav_%s" % (disc)] = Hist.Hist(            
            
lepflav_axis,
            region_axis,
            flav_axis,
            Hist.axis.Regular(50,0, 1, name=disc, label=disc),
            Hist.storage.Weight(),
        )
    
    _hist_event_dict['template_ll_mass'] = Hist.Hist(
            
            syst_axis,
            
lepflav_axis,
            region_axis,
            flav_axis,
            mass_axis,
            Hist.storage.Weight(),
        )
    _hist_event_dict['template_mTh'] = Hist.Hist(
            
            syst_axis,
            
lepflav_axis,
            region_axis,
            flav_axis,
            mt_axis,
            Hist.storage.Weight(),
        )
    _hist_event_dict['template_mT1'] = Hist.Hist(
            
            syst_axis,
            
lepflav_axis,
            region_axis,
            flav_axis,
            mt_axis,
            Hist.storage.Weight(),
        )
    _hist_event_dict['template_top1_mass'] = Hist.Hist(
            
            syst_axis,
            
lepflav_axis,
            region_axis,
            flav_axis,
            mass_axis,
            Hist.storage.Weight(),
        )
    _hist_event_dict['template_top2_mass'] = Hist.Hist(
            
            syst_axis,
            lepflav_axis,
            region_axis,
            flav_axis,
            mass_axis,
            Hist.storage.Weight(),
        )
    _hist_event_dict['template_tt_mass'] = Hist.Hist(
            
           syst_axis,
            lepflav_axis,
            region_axis,
            flav_axis,
            Hist.axis.Regular(50,0,500, name="mass", label="$m$ [GeV]"),
            Hist.storage.Weight(),
        )
    _hist_event_dict['template_BDT'] = Hist.Hist(
            
            syst_axis,
            
lepflav_axis,
            region_axis,
            flav_axis,
            Hist.axis.Regular(50,0, 1, name="BDT", label="BDT"),
            Hist.storage.Weight(),
        )
    return _hist_event_dict
