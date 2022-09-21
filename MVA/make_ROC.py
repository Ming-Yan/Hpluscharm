import functools
import argparse
from coffea.util import load
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score

from BTVNanoCommissioning.utils.xs_scaler import scale_xs_arr
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.metrics import make_scorer, accuracy_score
def load_data(fname, varlist, isSignal=True, channel="emu", lumi=41500):
    # Read data from ROOT files
    data = load(fname)
    events = data["sumw"]
    data_arrays = data["array"]

    # data_arrays=data_arrays[(data_arrays['region']=='SR')|(data_arrays['region']=='SR2')]
    # put gen weight
    genwei = scale_xs_arr(events, lumi, "../../../metadata/xsection.json")
    if channel == "emu":
        w = np.hstack(
            [
                data_arrays[dataset]["weight"].value[
                    (
                        (data_arrays[dataset]["region"].value == "SR2")
                        | (data_arrays[dataset]["region"].value == "SR")
                    )
                    & (data_arrays[dataset]["lepflav"].value == "emu")
                ]
                * genwei[dataset]
                for dataset in data_arrays.keys()
            ]
        )
    else:
        w = np.hstack(
            [
                data_arrays[dataset]["weight"].value[
                    (
                        (data_arrays[dataset]["region"].value == "SR2")
                        | (data_arrays[dataset]["region"].value == "SR")
                    )
                    & (data_arrays[dataset]["lepflav"].value != "emu")
                ]
                * genwei[dataset]
                for dataset in data_arrays.keys()
            ]
        )
    # Convert inputs to format readable by machine learning tools
    if channel == "emu":
        x = np.vstack(
            [
                np.hstack(
                    [
                        data_arrays[dataset][var].value[
                            (
                                (data_arrays[dataset]["region"].value == "SR2")
                                | (data_arrays[dataset]["region"].value == "SR")
                            )
                            & (data_arrays[dataset]["lepflav"].value == "emu")
                        ]
                        for dataset in data_arrays.keys()
                    ]
                )
                for var in varlist
            ]
        ).T
    else:
        x = np.vstack(
            [
                np.hstack(
                    [
                        data_arrays[dataset][var].value[
                            (
                                (data_arrays[dataset]["region"].value == "SR2")
                                | (data_arrays[dataset]["region"].value == "SR")
                            )
                            & (data_arrays[dataset]["lepflav"].value != "emu")
                        ]
                        for dataset in data_arrays.keys()
                    ]
                )
                for var in varlist
            ]
        ).T

    # Create labels
    if isSignal:
        y = np.ones(x.shape[0], dtype=int)
    else:
        y = np.zeros(x.shape[0], dtype=int)
    if isSignal:
        w = w * 1000
    return x, y, w
if __name__ == "__main__":
    import sys

    # Load data
    varlist = [
        "ll_mass",
                "MET_pt",
                "jetflav_pt",
                "lep1_pt",
                "lep2_pt",
                "ll_pt",
                "mT1",
                "mT2",
                "jetflav_btagDeepFlavCvL",
                "jetflav_btagDeepFlavCvB",
                "lll1_dr",
                "lll2_dr",
                "llc_dr",
                "l1met_dphi",
                "l2met_dphi",
                "cW1_dphi",
    ]
    bkgx, bkgy, bkgw = load_data(
        "../../../coffea_output/hists_HWW2l2nu_mcbkg_UL17arrays.coffea",
        varlist,
        False,
        "emu",
    )
    sigx, sigy, sigw = load_data(
        "../../../coffea_output/hists_HWW2l2nu_signal_UL17_4f_arrays.coffea",
        varlist,
        True,
        "emu",
    )
    fig, ax = plt.subplots()
    
    x = np.vstack([bkgx, sigx])
    y = np.hstack([bkgy, sigy])
    dmatrix = xgb.DMatrix(x)
    # dbkg = xgb.DMatrix(bkgx)
    w = np.hstack([bkgw, sigw])
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_output/SR_emu_scangamma_2017_gamma2.json")
    
    y_pred = xgb_model.predict(dmatrix)
    fpr, tpr, _ = roc_curve(y, y_pred)
    # print(fpr,tpr)
    roc_auc = auc(fpr, tpr)
    # print(len(fpr))
    llsig = sigx[:,0]
    llbkg = bkgx[:,0]
    llmin = min(min(bkgx[:,0]),min(sigx[:,0]))
    llmax = max(max(bkgx[:,0]),max(sigx[:,0]))
    binning = (llmax-llmin)/len(fpr)
    tpr_ll=[]
    fpr_ll=[]
    for i in range(len(fpr)):
        tpr_ll.append(len(llsig[llsig<llmax-binning*i])/len(llsig))
        fpr_ll.append(len(llbkg[llbkg<llmax-binning*i])/len(llbkg))
    roc_auc2 = auc(fpr_ll,tpr_ll)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="tab:orange",
        lw=lw,
        label="BDT training (area = %0.2f)" % roc_auc,
    )
    plt.plot(
        
        fpr_ll,
        tpr_ll,

        color="tab:blue",
        lw=lw,
        label="$m_{\ell\ell}$ (area = %0.2f)" % roc_auc2,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FP(relative background efficiency)")
    plt.ylabel("TP (relative signal efficiency)")
    # plt.title(f"ROC{args.channel}")
    plt.legend(loc="lower right")
    plt.savefig("llmass_ROC.pdf")
    plt.savefig("llmass_ROC.png")