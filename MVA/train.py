import argparse
from coffea.util import load
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV,LeaveOneOut

from BTVNanoCommissioning.utils.xs_scaler import scale_xs_arr
import pandas as pd
import xgboost as xgb
import functools
from sklearn.model_selection import train_test_split
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.metrics import make_scorer

# from joblib import parallel_backend
# with parallel_backend('threading', n_jobs=2):


def load_data(fname, varlist, isSignal=True, channel="emu", lumi=41500):
    # Read data from ROOT files
    data = load(fname)
    events = data["sumw"]
    data_arrays = data["array"]

    # put gen weight
    genwei = scale_xs_arr(events, lumi,"../../../metadata/xsection.json")
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ch", "--channel", choices=["ll", "emu"], required=True, help="SF/DF"
    )
    parser.add_argument("--year", default=2017, help="year")
    parser.add_argument("-v", "--version", type=str, required=True, help="version")
    args = parser.parse_args()
    # Load data
    varlist = [
        "ll_pt",
        "lll1_dr",
        "lll2_dr",
        "llc_dr",
        "lep1_pt",
        "lep2_pt",
        "ll_mass",
        "MET_pt",
        "jetflav_pt",
        "l1met_dphi",
        "l2met_dphi",
        "cW1_dphi",
        "mT1",
        "mT2",
        "jetflav_btagDeepFlavCvL",
        "jetflav_btagDeepFlavCvB",
    ]
    bkgx, bkgy, bkgw = load_data(
        "../../../coffea_output/hists_HWW2l2nu_mcbkg_UL17arrays.coffea", varlist, False, args.channel
    )
    sigx, sigy, sigw = load_data(
        "../../../coffea_output/hists_HWW2l2nu_signal_UL17_4f_arrays.coffea", varlist, True, args.channel
    )
    fig, ax = plt.subplots()

    x = np.vstack([bkgx, sigx])
    y = np.hstack([bkgy, sigy])
    w = np.hstack([bkgw, sigw])
    # # # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    # # print(X_train,y_train)
    # params = { 'max_depth': [3,6,10],
    #        'learning_rate': [0.01, 0.05, 0.1],
    #        'n_estimators': [100, 500, 1000],
    #        'colsample_bytree': [0.3, 0.7]}
    # kfold = StratifiedKFold(n_splits=5, shuffle=True)
    focal_params = {"focal_gamma": [ 1.0,1.1, 1.2, 1.3,1.4]}
    fix_params = {
        "eval_metric": ["logloss", "auc"],
        "max_depth": 3,
        "colsample_bytree":0.5,
        "eta":0.05,
        "subsample":0.5
    }
    
    clf = imb_xgb(fix_params, special_objective="focal", num_round=1000)
    score_eval_func = functools.partial(clf.score_eval_func, mode='accuracy')
    bdt = GridSearchCV(clf, param_grid=focal_params, n_jobs=5,cv=5,scoring=make_scorer(score_eval_func))
    model = bdt.fit(x, y, sample_weight=np.abs(w))
    best_model = model.best_estimator_

    best_model.boosting_model.save_model(
        f"SR_{args.channel}_scangamma_{args.year}_{args.version}.json"
    )
    ### Save plots
    fig, ax = plt.subplots()
    ax = xgb.plot_importance(best_model.boosting_model)
    flab = [f"f{i}" for i in range(len(varlist))]
    label = dict(zip(flab, varlist))
    ylab = [item.get_text() for item in ax.get_yticklabels()]
    ax.set_yticklabels([label[y] for y in ylab])
    plt.xlabel("Feature Importance")
    plt.savefig(f"importance_plot_balance{args.channel}_{args.year}_{args.version}.pdf")
    plt.cla()
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.5, random_state=42
    )

    sigpred_test = best_model.predict_sigmoid(X_test[y_test > 0.5])
    bkgpred_test = best_model.predict_sigmoid(X_test[y_test < 0.5])

    sigpred_train = best_model.predict_sigmoid(X_train[y_train > 0.5])
    bkgpred_train = best_model.predict_sigmoid(X_train[y_train < 0.5])

    bin_counts, bin_edges, patches = plt.hist(sigpred_test, bins=np.linspace(0, 1, 20))
    bin_counts2, bin_edges2, patches2 = plt.hist(
        bkgpred_test, bins=np.linspace(0, 1, 20)
    )
    fig, ax = plt.subplots()

    ax = plt.hist(
        sigpred_train,
        bins=np.linspace(0, 1, 20),
        histtype="step",
        color="blue",
        facecolor=None,
        label="signal:train",
    )
    ax = plt.hist(
        bkgpred_train,
        bins=np.linspace(0, 1, 20),
        histtype="step",
        color="red",
        label="background:train",
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centres2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
    from sklearn.metrics import roc_curve, auc, accuracy_score

    ax = plt.errorbar(
        x=bin_centres,
        y=bin_counts,
        yerr=np.zeros_like(bin_counts),
        fmt="o",
        capsize=2,
        color="blue",
        label="signal:test",
    )
    ax = plt.errorbar(
        x=bin_centres2,
        y=bin_counts2,
        yerr=np.zeros_like(bin_counts2),
        fmt="o",
        capsize=2,
        color="red",
        label="background:test",
    )
    plt.savefig(f"discri_balance{args.channel}_{args.year}_{args.version}.pdf")
    plt.semilogy()
    plt.legend()
    accuracy = accuracy_score(y_test, best_model.predict_determine(X_test))
    plt.title(f"Accuracy: {accuracy * 100.0} %")
    plt.savefig(f"discri_balance{args.channel}_{args.year}_{args.version}_log.pdf")
    plt.cla()
    y_pred_test = best_model.predict_sigmoid(X_test)
    y_pred_train = best_model.predict_sigmoid(X_train)

    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr, tpr)
    fpr2, tpr2, _ = roc_curve(y_train, y_pred_train)
    roc_auc2 = auc(fpr2, tpr2)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="tab:orange",
        lw=lw,
        label="Test:ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot(
        fpr2,
        tpr2,
        color="tab:blue",
        lw=lw,
        label="Train:ROC curve (area = %0.2f)" % roc_auc2,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FP(relative background efficiency)")
    plt.ylabel("TP (relative signal efficiency)")
    plt.title(f"ROC{args.channel}")
    plt.legend(loc="lower right")
    plt.savefig(f"ROC{args.channel}_{args.year}_{args.version}.pdf")
    print(type(best_model))
    results = best_model.evals_result
    print(results)

    epochs = len(results["valid"]["logloss"])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["valid"]["logloss"], label="Test")
    ax.plot(x_axis, results["train"]["logloss"], label="Train")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")
    plt.savefig(f"log_loss_{args.channel}_{args.year}_{args.version}.pdf")

    # plot classification error
    epochs = len(results["valid"]["error"])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["valid"]["error"], label="Test")
    ax.plot(x_axis, results["train"]["error"], label="Train")
    ax.legend()
    plt.ylabel("Classification Error")
    plt.title("XGBoost Classification Error")
    plt.savefig(f"xgb_err_{args.channel}_{args.year}_{args.version}.pdf")

    print("Best parameters:", bdt.best_params_)
    print("Best AUC Score: {}".format(model.best_score_))
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    params = model.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
