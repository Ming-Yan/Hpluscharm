import argparse
from coffea.util import load
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from utils.xs_scaler import scale_xs_arr
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    make_scorer,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb


def load_data_jet(fname, varlist, channel="emu", lumi=41500):
    # Read data from ROOT files
    data = load(fname)
    events = data["sumw"]
    data_arrays = data["jetarray"]

    # data_arrays=data_arrays[(data_arrays['region']=='SR')|(data_arrays['region']=='SR2')]
    # put gen weight

    # Create labels
    # if isSignal :
    #     if channel=='emu':w = np.hstack([data_arrays[dataset]['weight'].value[(data_arrays[dataset]['lepflav'].value=='emu')&(data_arrays[dataset]['flav'].value==4)]*genwei[dataset] for dataset in data_arrays.keys()])
    #     else :w = np.hstack([data_arrays[dataset]['weight'].value[(data_arrays[dataset]['lepflav'].value!='emu')&(data_arrays[dataset]['flav'].value==4)]*genwei[dataset] for dataset in data_arrays.keys()])
    #     if channel =='emu' : x = np.vstack([np.hstack([data_arrays[dataset][var].value[(data_arrays[dataset]['lepflav'].value=='emu')&(data_arrays[dataset]['flav'].value==4)] for dataset in data_arrays.keys()]) for var in varlist]).T
    #     else: x = np.vstack([np.hstack([data_arrays[dataset][var].value[(data_arrays[dataset]['lepflav'].value!='emu')&(data_arrays[dataset]['flav'].value==4)] for dataset in data_arrays.keys()]) for var in varlist]).T
    #     y=np.ones(x.shape[0],dtype=int)
    # else:
    #     if channel=='emu':w = np.hstack([data_arrays[dataset]['weight'].value[(data_arrays[dataset]['lepflav'].value=='emu')&(data_arrays[dataset]['flav'].value!=4)]*genwei[dataset] for dataset in data_arrays.keys()])
    #     else :w = np.hstack([data_arrays[dataset]['weight'].value[(data_arrays[dataset]['lepflav'].value!='emu')&(data_arrays[dataset]['flav'].value!=4)]*genwei[dataset] for dataset in data_arrays.keys()])
    #     if channel =='emu' : x = np.vstack([np.hstack([data_arrays[dataset][var].value[(data_arrays[dataset]['lepflav'].value=='emu')&(data_arrays[dataset]['flav'].value!=4)] for dataset in data_arrays.keys()]) for var in varlist]).T
    #     else: x = np.vstack([np.hstack([data_arrays[dataset][var].value[(data_arrays[dataset]['lepflav'].value!='emu')&(data_arrays[dataset]['flav'].value!=4)] for dataset in data_arrays.keys()]) for var in varlist]).T
    #     y = np.zeros(x.shape[0],dtype=int)
    if channel == "emu":
        w = np.hstack(
            [
                data_arrays[dataset]["weight"].value[
                    (data_arrays[dataset]["lepflav"].value == "emu")
                ]
                for dataset in data_arrays.keys()
            ]
        )
    else:
        w = np.hstack(
            [
                data_arrays[dataset]["weight"].value[
                    (data_arrays[dataset]["lepflav"].value != "emu")
                ]
                for dataset in data_arrays.keys()
            ]
        )
    if channel == "emu":
        x = np.vstack(
            [
                np.hstack(
                    [
                        data_arrays[dataset][var].value[
                            (data_arrays[dataset]["lepflav"].value == "emu")
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
                            (data_arrays[dataset]["lepflav"].value != "emu")
                        ]
                        for dataset in data_arrays.keys()
                    ]
                )
                for var in varlist
            ]
        ).T
    if channel == "emu":
        y = np.hstack(
            [
                data_arrays[dataset]["jetflav"].value[
                    (data_arrays[dataset]["lepflav"].value == "emu")
                ]
                for dataset in data_arrays.keys()
            ]
        )
    else:
        y = np.hstack(
            [
                data_arrays[dataset]["jetflav"].value[
                    (data_arrays[dataset]["lepflav"].value != "emu")
                ]
                for dataset in data_arrays.keys()
            ]
        )
    y[y == 1] = 0  # replace pu flag

    # print(x.shape,y.sh/,len(w))
    # y=np.ones(x.shape[0],dtype=int)
    return x, y, w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ch", "--channel", choices=["ll", "emu"], required=True, help="SF/DF"
    )
    parser.add_argument("--year", default=2017, help="year")
    parser.add_argument("--ntree", default=500, type=int, help="ntrees")
    parser.add_argument("--depth", default=2, type=int, help="Ndepth")
    parser.add_argument("--lrate", default=0.05, type=float, help="Learning rate")
    parser.add_argument("--subsample", default=0.5, type=float, help="subsample")
    parser.add_argument("--gamma", default=3, help="gamma")
    parser.add_argument("--min_child_weight", default=1, help="min_child_weight")
    parser.add_argument("--colsample_bytree", default=0.5, help="colsample_bytree")
    parser.add_argument("-v", "--version", type=str, required=True, help="version")
    from training_config import config2017

    args = parser.parse_args()
    # Load data
    varlist = config2017["varlist"][args.channel][args.version]
    # bkgx, bkgy, bkgw = load_data_jet(config2017['coffea']['sigjet'],varlist,False,args.channel)
    varlist.append("event")
    x, y, w = load_data_jet(config2017["coffea"]["sigjet"], varlist, True, args.channel)
    # print(x,np.shape(x))
    # print('x-2',x[:,:-1],np.shape(x[:,:-1]))

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        x[:, :-1], y, w, test_size=0.5, random_state=42
    )
    #     #w_train_bkg = w_train[y_train<0.5]
    #     #w_train_sig = w_train[y_train>0.5]*1e4
    #     #w_train = np.hstack([w_train_bkg,w_train_sig])

    bdt = XGBClassifier(
        objective="multi:softmax",
        max_depth=args.depth,
        n_estimators=args.ntree,
        learning_rate=args.lrate,
        n_jobs=-1,
        subsample=args.subsample,
        gamma=args.gamma,
        min_child_weight=args.min_child_weight,
        colsample_bytree=args.colsample_bytree,
    )
    bdt.fit(
        X_train,
        y_train,
        early_stopping_rounds=20,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=["merror", "mlogloss", "auc"],
    )
    # pickle.
    #
    # (bdt, open(f"SR_{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}.pkl", "wb"))
    bdt.save_model(
        f"jetbdt_{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_jet.json"
    )
    predict = bdt.predict(X_test)
    # print("Best parameters:", bdt.best_params_)
    # print('Best AUC Score: {}'.format(model.best_score_))
    # print('Accuracy: {}'.format(accuracy_score(y_test, predict)))
    # print(confusion_matrix(y_test,predict))
    idx = bdt.feature_importances_.argsort()
    varlist = np.array(varlist)
    plt.title(f"feature importance {args.channel}")
    plt.barh(varlist[idx], bdt.feature_importances_[idx])

    plt.xlabel("Feature Importance")
    plt.savefig(
        f"importance_plot_balance{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_jet.pdf"
    )
    plt.cla()
    results = bdt.evals_result()
    for i in range(len(varlist[:-1])):
        lvar = X_test[y_test == 0][:, i]
        cvar = X_test[y_test == 4][:, i]
        bvar = X_test[y_test == 5][:, i]
        if varlist[i] == "jetCvL" or varlist[i] == "jetCvB":
            binning = np.linspace(0, 1, 25)
        elif "pt" in varlist[i]:
            binning = np.linspace(0, 200, 25)
        elif "eta" in varlist[i]:
            binning = np.linspace(-2.5, 2.5, 25)
        elif "phi" in varlist[i]:
            binning = np.linspace(-3, 3, 30)
        elif "dr" in varlist[i]:
            binning = np.linspace(0, 5, 25)
        elif "nj" == varlist[i]:
            binning = np.linspace(0, 10, 10)

        plt.hist(
            lvar,
            bins=binning,
            histtype="step",
            label="light",
            color="tab:blue",
            density=True,
        )
        plt.hist(
            cvar,
            bins=binning,
            histtype="step",
            label="c",
            color="tab:orange",
            density=True,
        )
        plt.hist(
            bvar,
            bins=binning,
            histtype="step",
            label="b",
            color="tab:green",
            density=True,
        )
        plt.xlabel(varlist[i])
        plt.legend()
        plt.savefig(
            f"discri_balance{args.channel}_{args.year}_{args.version}_{varlist[i]}_jet.pdf"
        )
        plt.cla()
    lpred = bdt.predict_proba(X_test[y_test == 0])
    cpred = bdt.predict_proba(X_test[y_test == 4])
    bpred = bdt.predict_proba(X_test[y_test == 5])
    lpred2 = bdt.predict_proba(X_train[y_train == 0])
    cpred2 = bdt.predict_proba(X_train[y_train == 4])
    bpred2 = bdt.predict_proba(X_train[y_train == 5])
    from sklearn.metrics import accuracy_score

    fig, ax = plt.subplots()
    plt.hist(
        lpred[:, 0],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:test",
        color="tab:blue",
        alpha=0.5,
    )
    plt.hist(
        cpred[:, 0],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:test",
        color="tab:orange",
        alpha=0.5,
    )
    plt.hist(
        bpred[:, 0],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:test",
        color="tab:green",
        alpha=0.5,
    )
    plt.hist(
        lpred2[:, 0],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:train",
        color="tab:blue",
    )
    plt.hist(
        cpred2[:, 0],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:train",
        color="tab:orange",
    )
    plt.hist(
        bpred2[:, 0],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:train",
        color="tab:green",
    )
    plt.xlabel("Prob(l)")
    # plt.semilogy()
    plt.legend()
    accuracy = accuracy_score(y_test, bdt.predict(X_test))
    plt.title(f"Accuracy: {accuracy * 100.0} %")
    plt.savefig(
        f"discri_balance{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_lprob_jet.pdf"
    )
    plt.cla()
    plt.hist(
        lpred[:, 1],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:test",
        color="tab:blue",
        alpha=0.5,
    )
    plt.hist(
        cpred[:, 1],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:test",
        color="tab:orange",
        alpha=0.5,
    )
    plt.hist(
        bpred[:, 1],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:test",
        color="tab:green",
        alpha=0.5,
    )
    plt.hist(
        lpred2[:, 1],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:train",
        color="tab:blue",
    )
    plt.hist(
        cpred2[:, 1],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:train",
        color="tab:orange",
    )
    plt.hist(
        bpred2[:, 1],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:train",
        color="tab:green",
    )
    plt.xlabel("Prob(c)")
    # plt.semilogy()
    plt.legend()
    accuracy = accuracy_score(y_test, bdt.predict(X_test))
    plt.title(f"Accuracy: {accuracy * 100.0} %")
    plt.savefig(
        f"discri_balance{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_cprob_jet.pdf"
    )
    plt.cla()
    plt.hist(
        lpred[:, 2],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:test",
        color="tab:blue",
        alpha=0.5,
    )
    plt.hist(
        cpred[:, 2],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:test",
        color="tab:orange",
        alpha=0.5,
    )
    plt.hist(
        bpred[:, 2],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:test",
        color="tab:green",
        alpha=0.5,
    )
    plt.hist(
        lpred2[:, 2],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:test",
        color="tab:blue",
    )
    plt.hist(
        cpred2[:, 2],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:test",
        color="tab:orange",
    )
    plt.hist(
        bpred2[:, 2],
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:test",
        color="tab:green",
    )
    plt.xlabel("Prob(b)")
    # plt.semilogy()
    plt.legend()
    accuracy = accuracy_score(y_test, bdt.predict(X_test))
    plt.title(f"Accuracy: {accuracy * 100.0} %")
    plt.savefig(
        f"discri_balance{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_bprob_jet.pdf"
    )
    plt.cla()
    plt.hist(
        lpred[:, 1] / (lpred[:, 2] + lpred[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:test",
        color="tab:blue",
        alpha=0.5,
    )
    plt.hist(
        cpred[:, 1] / (cpred[:, 2] + cpred[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:test",
        color="tab:orange",
        alpha=0.5,
    )
    plt.hist(
        bpred[:, 1] / (bpred[:, 2] + bpred[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:test",
        color="tab:green",
        alpha=0.5,
    )
    plt.hist(
        lpred2[:, 1] / (lpred2[:, 2] + lpred2[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:train",
        color="tab:blue",
    )
    plt.hist(
        cpred2[:, 1] / (cpred2[:, 2] + cpred2[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:train",
        color="tab:orange",
    )
    plt.hist(
        bpred2[:, 1] / (bpred2[:, 2] + bpred2[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:train",
        color="tab:green",
    )
    plt.xlabel("CvB")
    # plt.semilogy()
    plt.legend()
    plt.savefig(
        f"discri_balance{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_CvB_jet.pdf"
    )
    plt.cla()
    plt.hist(
        lpred[:, 1] / (lpred[:, 0] + lpred[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:test",
        color="tab:blue",
        alpha=0.5,
    )
    plt.hist(
        cpred[:, 1] / (cpred[:, 0] + cpred[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:test",
        color="tab:orange",
        alpha=0.5,
    )
    plt.hist(
        bpred[:, 1] / (bpred[:, 0] + bpred[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:test",
        color="tab:green",
        alpha=0.5,
    )
    plt.hist(
        lpred2[:, 1] / (lpred2[:, 0] + lpred2[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="light:train",
        color="tab:blue",
    )
    plt.hist(
        cpred2[:, 1] / (cpred2[:, 0] + cpred2[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="c:train",
        color="tab:orange",
    )
    plt.hist(
        bpred2[:, 1] / (bpred2[:, 0] + bpred2[:, 1]),
        bins=np.linspace(0, 1, 20),
        histtype="step",
        label="b:train",
        color="tab:green",
    )
    plt.xlabel("CvL")
    plt.legend()
    plt.savefig(
        f"discri_balance{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_CvL_jet.pdf"
    )
    # plt.semilogy()

    plt.cla()
    y_pred_test = bdt.predict_proba(X_test)
    y_pred_train = bdt.predict_proba(X_train)
    y_test_cvb = y_test[y_test > 3]
    y_test_cvb = np.where(y_test_cvb == 4, 1, 0)
    y_train_cvb = y_train[y_train > 3]
    y_train_cvb = np.where(y_train_cvb == 4, 1, 0)
    fpr_cvb, tpr_cvb, _ = roc_curve(
        y_test_cvb, y_pred_test[y_test > 3][:, 1]
    )  # ,sample_weight=w_test)
    roc_auc_cvb = auc(fpr_cvb, tpr_cvb)
    fpr2_cvb, tpr2_cvb, _ = roc_curve(
        y_train_cvb, y_pred_train[y_train > 3][:, 1]
    )  # ,sample_weight=w_train)
    roc_auc2_cvb = auc(fpr2_cvb, tpr2_cvb)
    plt.figure()
    lw = 2
    plt.plot(
        fpr_cvb,
        tpr_cvb,
        color="tab:orange",
        lw=lw,
        label="Test:ROC curve (area = %0.2f)" % roc_auc_cvb,
    )
    plt.plot(
        fpr2_cvb,
        tpr2_cvb,
        color="tab:blue",
        lw=lw,
        label="Train:ROC curve (area = %0.2f)" % roc_auc2_cvb,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Prob(b)")
    plt.ylabel("Prob(c)")
    plt.title(f"ROC{args.channel}")
    plt.legend(loc="lower right")
    plt.savefig(
        f"ROC_cvb_{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_jet.pdf"
    )
    plt.cla()
    y_test_cvl = y_test[y_test < 5]
    y_test_cvl = np.where(y_test_cvl == 4, 1, 0)
    y_train_cvl = y_train[y_train < 5]
    y_train_cvl = np.where(y_train_cvl == 4, 1, 0)
    fpr_cvl, tpr_cvl, _ = roc_curve(
        y_test_cvl, y_pred_test[y_test < 5][:, 1]
    )  # ,sample_weight=w_test)
    roc_auc_cvl = auc(fpr_cvl, tpr_cvl)
    fpr2_cvl, tpr2_cvl, _ = roc_curve(
        y_train_cvl, y_pred_train[y_train < 5][:, 1]
    )  # ,sample_weight=w_train)
    roc_auc2_cvl = auc(fpr2_cvl, tpr2_cvl)
    plt.figure()
    lw = 2
    plt.plot(
        fpr_cvl,
        tpr_cvl,
        color="tab:orange",
        lw=lw,
        label="Test:ROC curve (area = %0.2f)" % roc_auc_cvl,
    )
    plt.plot(
        fpr2_cvl,
        tpr2_cvl,
        color="tab:blue",
        lw=lw,
        label="Train:ROC curve (area = %0.2f)" % roc_auc2_cvl,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Prob(l)")
    plt.ylabel("Prob(c)")
    plt.title(f"ROC{args.channel}")
    plt.legend(loc="lower right")
    plt.savefig(
        f"ROC_cvl_{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_jet.pdf"
    )
    print("Confusion matrix \n", confusion_matrix(y_test, bdt.predict(X_test)))
    results = bdt.evals_result()
    # # retrieve performance metrics

    epochs = len(results["validation_0"]["merror"])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
    ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")
    plt.savefig(
        f"log_loss_{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_jet.pdf"
    )

    #     plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_0"]["merror"], label="Train")
    ax.plot(x_axis, results["validation_1"]["merror"], label="Test")
    ax.legend()
    plt.ylabel("Classification Error")
    plt.title("XGBoost Classification Error")
    plt.savefig(
        f"xgb_err_{args.channel}_depth{args.depth}_ntree{args.ntree}_{args.year}_{args.version}_jet.pdf"
    )
    # plt.show()

    print("ranking!!!")
    print("CvL")
    event_num = x[:, -1]

    events = np.unique(event_num)
    import awkward as ak

    y_cvl = np.array(
        bdt.predict_proba(x[:, :-1])[:, 1]
        / (bdt.predict_proba(x[:, :-1])[:, 0] + bdt.predict_proba(x[:, :-1])[:, 1])
    )
    y_cvb = np.array(
        bdt.predict_proba(x[:, :-1])[:, 1]
        / (bdt.predict_proba(x[:, :-1])[:, 2] + bdt.predict_proba(x[:, :-1])[:, 1])
    )
    cvls = {}
    cvbs = {}
    cvl_cvbs = {}
    cvls_jet = {}
    cvbs_jet = {}
    cvl_cvbs_jet = {}
    flav = {}
    for ev in events:
        cvls[ev] = y_cvl[ev == event_num]
        cvbs[ev] = y_cvb[ev == event_num]
        cvl_cvbs[ev] = y_cvl[ev == event_num] + y_cvb[ev == event_num]
        cvls_jet[ev] = x[:, -3][ev == event_num]
        cvbs_jet[ev] = x[:, -2][ev == event_num]
        cvl_cvbs_jet[ev] = x[:, -3][ev == event_num] + x[:, -2][ev == event_num]
        flav[ev] = y[ev == event_num]
    cvl_arr = ak.Array(cvls.values())
    cvb_arr = ak.Array(cvbs.values())
    cvl_cvb_arr = ak.Array(cvl_cvbs.values())
    cvl_arr2 = ak.Array(cvls_jet.values())
    cvb_arr2 = ak.Array(cvbs_jet.values())
    cvl_cvb_arr2 = ak.Array(cvl_cvbs_jet.values())
    flav_arr = ak.Array(flav.values())
    flav_cvl = flav_arr[ak.argsort(cvl_arr, ascending=False)]
    flav_cvb = flav_arr[ak.argsort(cvb_arr, ascending=False)]
    flav_cvl_cvb = flav_arr[ak.argsort(cvl_cvb_arr, ascending=False)]
    flav_cvl2 = flav_arr[ak.argsort(cvl_arr2, ascending=False)]
    flav_cvb2 = flav_arr[ak.argsort(cvb_arr2, ascending=False)]
    flav_cvl_cvb2 = flav_arr[ak.argsort(cvl_cvb_arr2, ascending=False)]
    totalev = ak.count(flav_cvl[:, 0])
    print("====== train CvL ranking ======")
    print(
        "b: ",
        float(ak.sum(flav_cvl[:, 0] == 5) / ak.count(flav_cvl[:, 0])),
        "c: ",
        float(ak.sum(flav_cvl[:, 0] == 4) / ak.count(flav_cvl[:, 0])),
        "l: ",
        float(ak.sum(flav_cvl[:, 0] == 0) / ak.count(flav_cvl[:, 0])),
    )
    print("====== train CvB ranking ======")
    print(
        "b: ",
        float(ak.sum(flav_cvb[:, 0] == 5) / ak.count(flav_cvb[:, 0])),
        "c: ",
        float(ak.sum(flav_cvb[:, 0] == 4) / ak.count(flav_cvb[:, 0])),
        "l: ",
        float(ak.sum(flav_cvb[:, 0] == 0) / ak.count(flav_cvb[:, 0])),
    )
    print("====== train CvL/B ranking ======")
    print(
        "b: ",
        float(ak.sum(flav_cvl_cvb[:, 0] == 5) / ak.count(flav_cvl_cvb[:, 0])),
        "c: ",
        float(ak.sum(flav_cvl_cvb[:, 0] == 4) / ak.count(flav_cvl_cvb[:, 0])),
        "l: ",
        float(ak.sum(flav_cvl_cvb[:, 0] == 0) / ak.count(flav_cvl_cvb[:, 0])),
    )

    print("====== CvL ranking ======")
    print(
        "b: ",
        float(ak.sum(flav_cvl2[:, 0] == 5) / ak.count(flav_cvl2[:, 0])),
        "c: ",
        float(ak.sum(flav_cvl2[:, 0] == 4) / ak.count(flav_cvl2[:, 0])),
        "l: ",
        float(ak.sum(flav_cvl2[:, 0] == 0) / ak.count(flav_cvl2[:, 0])),
    )
    print("====== CvB ranking ======")
    print(
        "b: ",
        float(ak.sum(flav_cvb2[:, 0] == 5) / ak.count(flav_cvb2[:, 0])),
        "c: ",
        float(ak.sum(flav_cvb2[:, 0] == 4) / ak.count(flav_cvb2[:, 0])),
        "l: ",
        float(ak.sum(flav_cvb2[:, 0] == 0) / ak.count(flav_cvb2[:, 0])),
    )
    print("====== CvL/B ranking ======")
    print(
        "b: ",
        float(ak.sum(flav_cvl_cvb2[:, 0] == 5) / ak.count(flav_cvl_cvb2[:, 0])),
        "c: ",
        float(ak.sum(flav_cvl_cvb2[:, 0] == 4) / ak.count(flav_cvl_cvb2[:, 0])),
        "l: ",
        float(ak.sum(flav_cvl_cvb2[:, 0] == 0) / ak.count(flav_cvl_cvb2[:, 0])),
    )
