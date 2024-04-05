import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt


import xgboost as xgb

from sklearn.metrics import make_scorer,classification_report, confusion_matrix,ConfusionMatrixDisplay
# from multi_imbalance import MutliClassifierFocalLoss as multi_loss
from sklearn.model_selection import train_test_split,GridSearchCV, LeaveOneOut
from sklearn.utils import class_weight

import functools
from xgboost import XGBClassifier
###########user define
from BTVNanoCommissioning.utils.xs_scaler import collate
from BTVNanoCommissioning.utils.plot_utils import load_coffea
from BTVNanoCommissioning.helpers.xsection import xsection
from training_config import config2017  as config
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--campaign", default="UL17",choices=["UL17","UL18","UL16_preAPV","UL16_postAPV"], help="campaign")
    parser.add_argument("--prefix", type=str, required=True, help="prefix")
    parser.add_argument("-v", "--version", type=str, required=True, help="version")
    parser.add_argument("--weights",type=float,default=1.,help="signal weihgts")
    args = parser.parse_args()
    
    output = load_coffea(config[args.version]["input"]["bkg"],False)
    signal = load_coffea(config[args.version]["input"]["sig"],False)
    sumw = {}
    scales={}
    if args.campaign=="UL17":lumi=41500
    elif args.campaign=="UL16_preAPV":lumi=19500
    elif args.campaign=="UL16_postAPV":lumi=16800
    elif args.campaign=="UL18":lumi=59800
    collect_var={}
    varlist=config[args.version]["varlist"]+["weight"]
    
    for f in output.keys():
        for s in output[f].keys():
            if s not in sumw.keys():sumw[s]=output[f][s]['sumw']
            else:sumw[s] += output[f][s]['sumw']
            
            if s not in collect_var.keys():collect_var[s]={}
            for r in output[f][s]['array'].keys():
                
                if s not in collect_var.keys() or r not in collect_var[s].keys():collect_var[s][r]={}
                for var in varlist:
                    if r =='top_CR' and var=='BDT' : continue
                    if var not in list(collect_var[s][r].keys()):collect_var[s][r][var]=output[f][s]['array'][r][var].value
                    else:collect_var[s][r][var]=np.concatenate((collect_var[s][r][var],output[f][s]['array'][r][var].value))
                
    xs_dict = {}
    for obj in xsection:
        xs_dict[obj["process_name"]] = float(obj["cross_section"])
    for s in sumw.keys():
        scales[s] = xs_dict[s]*lumi/sumw[s]
        for r in collect_var[s].keys():
            collect_var[s][r]['mcwei']=np.full_like(collect_var[s][r]['weight'],scales[s])
    
    signal_var={'gchcWW2L2Nu_4f':{}}

    signal_dict=signal[list(signal.keys())[0]]['gchcWW2L2Nu_4f']['array']
    for r in signal_dict.keys():
        s='gchcWW2L2Nu_4f'
        signal_var[s][r]={}
        
        for var in varlist:
            if r =='top_CR' and var=='BDT' : continue
            if var not in signal_var[s][r].keys():signal_var[s][r][var]=signal_dict[r][var].value
            else:np.concatenate(signal_var[s][r][var],signal_dict[r][var].value)
        signal_var[s][r]['mcwei']=np.full_like(signal_var[s][r]['weight'],xs_dict['gchcWW2L2Nu_4f']*lumi/signal[list(signal.keys())[0]]['gchcWW2L2Nu_4f']['sumw'])
    mergemap=config[args.version]["mergemap"]
    trainvar=config[args.version]["varlist"]
    MCvar={}
    weivar={}
    Nsigevent=len(signal_var['gchcWW2L2Nu_4f']['SR_LM']["weight"])
    print(Nsigevent)
    for var in trainvar :
        MCbkgLM = []
        MCvar[var]={}
        for m in mergemap:
            tmpml,tmpml2=[],[]
            tmpwei,tmpwei2=[],[]
            bkglen = int(Nsigevent*1.5)
            for ml in mergemap[m]:
                 #int(len(collect_var[ml]['SR_LM']['weight']))
                
                tmpml=np.concatenate((tmpml,collect_var[ml]['SR_LM'][var])) 
                tmpwei=np.concatenate((tmpwei,collect_var[ml]['SR_LM']['mcwei']*collect_var[ml]['SR_LM']['weight'])) 
            MCvar[var][m]=tmpml[:bkglen]
            weivar[m]=tmpwei[:bkglen]
            MCbkgLM+=[tmpml[:bkglen]]
        MCvar[var]["Hc"]=signal_var['gchcWW2L2Nu_4f']['SR_LM'][var]
        weivar["Hc"]=signal_var['gchcWW2L2Nu_4f']['SR_LM']["mcwei"]*signal_var['gchcWW2L2Nu_4f']['SR_LM']["weight"]
    
    x = np.vstack([np.hstack([MCvar[var][s] for s in MCvar[var].keys()]) for var in MCvar.keys()]).T
    w = np.hstack([weivar[s] for s in weivar.keys()])
    y= np.hstack([np.ones(weivar[s].shape[0],dtype=int)*i for i,s in enumerate(weivar.keys())])
    # for i,s in enumerate(weivar.keys()):
    #     y=np.hstack([y,np.ones(weivar[s].shape[0],dtype=int)])
    # clf = multi_loss(eta=0.1,max_depth=3,num_boost_round=1,verbose_eval=1,  early_stopping_rounds=None, alpha=1.,gamma=1.,maximize=False)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.5, random_state=42
    )
    clf = XGBClassifier(early_stopping_rounds=10,max_depth=3, objective='multi:softproba',gamma=3,eval_metric=['mlogloss','merror'],eta=0.05,colsample_bytree=0.6,min_child_weight=5,subsample=0.8,alpha=0.5)#,'mae','auc'])
    param_grid = {
        "max_depth":[3,4],
        # "max_depth":[2],
        "n_estimators": [200,600,800], 
        # "n_estimators": [200]
        "eta": [0.1,0.05,0.01], 
        "colsample_bytree": [0.6,0.8,1], 
        "gamma":[0,2,4],
        "min_child_weight": [4,5,6],
        # "subsample":[0.6,0.8,1.],
        "alpha":[0.1,0.5,0.8]
    }
                            
    bdt = GridSearchCV(
        clf, param_grid=param_grid, n_jobs=-1,scoring='accuracy',cv=4,return_train_score=True#,error_score='raise'
    )
    
    # print(np.shape(x),np.shape(y),"shape before fit")
    model = bdt.fit(X_train, y_train,eval_set=[(X_train, y_train),(X_test, y_test)])
    best_model = model.best_estimator_
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",best_model)
    print("\n The best score across ALL searched params:\n",model.best_score_)
    print("\n The best parameters across ALL searched params:\n",model.best_params_)
    # best_model.save_config()
    best_model.save_model(
        f"{args.prefix}_{args.version}_{args.campaign}__mult_nofocal.json"
    )

    

    # ### Save plots
    fig, ax = plt.subplots()
    ax = xgb.plot_importance(best_model)
    flab = [f"f{i}" for i in range(len(varlist))]
    label = dict(zip(flab, varlist))
    ylab = [item.get_text() for item in ax.get_yticklabels()]
    ax.set_yticklabels([label[y] for y in ylab])
    plt.xlabel("Feature Importance")
    plt.savefig(f"importance_plot_balance_{args.prefix}_{args.version}_{args.campaign}__mult_nofocal.pdf")
    plt.cla()
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.5, random_state=42
    )
    print(np.shape(best_model.predict_proba(X_test)),y_test)
    from sklearn.metrics import roc_curve, auc, accuracy_score
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    # for n in range(6):
    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for  n,prob in  enumerate(weivar.keys()):
        fig, ax = plt.subplots()
        sig_ks = []
        all_bincount=[]
        for label,s in enumerate(weivar.keys()):
        
            sigpred_test = best_model.predict_proba(X_test[y_test==label])[:,n]
            sigpred_train = best_model.predict_proba(X_train[y_train==label])[:,n]
            from scipy.stats import ks_2samp
            # sig_ks.append()
            ks=ks_2samp(sigpred_test,sigpred_train).pvalue
            bin_counts, bin_edges = np.histogram(sigpred_test, bins=np.linspace(0, 1, 20))
            # bin_counts_train, bin_edges_train = np.histogram(sigpred_train, bins=np.linspace(0, 1, 20),density=True)
            all_bincount.append(bin_counts)
            ax.hist(
                sigpred_train,
                bins=np.linspace(0, 1, 20),
                histtype="step",
                color=colors[label],
                facecolor=None,
                label=f"{s}:train",
                # density=True
            )
            
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.errorbar(
                x=bin_centres,
                y=bin_counts,
                yerr=np.sqrt(bin_counts),
                fmt="o",
                capsize=2,
                color=colors[label],
                label=f"{s}:test KS:{round(ks,2)}",
            )
            plt.title(f"Prob({prob})")
            # plt.title(f"Accuracy: {round(accuracy * 100.0,1)}%, SIG KS test: {round(sig_ks,3)},BKG KS test: {round(bkg_ks,3)}")
            
            # plt.text()
        ax.legend(ncol=2)
        plt.savefig(f"discri_balance_emu_{args.prefix}_{args.version}_{args.campaign}_{prob}_mult_nofocal.pdf")
        plt.semilogy()
        plt.savefig(f"discri_balance_emu_{args.prefix}_{args.version}_{args.campaign}_{prob}_log_mult_nofocal.pdf")
        plt.cla()

    results = best_model.evals_result()
    # print(results)
    epochs = len(results["validation_0"]["mlogloss"])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
    ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")
    plt.savefig(f"log_loss__emu_{args.prefix}_{args.version}_{args.campaign}__mult_nofocal.pdf")

    # plot classification error
    plt.cla()
    ax.plot(x_axis, results["validation_1"]["merror"], label="Test")
    ax.plot(x_axis, results["validation_0"]["merror"], label="Train")
    ax.legend()
    plt.ylabel("Classification Error")
    plt.title("XGBoost Classification Error")
    plt.savefig(f"xgb_err_emu_{args.prefix}_{args.version}_{args.campaign}__mult_nofocal.pdf")
    # plt.cla()
    # ax.plot(x_axis, results["validation_1"]["mae"], label="Test")
    # ax.plot(x_axis, results["validation_0"]["mae"], label="Train")
    # ax.legend()
    # plt.ylabel("Classification MAE")
    # plt.title("XGBoost Classification MAE")
    # plt.savefig(f"xgb_mae_emu_{args.prefix}_{args.version}_{args.campaign}__mult_nofocal.pdf")
    # plt.cla()
    # ax.plot(x_axis, results["validation_1"]["aucpr"], label="Test")
    # ax.plot(x_axis, results["validation_0"]["aucpr"], label="Train")
    # ax.legend()
    # plt.ylabel("Classification aucpr")
    # plt.title("XGBoost Classification aucpr")
    # plt.savefig(f"xgb_aucpr_emu_{args.prefix}_{args.version}_{args.campaign}__mult_nofocal.pdf")
    # plt.cla()
    # ax.plot(x_axis, results["validation_1"]["auc"], label="Test")
    # ax.plot(x_axis, results["validation_0"]["auc"], label="Train")
    # ax.legend()
    # plt.ylabel("Classification auc")
    # plt.title("XGBoost Classification auc")
    # plt.savefig(f"xgb_auc_emu_{args.prefix}_{args.version}_{args.campaign}__mult_nofocal.pdf")
    # cm = confusion_matrix(y_test, best_model.predict(X_test))
    # plt.figure(figsize=(15,10))
    # plt.clf()
    # plt.imshow(cm, interpolation='nearest')
    # classNames = list(weivar.keys())
    # plt.title('Confusion matrix')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # tick_marks = np.arange(len(classNames))
    # plt.xticks(tick_marks, classNames, rotation=45)
    # plt.yticks(tick_marks, classNames)
    # # s = [['TN','FP'], ['FN', 'TP']]

    # for i,si in enumerate(classNames):
    #     for j,sj in enumerate(classNames):
    #         plt.text(j,i, str(cm[i][j]))
    #         print(cm[i][j],si,sj)
    # plt.invert_xaxis()
    disp =ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        display_labels=list(weivar.keys()),
        normalize="true",
        values_format=".3g"
    )
    disp.ax_.set_title("confusion matrix")
    disp.ax_.invert_xaxis()
    plt.savefig(f"confusion_matrix_{args.prefix}_{args.version}_{args.campaign}_mult_nofocal.pdf")
