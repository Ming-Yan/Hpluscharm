import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt


import xgboost as xgb
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import make_scorer,classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,GridSearchCV, LeaveOneOut
from sklearn.utils import class_weight
import functools
from xgboost import XGBClassifier
###########user define
from BTVNanoCommissioning.utils.xs_scaler import collate
from BTVNanoCommissioning.utils.plot_utils import load_coffea
from BTVNanoCommissioning.helpers.xsection import xsection
from training_config import config2017  as config
from training_config import train_collect
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c","--campaign", default="UL17",choices=["UL17","UL18","UL16_preAPV","UL16_postAPV"], help="campaign")
    parser.add_argument("-v", "--version", type=str, required=True, help="version")
    parser.add_argument("-p","--prefix", type=str, help="prefix")
    parser.add_argument("--weights",type=float,default=1.,help="signal weihgts")
    args = parser.parse_args()
    
    output = load_coffea(config[args.version]["input"][args.campaign]["bkg"],False)
    signal = load_coffea(config[args.version]["input"][args.campaign]["sig"],False)
    sumw = {}
    scales={}
    if args.campaign=="UL17":lumi=41500
    elif args.campaign=="UL16_preAPV":lumi=19500
    elif args.campaign=="UL16_postAPV":lumi=16800
    elif args.campaign=="UL18":lumi=59800
    collect_var={}
    varlist=config[args.version]["varlist"]+["weight"]
    mergemap=config[args.version]["mergemap"]
    trainvar=config[args.version]["varlist"]
    bkgs=[]
    for s in mergemap.keys():bkgs.extend(mergemap[s])
    output[list(signal.keys())[0]]=signal[list(signal.keys())[0]]
    Nbkg,Nsigevent=0,0
    for f in output.keys():
        for s in output[f].keys():   
            if s not in sumw.keys():sumw[s]=output[f][s]['sumw']
            else:sumw[s] += output[f][s]['sumw']
            if s not in collect_var.keys():collect_var[s]={}
            
            for r in output[f][s]['array'].keys(): 
                
                if s not in collect_var.keys() or r not in collect_var[s].keys():collect_var[s][r]={}
                for c in ['emu']:
                    varlist = list(output[f][s]['array'][r][c].keys())
                    
                    if s=="HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8" and( r=="SR_LM" or r=="SR2_LM"):Nsigevent=Nsigevent+len(output[f][s]['array'][r][c][varlist[0]].value)
                    elif  (r=="SR_LM" or r=="SR2_LM" ) and s in bkgs:
                        
                        if len(output[f][s]['array'][r][c].keys())>0:Nbkg=Nbkg+len(output[f][s]['array'][r][c][varlist[0]].value)
                    for var in varlist:
                        val=output[f][s]['array'][r][c][var].value
                        if 'jetflav2_Cv' in var:val[val==-99]=0.
                        if 'jetflav2_pt' ==var:val[val==-99]=-1.
                        if var not in list(collect_var[s][r].keys()):collect_var[s][r][var]=val
                        else:collect_var[s][r][var]=np.concatenate((collect_var[s][r][var],val))
    print(Nbkg,Nsigevent)             
    xs_dict = {}
    for obj in xsection:
        xs_dict[obj["process_name"]] = float(obj["cross_section"])
    for s in sumw.keys():
        scales[s] = xs_dict[s]*lumi/sumw[s]
        for r in collect_var[s].keys():
            if 'weight' not in collect_var[s][r].keys():continue
            collect_var[s][r]['mcwei']=np.full_like(collect_var[s][r]['weight'],scales[s])
    
    
    ### Resize to make it balanced
    for s in collect_var.keys():
        if "HPlusCharm" in s :continue
        for r in collect_var[s].keys():
            for var in collect_var[s][r].keys():
                    collect_var[s][r][var] = collect_var[s][r][var][:int(len(collect_var[s][r][var])/(Nbkg/Nsigevent)*1.5)]
                    

    
    MCvar={}
    weivar={}
    for var in trainvar :
        MCbkgLM = []
        MCvar[var]={}
        for m in mergemap:
            tmpml,tmpwei=[],[]
            for ml in mergemap[m]:
                if args.campaign!="UL17":
                    #if var=="ll_pt" and len(collect_var[ml]['SR_LM'].keys())==0:print(ml,"skipped")
                    if len(collect_var[ml]['SR_LM'].keys())>0 and len(collect_var[ml]['SR2_LM'].keys())>0:
                        tmpml=np.concatenate((tmpml,collect_var[ml]['SR_LM'][var],collect_var[ml]['SR2_LM'][var])) 
                        tmpwei=np.concatenate((tmpwei,collect_var[ml]['SR_LM']['mcwei']*collect_var[ml]['SR_LM']['weight'],collect_var[ml]['SR2_LM']['mcwei']*collect_var[ml]['SR2_LM']['weight'])) 
                    elif len(collect_var[ml]['SR_LM'].keys())>0:
                        tmpml=np.concatenate((tmpml,collect_var[ml]['SR_LM'][var])) 
                        tmpwei=np.concatenate((tmpwei,collect_var[ml]['SR_LM']['mcwei']*collect_var[ml]['SR_LM']['weight'])) 
                    elif len(collect_var[ml]['SR2_LM'].keys())>0:
                        tmpml=np.concatenate((tmpml,collect_var[ml]['SR2_LM'][var])) 
                        tmpwei=np.concatenate((tmpwei,collect_var[ml]['SR2_LM']['mcwei']*collect_var[ml]['SR2_LM']['weight'])) 
                else:
                    if len(collect_var[ml]['SR_LM'].keys())==0:continue
                    tmpml=np.concatenate((tmpml,collect_var[ml]['SR_LM'][var])) 
                    tmpwei=np.concatenate((tmpwei,collect_var[ml]['SR_LM']['mcwei']*collect_var[ml]['SR_LM']['weight'])) 
            MCvar[var][m]=tmpml
            weivar[m]=tmpwei
            MCbkgLM+=[tmpml]
        tmpml,tmpwei=[],[]
        
        MCvar[var]["H+c"]=np.concatenate((collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8']['SR_LM'][var],collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8']['SR2_LM'][var]))
        weivar["H+c"]=np.concatenate((collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8']['SR_LM']["mcwei"]*collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8']['SR_LM']["weight"],collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8']['SR2_LM']["mcwei"]*collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8']['SR2_LM']["weight"]))
        
        
    # print(MCvar["ll_pt"].keys())
    
    x = np.vstack([np.hstack([MCvar[var][s] for s in MCvar[var].keys()]) for var in MCvar.keys()]).T
    
    w = np.hstack([weivar[s] for s in weivar.keys()])
    
    
    y=np.array([])
    
    for i,s in enumerate(weivar.keys()):
        if s!='H+c':y=np.hstack([y,np.zeros(weivar[s].shape[0],dtype=int)*0])
        else:y=np.hstack([y,np.ones(weivar[s].shape[0],dtype=int)*1])
    w[y==0]=np.sum(w[y==1])/np.sum(w[y==0])*w[y==0]
    w=w/np.amin(abs(w))
    print(np.amin(abs(w)),np.sum(w[y==0]),np.sum(w[y==1]),len(w[y==1]),len(w[y==0]))
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.6, random_state=7
    )
    clf = XGBClassifier(early_stopping_rounds=10,max_depth=3, objective='binary:logistic',gamma=2.,colsample_bytree=0.6,eval_metric=['aucpr','logloss','error','mae','auc'])
    if "higgs" in args.version:
        param_grid = {
            "max_depth":[3],
            "n_estimators": [80,100,120],  
            "eta": [0.1,0.2,0.25], 
            # "colsample_bytree": [1], 
            "gamma":[3,4,5],
            "min_child_weight": [5,6,7],
            "subsample":[0.4,0.6],
            "alpha":[0.2,0.4]
        }
    else:
        param_grid = {
            "max_depth":[3],
            "n_estimators": [300,500,600],#,400,500,800],  
            "eta": [0.2,0.1,0.05,0.02], 
            # "colsample_bytree": [0.2,0.4,0.6,0.8,1], 
            "gamma":[0,1,2],
            # "min_child_weight": [4,5,6],
            # "subsample":[0.6,0.8,1.],
            # "alpha":[0.4,0.5,0.8]
        }

    def weighted_accuracy(y_true, y_pred, sample_weight):
        weighted_correct = np.sum(sample_weight * (y_true == y_pred))
        weighted_total = np.sum(sample_weight)
        return weighted_correct / weighted_total
    def weighted_auc_eval(y_pred, eval_set, sample_weight):
        X_eval, y_eval = eval_set[0]
        weighted_auc = roc_auc_score(y_eval, y_pred, sample_weight=sample_weight)
        return 'weighted_auc', weighted_auc, True
    scorer = make_scorer(weighted_accuracy, sample_weight=abs(w_test))

    bdt = GridSearchCV(
        clf, param_grid=param_grid, n_jobs=3,scoring='accuracy'#, return_train_score=True#,error_score='raise'
    )
    
    # print(np.shape(x),np.shape(y),"shape before fit")
    model = bdt.fit(X_train, y_train,eval_set=[(X_train, y_train),(X_test, y_test)],sample_weight=np.abs(w_train),sample_weight_eval_set=[np.abs(w_train),np.abs(w_test)])
    best_model = model.best_estimator_
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",best_model)
    print("\n The best score across ALL searched params:\n",model.best_score_)
    print("\n The best parameters across ALL searched params:\n",model.best_params_)
    # best_model.save_config()
    best_model.save_model(
        f"{args.prefix}_{args.version}_{args.campaign}_nofocal.json"
    )

    
    

    

    # ### Save plots
    fig, ax = plt.subplots(figsize=(15,10))
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=0.5, random_state=42
    )
    
    sigpred_test = best_model.predict_proba(X_test[y_test > 0.5])[:,1]
    bkgpred_test = best_model.predict_proba(X_test[y_test < 0.5])[:,1]
    sigpred_train = best_model.predict_proba(X_train[y_train > 0.5])[:,1]
    bkgpred_train = best_model.predict_proba(X_train[y_train < 0.5])[:,1]
    from scipy.stats import ks_2samp
    sig_ks= ks_2samp(sigpred_test,sigpred_train).pvalue
    bkg_ks= ks_2samp(bkgpred_test,bkgpred_train).pvalue
    bin_counts, bin_edges, patches = plt.hist(sigpred_test, bins=np.linspace(0, 1,40))
    bin_counts2, bin_edges2, patches2 = plt.hist(
        bkgpred_test, bins=np.linspace(0, 1,40)
    )
    fig, ax = plt.subplots()

    ax = plt.hist(
        sigpred_train,
        bins=np.linspace(0, 1,40),
        histtype="step",
        color="blue",
        facecolor=None,
        label="signal:train",
    )
    ax = plt.hist(
        bkgpred_train,
        bins=np.linspace(0, 1,40),
        histtype="step",
        color="red",
        label="background:train",
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centres2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
    
    
    
    ax = plt.errorbar(
        x=bin_centres,
        y=bin_counts,
        yerr=np.sqrt(bin_counts),
        fmt="o",
        capsize=2,
        color="blue",
        label="signal:test",
    )
    ax = plt.errorbar(
        x=bin_centres2,
        y=bin_counts2,
        yerr=np.sqrt(bin_counts2),
        fmt="o",
        capsize=2,
        color="red",
        label="background:test",
    )
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    from scipy.stats import chi2_contingency 
    
    plt.title(f"Accuracy: {round(accuracy * 100.0,1)}% \n SIG KS test: {round(sig_ks,3)},BKG KS test: {round(bkg_ks,3)}")
    # plt.text()
    plt.legend()
    plt.savefig(f"{args.prefix}_{args.version}_{args.campaign}_discri_balance_emu_nofocal.pdf")
    plt.semilogy()
    
    
    plt.savefig(f"{args.prefix}_{args.version}_{args.campaign}_discri_balance_emu_log_nofocal.pdf")
    plt.cla()
    y_pred_test = best_model.predict_proba(X_test)[:,1]
    y_pred_train = best_model.predict_proba(X_train)[:,1]
    
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
    plt.title(f"ROC_emu")
    plt.legend(loc="lower right")
    plt.savefig(f"{args.prefix}_{args.version}_{args.campaign}_ROC_emu_nofocal.pdf")
    
    # for evals in ['accuracy','precision','recall','f1','MCC']:
        # print(evals,best_model.score_eval_func(y_test, best_model.predict(X_test), mode=evals))
    results = best_model.evals_result()
    # print(results)
    epochs = len(results["validation_0"]["logloss"])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
    ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")
    plt.savefig(f"log_loss__emu_{args.prefix}_{args.version}_{args.campaign}_nofocal.pdf")

    # plot classification error
    plt.cla()
    ax.plot(x_axis, results["validation_1"]["error"], label="Test")
    ax.plot(x_axis, results["validation_0"]["error"], label="Train")
    ax.legend()
    plt.ylabel("Classification Error")
    plt.title("XGBoost Classification Error")
    plt.savefig(f"xgb_err_emu_{args.prefix}_{args.version}_{args.campaign}_nofocal.pdf")
    plt.cla()
    ax.plot(x_axis, results["validation_1"]["mae"], label="Test")
    ax.plot(x_axis, results["validation_0"]["mae"], label="Train")
    ax.legend()
    plt.ylabel("Classification MAE")
    plt.title("XGBoost Classification MAE")
    plt.savefig(f"xgb_mae_emu_{args.prefix}_{args.version}_{args.campaign}_nofocal.pdf")
    plt.cla()
    ax.plot(x_axis, results["validation_1"]["aucpr"], label="Test")
    ax.plot(x_axis, results["validation_0"]["aucpr"], label="Train")
    ax.legend()
    plt.ylabel("Classification aucpr")
    plt.title("XGBoost Classification aucpr")
    plt.savefig(f"{args.prefix}_{args.version}_{args.campaign}_xgb_aucpr_emu_nofocal.pdf")
    plt.cla()
    ax.plot(x_axis, results["validation_1"]["auc"], label="Test")
    ax.plot(x_axis, results["validation_0"]["auc"], label="Train")
    ax.legend()
    plt.ylabel("Classification auc")
    plt.title("XGBoost Classification auc")
    plt.savefig(f"{args.prefix}_{args.version}_{args.campaign}_xgb_auc_emu_nofocal.pdf")

    ax = xgb.plot_importance(best_model)
    flab = [f"f{i}" for i in range(len(trainvar))]     
    
    
    label = dict(zip(flab, trainvar))
    ylab = [item.get_text() for item in ax.get_yticklabels()]
    ax.set_yticklabels([train_collect[label[y]][0] for y in ylab])
    plt.xlabel("Feature Importance")
    plt.savefig(f"{args.prefix}_{args.version}_{args.campaign}_importance_plot_balance_nofocal.pdf")
   
