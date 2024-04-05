import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from coffea.util import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import pandas as pd
import xgboost as xgb
import functools
from sklearn.model_selection import train_test_split
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.metrics import make_scorer,classification_report, confusion_matrix
###########user define
from BTVNanoCommissioning.utils.xs_scaler import collate
from BTVNanoCommissioning.utils.plot_utils import load_coffea
from BTVNanoCommissioning.helpers.xsection import xsection
from training_config import config2017  as config
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--campaign", default="UL17",choices=["UL17","UL18","UL16_preAPV","UL16_postAPV"], help="campaign")
    parser.add_argument("-v", "--version", type=str, required=True, help="version")
    parser.add_argument("--weights",type=float,default=20000.,help="signal weihgts")
    args = parser.parse_args()
    
    output = load_coffea(config[args.version]["input"]["bkg"],False)
    data = load_coffea({"input":"/nfs/dust/cms/user/milee/CoffeaRunner/data_array_v00/*.coffea"},False)
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
    for var in trainvar :
        MCbkgLM = []
        MCvar[var]={}
        for m in mergemap:
            tmpml,tmpml2=[],[]
            tmpwei,tmpwei2=[],[]
            for ml in mergemap[m]:
                tmpml=np.concatenate((tmpml,collect_var[ml]['SR_LM'][var])) 

                tmpwei=np.concatenate((tmpwei,collect_var[ml]['SR_LM']['mcwei']*collect_var[ml]['SR_LM']['weight'])) 
            MCvar[var][m]=tmpml
            weivar[m]=tmpwei
            MCbkgLM+=[tmpml]
        MCvar[var]["Hc"]=signal_var['gchcWW2L2Nu_4f']['SR_LM'][var]
        weivar["Hc"]=signal_var['gchcWW2L2Nu_4f']['SR_LM']["mcwei"]*signal_var['gchcWW2L2Nu_4f']['SR_LM']["weight"]*args.weights

    x = np.vstack([np.hstack([MCvar[var][s] for s in MCvar[var].keys()]) for var in MCvar.keys()]).T
    w = np.hstack([weivar[s] for s in weivar.keys()])
    
    y=np.array([])
    for i,s in enumerate(weivar.keys()):
        if s!='Hc':y=np.hstack([y,np.zeros(weivar[s].shape[0],dtype=int)*0])
        else:y=np.hstack([y,np.ones(weivar[s].shape[0],dtype=int)*1])
    fig, ax = plt.subplots()
    # print(y,np.len())
    # x = np.vstack([bkgx, sigx])
    # y = np.hstack([bkgy, sigy])
    dmatrix = xgb.DMatrix(x)
    dsig=xgb.DMatrix(x[y>0.5])
    dbkg=xgb.DMatrix(x[y<0.5])
    # print(x[y<0])
    # dbkg = xgb.DMatrix(bkgx)
    # w = np.hstack([bkgw, sigw])
    xgb_model = xgb.Booster()
    xgb_model2 = xgb.Booster()
    # xgb_model.load_model("binary_LM_UL17_binary_opt.json")
    xgb_model2.load_model("binary_LM_UL17_nofocal.json")
    
    y_pred = 1.0/(1+np.exp(-xgb_model.predict(dsig)))
    y_pred2 = xgb_model2.predict(dsig)
    # print(y_pred,y_pred2)
    plt.hist(1.0/(1+np.exp(-xgb_model.predict(dsig))),label="sig:old",color='tab:blue',histtype='step',range=[0,1],weights=w[y>0.5],bins=20)
    plt.hist(1.0/(1+np.exp(-xgb_model.predict(dbkg))),label="bkg:old",color='tab:red',histtype='step',range=[0,1],weights=w[y<0.5],bins=20)
    plt.hist(xgb_model2.predict(dsig),label='sig:new',histtype='step',color='b',range=[0,1],weights=w[y>0.5],bins=20)
    plt.hist(xgb_model2.predict(dbkg),label='sig:new',histtype='step',color='r',range=[0,1],weights=w[y<0.5],bins=20)
    plt.savefig("oldnewscore.png")
    from sklearn.metrics import roc_curve, auc, accuracy_score

    # fpr, tpr, _ = roc_curve(y,1.0/(1+np.exp(-xgb_model.predict(dmatrix))))
    # fpr2, tpr2, _2 = roc_curve(y,xgb_model2.predict(dmatrix))#roc_curve(y,1.0/(1+np.exp(-xgb_model.predict(dmatrix))))
    # # # print(fpr,tpr)
    fpr,tpr, fpr2,tpr2=[],[],[],[]
    bins=100
    ymin,ymax=np.min(1.0/(1+np.exp(-xgb_model.predict(dmatrix)))),np.max(1.0/(1+np.exp(-xgb_model.predict(dmatrix))))
    ymin2,ymax2=np.min(xgb_model2.predict(dmatrix)),np.max(xgb_model2.predict(dmatrix))
    ybin,ybin2=(ymax-ymin)/100,(ymax2-ymin2)/100
    nbkgsum,nsigsum=np.sum(w[y<0.5]),np.sum(w[y>0.5])
    bkgw,sigw=w[y<0.5],w[y>0.5]
    # print(ymin,ymax,ymin2,ymax2)
    for i in range(100):
        ypred_sig=1.0/(1+np.exp(-xgb_model.predict(dsig)))
        ypred_bkg=1.0/(1+np.exp(-xgb_model.predict(dbkg)))
        # ypred2=1.0/(1+np.exp(-xgb_model.predict(dmatrix)))
        
        fpr.append(np.sum(bkgw[ypred_bkg>ymin+i*ybin])/nbkgsum)
        tpr.append(np.sum(sigw[ypred_sig>ymin+i*ybin])/nsigsum)
        fpr2.append(np.sum(bkgw[xgb_model2.predict(dbkg)>ymin2+i*ybin2])/nbkgsum)
        # print(np.sum(bkgw[xgb_model2.predict(dbkg)>ymin2+i*ybin2]))
        tpr2.append(np.sum(sigw[xgb_model2.predict(dsig)>ymin2+i*ybin2])/nsigsum)
    # print(fpr,tpr)
    # print(fpr2,tpr2)
    # roc_auc = auc(fpr, tpr)
    # roc_auc2 = auc(fpr2,tpr2)
    print(np.around(np.array(tpr),2),np.around(np.array(tpr2),2))
    print(np.sum(tpr),np.sum(tpr2))
    plt.cla()
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="tab:orange",
        lw=lw,
        label="imbalanced",#(area = %0.2f)" % round(np.sum(tpr)/100,2),
    )
    plt.plot(
        
        fpr2,
        tpr2,

        color="tab:blue",
        lw=lw,
        label="balanced",#(area = %0.2f)" % round(np.sum(tpr2)/100,2),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FP(relative background efficiency)")
    plt.ylabel("TP (relative signal efficiency)")
    # # plt.title(f"ROC{args.channel}")
    plt.legend(loc="lower right")
    # plt.savefig("llmass_ROC.pdf")
    plt.savefig("new_model_ROC.png")
