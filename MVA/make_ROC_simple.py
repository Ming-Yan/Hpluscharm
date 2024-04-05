import xgboost as xgb
import numpy as np
def make_ROC(MCvar,trainingvar,weivar,model,signal="H+c"):
    # get BDT score
    bdt={}
    for s in MCvar[list(MCvar.keys())[0]].keys():
        x= np.vstack([MCvar[var][s] for var in trainingvar]).T
        dmatrix = xgb.DMatrix(x)
        bdt[s]=xgb_model.predict(dmatrix)
    hist={}
    for s in bdt.keys():
        hist[s],egdges=np.histogram(bdt[s],bins=100,range=(np.amin(bdt[s]),np.amax(bdt[s])),weights=weivar[s])
        hist[s][hist[s]<0]=0
    pr={}
    sum_hist,pr_sum=np.zeros(100),np.zeros(101)
    for b in hist.keys():
        sum_hist=sum_hist+hist[b]
        pr[b]=[]
        for i in range(101):
            pr[b].append(np.sum(hist[b][:i])/np.sum(hist[b]))
    for i in range(101):pr_sum[i]=(np.sum(sum_hist[:i])/np.sum(sum_hist))
    return pr,pr_sum
    
    