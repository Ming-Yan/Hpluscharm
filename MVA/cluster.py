import argparse,warnings,hist,uproot
import mplhep as hep
import xgboost as xgb
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
###########user define
from BTVNanoCommissioning.utils.plot_utils import load_coffea
import numpy as np
from BTVNanoCommissioning.helpers.xsection import xsection
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt, mplhep as hep
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnchoredText
from BTVNanoCommissioning.utils.plot_utils import (
    load_coffea,
    plotratio,
)
from training_config import config2017, train_collect

from joblib import dump,load
plt.style.use(hep.style.ROOT)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c","--campaign", default="UL17",choices=["UL17","UL18","UL16_preAPV","UL16_postAPV"], help="campaign")
    parser.add_argument("-v", "--version", type=str, required=True, help="version")
    parser.add_argument("-r","--region", default="SR2_LM",type=str,help="categories")
    parser.add_argument("-n","--ncluster",type=str,default="20,25,30,40,45,50,55",help="nclusters")
    args = parser.parse_args()
    n_clusters_list=args.ncluster.split(",")
    n_clusters_list=[int(i) for i in n_clusters_list]
    data = load_coffea(config2017[args.version]["input"][args.campaign]["data"],False)
    output = load_coffea(config2017[args.version]["input"][args.campaign]["bkg"],False)
    signal = load_coffea(config2017[args.version]["input"][args.campaign]["sig"],False)
    sumw = {}
    scales={}
    if args.campaign=="UL17":lumi=41500
    elif args.campaign=="UL16_preAPV":lumi=19500
    elif args.campaign=="UL16_postAPV":lumi=16800
    elif args.campaign=="UL18":lumi=59800
    color_map={"TT":"#38A6A5","ST":"#73AF48","VV":"#EDAD08","Z+jets":"#554e99","Higgs (WW+ZZ)":"#666666","H+c":"#CC503E",'ggH':'gray'}
    name_map={"TT":'ttbar',"ST":"st","Z+jets":"zjets","VV":'vv',"Higgs (WW+ZZ)":"higgs","H+c":"hc","data":"data_obs"}

    collect_var={}
    varlist=config2017[args.version]["varlist"]+["weight"]+['nselj']
    output[list(signal.keys())[0]]=signal[list(signal.keys())[0]]
    at = AnchoredText("",  frameon=False,loc=2)
    
    for h in data.keys():output[h] = data[h]
    for f in output.keys():
        for s in output[f].keys():   
            
            if s not in sumw.keys():sumw[s]=output[f][s]['sumw']
            else:sumw[s] += output[f][s]['sumw']
            if s not in collect_var.keys():collect_var[s]={}
            for r in output[f][s]['array'].keys():   
                if s not in collect_var.keys() or r not in collect_var[s].keys():collect_var[s][r]={}
                for c in ['emu']:
                    varlist = list(output[f][s]['array'][r][c].keys())
                    for var in varlist:
                        val=output[f][s]['array'][r][c][var].value
                        if 'jetflav2_Cv' in var:val[val==-99]=0.
                        if 'jetflav2_pt' ==var:val[val==-99]=-1.
                        if var not in list(collect_var[s][r].keys()):collect_var[s][r][var]=val
                        else:collect_var[s][r][var]=np.concatenate((collect_var[s][r][var],val))
                
    xs_dict = {}
    for obj in xsection:
        xs_dict[obj["process_name"]] = float(obj["cross_section"])
    for s in sumw.keys():        
        if "MuonEG" in s :continue
        scales[s] = xs_dict[s]*lumi/sumw[s]
        for r in collect_var[s].keys():
            if 'weight' not in collect_var[s][r].keys():continue
            
            collect_var[s][r]['mcwei']=np.full_like(collect_var[s][r]['weight'],scales[s])
    mergemap=config2017[args.version]["mergemap"]
    trainvar=config2017[args.version]["varlist"]
    MCvar={}
    weivar={}
    for var in trainvar :
        MCbkgLM = []
        MCvar[var]={}
        for m in mergemap:
            tmpml,tmpwei=[],[]
            for ml in mergemap[m]:
                if args.campaign!="UL17":
                    if ml not in collect_var.keys():continue
                    if len(collect_var[ml][args.region].keys())==0 :continue
                    tmpml=np.concatenate((tmpml,collect_var[ml][args.region][var])) 
                    tmpwei=np.concatenate((tmpwei,collect_var[ml][args.region]['mcwei']*collect_var[ml][args.region]['weight'])) 
                    
                    
                else:
                    if len(collect_var[ml]['SR_LM'].keys())==0:continue
                    mask=collect_var[ml]['SR_LM']['nselj']==1 if args.region=="SR2_LM" else collect_var[ml]['SR_LM']['nselj']>1
                    
                    tmpml=np.concatenate((tmpml,collect_var[ml]['SR_LM'][var][mask])) 
                    if m!="data":
                        tmpwei=np.concatenate((tmpwei,collect_var[ml]['SR_LM']['mcwei'][mask]*collect_var[ml]['SR_LM']['weight'][mask])) 
                    else:tmpwei=np.ones(len(tmpml))
            MCvar[var][m]=tmpml
            if m!="data":weivar[m]=tmpwei
            MCbkgLM+=[tmpml]
        tmpml,tmpwei=[],[]
        
        MCvar[var]["H+c"]=collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8'][args.region][var]
        weivar["H+c"]=collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8'][args.region]["mcwei"]*collect_var['HPlusCharm_4FS_MuRFScaleDynX0p50_HToWWTo2L2Nu_M125_TuneCP5_13TeV-amcatnloFXFX-pythia8'][args.region]["weight"]
        if args.campaign!="UL17":MCvar[var]["data"]=np.hstack([collect_var[s][args.region][var] for s in collect_var.keys() if len(collect_var[s][args.region])>0 and "MuonEG_Run" in s ])
        else:
            if "SR_LM"==args.region:
                MCvar[var]["data"]=np.hstack([collect_var[s]["SR_LM"][var][collect_var[s]['SR_LM']['nselj']>1] for s in collect_var.keys() if len(collect_var[s]["SR_LM"])>0 and "MuonEG_Run" in s ])
            elif "SR2_LM"==args.region:
                MCvar[var]["data"]=np.hstack([collect_var[s]["SR_LM"][var][collect_var[s]['SR_LM']['nselj']==1] for s in collect_var.keys() if len(collect_var[s]["SR_LM"])>0 and "MuonEG_Run" in s ])
        
    xgb_model_bkg,xgb_model_higgs = xgb.Booster(),xgb.Booster()
    xgb_model_bkg.load_model(f"None_{args.version}_bkg_{args.campaign}_nofocal.json")
    # xgb_model_higgs.load_model(f"None_{args.version}_{args.campaign}_nofocal.json")

    xgb_model_higgs.load_model(f"None_{args.version}_higgs_{args.campaign}_nofocal.json")
    bdt_bkg,bdt_higgs={},{}
    hist_bkg_BDT, hist_higgs_BDT,hist_bkg_BDT_bin, hist_higgs_BDT_bin={},{},{},{}
    for w in weivar.keys():
        if "UL16_postAPV" in args.campaign and w != "Higgs (WW+ZZ)" and w!="H+c":weivar[w]=weivar[w]*0.141
        if "UL16_preAPV"  in args.campaign: weivar[w] = weivar[w]
    for s in sumw.keys():
        if "MuonEG" in s :continue
    
        for r in collect_var[s].keys():
            if "SR_LM"!=r and "SR2_LM" !=r: continue
            if len(collect_var[s][r].keys())==0:continue
            
            # print(r,s,sumw[s],scales[s],lumi,xs_dict[s]),len(collect_var[s][r]['mcwei'])
    for s in MCvar[trainvar[0]].keys():
        x= np.vstack([MCvar[var][s] for var in trainvar]).T
        dmatrix = xgb.DMatrix(x)
        h = hist.Hist(hist.axis.Regular(40,0,1., name="discr", label="BDT"),hist.storage.Weight())
        hh = hist.Hist(hist.axis.Regular(40,0,1., name="discr", label="BDT"),hist.storage.Weight())
        bdt_bkg[s]=xgb_model_bkg.predict(dmatrix)
        bdt_higgs[s]=xgb_model_higgs.predict(dmatrix)
        weight=weivar[s] if s!="data" else np.ones_like(bdt_bkg[s])
        h.fill(bdt_bkg[s],weight=weight)
        hh.fill(bdt_higgs[s],weight=weight)
        hist_bkg_BDT[s],hist_higgs_BDT[s]=h,hh
    fig, ((ax), (rax)) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True
            )
    hep.cms.label(
            "Private Work",
            data=True,
            lumi=lumi/ 1000.0,
            com="13",
            loc=0,
            ax=ax,
        )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    # scale=0.917
    hep.histplot([hist_bkg_BDT[s] for s in weivar.keys()],label=[s for s in weivar.keys()],stack=True,histtype='fill',yerr=True,ax=ax,color=[color_map[s]for s in weivar.keys()])
    hep.histplot(hist_bkg_BDT["H+c"]*10000,label='H+c x10000',lw=3,ax=ax,histtype='step',color=color_map["H+c"])
    hep.histplot(hist_bkg_BDT["Higgs (WW+ZZ)"]*100,label='Higgs x100',lw=2,ls=":",ax=ax,histtype='step',color=color_map["Higgs (WW+ZZ)"])
    hmc = hist.Hist(hist.axis.Regular(40,0,1., name="discr", label="BDT"),hist.storage.Weight())
    for s in weivar.keys():
        hmc=hmc+hist_bkg_BDT[s]
    scale = np.sum(hist_bkg_BDT["data"].values())/np.sum(hmc.values())
    print(np.sum(hist_bkg_BDT["data"].values()),scale)
    data = hist_bkg_BDT["data"].values()
    ax.set_ylim(0,np.amax(data)*1.2)
    data[-10:] = np.nan
    hep.histplot(data,hist_bkg_BDT["Higgs (WW+ZZ)"].axes[0].edges,histtype='errorbar',color="k",label="data",ax=ax)    
    
    
    
    ax.legend(ncols=2)
    
    
    plotratio(data,hmc,ax=rax,data_is_np=True)
    
    
    
    rax.axhline(y=scale)
    
    rax.set_ylim(0.5,1.5)
    ax.set_xlabel(None)
    rax.set_ylabel("data/MC")
    ax.set_ylabel("Events")
    rax.set_xlabel("BDT (H+c, Bkg)")
    
    
    at = AnchoredText("",  frameon=False,loc=2)
    ax.add_artist(at)
    hep.mpl_magic(ax=ax)
    
    fig.savefig(f"BDT_bkg_{args.version}_{args.region}_{args.campaign}.pdf")
    fig, ((ax), (rax)) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True
            )
    hep.cms.label(
            "Private Work",
            data=True,
            lumi=lumi/ 1000.0,
            com="13",
            loc=0,
            ax=ax,
        )
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    hep.histplot([hist_higgs_BDT[s] for s in weivar.keys()],label=[s for s in weivar.keys()],stack=True,histtype='fill',yerr=True,ax=ax,color=[color_map[s] for s in weivar.keys()])
    hep.histplot(hist_higgs_BDT["H+c"]*10000,label='H+c x10000',lw=3,ax=ax,histtype='step',color=color_map["H+c"])
    hep.histplot(hist_higgs_BDT["Higgs (WW+ZZ)"]*100,label='Higgs x100',lw=2,ls=":",ax=ax,histtype='step',color=color_map["Higgs (WW+ZZ)"])
    data = hist_higgs_BDT["data"].values()
    data[-10:] = np.nan
    hep.histplot(data,hist_higgs_BDT["Higgs (WW+ZZ)"].axes[0].edges,histtype='errorbar',color="k",label="data",ax=ax)
    
    ax.legend(ncols=2)
    hmc = hist.Hist(hist.axis.Regular(40,0,1., name="discr", label="BDT"),hist.storage.Weight())
    for s in weivar.keys():hmc=hmc+hist_higgs_BDT[s]
    
    plotratio(data,hmc,ax=rax,data_is_np=True)
    rax.axhline(scale)
    rax.set_ylim(0.5,1.5)
    ax.set_xlabel(None)
    rax.set_ylabel("data/MC")
    ax.set_ylabel("Events")
    rax.set_xlabel("BDT (H+c, Higgs)")

    at = AnchoredText("",  frameon=False,loc=2)
    ax.add_artist(at)
    hep.mpl_magic(ax=ax)
    
    fig.savefig(f"BDT_higgs_{args.region}_{args.campaign}.pdf")
    
   
    # x = np.vstack([np.hstack([bdt_bkg[s] for s in bdt_bkg.keys() if s!="data"]),np.hstack([bdt_higgs[s] for s in bdt_higgs.keys()if s!="data"])]).T
    # y=np.array([])
    # for i,s in enumerate(weivar.keys()):
    #     if s=='H+c':y=np.hstack([y,np.ones(weivar[s].shape[0],dtype=int)*1])
    #     elif s!="data":y=np.hstack([y,np.zeros(weivar[s].shape[0],dtype=int)*0])
    # w = np.hstack([weivar[s] for s in weivar.keys()])
    # w[y==0]=np.sum(w[y==1])/np.sum(w[y==0])*w[y==0]
    # w=w/np.amin(abs(w))
    # X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    #         x, y, w, test_size=0.55, random_state=7
    #     )
    
    # clustering_algorithms = {
    # "K-Means": KMeans,
    # }


    # random_state=0
    # for j, n_clusters in enumerate(n_clusters_list):
    #     fig, axs = plt.subplots(1, 2, figsize=(12,6),sharex=True,sharey=True)
    #     hep.cms.label(
    #         "Private Work",
    #         data=True,
    #         lumi=lumi/ 1000.0,
    #         com="13",
    #         loc=0,
    #         ax=ax,
    #     )
    #     algo = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    #     algo.fit(X_train,y_train,sample_weight=w_train)
    #     test=algo.predict(X_test,sample_weight=w_test)
    #     centers = algo.cluster_centers_

    #     axs[0].scatter(X_train[:, 0], X_train[:, 1], s=10, c=algo.labels_)
    #     axs[1].scatter(X_test[:, 0], X_test[:, 1], s=10, c=test)
    #     axs[1].scatter(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], s=10, c='tab:orange',alpha=0.3)
    #     for i in range(len(centers)):
    #         axs[0].text(centers[i, 0], centers[i, 1],str(i), c="r",fontsize=15)
    #     axs[0].set_title(f"train: {n_clusters} clusters")
    #     axs[1].set_title(f"test: {n_clusters} clusters")
    #     dump(algo, f'kmeans_model_{args.campaign}_{n_clusters}_{args.region}.joblib')
    #     axs[0].set_xticks([])
    #     axs[0].set_yticks([])
    #     axs[1].set_xticks([])
    #     axs[1].set_yticks([])
    #     axs[0].set_ylabel("BDT(H+c,Higgs)")
    #     axs[0].set_xlabel("BDT(H+c,Other bkg($t\\bar{t}$))")
    #     axs[1].set_ylabel("BDT(H+c,Higgs)")
    #     axs[1].set_xlabel("BDT(H+c,Other bkg($t\\bar{t}$))")
        
    #     fig.savefig(f"{args.region}_{args.campaign}_{n_clusters}cluster.png",dpi=100)
    #     plt.cla()

        
    
        
    #     clus_id,weight_sum,fs={},{},{}
    #     weight_bkg_sum=np.zeros(n_clusters)
    #     for s in bdt_bkg.keys():
    #         X_all=np.vstack([np.array(bdt_bkg[s]),np.array(bdt_higgs[s])]).T
    #         clus_id[s]=algo.predict(X_all)
    #     for c in weivar.keys():
    #         if c=="H+c":weight_sig=np.array([np.sum(np.ones(len(clus_id["H+c"][clus_id["H+c"]==i]),dtype='float64')*weivar["H+c"][clus_id["H+c"]==i]) for i in range(n_clusters)])
    #         else:weight_bkg_sum=np.array([np.sum(np.ones(len(clus_id[c][clus_id[c]==i]),dtype='float64')*weivar[c][clus_id[c]==i]) for i in range(n_clusters)])+weight_bkg_sum
    #     index_map=list(np.argsort(weight_sig/np.sqrt(weight_bkg_sum)))
    #     f = uproot.recreate(f"/nfs/dust/cms/user/milee/card_maker/shape/{args.campaign}_{args.region}_template_bin{n_clusters}_{args.region}.root")
    #     for s in clus_id.keys():
    #         weight_sum[s]=np.zeros(n_clusters)
            
    #         hist_index=list(map(lambda x:index_map.index(x),clus_id[s]))
    #         h=hist.Hist(hist.axis.Integer(0,n_clusters,name="bin",label='bin numbers'),hist.storage.Weight())
    #         weight=1.
    #         if s!='data':weight=weivar[s]
    #         h.fill(list(map(lambda x:index_map.index(x),clus_id[s])),weight=weight)
    #         for i in range(n_clusters):
    #             if s!="data":weight=weivar[s][clus_id[s]==index_map[i]]
    #             else:weight=1.
                
    #             weight_sum[s][i]=np.sum(np.ones(len(clus_id[s][clus_id[s]==index_map[i]]),dtype='float64')*weight)
            
    #         f[name_map[s]]=h
    #         fs[s]=h
    #     fig, ((ax), (rax)) = plt.subplots(
    #             2, 1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True,figsize=(3*int(n_clusters/5),8)
    #         )
    #     fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    #     hep.cms.label(
    #         "Private Work",
    #         data=True,
    #         lumi=lumi/ 1000.0,
    #         com="13",
    #         loc=0,
    #         ax=ax,
    #     )
    #     hep.histplot([fs[s] for s in fs.keys() if 'data' not in s],stack=True,ax=ax,histtype='fill',color=[color_map[s]for s in weight_sum.keys() if 'data' not in s],label=[s for s in weight_sum.keys() if s!="data"])

    #     hep.histplot(fs['Higgs (WW+ZZ)']*100,histtype='step',label="Higgs$\\times$100",ls=":",lw=2,color=color_map["Higgs (WW+ZZ)"],ax=ax)
    #     hep.histplot(fs['H+c']*10000,histtype='step',label="H+c$\\times$10000",lw=3,color=color_map["H+c"],ax=ax)
        
        
    #     allMC,mcunc =np.zeros_like(weight_sum['data']),np.zeros_like(weight_sum['data'])
    #     for s in fs.keys():
    #         if s!="data" : 
    #             allMC =fs[s].values()+allMC
    #             mcunc= fs[s].variances()+mcunc
    #     ax.set_ylim(0,max(np.amax(fs['Higgs (WW+ZZ)'].values()*100),np.amax(weight_sum['data']))*1.3)
    #     weight_sum['data'][-int(n_clusters*0.25):]=np.nan
    #     hep.histplot(weight_sum['data'],np.arange(0,n_clusters+1,1),yerr=True,histtype='errorbar',color='k',ax=ax,label='data')
    #     hep.histplot(weight_sum['data']/allMC,np.arange(0,n_clusters+1,1),ax=rax,histtype='errorbar',color='k',yerr=np.sqrt((np.sqrt(weight_sum['data'])/weight_sum['data'])**2+(np.sqrt(mcunc)/allMC)**2))

    #     ax.legend(ncols=3,loc='upper right',fontsize=20)
    #     rax.set_xlim(0,n_clusters)
    #     rax.set_ylim(0.5,1.5)
    #     ax.set_xlabel(None)
    #     # rax.set_xticklabels(None)
    #     rax.set_xticklabels('')
    #     rax.set_ylabel("data/MC")
    #     rax.set_xticks(np.arange(0,n_clusters,1)+0.5,index_map)
    #     rax.minorticks_off()


    #     # ax.set_title(f"KMeans: {n_clusters} cluster")
    #     ax.set_ylabel("Events")
    #     rax.set_xlabel("Bin number")
    #     fig.savefig(f"{args.campaign}_kmeans_{n_clusters}_{args.region}.pdf")
    #     plt.cla()
            

    
