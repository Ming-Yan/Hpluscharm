from collections import defaultdict
import os,sys,json, argparse
from this import d
from coffea.util import load
import numpy as np
import matplotlib.pyplot as plt
from utils.xs_scaler import scale_xs_arr
import mplhep as hep
import xgboost as xgb
from coffea import hist
from coffea.hist import plot
import mplhep as hep
from cycler import cycler
import matplotlib as mpl
import pandas as pd
from matplotlib.offsetbox import AnchoredText
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import uproot3
plt.style.use(hep.style.ROOT)


data_err_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.,
    "color": "k",
    "elinewidth": 1,
}
from cycler import cycler
import matplotlib as mpl

colors = ["#666666","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#554e99","#6f4e99","#854e99","#994e85","#666666"]

mpl.rcParams["axes.prop_cycle"] = cycler("color", colors)

with open("metadata/mergemap.json") as json_file:  
    merge_map = json.load(json_file)

data_err_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.,
    "color": "k",
    "elinewidth": 1,
}
colors = ["#666666","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#554e99","#6f4e99","#854e99","#994e85","#666666"]
mpl.rcParams["axes.prop_cycle"] = cycler("color", colors)

plt.style.use(hep.style.ROOT)
fig, ((ax),(rax)) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
hep.cms.label("Work in progress", data=True, lumi=41.5, year=2017,ax=ax)
plt.style.use(hep.style.ROOT)

with open("metadata/mergemap.json") as json_file:  
    merge_map = json.load(json_file)
def load_dataplot(data, varlist,region,channel,isData=False,lumi=41500):
    # Read data from ROOT files
   
    events=data["sumw"]
    data_arrays=data["array"]
    
    # put gen weight
    genwei = scale_xs_arr(events,lumi)
    
    # Convert inputs to format readable by machine learning tools    
    x = np.vstack([np.hstack([data_arrays[dataset][var].value[(data_arrays[dataset]["region"].value==region)&(data_arrays[dataset]["lepflav"].value==channel)] for dataset in data_arrays.keys()]) for var in varlist]).T    
    datasets=np.hstack([np.full_like(data_arrays[dataset]["weight"].value[(data_arrays[dataset]["region"].value==region)&(data_arrays[dataset]["lepflav"].value==channel)],dataset,dtype="U128") for dataset in data_arrays.keys()])
    jetflav = np.hstack([data_arrays[dataset]["jetflav"].value[(data_arrays[dataset]["region"].value==region)&(data_arrays[dataset]["lepflav"].value==channel)]for dataset in data_arrays.keys()])
    if isData:w=np.ones(x.shape[0],dtype=float)
    else:w = np.hstack([data_arrays[dataset]["weight"].value[(data_arrays[dataset]["region"].value==region)&(data_arrays[dataset]["lepflav"].value==channel)]*genwei[dataset] for dataset in data_arrays.keys()])
    return x,w,datasets,jetflav
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ch","--channel",required=True,choices=["ee","mumu","emu"],help="SF/DF")
    parser.add_argument("-r","--region",required=True,choices=["SR","SR2"],help="regions")
    parser.add_argument("--year",default=2017,help="year")
    parser.add_argument("-o","--dirname",default="HWW2l2nu_0521",help="outputdir")
    parser.add_argument("--bin", type=int, default=50,help="bin size")
    parser.add_argument("--blind",type=float,required=True,help="blind bins")
    parser.add_argument("-v","--version",type=str,required=True,help="version")
    args = parser.parse_args()
    
    from training_config import config2017
    
    if args.channel=="emu":chs="emu"
    else:chs="ll"
    varlist = config2017['varlist'][chs][args.version]
   
    bkgoutput = load(config2017['coffea']['bkg'])
    sigoutput = load(config2017['coffea']['sig'])
    dataoutput = load(config2017['coffea']['data'])
    dyoutput= load(config2017['coffea']['dy'])
    bkgx, bkgw,bkgdataset,bkgjetflav = load_dataplot(bkgoutput,varlist,args.region,args.channel,False,41500)
    sigx, sigw,sigdataset,sigjetflav = load_dataplot(sigoutput,varlist,args.region,args.channel,False,41500)
    datax, dataw,datadataset,dataflav = load_dataplot(dataoutput,varlist,args.region,args.channel,True,41500)
    dyx, dyw, dydataset,dyjetflav = load_dataplot(dyoutput,varlist,args.region,args.channel,False,41500)
    # print(np.shape(bkgx),np.shape(bkgw))
    # bkgall= np.vstack([bkgx.T,bkgw,np.full_like(bkgw,"bkgMC",dtype='U64')])
    # sigall= np.vstack([sigx.T,sigw,np.full_like(sigw,"sigMC",dtype='U64')])
    # dataall= np.vstack([datax.T,dataw,np.full_like(dataw,"data",dtype='U64')])
    # print(np.shape(np.hstack([bkgall,sigall,dataall])))
    # df = pd.DataFrame(np.hstack([bkgall,sigall]).T)
    # colname= varlist
    # colname.append('weight')
    # colname.append('type')
    # df.columns= colname
    # print(colname,len(colname))
    # df.set_axis(colname, axis='columns')
    # df.columns = colname
    # print(df)
    # df.to_csv(f"{args.channel}_{args.year}.csv")
    # print(np.shape(bkgall))
    # dyx, dyw,dydataset,dyflav = load_dataplot(dyoutput,varlist,args.region,args.channel,False,41500)
    
    if 'focal' in args.version:xgb_model = xgb.Booster()
    else :xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f"{config2017['input_json'][args.version][chs]}.json")

    dataset_axis = hist.Cat("dataset", "Primary dataset")
    flav_axis = hist.Bin("flav", r"Genflavour",[0,1,4,5,6])
    lepflav_axis = hist.Cat("lepflav",["ee","mumu","emu"])
    
    # maxi = xgb_model.predict_proba(datax)[:,1].

    
    dsig = xgb.DMatrix(sigx)
    dbkg = xgb.DMatrix(bkgx)
    ddata = xgb.DMatrix(datax)
    if 'focal' in args.version:maxi = np.around(max(max(np.max(1./(1+np.exp(-xgb_model.predict(ddata)))),np.max(1./(1+np.exp(-xgb_model.predict(dsig))))),np.max(1./(1+np.exp(-xgb_model.predict(dbkg))))),1)
    else :maxi = np.around(max(max(np.max(xgb_model.predict_proba(datax)[:,1]),np.max(xgb_model.predict_proba(sigx)[:,1])),np.max(xgb_model.predict_proba(bkgx)[:,1])),1)
   
    bdt_axis = hist.Bin("bdt",r"SR BDT", args.bin,0,maxi)
    # llmass_axis = hist.Bin("ll_mass",r"m_{\ell\ell}",)
    histo = hist.Hist("Counts", dataset_axis, lepflav_axis, flav_axis,bdt_axis)
    scales=50000
    for dataset in bkgoutput["array"].keys():
        if 'focal' in args.version:
            dbkg = xgb.DMatrix(bkgx[bkgdataset==dataset])
            if len(bkgjetflav[bkgdataset==dataset])>0 :histo.fill(dataset=dataset,lepflav=args.channel,flav=bkgjetflav[bkgdataset==dataset],bdt=1./(1+np.exp(-xgb_model.predict(dbkg))),weight=bkgw[bkgdataset==dataset])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
        else:
            if len(bkgjetflav[bkgdataset==dataset])>0 :histo.fill(dataset=dataset,lepflav=args.channel,flav=bkgjetflav[bkgdataset==dataset],bdt=xgb_model.predict_proba(bkgx[bkgdataset==dataset])[:,1],weight=bkgw[bkgdataset==dataset])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
    for dataset in sigoutput["array"].keys():
        if 'focal' in args.version:
            
            dsig = xgb.DMatrix(sigx[sigdataset==dataset])
            if len(sigjetflav[sigdataset==dataset])>0:histo.fill(dataset=dataset,lepflav=args.channel,flav=sigjetflav[sigdataset==dataset],bdt=1./(1+np.exp(-xgb_model.predict(dsig))),weight=scales*sigw[sigdataset==dataset])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
        else:
            if len(sigjetflav[sigdataset==dataset])>0:histo.fill(dataset=dataset,lepflav=args.channel,flav=sigjetflav[sigdataset==dataset],bdt=xgb_model.predict_proba(sigx[sigdataset==dataset])[:,1],weight=scales*sigw[sigdataset==dataset])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
    for dataset in dyoutput["array"].keys():
        if 'focal' in args.version:
            
            ddy = xgb.DMatrix(dyx[dydataset==dataset])
            if len(dyjetflav[dydataset==dataset])>0:histo.fill(dataset=dataset,lepflav=args.channel,flav=dyjetflav[dydataset==dataset],bdt=1./(1+np.exp(-xgb_model.predict(ddy))),weight=dyw[dydataset==dataset])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
        else:
            if len(dyjetflav[dydataset==dataset])>0:histo.fill(dataset=dataset,lepflav=args.channel,flav=dyjetflav[dydataset==dataset],bdt=xgb_model.predict_proba(dyx[dydataset==dataset])[:,1],weight=dyw[dydataset==dataset])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
    for dataset in dataoutput["array"].keys():
        if 'focal' in args.version:
            ddata = xgb.DMatrix(datax[datadataset==dataset])
            if (len(dataw[datadataset==dataset])>0):
                histo.fill(dataset=dataset,lepflav=args.channel,flav=5,bdt=np.where(1./(1+np.exp(-xgb_model.predict(ddata)))<args.blind,1./(1+np.exp(-xgb_model.predict(ddata))),0))
                #histo.fill(dataset=dataset,lepflav=args.channel,flav=5,bdt=xgb_model.predict_proba(datax[datadataset==dataset])[:,1])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
        else:
            if (len(dataw[datadataset==dataset])>0):
                histo.fill(dataset=dataset,lepflav=args.channel,flav=5,bdt=np.where(xgb_model.predict_proba(datax[datadataset==dataset])[:,1]<args.blind,xgb_model.predict_proba(datax[datadataset==dataset])[:,1],0))
                # histo.fill(dataset=dataset,lepflav=args.channel,flav=5,bdt=xgb_model.predict_proba(datax[datadataset==dataset])[:,1])
            else:histo.fill(dataset=dataset,lepflav=args.channel,flav=0,bdt=-1,weight=0.)
    histo = histo.group("dataset",hist.Cat("plotgroup", "plotgroup"),merge_map["hww"])
    # histo = histo.group("dataset",hist.Cat("plotgroup", "plotgroup"),merge_map["hww_tem_zptbin"])

    fig.subplots_adjust(hspace=.07) 
    ax = plot.plot1d(histo.sum("flav").integrate("lepflav",args.channel),overlay="plotgroup",stack=True,order=["Z+jets","W+jets","tt-dilep","tt-semilep","ST","WW","WZ","ZZ"],ax=ax)
    plot.plot1d(histo.integrate("plotgroup","Z+jets").integrate("lepflav",args.channel),overlay="flav",stack=True,ax=ax,clear=False)
    plot.plot1d(histo.sum("flav").integrate("lepflav",args.channel).integrate("plotgroup","signal"),clear=False,ax=ax)
    plot.plot1d(histo.integrate("lepflav",args.channel).integrate("plotgroup","data_%s"%(args.channel)).sum("flav"),clear=False,error_opts=data_err_opts,ax=ax)
    
    ax.set_ylim(bottom=0.1)
    ax.semilogy()
    rax = plot.plotratio(num=histo.integrate("lepflav",args.channel).integrate("plotgroup","data_%s"%(args.channel)).sum("flav"),
                         denom=histo.sum("flav").integrate("lepflav",args.channel).integrate("plotgroup",["Z+jets","W+jets","tt-dilep","tt-semilep","ST","WW","WZ","ZZ"]),
                         ax=rax,
                         error_opts=data_err_opts,
                         denom_fill_opts={},
                         unc="num",
                         )
                  
                 
    rax.set_ylim(0.5,1.5)
    rax.set_ylabel("Data/Background")
    rax.set_xlabel("SR BDT")
    chl = args.channel
    if args.channel =="mumu" :chs="$\mu\mu$"
    elif args.channel =="emu" :chs="e$\mu$"
    else :chs= "ee"
    at = AnchoredText(chs+"  "+ args.region+"\n" +r"HWW$\rightarrow 2\ell 2\nu$" , loc="upper left",frameon=False)
    ax.add_artist(at)
    leg_label = ax.get_legend_handles_labels()[1][1:]
    leg_label[-6]="Z+l"
    leg_label[-5]="Z+pu"
    leg_label[-4]="Z+c"
    leg_label[-3]="Z+b"
    leg_label[-1]="data"
    leg_label[-2]=f"Signalx{scales}"
    ax.legend(loc="upper right",handles=ax.get_legend_handles_labels()[0][1:],ncol=2,labels=leg_label)
    hep.mpl_magic(ax= ax)
    ax.set_xlabel("")
    fig.savefig(f"{args.dirname}_{args.region}_{args.channel}_BDT_split_{args.version}_dy.pdf" )
    # fig.savefig(f"{args.dirname}_{args.region}_{args.channel}_BDT_split_{args.version}.png" )
    
    # template_file = f"../stat_analysis/shape/templates_{args.region}_{args.channel}_srbdt_{args.year}_{args.version}.root"
    # if os.path.exists(template_file):
    #     os.remove(template_file)
    # print(f"Will save templates to {template_file}")
    # fout = uproot3.create(template_file)
    # name = "hc" 
    # histo.scale({"signal":1./scales},axis="plotgroup")
  
    
    # fout[name] = hist.export1d(histo.sum("flav").integrate("lepflav",args.channel).integrate("plotgroup","signal"))
    # name = "data_obs"
    # fout[name] = hist.export1d(histo.integrate("lepflav",args.channel).integrate("plotgroup","data_%s"%(args.channel)).sum("flav"))
    # name = "vjets"
    # fout[name] = hist.export1d(histo.sum("flav").integrate("lepflav",args.channel).integrate("plotgroup",["Z+jets","W+jets"]))
    # name = "ttbar"
    # fout[name] = hist.export1d(histo.sum("flav").integrate("lepflav",args.channel).integrate("plotgroup",["tt-dilep","tt-semilep"]))
    # name = "vv"
    # fout[name] = hist.export1d(histo.sum("flav").integrate("lepflav",args.channel).integrate("plotgroup",["WW","WZ","ZZ"]))
    # name = "st"
    # fout[name] = hist.export1d(histo.sum("flav").integrate("lepflav",args.channel).integrate("plotgroup","ST"))
    # fout.close()
