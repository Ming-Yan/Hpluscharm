import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import uproot
import json
import hist 
from coffea.util import load
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.offsetbox import AnchoredText
import numpy as np
import importlib.resources
from tqdm.auto import tqdm
from BTVNanoCommissioning.utils.xs_scaler import getSumW,collate,scaleSumW,additional_scale
plt.style.use(hep.style.ROOT)
from dylist import dylist
with open("metadata/mergemap.json") as json_file:
    merge_map = json.load(json_file)

correlation_map = ["JERUp","JESUp","UESUp","eleSFsUp","muSFsUp","puweightUp","cjetSFsUp","JERDown","JESDown","UESDown","eleSFsDown","muSFsDown","puweightDown","cjetSFsDown"]
region_map = {
    "SR": "SR $N_j>$1",
    "DY_CRb": "DY+b CR",
    "DY_CRl": "DY+l CR",
    "DY_CRc": "DY+c CR",
    "top_CR": "top CR",
    "DY_CR": "DY CR",
    "SR2": "SR $N_j$==1",
}
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-wf",
        "--workflow",
        default=r"HWW2l2nu",
        help="File identifier to carry through (default: %(default)s)"
        ,
    )
    parser.add_argument(
        "--year",
        default=2017,
        type=str,
        required=True,
        help="Scale by appropriate lumi",
    )
    parser.add_argument(
        "--systs", action="store_true", default=False, help="Process systematics"
    )
    parser.add_argument(
        "--valid", action="store_true", default=False, help="add"
    )
    parser.add_argument(
        "--flav",
        default="ee",
        choices=["ee", "mumu", "emu", "all"],
        type=str,
        required=True,
        help="flavor",
    )
    parser.add_argument(
        "--region",
        default="SR2",
        choices=["SR", "SR2", "top_CR", "DY_CR", "all"],
        type=str,
        required=True,
        help="Which region in templates",
    )
    parser.add_argument(
        "-obs",
        "--observable",
        type=str,
        default="ll_mass",
        help="observable to the fit",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="input.json",
        help="Input files",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="version",
    )
    parser.add_argument("--axis",type=str,help="axis name")
    parser.add_argument("--prefix", type=str, default="", help="prefix of output")
    # parser.add_argument("--xs_path",type=str,default="../../metadata/xsection.json",help="xsection json file path")
    args = parser.parse_args()
    print("Running with the following options:")
    print(args)
    with open(f"metadata/{args.input}") as inputs:
        input_map = json.load(inputs)
    output = {
        i: load(f"../{input_map[args.version][i]}")
        for i in input_map[args.version].keys()
    }
    if args.year == "2016":
        lumis = 36100
    elif args.year == "2017":
        lumis = 41500
    elif args.year == "2018":
        lumis = 59800
    
    
    
    for out in output.keys():
        output[out] = scaleSumW(output[out],lumis,getSumW(output[out]))
        output[out] = additional_scale(output[out],0.5,dylist)
    
    
    
    collated = collate(output,merge_map["HWW_template_moredy"])    
        
    regions = ["SR", "SR2", "top_CR", "DY_CR"]
    flavs = ["ee", "mumu", "emu"]
    for region in regions:
        if args.region != "all" and args.region != region:
            continue

        for flav in ["ee", "mumu", "emu"]:
            if args.flav != "all" and args.flav != flav:
                continue
            template_file = f"../../../card_maker/shape/templates_{region}_{flav}_{args.observable}_{args.year}{args.prefix}.root"
            if os.path.exists(template_file):
                os.remove(template_file)
            print(f"Will save templates to {template_file}")

            #fout = uproot.create(template_file)
            for syst in  tqdm(collated['hc'][args.observable].axes['syst'], desc='syst', leave=False):
            # for syst in ["nominal"]:
                if not args.systs :break
                if 'syst' in collated[f'data_{flav}'][args.observable].axes.name:hist_axes = {'lepflav':flav,'region':region,'flav':sum,'syst':syst}
                else :hist_axes = {'lepflav':flav,'region':region}
                if syst == 'nominal':
                    name = "data_obs"
                    #fout[name] =  collated[f'data_{flav}'][args.observable][hist_axes]
                
                for proc in collated.keys():
                    if 'data' not in proc:
                        if syst != 'nominal' :
                            if syst in correlation_map: 
                                if 'Up' in syst:name = proc+"_CMS_"+syst[:-2]+"_13TeV_" + args.year+"Up"
                                else:name = proc+"_CMS_"+syst[:-4]+"_13TeV_" + args.year+"Down"
                            else: name = proc+"_CMS_"+syst
                        else : name = proc
                        if proc == "higgs" and args.observable=='template_mT1'  :
                            binning=collated['data_ee'][args.observable][{'lepflav':flav,'region':region,'flav':sum,'syst':'nominal'}].to_numpy()[1]
                            #fout[name] = np.histogram([],binning)
                        
                        # else:#fout[name] = collated[proc][args.observable][hist_axes]
                        print(proc, syst)
                        # if proc == 'vv' and syst not in ['JESDown', 'UESDown', 'JERDown'] and (args.observable=='template_tt_mass' or args.observable=='template_llbb_mass') :
                        
                
                
            
            if args.valid:
                fig, ((ax), (rax)) = plt.subplots(
                2,1,
                figsize=(12, 12),
                gridspec_kw={"height_ratios": (3, 1)},
                    #sharex=True,
             )
                # fig, ax = plt.subplots(figsize=(8, 8))

                fig.subplots_adjust(hspace=0.05)
                hep.cms.label(
                    "Work in progress",
                    data=True,
                    # lumi=lumis / 1000.0,
                    year=args.year,
                    loc=0,
                    ax=ax,
                )

                import boost_histogram as bh
                if 'syst' in collated[f'data_{flav}'][args.observable].axes.name: vhist_axes = {'lepflav':flav,'region':region,'flav':sum,'syst':'nominal'}
                # elif 'npv' in 
                else:  vhist_axes = {'lepflav':flav,'flav':sum,'region':region,'syst':args.prefix} #vhist_axes = {'lepflav':flav,'flav':sum,'region':region}
                # print(collated[f'data_{flav}'])
                # vhist_axes = {'lepflav':flav,'flav':sum,'region':region,'syst':args.prefix}
                hbkglist = [
                    collated[sample][args.observable][vhist_axes]
                        for sample in collated.keys() if 'data' not in sample
                    ]
                # hbkglist = [
                #     values[flav][sample]
                
                #         for sample in collated.keys() if 'data' not in sample
                #     ]

                label = [sample for  sample in collated.keys() if 'data' not in sample]
                
                i=0
                for sample in collated.keys():
                    if 'data' in sample: continue
                    if i==0: hmc = collated[sample][args.observable][vhist_axes]
                    else: hmc = collated[sample][args.observable][vhist_axes] + hmc
                    i = i+1
                from hist.intervals import ratio_uncertainty
                
                hdata = collated[f'data_{flav}'][args.observable][{'lepflav':flav,'flav':sum,'region':region}]
                rax.errorbar(
                x= hdata.axes[0].centers,
                y= hdata.values() / hmc.values(),
                yerr=ratio_uncertainty(hdata.values(), hmc.values()),
                color="k",
                linestyle="none",
                marker="o",
                elinewidth=1,
                )   
                
                rax.axhline(y=1.0, linestyle="dashed", color="gray")

                
                rax.set_ylim(0.5, 1.5)

                hep.histplot(
                    hbkglist,
                     stack=True,
                     histtype="fill",
                    label=["V+jets","ttbar","Single Top", "Diboson","Higgs x500","H+c x50000"],
                    color=["#554e99","#38A6A5","#73AF48","#EDAD08","#CC503E","#666666"],
                    # density=True,
                    # yerr=False,
                    ax=ax,
                )
                #print(args.prefix,np.sum(collated["ttbar"][args.observable][vhist_axes].values()))
                # print(np.sum(collated["ttbar"][args.observable][{'lepflav':flav,'flav':sum,'region':region,'syst':"nominal"}].values()),np.sum(collated["ttbar"][args.observable][{'lepflav':flav,'flav':sum,'region':region,'syst':"puweightUp"}].values()),np.sum(collated["ttbar"][args.observable][{'lepflav':flav,'flav':sum,'region':region,'syst':"puweightDown"}].values()))
                hep.histplot(
                    collated["higgs"][args.observable][vhist_axes]*500,
                    label=[""],
                    yerr=False,
                    color="#CC503E",
                    ax=ax,
                )
                hep.histplot(
                    collated["hc"][args.observable][vhist_axes]*50000,
                    label=[""],
                    yerr=False,
                    color="#666666",
                    ax=ax,
                )
                hep.histplot(
                   
                   hdata,
                   histtype="errorbar",
                   color="black",
                   label=f"Data",
                   yerr=True,
                   ax=ax,
                )
                
                flavs = flav.replace("mu","$\mu$")
                at = AnchoredText(
                    flavs
                    + "  "
                    + region_map[region]
                    + "\n"
                    + r"HWW$\rightarrow 2\ell 2\nu$",
                    loc="upper left",
                    frameon=False,
                    # prop=dict(size=20),
                    
                )
                ax.add_artist(at)
                ax.set_ylim(0.,3000)
                #ax.semilogy()
                rax.set_ylabel("data/MC")
                ax.set_xlabel(args.axis)
                ax.set_ylabel("Counts")
                ax.legend(
                    loc="upper right",
                    # ncol=2,
                    # fontsize=20
                )
                # ax.set_ylim(0.,3000)
                ax.set_ylim(bottom=0.)
                #hep.mpl_magic(ax=ax)
                fig.savefig(f"validate_{region}_{flav}_{args.observable}{args.prefix}.pdf")
    # #fout.close()
