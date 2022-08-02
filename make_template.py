import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import uproot3
import json
# from coffea import hist
import hist 
from coffea.util import load
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.offsetbox import AnchoredText

import importlib.resources

from BTVNanoCommissioning.utils.xs_scaler import scale_xs


with open("metadata/mergemap.json") as json_file:
    merge_map = json.load(json_file)

data_err_opts = {
    "linestyle": "none",
    "marker": ".",
    "markersize": 10.0,
    "color": "k",
    "elinewidth": 1,
}
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
        type=int,
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
    if args.year == 2016:
        lumis = 36100
    elif args.year == 2017:
        lumis = 41500
    elif args.year == 2018:
        lumis = 59800
    dyjet = [
        "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DY1JetsToLL_M-10to50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DY2JetsToLL_M-10to50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DY3JetsToLL_M-10to50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DY4JetsToLL_M-10to50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DY1JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DY2JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DY3JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DY4JetsToLL_M-50_MatchEWPDG20_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_Pt-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_Pt-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_Pt-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_Pt-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_Pt-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    ]
    dyscaler = {}
    for dy in dyjet:
        if "M-10to50" in dy:
            dyscaler[dy] = 1.0 / 2.0
        else:
            dyscaler[dy] = 1.0 / 3.0
    # print(dyscaler)
    for out in output.keys():
        ## Scale XS
        if out == "data":
                output[out] = collate(output[out],merge_map["data"])
        else:
                output[out] = scaleSumW(output[out],lumis,getSumw(output[out]),"../metadata/xsection.json")
                output[out] = collate(output[out],merge_map["HWW_template_moredy"])
        

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

            fout = uproot3.create(template_file)
            for syst in ['nominal', 'aS_weightUp', 'UEPS_ISRDown', 'cjetSFsUp', 'L1prefireweightDown', 'PDF_weightDown', 'scalevar_7ptUp', 'PDFaS_weightDown', 'UEPS_FSRUp', 'scalevar_7ptDown', 'scalevar_3ptDown', 'eleSFsDown', 'L1prefireweightUp', 'aS_weightDown', 'scalevar_3ptUp', 'muSFsUp', 'UEPS_FSRDown', 'PDF_weightUp', 'cjetSFsDown', 'muSFsDown', 'puweightDown', 'eleSFsUp', 'puweightUp', 'UEPS_ISRUp', 'PDFaS_weightUp', 'JESUp', 'JESDown', 'UESUp', 'UESDown', 'JERUp', 'JERDown']:
                if syst=='nominal':s=''
                else : 
                    s=syst+'_'
                    if not args.systs:break
                name = s+"hc"
                fout[name] = output['signal'][var][{'lepflav':flav,'region':region,'flav':sum,'syst':syst}].project(output['signal'][var].axes[-1])
                if syst == 'nominal:'
                    name = "data_obs"
                    fout[name] =  output[f'data_{flav}'][var][{'lepflav':flav,'region':region,'flav':sum,'syst':syst}].project(output[f'data_{flav}'][var].axes[-1])
                
                for proc in output.keys():
                    name = s+str(proc)
                    fout[name] = output[proc][var][{'lepflav':flav,'region':region,'flav':sum}].project(output[proc][var].axes[-1])
            
                
            fig, ((ax), (rax)) = plt.subplots(
                2,1,
                figsize=(12, 12),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True,
            )
            if args.valid:
                fig.subplots_adjust(hspace=0.07)
                hbkglist = [
                        output[sample][var][{'lepflav':flav,'region':region,'flav':sum}].project(output[sample][var].axes[-1])
                        for sample in output.keys()
                    ]
                
                ax = hep.histplot(
                    hbkglist,
                    stack=True,
                    histtype="fill",
                    ax=ax,
                )
                hep.histplot(
                    output[f'data_{flav}'][var][{'lepflav':flav,'region':region,'flav':sum}].project(output[f'data_{flav}'][var].axes[-1]),
                    histtype="errorbar",
                    color="black",
                    label=f"Data",
                    yerr=True,
                    ax=ax,
                )
                
                
                at = AnchoredText(
                    flav
                    + "  "
                    + region_map[region]
                    + "\n"
                    + r"HWW$\rightarrow 2\ell 2\nu$",
                    loc="upper left",
                    frameon=False,
                )
                ax.add_artist(at)
                # ax.set_ylim(bottom=0.0001)
                # ax.semilogy()
                for sample in output.keys():
                    if 'data' not in sample: hmc=output[sample][var][{'lepflav':chs,'region':region,'flav':sum}].project(output[sample][var].axes[-1]) +hmc 
                rax = hist.plotratio(
                    num=output[f'data_{flav}'][var][{'lepflav':flav,'region':region,'flav':sum}].project(output[f'data_{flav}'][var].axes[-1]),
                    denom=hmc,
                    ax=rax,
                    error_opts=data_err_opts,
                    denom_fill_opts={},
                    #
                    unc="num",
                )
                rax.set_ylim(0.5, 1.5)
                hep.mpl_magic(ax=ax)
                fig.savefig(f"validate_{region}_{flav}_{args.observable}{args.prefix}.pdf")
    fout.close()
