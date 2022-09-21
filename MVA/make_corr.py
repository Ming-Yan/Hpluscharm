from collections import defaultdict
import os, sys, json, argparse
from coffea.util import load
import numpy as np
from BTVNanoCommissioning.utils.xs_scaler import scale_xs_arr
import pandas as pd

# import imgkit


with open("../metadata/mergemap.json") as json_file:
    merge_map = json.load(json_file)


def load_data(fname, varlist, isSignal=True, channel="emu", lumi=41500):
    # Read data from ROOT files
    data = load(fname)
    events = data["sumw"]
    data_arrays = data["array"]

    # put gen weight
    genwei = scale_xs_arr(events, lumi, "../../../metadata/xsection.json")
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ch", "--channel", required=True, choices=["ll", "emu"], help="SF/DF"
    )
    parser.add_argument("-c", "--campaign", default="UL17", help="year")
    parser.add_argument("-v", "--version", type=str, required=True, help="version")
    args = parser.parse_args()
    if "16" in args.campaign:
        year = 2016
        if "UL16" in args.campaign:
            lumis = 36100
        else:
            lumis = 35900
    elif "17" in args.campaign:
        year = 2017
        lumis = 41500
    elif "18" in args.campaign:
        year = 2018
        lumis = 59800
    if year == 2016:
        from training_config import config2016 as config
    if year == 2017:
        from training_config import config2017 as config
    if year == 2018:
        from training_config import config2018 as config

    varlist = config["varlist"][args.channel][args.version]

    sigx, sigy, sigw = load_data(
        config["coffea_new"]["sig"], varlist, True, args.channel
    )
    bkgx, bkgy, bkgw = load_data(
        config["coffea_new"]["bkg"], varlist, False, args.channel
    )
    # bkgall= np.vstack([bkgx.T,bkgw,np.full_like(bkgw,"bkgMC",dtype='U64')])
    # sigall= np.vstack([sigx.T,sigw,np.full_like(sigw,"sigMC",dtype='U64')])
    dfbkg = pd.DataFrame(bkgx)
    dfsig = pd.DataFrame(sigx)
    # df = pd.DataFrame(np.hstack([bkgall,sigall]).T)
    # colname = [plot_map['var_map'][var] for var in varlist]
    dfsig.set_axis(varlist, axis="columns")
    dfsig.columns = varlist
    corrsig = dfsig.corr()

    htmlsig = (
        corrsig.style.background_gradient(cmap="coolwarm").set_precision(3).render()
    )
    corrsig.to_csv(f"corrsig_{args.channel}_{args.campaign}_{args.version}.csv")
    # corrsig.style.background_gradient(cmap='coolwarm').set_precision(3).to_csv("corrsig_{args.channel}_{args.campaign}_{args.version}.csv")
    # imgkit.from_string(htmlsig, f"corrsig_{args.channel}_{args.campaign}_{args.version}.png")
    dfbkg.set_axis(varlist, axis="columns")
    dfbkg.columns = varlist
    corrbkg = dfbkg.corr()
    htmlbkg = (
        corrbkg.style.background_gradient(cmap="coolwarm").set_precision(3).render()
    )
    corrbkg.to_csv(f"corrbkg_{args.channel}_{args.campaign}_{args.version}.csv")
    # corrbkg.style.background_gradient(cmap='coolwarm').set_precision(3).to_csv("corrbkg_{args.channel}_{args.campaign}_{args.version}.csv")
    # imgkit.from_string(htmlbkg, f"corrbkg_{args.channel}_{args.campaign}_{args.version}.png")
