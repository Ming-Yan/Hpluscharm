import os
import sys
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import uproot3
import json
from coffea import hist
from coffea.util import load
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.offsetbox import AnchoredText
sys.path.append('..')
from Hpluscharm_processor.utils.xs_scaler import scale_xs


with open('metadata/mergemap.json') as json_file:  
    merge_map = json.load(json_file)

data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 10.,
    'color': 'k',
    'elinewidth': 1,
}
region_map = {'SR':'SR $N_j>$1','DY_CRb':'DY+b CR','DY_CRl':'DY+l CR','DY_CRc':'DY+c CR','top_CR':'top CR','DY_CR':'DY CR','SR2':'SR $N_j$==1'}
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-wf', '--workflow', default=r'HWW2l2nu', help='File identifier to carry through (default: %(default)s)')
    parser.add_argument("--year", default=2017, type=int, required=True, help="Scale by appropriate lumi")
    parser.add_argument("--systs", action='store_true',default=False, help='Process systematics')
    parser.add_argument("--flav", default='ee', choices=['ee', 'mumu', 'emu','all'], type=str, help="flavor")
    parser.add_argument("--region", default='SR2', choices=['SR','SR2','top_CR', 'DY_CR','all'], type=str, help="Which region in templates")
    parser.add_argument("--input_sig",default='hists_HWW2l2nu_signal_UL17array_4f.coffea',type=str,help="Input signal files")
    parser.add_argument("--input_bkg",default='hists_HWW2l2nu_mcbkg_UL17array.coffea',type=str,help="Input background files")
    parser.add_argument("--input_data",default='hists_HWW2l2nu_data2017array.coffea',type=str,help="Input data files")
    # parser.add_argument("-obs_sig", "--observable_sig",type=str,default='ll_mass',help='observable to the fit')
    # parser.add_argument("-obs_bkg", "--observable_bkg",type=str,default='ll_mass',help='observable to the fit')
    parser.add_argument("-obs", "--observable",type=str,default='ll_mass',help='observable to the fit')
    parser.add_argument("--prefix",type=str,default='',help='prefix of output')

    args = parser.parse_args()
    print("Running with the following options:")
    print(args)

    outputWWl_s = load(args.input_sig)
    outputWWl_b = load(args.input_bkg)
    outputWWl_data = load(args.input_data)
    outputWWl_dy = load('hists_HWW2l2nu_dym10CR_all.coffea')
    outputWWl_dynlo = load('hists_HWW2l2nu_dy_nlCR_all.coffea')
    eventWWl_s = outputWWl_s['sumw']
    eventWWl_b = outputWWl_b['sumw']
    eventWWl_dy = outputWWl_dy['sumw']
    eventWWl_dynlo = outputWWl_dynlo['sumw']
    if args.year==2017:lumi=41500
    if args.year==2016:lumi=36300
    if args.year==2018:lumi=59800
    outputWWl_s[args.observable]=scale_xs(outputWWl_s[args.observable],lumi,eventWWl_s)
    outputWWl_b[args.observable]=scale_xs(outputWWl_b[args.observable],lumi,eventWWl_b)
    outputWWl_dy[args.observable]=scale_xs(outputWWl_dy[args.observable],lumi,eventWWl_dy)
    outputWWl_dynlo[args.observable]=scale_xs(outputWWl_dynlo[args.observable],lumi,eventWWl_dynlo)
    outputWWl_b[args.observable]=outputWWl_b[args.observable].group("dataset",hist.Cat("plotgroup", "plotgroup"),merge_map['hww_tem_zptbin'])
    outputWWl_dy[args.observable]=outputWWl_dy[args.observable].group("dataset",hist.Cat("plotgroup", "plotgroup"),merge_map['hww_tem_zptbin'])
    outputWWl_dynlo[args.observable]=outputWWl_dynlo[args.observable].group("dataset",hist.Cat("plotgroup", "plotgroup"),merge_map['hww_tem_zptbin'])    
    outputWWl_data[args.observable]=outputWWl_data[args.observable].group("dataset",hist.Cat("plotgroup", "plotgroup"),merge_map['data'])
    proc_names_bkg = outputWWl_b[args.observable].axis('plotgroup').identifiers()
    regions = ['SR','SR2','top_CR', 'DY_CR']
    flavs = ['ee','mumu','emu']
    for region in ['SR','SR2','top_CR', 'DY_CR']:
        if args.region != 'all' and args.region !=region:continue
        
        for flav in ['ee','mumu','emu']:
            if args.flav != 'all' and args.flav != flav:continue
            template_file = f"../../card_maker/shape/templates_{region}_{flav}_{args.observable}_{args.year}{args.prefix}.root"
            if os.path.exists(template_file):
                os.remove(template_file)
            print(f'Will save templates to {template_file}')
            
            fout = uproot3.create(template_file)
            name = "hc" 
            fout[name] = hist.export1d(outputWWl_s[args.observable].integrate("lepflav",flav).integrate("region",region).sum('flav').integrate("dataset","gchcWW2L2Nu"))
            name = "data_obs"
            fout[name] = hist.export1d(outputWWl_data[args.observable].integrate("lepflav",flav).integrate("region",region).sum('flav').integrate('plotgroup',f'data_{flav}'))
            for proc in proc_names_bkg:
                name = str(proc)
                output_bkg = outputWWl_b[args.observable].integrate("lepflav",flav).integrate("region",region).sum('flav').add(outputWWl_dy[args.observable].integrate("lepflav",flav).integrate("region",region).sum('flav'))
                output_bkg = output_bkg.add(outputWWl_dynlo[args.observable].integrate("lepflav",flav).integrate("region",region).sum('flav'))
                fout[name] = hist.export1d(output_bkg.integrate('plotgroup',proc))
            fig,ax = plt.subplots()
            ax = hist.plot1d(output_bkg,overlay='plotgroup',stack=True)
            hist.plot1d(outputWWl_data[args.observable].integrate("lepflav",flav).integrate("region",region).sum('flav').integrate('plotgroup',f'data_{flav}'),ax=ax,clear=False,error_opts=data_err_opts)
            hist.plot1d(outputWWl_s[args.observable].integrate("lepflav",flav).integrate("region",region).sum('flav').integrate("dataset","gchcWW2L2Nu"),ax=ax,clear=False)
            # leg_label=ax.get_legend_handles_labels()[1]
            # leg_label[-1]='Signal'
            # leg_label[-2]='Data'
            # ax.legend(loc="upper right",labels=leg_label)
            at = AnchoredText(flav+"  "+region_map[region]+"\n" +r"HWW$\rightarrow 2\ell 2\nu$"                                                         
                                            , loc='upper left',frameon=False)
            ax.add_artist(at)
            ax.set_ylim(bottom=0.0001)
            # ax.semilogy()
            hep.mpl_magic(ax=ax)
            
            fig.savefig(f'plot/validate_{region}_{flav}_{args.observable}.pdf')
    fout.close()
