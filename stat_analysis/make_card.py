import CombineHarvester.CombineTools.ch as ch
import ROOT as R
import glob
import numpy as np
import os
import sys
import argparse
cb = ch.CombineHarvester()
def adjust_shape(proc,nbins):
  new_hist = proc.ShapeAsTH1F();
  new_hist.Scale(proc.rate())
  for i in range(1,new_hist.GetNbinsX()+1-nbins):
    new_hist.SetBinContent(i,0.)
  proc.set_shape(new_hist,True)

def drop_zero_procs(chob,proc):
  null_yield = not (proc.rate() > 0.)
  if(null_yield):
    chob.FilterSysts(lambda sys: matching_proc(proc,sys)) 
  return null_yield

def drop_zero_systs(syst):
  null_yield = (not (syst.value_u() > 0. and syst.value_d()>0.) ) and syst.type() in 'shape'
  if(null_yield):
    print 'Dropping systematic ',syst.name(),' for region ', syst.bin(), ' ,process ', syst.process(), '. up norm is ', syst.value_u() , ' and down norm is ', syst.value_d()
    #chob.FilterSysts(lambda sys: matching_proc(proc,sys)) 
  return null_yield

def matching_proc(p,s):
  return ((p.bin()==s.bin()) and (p.process()==s.process()) and (p.signal()==s.signal()) 
          and (p.analysis()==s.analysis()) and  (p.era()==s.era()) 
          and (p.channel()==s.channel()) and (p.bin_id()==s.bin_id()) and (p.mass()==s.mass()))


def remove_norm_effect(syst):
  syst.set_value_u(1.0)
  syst.set_value_d(1.0)

def symm(syst,nominal):
  print 'Symmetrising systematic ', syst.name(), ' in region ', syst.bin(), ' for process ', syst.process()
  hist_u = syst.ShapeUAsTH1F()
  hist_u.Scale(nominal.Integral()*syst.value_u())
  hist_d = nominal.Clone()
  hist_d.Scale(2)
  hist_d.Add(hist_u,-1)
  syst.set_shapes(hist_u,hist_d,nominal)
  
  
def symmetrise_syst(chob,proc,sys_name):
  nom_hist = proc.ShapeAsTH1F()
  nom_hist.Scale(proc.rate())
  chob.ForEachSyst(lambda s: symm(s,nom_hist) if (s.name()==sys_name and matching_proc(proc,s)) else None)

def increase_bin_errors(proc):
  print 'increasing bin errors for process ', proc.process(), ' in region ', proc.bin()
  new_hist = proc.ShapeAsTH1F();
  new_hist.Scale(proc.rate())
  for i in range(1,new_hist.GetNbinsX()+1):
    new_hist.SetBinError(i,np.sqrt(2)*new_hist.GetBinError(i))
  proc.set_shape(new_hist,False)

def decrease_bin_errors(proc):
  print 'decreasing bin errors for process ', proc.process(), ' in region ', proc.bin()
  new_hist = proc.ShapeAsTH1F();
  new_hist.Scale(proc.rate())
  for i in range(1,new_hist.GetNbinsX()+1):
    new_hist.SetBinError(i,new_hist.GetBinError(i)/2.0)
  proc.set_shape(new_hist,False)


def drop_noRealShape_systs(proc,syst):
  diff_lim=0.00025
  if syst.type()=='shape' : 
    hist_u = syst.ShapeUAsTH1F()
    hist_d = syst.ShapeDAsTH1F()
    hist_nom = proc.ShapeAsTH1F()
    hist_nom.Scale(1./hist_nom.Integral())
    up_diff=0
    down_diff=0
    print "SYSTEMATICS = ",syst.name(),syst.process(),syst.bin()
    for i in range(1,hist_u.GetNbinsX()+1):
      if hist_nom.GetBinContent(i)!=0:
        up_diff+=2*(abs(hist_u.GetBinContent(i)-hist_nom.GetBinContent(i)))/(abs(hist_u.GetBinContent(i))+abs(hist_nom.GetBinContent(i)))
        down_diff+=2*(abs(hist_d.GetBinContent(i)-hist_nom.GetBinContent(i)))/(abs(hist_u.GetBinContent(i))+abs(hist_nom.GetBinContent(i)))
    else:
        up_diff+=0
        down_diff+=0
    null_yield = (up_diff<diff_lim and down_diff<diff_lim)
    if(null_yield):
      #print "Uncertainty has no real shape effect. Summed rel. diff. per bin between norm. nominal and up/down shape: ",up_diff, down_diff
      print 'Dropping systematic ',syst.name(),' for region ', syst.bin(), ' ,process ', syst.process(), '. up int ', hist_u.Integral() , ' and down int is ', hist_d.Integral()
    return null_yield  

def PrintProc(proc):
  print  proc.channel(), proc.bin_id(), proc.process()

def PrintSyst(syst,proc):
  print  syst.channel(), syst.bin_id(), syst.process(), syst.name(), proc.process()
  
parser = argparse.ArgumentParser()
parser.add_argument('-o','--output_folder', default='H+c_cards', help="""Subdirectory of ./output/ where the cards are written out to""")
parser.add_argument('--year', default='2017', help="""Year to produce datacards for (2018, 2017 or 2016)""")
parser.add_argument("-obs", "--observable",type=str,default='ll_mass',help='observable to the fit')
parser.add_argument("--chn", default='ee', choices=['ee', 'mumu', 'emu','all'], type=str, help="chnor")
parser.add_argument("--rebin", default=None,type=int,help='rebin')
args = parser.parse_args()
shapes = os.getcwd()
mass = ['125']
chns=['ee','mumu','emu']
regions=['SR','SR2','DY_CR','top_CR','HM_CR']
year = args.year
bkg_proc = ['st','vv','vjets','ttbar']
sig_proc = ['hc']
cats = {
  'ee':[(1,'SR'),(2,'SR2'),(3,'top_CR'),(4,'DY_CR')],#,(5,'HM_CR')],
  'mumu':[(1,'SR'),(2,'SR2'),(3,'top_CR'),(4,'DY_CR')],#,(5,'HM_CR')],
  'emu':[(1,'SR'),(2,'SR2'),(3,'top_CR')]#,(5,'HM_CR')],
}
for chn in chns:
  if args.chn != 'all' and args.chn != chn:continue
  
  cb.AddObservations( ['*'], ['hc'], ['13TeV'], [chn], cats[chn])
  cb.AddProcesses( ['*'], ['hc'], ['13TeV'], [chn], bkg_proc, cats[chn], False)
  cb.AddProcesses( ['*'], ['hc'], ['13TeV'], [chn], sig_proc, cats[chn], True)

  ## read shapes
  i=1
  for region in regions:
    file = 'shape/templates_%s_%s_%s_%s.root' %(region,chn,args.observable,args.year)
    if region == 'top_CR' : file='shape/templates_%s_%s_nj_%s.root' %(region,chn,args.year)
    # elif 'SR' in region : file='shape/templates_%s_LM_%s_ll_mass_%s.root' %(region,chn,args.year)
    if chn == 'emu' and region == 'DY_CR' : continue
    # if chn=='emu' and region =='HM_CR':i=5
    cb.cp().channel([chn]).signals().bin_id([i]).ExtractShapes(file, '$PROCESS', '$PROCESS')
    cb.cp().channel([chn]).backgrounds().bin_id([i]).ExtractShapes(file, '$PROCESS', '$PROCESS')
    if region == 'top_CR' or region=='SR':
      binning=np.linspace(0,300,num=args.rebin)
      cb.cp().channel(chn).bin_id([i]).VariableRebin(binning)
    cb.cp().AddSyst(cb,'lumi_13TeV','lnN', ch.SystMap()(1.023))
    # cb.cp().AddSyst(cb,'adhoc_13TeV','shape',ch.SystMap()(1.25))
    # cb.cp().process(['hc']).AddSyst(cb,'CMS_hc','lnN',ch.SystMap()(1.50))
    cb.cp().process(['st','vv']).AddSyst(cb,'CMS_vvst','lnN',ch.SystMap()(1.15))
    cb.cp().process(['ttbar']).AddSyst(cb,'CMS_ttbar','lnN',ch.SystMap()(1.005))
    cb.cp().process(['vjets']).AddSyst(cb,'CMS_vjet','lnN',ch.SystMap()(1.05))
    cb.cp().channel([chn]).process(['ttbar']).AddSyst(cb, 'SF_tt_%s' %(chn),'rateParam',ch.SystMap()
     (1.0))
    if chn !='emu' :cb.cp().channel([chn]).process(['vjets']).AddSyst(cb, 'SF_vjets_%s' %(chn),'rateParam',ch.SystMap()
     (1.0))
    i=i+1
    
  
ch.SetStandardBinNames(cb)
writer=ch.CardWriter("cards/"+args.output_folder+args.year+args.observable+"/$BIN"+".txt","cards/"+args.output_folder+args.year+args.observable+"/input$BIN"+".root")  
writer.SetWildcardMasses([])

for chn in chns:
  writer.WriteCards(chn,cb.cp().channel([chn]))
  cb.AddDatacardLineAtEnd("* autoMCStats 0")
cb.cp().mass("*").WriteDatacard("cards/"+args.output_folder+args.year+args.observable+"/combine"+args.chn+".txt","cards/"+args.output_folder+args.year+args.observable+"/combineinput"+args.chn+".root");
cb.AddDatacardLineAtEnd("* autoMCStats 0")
  

  

