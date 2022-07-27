import numpy as np
import awkward as ak
# import vector
from coffea.nanoevents.methods import vector

mW = 80.38
def real_result(lep,met,lam, discri):
    # ak.behavior.update(vector.behavior)
    neupz = np.where(np.abs(lam*lep.pz/lep.p**2+1./lep.pt**2*np.sqrt(discri))<np.abs(lam*lep.pz/lep.p**2-1./lep.pt**2*np.sqrt(discri)),lam*lep.pz/(lep.p**2)+1./lep.pt**2*np.sqrt(discri),lam*lep.pz/lep.p**2-1./lep.pt**2*np.sqrt(discri))
    # print("split ",lam*lep.pz/lep.p**2,1./lep.pt**2*np.sqrt(discri))
    # print("discri: " ,discri)
    neu = ak.zip({
    "x" : met.pt*np.cos(met.phi)/np.sqrt(met.pt**2+neupz**2),
    "y" : met.pt*np.sin(met.phi)/np.sqrt(met.pt**2+neupz**2),
    "z" : neupz/np.sqrt(met.pt**2+neupz**2),
    "t": np.sqrt( met.pt*np.cos(met.phi)**2+met.pt*np.sin(met.phi)**2+neupz**2),
}, with_name="LorentzVector")
    ak.behavior.update(vector.behavior)
    
    return neu
def img_result(lep,met,lam, discri):
    # 
    a = lep.pt**2*met.pt**2*(lep.pz**2+lep.pt**2)
    b = mW**2*lep.pt*met.pt*(lep.pz**2+lep.pt**2)
    c = mW**4/4.*(lep.pz**2+lep.pt**2)-lep.pt**2*lep.energy**2*met.pt**2
    print(-b,np.sqrt(4.*a*c)/(2.*a))
    pos= np.arccos(-b+np.sqrt(4.*a*c)/(2.*a))
    # np.arccos(-met.pt*lep.pt*mW**2*(lep.pt**2+lep.pz**2)+np.sqrt(met.pt**2*lep.pt**2*mW**4*(lep.pt**2+lep.pz**2)*(lep.pt**2+lep.pz**2)-mW**4*(met.pt**2*lep.pt**2-lep.pt**2*lep.energy**2)))
    neg = np.arccos(-b-np.sqrt(4.*a*c)/(2.*a))
    # np.arccos(-met.pt*lep.pt*mW**2*(lep.pt**2+lep.pz**2)-np.sqrt(met.pt**2*lep.pt**2*mW**4*(lep.pt**2+lep.pz**2)*(lep.pt**2+lep.pz**2)-mW**4*(met.pt**2*lep.pt**2-lep.pt**2*lep.energy**2)))
    print("angle",pos,neg)
    angle = np.where(np.abs(pos-met.phi)<np.abs(neg-met.phi),pos,neg)
    neupx = np.cos(angle)*met.pt
    neupy = np.sin(angle)*met.pt
    neupz = (np.cos(angle)*lep.pt*met.pt+mW**2/2.)*lep.pz/lep.pt**2
    neu = ak.zip({
        "x" : neupx/np.sqrt(neupx**2+neupy**2+neupz**2),
        "y" : neupy/np.sqrt(neupx**2+neupy**2+neupz**2),
        "z" : neupz/np.sqrt(neupx**2+neupy**2+neupz**2),
        "t": np.sqrt(neupx**2+neupy**2+neupz**2),
    }, with_name="LorentzVector")
    ak.behavior.update(vector.behavior)
    print("neupz: ",neupz)
    print(neu.pt,neu.mass,neu.eta,neu.phi)
    return neu
def get_nu4vec(lep,met):
    lam= mW*mW/2.+lep.pt*met.pt*np.cos(lep.delta_phi(met))
    discr = lam**2*lep.pz**2-lep.pt**2*(lep.energy**2*met.pt**2-lam**2) 
    neu = ak.where(discr>0,real_result(lep,met,lam,discr),img_result(lep,met,lam,discr))
    print("discr: ",discr)
    # neu = real_result(lep,met,lam,discr)
    # print(neu.pt,neu.eta,neu.phi,neu.mass)
    return neu
# def get_topmass(lep, jet, met):
