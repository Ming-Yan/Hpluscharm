import csv
from curses import meta
from dataclasses import dataclass
import gzip
import pickle, os, sys, mplhep as hep, numpy as np
from select import select

from matplotlib.pyplot import jet

import coffea
from coffea import hist, processor
from coffea.nanoevents.methods import vector
import awkward as ak
from coffea.analysis_tools import Weights
from functools import partial

# import numba
from helpers.util import reduce_and, reduce_or, nano_mask_or, get_ht, normalize, make_p4


def mT(obj1, obj2):
    return np.sqrt(2.0 * obj1.pt * obj2.pt * (1.0 - np.cos(obj1.phi - obj2.phi)))


def mT2(lep, met):
    mt2 = np.power(
        np.sqrt(lep.pt**2 + lep.mass**2) + np.sqrt(met.pt**2 + 91.18**2), 2
    ) - np.power(lep.pt + met.pt, 2)
    return np.sqrt(abs(mt2))


def flatten(ar):  # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)


def normalize(val, cut):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar


class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self, year="2017"):
        self._year = year

        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r" $m_{\\ell\\ell}$ [GeV]", 60, 0, 150)
        cmass_axis = hist.Bin("mass", r" $m_{\\ell\\ell}$ [GeV]", 50, 0, 200)
        pt_axis = hist.Bin("pt", r" $p_T$ [GeV]", 60, 0, 150)
        eta_axis = hist.Bin("eta", r" $\\eta$", 50, -2.5, 2.5)
        phi_axis = hist.Bin("phi", r" $\\phi$", 60, -3, 3)
        dr_axis = hist.Bin("dr", r" $\\Delta R$", 50, 0, 5)

        flav_axis = hist.Bin("flav", r"Genflavour", [0, 1, 4, 5, 6])
        cvb_axis = hist.Bin("CvB", r" CvB", 50, 0, 1)
        cvl_axis = hist.Bin("CvL", r" CvL", 50, 0, 1)

        _hist_event_dict = {
            "genlep1_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genlep1_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genlep1_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "genlep2_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genlep2_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genlep2_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "genjet1_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genjet1_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genjet1_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "genjet2_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genjet2_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genjet2_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "gennu_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "gennu_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genmet_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genmet_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genll_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genll_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "genll_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genll_mass": hist.Hist("Counts", dataset_axis, mass_axis),
            "genjj_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genjj_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "genjj_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genjj_mass": hist.Hist("Counts", dataset_axis, cmass_axis),
            "genc_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "genc_phi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genc_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "genc_mass": hist.Hist("Counts", dataset_axis, cmass_axis),
            "genll_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "genjj_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "genllc_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "genjjc_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "genl1c_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "genl2c_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "genj1c_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "genj2c_dr": hist.Hist("Counts", dataset_axis, dr_axis),
            "gennuc_dphi": hist.Hist("Counts", dataset_axis, phi_axis),
            "gennul_dphi": hist.Hist("Counts", dataset_axis, phi_axis),
            "gennull_dphi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genmetc_dphi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genmetl_dphi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genmetll_dphi": hist.Hist("Counts", dataset_axis, phi_axis),
            "genmet_mt": hist.Hist("Counts", dataset_axis, mass_axis),
            "genmet_mt2": hist.Hist("Counts", dataset_axis, cmass_axis),
            "matched_deepJet": hist.Hist(
                "Counts", dataset_axis, flav_axis, cvl_axis, cvb_axis
            ),
            "matched_deepCSV": hist.Hist(
                "Counts", dataset_axis, flav_axis, cvl_axis, cvb_axis
            ),
        }

        self.event_hists = list(_hist_event_dict.keys())

        self._accumulator = processor.dict_accumulator(
            {
                **_hist_event_dict,
                "cutflow": processor.defaultdict_accumulator(
                    # we don't use a lambda function to avoid pickle issues
                    partial(processor.defaultdict_accumulator, int)
                ),
            }
        )
        self._accumulator["sumw"] = processor.defaultdict_accumulator(float)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = processor.PackedSelection()
        if isRealData:
            output["sumw"][dataset] += 1.0
        else:
            output["sumw"][dataset] += ak.sum(events.genWeight / abs(events.genWeight))
        weights = Weights(len(events), storeIndividual=True)
        if isRealData:
            weights.add("genweight", np.ones(len(events)))
        else:
            weights.add("genweight", events.genWeight / abs(events.genWeight))
            # weights.add('puweight', compiled['2017_pileupweight'](events.Pileup.nPU))
        ##############
        if isRealData:
            output["cutflow"][dataset]["all"] += 1.0
        else:
            output["cutflow"][dataset]["all"] += ak.sum(
                events.genWeight / abs(events.genWeight)
            )

        genc = events.GenPart[
            (abs(events.GenPart.pdgId) == 4)
            & (events.GenPart.pt != 0)
            & (events.GenPart.hasFlags(["fromHardProcess"]) == True)
            & (events.GenPart.hasFlags(["isHardProcess"]) == True)
        ]
        matchj = genc.nearest(events.Jet, threshold=0.1)

        # print(ak.type(matchj))
        # if "WW" in dataset : momid=24
        # else :momid=23
        genlep = events.GenPart[
            (
                (abs(events.GenPart.pdgId) == 11)
                | (abs(events.GenPart.pdgId) == 13)
                | (abs(events.GenPart.pdgId) == 15)
            )
            & (events.GenPart.hasFlags(["fromHardProcess"]) == True)
            & (events.GenPart.hasFlags(["isHardProcess"]) == True)
        ]

        if "2Nu" in dataset:
            gennu = events.GenPart[
                (
                    (abs(events.GenPart.pdgId) == 12)
                    | (abs(events.GenPart.pdgId) == 14)
                    | (abs(events.GenPart.pdgId) == 16)
                )
                & (events.GenPart.hasFlags(["fromHardProcess"]) == True)
                & (events.GenPart.hasFlags(["isHardProcess"]) == True)
            ]
            gennunu = ak.zip(
                {
                    "pt": (gennu[:, 0] + gennu[:, 1]).pt,
                    "phi": (gennu[:, 0] + gennu[:, 1]).phi,
                    "energy": (gennu[:, 0] + gennu[:, 1]).energy,
                },
                with_name="PtEtaPhiMLorentzVector",
            )
            genmet = ak.zip(
                {
                    "pt": events.GenMET.pt,
                    "phi": events.GenMET.phi,
                },
                with_name="PtEtaPhiMLorentzVector",
            )

        else:
            genjet = events.GenPart[
                (abs(events.GenPart.pdgId) < 6)
                & (events.GenPart.hasFlags(["fromHardProcess"]) == True)
                & (events.GenPart.hasFlags(["isHardProcess"]) == True)
            ]
            genjet = genjet[genjet.parent.pdgId == 23]

            genzj = genjet[:, 0] + genjet[:, 1]

        genzl = genlep[:, 0] + genlep[:, 1]
        # print(genlep[:,0].pt)
        # print(genz.mass)
        output["genlep1_pt"].fill(dataset=dataset, pt=flatten(genlep[:, 0].pt))
        output["genlep1_eta"].fill(dataset=dataset, eta=flatten(genlep[:, 0].eta))
        output["genlep1_phi"].fill(dataset=dataset, phi=flatten(genlep[:, 0].phi))
        output["genlep2_pt"].fill(dataset=dataset, pt=flatten(genlep[:, 1].pt))
        output["genlep2_eta"].fill(dataset=dataset, eta=flatten(genlep[:, 1].eta))
        output["genlep2_phi"].fill(dataset=dataset, phi=flatten(genlep[:, 1].phi))
        output["genll_pt"].fill(dataset=dataset, pt=flatten(genzl.pt))
        output["genll_eta"].fill(dataset=dataset, eta=flatten(genzl.eta))
        output["genll_phi"].fill(dataset=dataset, phi=flatten(genzl.phi))
        output["genll_mass"].fill(dataset=dataset, mass=flatten(genzl.mass))
        output["genc_pt"].fill(dataset=dataset, pt=flatten(genc.pt))
        output["genc_eta"].fill(dataset=dataset, eta=flatten(genc.eta))
        output["genc_phi"].fill(dataset=dataset, phi=flatten(genc.phi))
        output["genc_mass"].fill(dataset=dataset, mass=flatten(genc.mass))
        output["genll_dr"].fill(
            dataset=dataset, dr=flatten(genlep[:, 0].delta_r(genlep[:, 1]))
        )
        output["genllc_dr"].fill(dataset=dataset, dr=flatten(genzl.delta_r(genc)))
        output["genl1c_dr"].fill(
            dataset=dataset, dr=flatten(genlep[:, 0].delta_r(genc))
        )
        output["genl2c_dr"].fill(
            dataset=dataset, dr=flatten(genlep[:, 1].delta_r(genc))
        )
        output["matched_deepJet"].fill(
            dataset=dataset,
            flav=flatten(
                matchj.hadronFlavour
                + 1 * ((matchj.partonFlavour == 0) & (matchj.hadronFlavour == 0))
            ),
            CvL=flatten(matchj.btagDeepFlavCvL),
            CvB=flatten(matchj.btagDeepFlavCvB),
        )
        output["matched_deepCSV"].fill(
            dataset=dataset,
            flav=flatten(
                matchj.hadronFlavour
                + 1 * ((matchj.partonFlavour == 0) & (matchj.hadronFlavour == 0))
            ),
            CvL=flatten(matchj.btagDeepCvL),
            CvB=flatten(matchj.btagDeepCvB),
        )

        if "2Nu" in dataset:
            output["gennu_pt"].fill(dataset=dataset, pt=flatten(gennunu.pt))
            output["gennu_phi"].fill(dataset=dataset, phi=flatten(gennunu.phi))
            output["gennuc_dphi"].fill(
                dataset=dataset, phi=flatten(gennunu.delta_phi(genc))
            )
            output["gennul_dphi"].fill(
                dataset=dataset, phi=flatten(gennunu.delta_phi(genlep))
            )
            output["gennull_dphi"].fill(
                dataset=dataset, phi=flatten(gennunu.delta_phi(genzl))
            )
            output["genmet_pt"].fill(dataset=dataset, pt=flatten(events.GenMET.pt))
            output["genmet_phi"].fill(dataset=dataset, phi=flatten(events.GenMET.phi))
            output["genmetc_dphi"].fill(
                dataset=dataset, phi=flatten(genmet.delta_phi(genc))
            )
            output["genmetl_dphi"].fill(
                dataset=dataset, phi=flatten(genmet.delta_phi(genlep))
            )
            output["genmetll_dphi"].fill(
                dataset=dataset, phi=flatten(genmet.delta_phi(genzl))
            )
            output["genmet_mt"].fill(dataset=dataset, mass=flatten(mT(genmet, genzl)))

            output["genmet_mt2"].fill(dataset=dataset, mass=flatten(mT2(genzl, genmet)))

        else:
            output["genjet1_pt"].fill(dataset=dataset, pt=flatten(genjet[:, 0].pt))
            output["genjet1_eta"].fill(dataset=dataset, eta=flatten(genjet[:, 0].eta))
            output["genjet1_phi"].fill(dataset=dataset, phi=flatten(genjet[:, 0].phi))
            output["genjet2_pt"].fill(dataset=dataset, pt=flatten(genjet[:, 1].pt))
            output["genjet2_eta"].fill(dataset=dataset, eta=flatten(genjet[:, 1].eta))
            output["genjet2_phi"].fill(dataset=dataset, phi=flatten(genjet[:, 1].phi))
            output["genjj_pt"].fill(dataset=dataset, pt=flatten(genzj.pt))
            output["genjj_eta"].fill(dataset=dataset, eta=flatten(genzj.eta))
            output["genjj_phi"].fill(dataset=dataset, phi=flatten(genzj.phi))
            output["genjj_mass"].fill(dataset=dataset, mass=flatten(genzj.mass))
            output["genjj_dr"].fill(
                dataset=dataset, dr=flatten(genjet[:, 0].delta_r(genjet[:, 1]))
            )
            output["genjjc_dr"].fill(dataset=dataset, dr=flatten(genzj.delta_r(genc)))
            output["genj1c_dr"].fill(
                dataset=dataset, dr=flatten(genjet[:, 0].delta_r(genc))
            )
            output["genj2c_dr"].fill(
                dataset=dataset, dr=flatten(genjet[:, 1].delta_r(genc))
            )

            # if 'LNu' in dataset:
            #     output['genmetcphi'].fill(dataset=dataset,phi=flatten(genmet.delta_phi(genc)))
            #     output['genmetlphi'].fill(dataset=dataset,phi=flatten(genmet.delta_phi(genlep)))
            #     output['genmetllphi'].fill(dataset=dataset,phi=flatten(genmet.delta_phi(genzl)))

        return output

    def postprocess(self, accumulator):
        print(accumulator)
        return accumulator
