# coding: utf-8

import functools
import logging
from collections import namedtuple
from functools import reduce
from operator import and_, or_

import awkward as ak
import numpy as np
from coffea import processor


logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


def df_object_overlap(toclean, cleanagainst, dr=0.4):
    particle_pair = toclean["p4"].cross(cleanagainst["p4"], nested=True)
    return (particle_pair.i0.delta_r(particle_pair.i1)).min() > dr


def nano_object_overlap(toclean, cleanagainst, dr=0.4):
    return ak.all(toclean.metric_table(cleanagainst) > dr, axis=-1)


def df_mask_or(df, masks):
    decision = reduce(or_, (df[mask] for mask in masks))
    return decision


def df_mask_and(df, masks):
    decision = reduce(and_, (df[mask] for mask in masks))
    return decision


def nano_mask_or(events, masks, skipable=()):
    masks = set(map(str, masks))
    if skipable is True:
        skipable = masks
    else:
        skipable = set(map(str, skipable))
    return reduce(
        or_,
        (
            getattr(events, mask)
            for mask in masks
            if mask not in skipable or hasattr(events, mask)
        ),
    )


def nano_mask_and(events, masks):
    decision = reduce(and_, (getattr(events, str(mask)) for mask in masks))
    return decision


def nano_cut(what, *cuts):
    return what[reduce(and_, cuts)]


def reduce_and(*what):
    return reduce(and_, what)


def reduce_or(*what):
    return reduce(or_, what)


def top_pT_sf_formula(pt):
    return np.exp(
        -2.02274e-01
        + 1.09734e-04 * pt
        + -1.30088e-07 * pt**2
        + (5.83494e01 / (pt + 1.96252e02))
    )


def top_pT_reweighting(gen):
    """
    Apply this SF only to TTbar datasets!

    Documentation:
        - https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting
        - https://indico.cern.ch/event/904971/contributions/3857701/attachments/2036949/3410728/TopPt_20.05.12.pdf
    """
    top = gen[(gen.pdgId == 6) & gen.hasFlags(["isLastCopy"])]
    anti_top = gen[(gen.pdgId == -6) & gen.hasFlags(["isLastCopy"])]
    return np.sqrt(top_pT_sf_formula(top.pt) * top_pT_sf_formula(anti_top.pt))


@parametrized
def padflat(func, n_particles=1):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return ak.flatten(ak.fill_none(ak.pad_none(res, n_particles), np.nan))

    return wrapper


def bregcorr(jets):
    return ak.zip(
        {
            "pt": jets.pt * jets.bRegCorr,
            "eta": jets.eta,
            "phi": jets.phi,
            "energy": jets.energy * jets.bRegCorr,
        },
        with_name="PtEtaPhiELorentzVector",
    )


@padflat(1)
def m_bb(jets):
    lead_jets = jets[..., :2].distincts()
    bb = lead_jets.i0 + lead_jets.i1
    m_bb = bb.mass
    return m_bb


def get_ht(jets):
    return ak.sum(jets.pt, axis=-1)


def min_dr_part1_part2(part1, part2, getn=0, fill=np.nan):
    """
    For each particle in part1 returns the minimum
    delta_r between this and each particle in part2
    """
    a, b = ak.unzip(ak.cartesian({"p1": part1, "p2": part2}, nested=True))
    if not hasattr(a, "delta_r"):
        a = make_p4(a)
    if not hasattr(b, "delta_r"):
        b = make_p4(a)
    r = ak.min(a.delta_r(b), axis=-1)
    if 0 < getn:
        r = ak.fill_none(ak.pad_none(r, getn, clip=True), np.nan)
        return tuple(r[:, i] for i in range(getn))
    else:
        return r


def get_metp4(met):
    return ak.zip(
        {
            "pt": met.pt,
            "eta": met.pt * 0,
            "phi": met.phi,
            "mass": met.pt * 0,
        },
        with_name="PtEtaPhiMLorentzVector",
    )


def min_dr(particles):
    di_particles = ak.combinations(
        particles,
        n=2,
        replacement=False,
        axis=1,
        fields=["p1", "p2"],
    )
    return ak.min(
        make_p4(di_particles.p1).delta_r(make_p4(di_particles.p2)),
        axis=-1,
        mask_identity=False,
    )


def min_dphi(particles):
    di_particles = ak.combinations(
        particles,
        n=2,
        replacement=False,
        axis=1,
        fields=["p1", "p2"],
    )
    return ak.min(
        np.abs(make_p4(di_particles.p1).delta_phi(make_p4(di_particles.p2))),
        axis=-1,
        mask_identity=False,
    )


def get_met_ld(jets, leps, met, met_coef=0.6, mht_coef=0.4):
    mht = make_p4(jets).sum() + make_p4(leps).sum()
    return met_coef * met.pt + mht_coef * mht.pt


def get_cone_pt(part, n=1):
    padded_part = ak.pad_none(part, n)
    return [ak.fill_none(padded_part[..., i].cone_pt, np.nan) for i in range(n)]


def make_p4(obj, candidate=False):
    params = ["pt", "eta", "phi", "mass"]
    with_name = "PtEtaPhiMLorentzVector"
    if candidate:
        params.append("charge")
        with_name = "PtEtaPhiMCandidate"
    return ak.zip(
        {p: getattr(obj, p) for p in params},
        with_name=with_name,
    )


def lead_diobj(objs):
    two = objs[:, :2]
    a, b = ak.unzip(
        ak.combinations(
            two,
            n=2,
            replacement=False,
            axis=1,
            fields=["a", "b"],
        )
    )

    # make sure it is a CandidateArray
    if not hasattr(a, "delta_r"):
        a = make_p4(a, candidate=False)
    if not hasattr(b, "delta_r"):
        b = make_p4(b, candidate=False)
    diobj = a + b
    diobj["deltaR"] = a.delta_r(b)
    diobj["deltaPhi"] = a.delta_phi(b)
    return diobj


class chunked:
    def __init__(self, func, chunksize=10000):
        self.func = func
        self.chunksize = chunksize

    def __call__(self, *args, **kwargs):
        lens = set(map(len, args))
        if len(lens) != 1:
            raise ValueError("inconsistent *args len")
        return ak.concatenate(
            [
                self.func(*(a[off : off + self.chunksize] for a in args), **kwargs)
                for off in range(0, max(lens), self.chunksize)
            ]
        )


def linear_fit(x, y, eigen_decomp=False):
    coeff, cov = np.polyfit(x, y, 1, cov="unscaled")
    return linear_func(coeff, cov, eigen_decomp)


def linear_func(coeff, cov, eigen_decomp=False):
    c1, c0 = coeff
    nom = lambda v: c0 + c1 * v
    if eigen_decomp:
        eigenvals, eigenvecs = np.linalg.eig(cov)
        lambda0, lambda1 = np.sqrt(eigenvals)
        v00, v01 = eigenvecs[:, 0]  # 1st eigenvector
        v10, v11 = eigenvecs[:, 1]  # 2nd eigenvector
        var1_down = lambda v: c0 - lambda0 * v00 + (c1 - lambda0 * v01) * v
        var1_up = lambda v: c0 + lambda0 * v00 + (c1 + lambda0 * v01) * v
        var2_down = lambda v: c0 - lambda1 * v10 + (c1 - lambda0 * v11) * v
        var2_up = lambda v: c0 + lambda1 * v10 + (c1 + lambda0 * v11) * v
        return nom, (var1_down, var1_up), (var2_down, var2_up)
    else:
        return nom


import awkward as ak


def mT(obj1, obj2):
    return np.sqrt(2.0 * obj1.pt * obj2.pt * (1.0 - np.cos(obj1.phi - obj2.phi)))


def flatten(ar):  # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)


def normalize(val, cut=None):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar


def empty_column_accumulator():
    return processor.column_accumulator(np.array([], dtype=np.float64))


def defaultdict_accumulator():
    return processor.defaultdict_accumulator(empty_column_accumulator)
