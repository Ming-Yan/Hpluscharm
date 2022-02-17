# coding: utf-8

import functools
import logging
from collections import namedtuple
from functools import reduce
from operator import and_, or_

import awkward as ak
import numpy as np
import vector

from func import parametrized

logger = logging.getLogger(__name__)


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
        (getattr(events, mask) for mask in masks if mask not in skipable or hasattr(events, mask)),
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
        -2.02274e-01 + 1.09734e-04 * pt + -1.30088e-07 * pt ** 2 + (5.83494e01 / (pt + 1.96252e02))
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
    return ak.sum(jets.pt, axis=-1)  # jets[jets.pt > 50].pt.sum()


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

'''
def _eventshape_eigvals(met_or_nu, leps, jets):
    met_or_nu = vector.awk({"x": met_or_nu.x, "y": met_or_nu.y, "z": met_or_nu.z})
    leps = vector.awk({"x": leps.x, "y": leps.y, "z": leps.z})
    jets = vector.awk({"x": jets.x, "y": jets.y, "z": jets.z})
    assert len(met_or_nu) == len(leps) == len(jets)
    # fmt: off
    mxx = np.asarray(met_or_nu.x * met_or_nu.x + ak.sum(leps.x * leps.x, axis=-1) + ak.sum(jets.x * jets.x, axis=-1))
    myy = np.asarray(met_or_nu.y * met_or_nu.y + ak.sum(leps.y * leps.y, axis=-1) + ak.sum(jets.y * jets.y, axis=-1))
    mzz = np.asarray(met_or_nu.z * met_or_nu.z + ak.sum(leps.z * leps.z, axis=-1) + ak.sum(jets.z * jets.z, axis=-1))
    mxy = np.asarray(met_or_nu.x * met_or_nu.y + ak.sum(leps.x * leps.y, axis=-1) + ak.sum(jets.x * jets.y, axis=-1))
    mxz = np.asarray(met_or_nu.x * met_or_nu.z + ak.sum(leps.x * leps.z, axis=-1) + ak.sum(jets.x * jets.z, axis=-1))
    myz = np.asarray(met_or_nu.y * met_or_nu.z + ak.sum(leps.y * leps.z, axis=-1) + ak.sum(jets.y * jets.z, axis=-1))
    # fmt: on
    matrix = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]]) / (mxx + myy + mzz)
    matrix = np.ascontiguousarray(np.rollaxis(matrix, -1))
    eig = eigvalsh_with_nans(matrix)
    # check ascending order: lambda_1 < lambda_2 < lambda_3 for good (non-nan) eigvals
    good = ~np.any(np.isnan(eig), axis=-1)
    assert np.all(np.sort(eig[good], axis=-1) == eig[good])
    return eig


EventShapeVariables = namedtuple(
    "EventShapeVariables",
    "sphericity sphericity_T aplanarity C D Y foxwolfram centrality centrality_jets eigenvalues",
)

'''
def get_eventshape_variables(met_or_nu, leps, jets):
    """
    see:
        - https://gitlab.cern.ch/aachen-3a-cms/common/-/blob/master/analyses/ttH_bb/modules/ttH_bb_EventVariables/bdtvariables.h#L29
        - https://github.com/davidsheffield/EventShape/blob/master/Class/src/EventShape.cc
        - https://arxiv.org/pdf/hep-ph/0312283.pdf
        - https://arxiv.org/pdf/hep-ph/9308216.pdf
        - https://arxiv.org/pdf/1206.2135.pdf
        - https://journals.aps.org/prd/pdf/10.1103/PhysRevD.88.032004

    args:
        met_or_nu: (n_events,)  -- Caution: This assumes that every event has always exactly 1 MET or 1 Neutrino!
        leps: (n_events, var)
        jets: (n_events, var)


    This function calculates several event shape variables. The variables `sphericity`, `aplanarity`, `C`, `D`, `Y` are defined for a 3D momentum tensor,
    thus requiring physical knowledge about the px,py,pz component of all input particles. Since neutrinos escape the detector we have (at first hand) only
    access to the missing transverse energy (MET) of which the pz component is _always_ zero. Thus the above mentioned variables are not well defined in case
    the MET 4-vector is used. However there is also a transverse definition of the sphericity called `sphericity_T` which is defined for a 2D momentum tensor (px,py)
    and thus does not require the knowledge of the pz component.
    Additionally the 5 `foxwolfram` moments of the jets and the centrality of all particles/jets are calculated.
    The raw (sorted ascending) `eigenvalues` of the momentum tensor are also returned.
    """

    eigvals = _eventshape_eigvals(met_or_nu, leps, jets)
    # fmt: off
    return EventShapeVariables(
        sphericity=3.0 / 2.0 * (eigvals[:, 0] + eigvals[:, 1]),
        sphericity_T=2.0 * eigvals[:, 1] / (eigvals[:, 2] + eigvals[:, 1]),
        aplanarity=3.0 / 2.0 * eigvals[:, 0],
        C=3.0 * (eigvals[:, 2] * eigvals[:, 1] + eigvals[:, 2] * eigvals[:, 0] + eigvals[:, 1] * eigvals[:, 0]),
        D=27.0 * eigvals[:, 2] * eigvals[:, 1] * eigvals[:, 0],
        Y=np.sqrt(3.0) / 2.0 * (eigvals[:, 1] - eigvals[:, 0]),
        foxwolfram=_foxwolfram_moments(jets),
        centrality=normalize((ak.sum(jets.pt, axis=-1) + ak.sum(leps.pt, axis=-1) + met_or_nu.pt) / (ak.sum(jets.energy, axis=-1) + ak.sum(leps.energy, axis=-1) + met_or_nu.energy)),
        centrality_jets=normalize(ak.sum(jets.pt, axis=-1) / ak.sum(jets.energy, axis=-1)),
        eigenvalues=eigvals,
    )
    # fmt: on


def _foxwolfram_moments(jets):
    """see: https://gitlab.cern.ch/aachen-3a-cms/common/-/blob/master/analyses/ttH_bb/modules/ttH_bb_EventVariables/bdtvariables.h#L90"""
    j1, j2 = ak.unzip(
        ak.combinations(
            jets,
            n=2,
            replacement=False,
            axis=1,
        )
    )
    j1 = vector.awk({"x": j1.x, "y": j1.y, "z": j1.z})
    j2 = vector.awk({"x": j2.x, "y": j2.y, "z": j2.z})
    costheta = np.cos(j1.deltaangle(j2))
    p0 = 1.0
    p1 = costheta
    p2 = 0.5 * (3.0 * costheta * costheta - 1.0)
    p3 = 0.5 * (5.0 * costheta * costheta * costheta - 3.0 * costheta)
    p4 = 0.125 * (35.0 * costheta * costheta * costheta * costheta - 30.0 * costheta * costheta + 3.0)  # fmt: skip
    pipj = j1.mag * j2.mag
    norm = 1 / np.square(ak.sum(jets.energy, axis=-1))
    h0 = normalize(ak.sum(pipj * p0, axis=-1) * norm)
    h1 = normalize(ak.sum(pipj * p1, axis=-1) * norm)
    h2 = normalize(ak.sum(pipj * p2, axis=-1) * norm)
    h3 = normalize(ak.sum(pipj * p3, axis=-1) * norm)
    h4 = normalize(ak.sum(pipj * p4, axis=-1) * norm)
    return h0, h1, h2, h3, h4


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


def normalize(arr, dtype=np.float32, fill=np.nan, pad=None, clip=None):
    if dtype is not None:
        arr = ak.values_astype(arr, dtype)
    if pad is not None:
        assert clip is None or isinstance(clip, bool)
        arr = ak.pad_none(arr, pad, clip=bool(clip), axis=-1)
        clip = pad is True
    if clip is True:
        arr = arr[..., 0]
    elif clip is not False and isinstance(clip, int):
        arr = arr[..., :clip]
    if fill is not None:
        if callable(dtype):
            fill = dtype(fill)
        arr = ak.fill_none(arr, fill, axis=-1)
    return ak.to_numpy(arr)


def where_mul(cond, true, false):
    return cond * true + ~cond * false


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


# https://github.com/HEP-KBFI/hh-bbww/commit/739fd138e89013ad625ec03c50557bbb4c892337
class Trigger:
    def __init__(self, HLT, trigger, dataset):
        assert not isinstance(dataset.aux.get("channels", None), str)
        self.HLT = HLT
        self.trigger = trigger
        self.dataset = dataset

    @property
    def run(self):
        return self.dataset.aux["run"] if self.dataset.is_data else None

    @property
    def channels(self):
        return self.dataset.aux["channels"] if self.dataset.is_data else None

    def _has_channel(self, channel):
        return self.dataset.is_mc or channel in self.channels

    @staticmethod
    def _in_range(value, range):
        if value is None or range is all:
            return True
        if "-" in range:
            if isinstance(range, str):
                value = value.lower()
                range = range.lower()
            low, high = range.split("-")
            return low <= value <= high
        else:
            return value == range

    def _get(self, channel):
        ch = channel
        tr = self.trigger[ch]
        if isinstance(tr, dict):
            tr = [name for name, range in tr.items() if self._in_range(self.run, range)]
        return nano_mask_or(self.HLT, tr, skipable=True)

    def get(self, *channels):
        ret = False
        for i, ch in enumerate(channels):
            if not self._has_channel(ch):
                continue
            if self.dataset.is_data and i:
                vetos = set(channels[:i]) - set(self.channels)
            else:
                vetos = ()
            good = self._get(ch)
            for veto in vetos:
                good = good & ~self._get(veto)
            ret = ret | good
        return ret


def nthlargest(arr, n=1):
    max = -np.inf
    for i in range(n):
        max = arr.max()
        arr = arr[arr < max]
    return max


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
