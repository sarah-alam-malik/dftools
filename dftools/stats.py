import ROOT
ROOT.gROOT.SetBatch(True)

import numpy as np
import scipy.stats

__all__ = [
    "fc_poisson_efficiency_interval",
    "poisson_interval",
]

def fc_poisson_efficiency_interval(total, passed, level=0.68):
    up = np.zeros(total.shape, dtype='float')
    down = np.zeros(total.shape, dtype='float')
    for idx, (tot, pas) in enumerate(zip(total, passed)):
        up[idx] = ROOT.TEfficiency.FeldmanCousins(tot, pas, level, True)
        down[idx] = ROOT.TEfficiency.FeldmanCousins(tot, pas, level, False)
    return down, up

def poisson_interval(counts, scale=1, level=0.68):
    if np.isscalar(counts):
        _counts = np.array([counts])
    elif isinstance(counts, list):
        _counts = np.array(counts)
    else:
        _counts = counts

    l = scipy.stats.gamma.interval(level, _counts, scale=scale)[0]
    l[_counts==0] = 0.
    u = scipy.stats.gamma.interval(level, _counts+1, scale=scale)[1]

    ret = (l, u)
    if np.isscalar(counts):
        ret = (l[0], u[0])
    return ret
