import numpy as np
import scipy.stats

__all__ = [
    "fc_poisson_efficiency_interval",
    "poisson_interval",
]

def fc_poisson_efficiency_interval(total, passed, level=0.68):
    import ROOT
    ROOT.gROOT.SetBatch(True)

    up = np.zeros(total.shape, dtype='float')
    down = np.zeros(total.shape, dtype='float')
    for idx, (tot, pas) in enumerate(zip(total, passed)):
        up[idx] = ROOT.TEfficiency.FeldmanCousins(tot, pas, level, True)
        down[idx] = ROOT.TEfficiency.FeldmanCousins(tot, pas, level, False)
    return down, up

def poisson_interval(counts, scale=1, level=0.68):
    """
    Determine the up and down variations for the poisson interval (possibly
    scaled).

    Parameters
    ----------
    counts : ndarray-like
        The counts or effective number of entries in histogrammed bins

    scale : ndarray-like
        Scale to give to the gamma distribution interval for weighted events.
        Should be equivalent to sum_w/neff.

    level : float (default 0.68)
        The coverage interval. Default is 68%, i.e. near 1 sigma.
    """
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
