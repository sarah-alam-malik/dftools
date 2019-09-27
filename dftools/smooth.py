from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

def unispline_to_gausfilter(
    x, y, w=None, spline_kw={"k": 2}, filter_kw={"sigma": 1},
):
    if w is not None:
        spline_kw["w"] = w

    s = UnivariateSpline(x, y, **spline_kw)
    return gaussian_filter1d(s(x), **filter_kw)
