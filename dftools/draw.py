import functools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as opt
from .stats import poisson_interval

__all__ = [
    "cms_label", "legend_data_mc", "data_mc", "data", "mc", "heatmap",
    "annotate_heatmap",
    "process_names", "process_colours",
    "impacts", "nllscan",
]

def cms_label(ax, label, lumi=35.9, energy=13, extra_label=""):
    ax.text(
        0, 1, r'$\mathbf{CMS}\ \mathit{'+label+'}$',
        ha='left', va='bottom', transform=ax.transAxes,
    )
    ax.text(
        1, 1, r'${:.1f}\ \mathrm{{fb}}^{{-1}}$ ({:.0f} TeV)'.format(lumi, energy),
        ha='right', va='bottom', transform=ax.transAxes,
    )
    # label on centre top of axes
    ax.text(
        0.5, 1, extra_label,
        ha='center', va='bottom', transform=ax.transAxes,
    )

def legend_data_mc(
    ax, df_data, df_mc, label, add_ratios=True, offaxis=True, legend_kw={},
):
    handles, labels = ax[0].get_legend_handles_labels()

    if add_ratios:
        # sort by process total
        tdf_mc = pd.pivot_table(
            df_mc, index=label, columns="parent",
            values="sum_w", aggfunc=np.sum,
        )
        tdf_mc = tdf_mc[tdf_mc.sum(axis=0).sort_values().index]

        data_idx = labels.index("Data")
        data_label = labels.pop(data_idx)
        labels = (labels+[data_label])[::-1]
        data_handle = handles.pop(data_idx)
        handles = (handles+[data_handle])[::-1]

        df_data_sum = df_data.sum()
        tdf_mc_sum = tdf_mc.sum()

        fractions = [
            df_data_sum["sum_w"]/tdf_mc_sum.sum(), 1.,
        ] + list((tdf_mc_sum / tdf_mc_sum.sum()).values[::-1])
        fraction_labels = [
            "{:.3f} {}".format(fractions[idx], labels[idx])
            for idx in range(len(labels))
        ]
    else:
        handles = handles[::-1]
        fraction_labels = labels[::-1]

    kwargs = dict(legend_kw)
    kwargs_noloc = dict(kwargs)
    kwargs_noloc.pop("loc", None)
    if offaxis:
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax[0].legend(
            handles, fraction_labels, bbox_to_anchor=(1, 1), **kwargs_noloc
        )
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width*0.8, box.height])
    else:
        ax[0].legend(handles, fraction_labels, **kwargs)

    handles, labels = ax[1].get_legend_handles_labels()
    if offaxis:
        ax[1].legend(handles, labels, bbox_to_anchor=(1, 1), **kwargs_noloc)
    else:
        ax[1].legend(handles, labels, **kwargs)

def bin_lows_to_edges_cents(lows):
    edges = np.array(list(lows)+[2*lows[-1]-lows[-2]])
    cents = (edges[:-1] + edges[1:])/2.
    return edges, cents

def data(ax, df, label, bins, data_kw={}):
    bin_edges, bin_cents = bin_lows_to_edges_cents(bins)

    # draw
    kwargs = dict(fmt='o', lw=1, color='black', label='Data')
    kwargs.update(data_kw)

    mask = (df["sum_ww"]==0.)
    neff = df["sum_w"]**2 / df["sum_ww"]
    neff[mask] = 0.

    scale = df["sum_w"]/neff
    scale[mask] = 1.

    down, up = poisson_interval(neff, scale=scale)
    ax.errorbar(
        bin_cents, df["sum_w"], yerr=[df["sum_w"]-down, up-df["sum_w"]],
        **kwargs,
    )

def poisson_interval_with_checks(x, variance):
    down, up = poisson_interval(x**2/variance, scale=variance/x)
    mask = (variance==0.)
    down[mask] = 0.
    up[mask] = np.inf
    return down, up

def mc(
    ax, df, label, bins, mcstat=False, mc_kw={}, mcstat_kw={}, proc_kw={},
    zorder=0, interval_func=poisson_interval_with_checks, sort_by_process=True,
):
    stacked = mc_kw.pop("stacked") if "stacked" in mc_kw else False
    bin_edges, bin_cents = bin_lows_to_edges_cents(bins)

    # preprocess mc
    tdf = pd.pivot_table(
        df, index=label, columns="parent",
        values="sum_w", aggfunc=np.sum,
    )

    # sort by process total
    if sort_by_process:
        tdf_procsum = tdf.sum(axis=0)
        tdf = tdf[tdf_procsum.sort_values().index]

    # mc
    procs = tdf.columns.to_series()

    cumsum = tdf.iloc[:,0].copy(deep=True)
    cumsum.values[:] = 0.
    for idx, proc in enumerate(tdf.columns):
        if stacked:
            prev_cumsum = cumsum.copy(deep=True)
            cumsum += tdf[proc]
        else:
            cumsum = tdf[proc]

        color = proc_kw.get("colours", {}).get(proc, "blue")
        kwargs = {
            "color": color, "ec": color,
            "label": proc_kw.get("labels", {}).get(proc, proc),
        }
        kwargs.update(mc_kw)
        if stacked:
            kwargs.update({"ec": color, "lw": 0.1, "zorder": -idx})
        ax.hist(bin_cents, bins=bin_edges, weights=cumsum, **kwargs)

    if mcstat:
        tdf_ww_up = pd.pivot_table(
            df, index=label, columns="parent",
            values="sum_ww_up", aggfunc=np.sum,
        )
        _, up = interval_func(tdf.values[:,0], tdf_ww_up.values[:,0])

        tdf_ww_down = pd.pivot_table(
            df, index=label, columns="parent",
            values="sum_ww_down", aggfunc=np.sum,
        )
        down, _ = interval_func(tdf.values[:,0], tdf_ww_down.values[:,0])

        kwargs = dict(color='black', alpha=0.2)
        kwargs.update(mcstat_kw)

        ax.fill_between(
            bin_edges, list(up)+[list(up)[-1]],
            list(down)+[list(down)[-1]],
            step='post', **kwargs
        )

def data_mc(
    ax, df_data, df_mc, label, bins,
    sigs=[], blind=False, log=True, legend=True, ratio=True, sm_total=True,
    mcstat_top=False, mcstat=True, add_ratios=True, show_zeros=False,
    mc_kw={}, sig_kw={}, mcstat_kw={}, sm_kw={}, data_kw={}, proc_kw={},
    legend_kw={}, cms_kw={}, interval_func=poisson_interval_with_checks,
):
    _df_data = df_data.copy(deep=True)
    _df_mc = df_mc.copy(deep=True)

    if not show_zeros:
        _df_data.loc[_df_data["sum_w"]==0.,"sum_w"] = np.nan

    # only mc sum_ww can be asymmetric
    if "sum_ww_up" not in _df_mc:
        _df_mc["sum_ww_up"] = _df_mc["sum_ww"]
    if "sum_ww_down" not in _df_mc:
        _df_mc["sum_ww_down"] = _df_mc["sum_ww"]

    # collect signals if set
    sigs = sigs[::-1]
    sig_mask = ~_df_mc.index.get_level_values("parent").isin(sigs)
    df_sig = _df_mc.loc[~sig_mask].copy(deep=True)

    df_mc_sm = _df_mc.loc[sig_mask].copy(deep=True)

    # preprocessing
    df_mc_sum = df_mc_sm.groupby(label).sum()
    df_mc_sum.loc[:,"parent"] = "SMTotal"
    df_mc_sum = df_mc_sum.groupby(["parent", label]).sum()

    # draw
    if log:
        ax[0].set_yscale('log')

    bin_edges, _ = bin_lows_to_edges_cents(bins)
    ax[0].set_xlim(bin_edges.min(), bin_edges.max())

    # signals - top panel
    sig_kw_ = dict(histtype='step', zorder=10)
    sig_kw_.update(sig_kw)
    if len(sigs) > 0:
        mc(
            ax[0], df_sig, label, bins, mcstat=False, mc_kw=sig_kw_,
            proc_kw=proc_kw, interval_func=interval_func, sort_by_process=False,
        )

    # MC - top panel
    mc_kw_ = dict(stacked=True)
    mc_kw_.update(mc_kw)
    mc(
        ax[0], df_mc_sm, label, bins, mcstat=False,
        mc_kw=mc_kw_, proc_kw=proc_kw, interval_func=interval_func,
    )

    # SM total - top panel
    if sm_total:
        mc_kw_ = dict(histtype='step')
        mc_kw_.update(sm_kw)
        mcstat_kw_ = dict(label="", color="black", alpha=0.2)
        mcstat_kw_.update(mcstat_kw)
        mc(
            ax[0], df_mc_sum, label, bins, mcstat=mcstat_top, mc_kw=mc_kw_,
            mcstat_kw=mcstat_kw_, proc_kw=proc_kw, interval_func=interval_func,
        )

    # Data - top panel
    if not blind:
        data(ax[0], _df_data, label, bins, data_kw=data_kw)

    # CMS label - top panel
    kwargs = dict(label="Preliminary", lumi=35.9, energy=13)
    kwargs.update(cms_kw)
    #cms_label(ax[0], **kwargs)

    # SM total ratio - bottom panel
    df_mc_sum_ratio = df_mc_sum.copy()
    df_mc_sum_ratio.loc[:,"sum_w"] = 1.
    df_mc_sum_ratio.loc[:,"sum_ww_up"] = (
        df_mc_sum["sum_ww_up"]/df_mc_sum["sum_w"]**2
    )
    df_mc_sum_ratio.loc[:,"sum_ww_down"] = (
        df_mc_sum["sum_ww_down"]/df_mc_sum["sum_w"]**2
    )

    if ratio:
        mc_kw_ = dict(label="", histtype='step')
        mc_kw_.update(sm_kw)
        mcstat_kw_ = dict(label="MC stat. unc.", color="black", alpha=0.2)
        mcstat_kw_.update(mcstat_kw)

        mc(
            ax[1], df_mc_sum_ratio, label, bins, mcstat=mcstat, mc_kw=mc_kw_,
            mcstat_kw=mcstat_kw_, proc_kw=proc_kw, interval_func=interval_func,
        )

        # Data ratio - bottom panel
        if not blind:
            kwargs = dict(data_kw)
            kwargs["label"] = ""
            df_data_ratio = _df_data.copy()
            df_data_ratio.loc[:,"sum_w"] = _df_data["sum_w"]/df_mc_sum["sum_w"].values
            df_data_ratio.loc[:,"sum_ww"] = _df_data["sum_ww"]/df_mc_sum["sum_w"].values**2
            data(ax[1], df_data_ratio, label, bins, data_kw=kwargs)

        if legend:
            offaxis = legend_kw.pop("offaxis", True)
            kwargs = dict(labelspacing=0.05)
            kwargs.update(legend_kw)
            legend_data_mc(
                ax, _df_data, _df_mc, label, add_ratios=add_ratios,
                offaxis=offaxis, legend_kw=kwargs,
            )

    return ax

def heatmap(
    data, row_labels, col_labels, ax, cbar_kw=dict(fraction=0.046, pad=0.04),
    cbarlabel="", grid_kw={}, tick_kw={}, **kwargs,
):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(**tick_kw)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), ha="right", #rotation=-30,
        rotation_mode="anchor",
    )

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    gkw = dict(which="minor", color="w", linestyle='-', linewidth=2)
    gkw.update(grid_kw)
    ax.grid(**gkw)
    ax.tick_params(
        which="minor", bottom=False, left=False, top=False, right=False,
    )
    ax.tick_params(
        which="major", bottom=False, left=False, top=False, right=False,
    )

    return im, cbar

def annotate_heatmap(
    im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"],
    cthreshold=lambda z: True, vthreshold=lambda z: True, **textkw,
):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    kw = dict(ha="center", va="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(cthreshold(data[i, j]))])
            if not vthreshold(data[i, j]):
                continue
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


process_colours = {
    "SMTotal":          "black",
    "MET":              "black",
    "SingleMuon":       "black",
    "SingleElectron":   "black",
    "ZJetsToNuNu":      "#80b1d3",
    "WJetsToLNu":       "#b3de69",
    "WJetsToENu":       "#b2df8a",
    "WJetsToMuNu":      "#b3de69",
    "WJetsToTauNu":     "#8dd3c7",
    "WJetsToTauLNu":    "#8dd3c7",
    "WJetsToTauHNu":    "#8dd3c7",
    "Diboson":          "#fdb462",
    "DYJetsToLL":       "#ffed6f",
    "DYJetsToEE":       "#fff6b3",
    "DYJetsToMuMu":     "#ffed6f",
    "DYJetsToTauTau":   "#ffe41a",
    "DYJetsToTauLTauL": "#ffe41a",
    "DYJetsToTauHTauL": "#ffe41a",
    "DYJetsToTauHTauH": "#ffe41a",
    "EWKV2Jets":        "#bebada",
    "SingleTop":        "#fccde5",
    "TTJets":           "#bc80bd",
    "Top":              "#bc80bd",
    "QCD":              "#fb8072",
    "G1Jet":            "#ccebc5",
    "VGamma":           "#ffffb3",
    "Minor":            "#d9d9d9",
    "MinorBkgs":        "#d9d9d9",
}

process_names = {
    "SMTotal":          "SM total",
    "MET":              "MET",
    "SingleMuon":       "Single Muon",
    "SingleElectron":   "Single Electron",
    "ZJetsToNuNu":      "$Z(\\rightarrow \\nu\\nu)+j$",
    "WJetsToLNu":       "$W(\\rightarrow l\\nu)+j$",
    "WJetsToENu":       "$W(\\rightarrow e\\nu)+j$",
    "WJetsToMuNu":      "$W(\\rightarrow \\mu\\nu)+j$",
    "WJetsToTauNu":     "$W(\\rightarrow \\tau\\nu)+j$",
    "WJetsToTauLNu":    "$W(\\rightarrow \\tau_{l}\\nu)+j$",
    "WJetsToTauHNu":    "$W(\\rightarrow \\tau_{h}\\nu)+j$",
    "Diboson":          "Diboson",
    "DYJetsToLL":       "$Z/\\gamma^{*}(\\rightarrow ll)+j$",
    "DYJetsToEE":       "$Z/\\gamma^{*}(\\rightarrow ee)+j$",
    "DYJetsToMuMu":     "$Z/\\gamma^{*}(\\rightarrow \\mu\\mu)+j$",
    "DYJetsToTauTau":   "$Z/\\gamma^{*}(\\rightarrow \\tau\\tau)+j$",
    "DYJetsToTauLTauL": "$Z/\\gamma^{*}(\\rightarrow \\tau_{l}\\tau_{l})+j$",
    "DYJetsToTauHTauL": "$Z/\\gamma^{*}(\\rightarrow \\tau_{l}\\tau_{h})+j$",
    "DYJetsToTauHTauH": "$Z/\\gamma^{*}(\\rightarrow \\tau_{h}\\tau_{h})+j$",
    "EWKV2Jets":        "VBS",
    "SingleTop":        "Single Top",
    "TTJets":           "$t\\bar{t}+j$",
    "QCD":              "QCD multijet",
    "G1Jet":            "$\\gamma+j$",
    "VGamma":           "$V+\\gamma$",
    "Minor":            "Minor",
    "MinorBkgs":        "Minor",
}

nuisance_names = {
    "d1kqcd": r'$\delta^{(1)}k_{\mathrm{QCD}}$',
    "d2kqcd": r'$\delta^{(2)}k_{\mathrm{QCD}}$',
    "d3kqcd": r'$\delta^{(3)}k_{\mathrm{QCD}}$',
    "d1kew": r'$\delta^{(1)}k_{\mathrm{EW}}$',
    "d2keww": r'$\delta^{(2)}k_{\mathrm{EW}}^{\mathrm{W}}$',
    "d2kewz": r'$\delta^{(2)}k_{\mathrm{EW}}^{\mathrm{Z}}$',
    "d3keww": r'$\delta^{(3)}k_{\mathrm{EW}}^{\mathrm{W}}$',
    "d3kewz": r'$\delta^{(3)}k_{\mathrm{EW}}^{\mathrm{Z}}$',
    "dkmix": r'$\delta k_{\mathrm{mix}}$',
    "jesTotal": r'JES',
    "jerSF": r'JER',
    "unclust": r'Unclustered energy',
    "lhePdfWeight": r'PDF',
    "btagSF": r'$b$-tag veto',
    "photonIdLoose": r'Photon id. veto',
    "photonPixelSeedVeto": r'Photon pixel veto',
    "tauIdTight": r'$\tau_h$-tag id. selection',
    "tauIdVLoose": r'$\tau_h$-tag id. veto',
    "muonIdLooseSyst": r'Muon id. veto (syst.)',
    "muonIdLooseStat": r'Muon id. veto (stat.)',
    "muonIsoLooseSyst": r'Muon iso. veto (syst.)',
    "muonIsoLooseStat": r'Muon iso. veto (stat.)',
    "muonIdTightSyst": r'Muon id. selection (syst.)',
    "muonIdTightStat": r'Muon id. selection (stat.)',
    "muonIsoTightSyst": r'Muon iso. selection (syst.)',
    "muonIsoTightStat": r'Muon iso. selection (stat.)',
    "eleIdIsoVeto": r'Electron id. veto',
    "eleIdIsoTight": r'Electron id. selection',
    "eleReco": r'Electron reconstruction',
    "eleTrig": r'Electron trigger',
    "prefiring": r'ECAL timing',
    "pileup": r'Pileup',
    "lumi": r'Luminosity',
    "metTrig0MuSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($0\mu$)',
    "metTrig1MuSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($1\mu$)',
    "metTrig2MuSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($2\mu$)',
    "metTrigReferenceTriggerSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger (ref.)',
    "metTrigMonojetSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($\p_{\mathrm{T}}^{\mathrm{miss}}+\mathrm{jets}$)',
    "metTrigSingleMuonSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($\mu+\mathrm{jets}$)',
    "metTrigDoubleMuonSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($\mu\mu+\mathrm{jets}$)',
    "metTrigSingleTauSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($\tau_h+\mathrm{jets}$)',
    "metTrigSingleElectronSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($e+\mathrm{jets}$)',
    "metTrigDoubleElectronSyst": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ trigger ($ee+\mathrm{jets}$)',
}

def impacts(data, fig=None, ax=None, converter=nuisance_names):
    if fig is None:
        fig = plt.figure(figsize=(4,4), dpi=150)
    if ax is None:
        ax = fig.subplots(
            ncols=2, nrows=1,
            sharex=False, sharey=True,
            gridspec_kw={"hspace": 0., "wspace": 0.},
        )

    ax[0].minorticks_off()
    ax[1].minorticks_off()
    ax[1].set_yticklabels([])

    y = data["poi_paramdown"].values
    x = np.linspace(0., len(y), len(y)+1)
    ax[1].hist(
        x[:-1], bins=x, weights=y,
        color='#1f78b4', alpha=0.8,
        orientation='horizontal',
        label=r'$-1\sigma$',
    )
    y = data["poi_paramup"].values
    ax[1].hist(
        x[:-1], bins=x, weights=y,
        color='#e31a1c', alpha=0.8,
        orientation='horizontal',
        label=r'$+1\sigma$',
    )
    xmax = np.max(np.abs(ax[1].get_xlim()))
    ax[1].set_xlim(-1.1*xmax, 1.1*xmax)
    ax[1].set_ylim(0, len(y))
    ax[1].axvline(0, lw=1, color='gray', alpha=0.8)

    y = data["param_value"].values
    yerr = (
        -1*data["param_merrdown"].values,
        data["param_merrup"].values,
    )
    ax[0].errorbar(
        y, (x[:-1]+x[1:])/2., xerr=yerr,
        fmt='o', color='black',
        ms=4, capsize=4,
    )
    xmax = data.eval("param_value+param_merrup").max()
    xmax = max(xmax, data.eval("-(param_value+param_merrdown)").max())
    xmax = int(xmax)+1
    ax[0].set_xlim(-xmax, xmax)
    for pos in range(xmax):
        ax[0].axvline(pos, lw=1, color='gray', alpha=0.8)
        ax[0].axvline(-pos, lw=1, color='gray', alpha=0.8)
    ax[0].set_ylim(0, len(y))
    ax[0].set_xticks(np.arange(-(xmax-1), (xmax-1)+0.1, 1.))
    ax[0].set_yticks((x[:-1]+x[1:])/2.)
    labels = [
        converter.get(l, l.replace("_", "\\_"))
        for l in data.index.get_level_values("param").values
    ]
    ax[0].set_yticklabels(labels)
    ax[0].set_xlabel(r'$\theta$')
    ax[1].set_xlabel(r'$\Delta\hat{r}$')
    ax[1].legend(fancybox=True, edgecolor='#d9d9d9')
    return fig, ax

def nllscan(
    x, y, ax=None, marker_kw={}, spline_kw={}, splrep_kw={}, splev_kw={},
    opt_kw={}, root_kw={}, line_kw={}, text_kw={}, nsigs=[1],
    bestfit_guess=[0.], left_bracket=(-np.inf, 0), right_bracket=(0, np.inf),
):
    """
    Helper function to plot a -2*Delta(log(L)) scan from a pd.DataFrame with
    two columns: x variable and y variable (which should hold the
    -2*Delta(log(L)) values.

    Parameters
    ----------
    x : np.ndarray-like
        The input x variable.

    y : np.ndarray-like
        The input y variable. Should hold values of -2*Delta(log(L))

    ax : matplotlib.axes, optional (default=None)
        The axis to draw on.

    marker_kw : dict-like, optional (default={})
        kwargs to pass to ax.plot. Updates a dict with:
        dict(marker='o', ms=2, lw=0., label='Scan', color='#1f78bf')

    spline_kw : dict-like, optional (default={})
        kwargs to pass to ax.plot. Updates a dict with:
        dict(lw=1., label='Spline', color='#e31a1c')

    splrep_kw: dict-like, optional (default={})
        kwargs to pass to scipy.interpolate.splrep. Updates a dict with:
        dict(s=0)

    splev_kw: dict-like, optional (default={})
        kwargs to pass to scipy.interpolate.splev. Updates a dict with:
        dict(der=0)

    opt_kw: dict-like, optional (default={})
        kwargs to pass to scipy.optimize.optimize. Updates a dict with:
        dict(der=0)

    root_kw: dict-like, optional (default={})
        kwargs to pass to scipy.optimize.root_scalar. Updates a dict with:
        dict(method='brentq')

    line_kw: dict-like, optional (default={})
        kwargs to pass to axes.ax?line. Updates a dict with:
        dict(lw=1, ls='--', color='gray')

    text_kw: dict-like, optional (default={})
        kwargs to pass to axes.text. Updates a dict with:
        ict(ha='left', va='bottom', color='gray')

    nsigs : list of floats, optional (default=[1])
        List of number of sigmas to draw on the final plot

    bestfit_guess : list of floats, options (default=[0.])
        Best fit guess of the minimum for scipy.optimize

    left_bracket : tuple of floats, options (default=(-np.inf, 0))
        Guess for left root bracket.

    right_bracket : tuple of floats, options (default=(-np.inf, 0))
        Guess for right root bracket.

    Return
    ------
    pd.DataFrame with columns: nsig and x values
    """
    outdata = []
    if ax is None:
        fig, ax = plt.subplots()

    kw = dict(marker='o', ms=2, lw=0., label='Scan', color='#1f78bf')
    kw.update(marker_kw)
    ax.plot(x, y, **kw)

    # spline
    kw = dict(s=0)
    kw.update(splrep_kw)
    tck = interp.splrep(x, y, **kw)

    kw = dict(der=0)
    kw.update(splev_kw)
    kw["tck"] = tck
    func = functools.partial(interp.splev, **kw)

    xfine = np.linspace(x.min(), x.max(), 201)
    kw = dict(lw=1., label='Spline', color='#e31a1c')
    kw.update(spline_kw)
    ax.plot(xfine, func(xfine), **kw)

    kw = dict(method='L-BFGS-B')
    kw.update(opt_kw)
    bestfit = opt.minimize(func, bestfit_guess, **kw)
    outdata.append({"nsig": 0., "xval": bestfit.x[0]})

    for nsig in nsigs:
        kw = dict(method='brentq')
        kw.update(root_kw)
        kw["bracket"] = left_bracket
        left = opt.root_scalar(lambda x: func(x)-nsig**2, **kw)
        outdata.append({"nsig": nsig, "xval": left.root})

        kw = dict(method='brentq')
        kw.update(root_kw)
        kw["bracket"] = right_bracket
        right = opt.root_scalar(lambda x: func(x)-nsig**2, **kw)
        outdata.append({"nsig": -nsig, "xval": right.root})

        kw = dict(lw=1, ls='--', color='gray')
        kw.update(line_kw)
        ax.plot((left.root, left.root), (0., nsig**2), **kw)
        ax.plot((right.root, right.root), (0., nsig**2), **kw)
        ax.axhline(nsig**2, **kw)

        pos = ax.transData.inverted().transform(
            ax.transAxes.transform((0.025, 1))
        )
        kw = dict(ha='left', va='bottom', color='gray')
        kw.update(text_kw)
        #ax.text(pos[0], nsig**2, f'${nsig}\\sigma$', **kw)

    return pd.DataFrame(outdata)
