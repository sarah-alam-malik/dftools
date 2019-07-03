import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .stats import poisson_interval

__all__ = [
    "cms_label", "legend_data_mc", "data_mc", "data", "mc",
]

def cms_label(ax, label, lumi=35.9, energy=13):
    ax.text(
        0, 1, r'$\mathbf{CMS}\ \mathit{'+label+'}$',
        ha='left', va='bottom', transform=ax.transAxes,
    )
    ax.text(
        1, 1, r'${:.1f}\ \mathrm{{fb}}^{{-1}}$ ({:.0f} TeV)'.format(lumi, energy),
        ha='right', va='bottom', transform=ax.transAxes,
    )

def legend_data_mc(ax, df_data, df_mc, offaxis=True, legend_kw={}):
    handles, labels = ax[0].get_legend_handles_labels()

    # sort by process total
    tdf_mc = pd.pivot_table(
        df_mc, index="binvar0", columns="parent",
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

def data(df, ax, bins, data_kw={}):
    bin_edges, bin_cents = bin_lows_to_edges_cents(bins)

    # draw
    kwargs = dict(fmt='o', lw=1, color='black', label='Data')
    kwargs.update(data_kw)

    cent = df["sum_w"]
    scale = cent/df["count"]
    scale[df["count"]==0] = 0.
    down, up = poisson_interval(df["count"])
    down *= scale
    up *= scale
    ax.errorbar(bin_cents, cent, yerr=[cent-down, up-cent], **kwargs)

def mc(df, ax, bins, mcstat=False, mc_kw={}, mcstat_kw={}, proc_kw={}):
    bin_edges, bin_cents = bin_lows_to_edges_cents(bins)

    # preprocess mc
    tdf = pd.pivot_table(
        df, index="binvar0", columns="parent",
        values="sum_w", aggfunc=np.sum,
    )

    # sort by process total
    tdf_procsum = tdf.sum(axis=0)
    tdf = tdf[tdf_procsum.sort_values().index]

    # mc
    procs = tdf.columns.to_series()
    kwargs = dict(
        label=procs.replace(proc_kw.get("labels", {})),
        color=procs.apply(lambda x: proc_kw.get("colours", {}).get(x, "blue")),
    )
    kwargs.update(mc_kw)
    ax.hist([bin_cents]*tdf.shape[1], bins=bin_edges, weights=tdf.T, **kwargs)

    if mcstat:
        tdf_ww = pd.pivot_table(
            df, index="binvar0", columns="parent",
            values="sum_ww", aggfunc=np.sum,
        )
        up = (tdf + np.sqrt(tdf_ww)).values.ravel()
        down = (tdf - np.sqrt(tdf_ww)).values.ravel()
        kwargs = dict(color='black', alpha=0.2)
        kwargs.update(mcstat_kw)
        ax.fill_between(
            bin_edges, list(up)+[up[-1]], list(down)+[down[-1]],
            step='post', **kwargs
        )

def data_mc(
    df_data, df_mc, bins, blind=False, log=True, legend=True, subplots_kw={},
    mc_kw={}, mcstat_kw={}, sm_kw={}, data_kw={}, proc_kw={}, legend_kw={},
    cms_kw={},
):
    if blind:
        df_data = df_data*0.

    # preprocessing
    df_mc_sum = df_mc.groupby("binvar0").sum()
    df_mc_sum.loc[:,"parent"] = "SMTotal"
    df_mc_sum = df_mc_sum.groupby(["parent", "binvar0"]).sum()

    # draw
    kwargs = dict(
        figsize=(4.2,5.6), dpi=100,
        nrows=2, ncols=1, sharex=True, sharey=False,
        gridspec_kw=dict(height_ratios=[2,1]),
    )
    kwargs.update(subplots_kw)
    fig, ax = plt.subplots(**kwargs)
    if log:
        ax[0].set_yscale('log')

    bin_edges, _ = bin_lows_to_edges_cents(bins)
    ax[0].set_xlim(bin_edges.min(), bin_edges.max())

    # MC - top panel
    mc_kw_ = dict(stacked=True)
    mc_kw_.update(mc_kw)
    mc(df_mc, ax[0], bins, mc_kw=mc_kw_, proc_kw=proc_kw)

    # SM total - top panel
    mc_kw_ = dict(histtype='step')
    mc_kw_.update(sm_kw)
    mcstat_kw_ = dict(label="", color="black", alpha=0.2)
    mcstat_kw_.update(mcstat_kw)
    mc(
        df_mc_sum, ax[0], bins, mcstat=True, mc_kw=mc_kw_,
        mcstat_kw=mcstat_kw_, proc_kw=proc_kw,
    )

    # Data - top panel
    data(df_data, ax[0], bins, data_kw=data_kw)

    # CMS label - top panel
    kwargs = dict(label="Preliminary", lumi=35.9, energy=13)
    kwargs.update(cms_kw)
    cms_label(ax[0], **kwargs)

    # SM total ratio - bottom panel
    df_mc_sum_ratio = df_mc_sum.copy()
    df_mc_sum_ratio.loc[:,"count"] = df_mc_sum["count"]
    df_mc_sum_ratio.loc[:,"sum_w"] = 1.
    df_mc_sum_ratio.loc[:,"sum_ww"] = df_mc_sum["sum_ww"]/df_mc_sum["sum_w"]**2

    mc_kw_ = dict(label="", histtype='step')
    mc_kw_.update(sm_kw)
    mcstat_kw_ = dict(label="MC stat. unc.", color="black", alpha=0.2)
    mcstat_kw_.update(mcstat_kw)
    mc(
        df_mc_sum_ratio, ax[1], bins, mcstat=True, mc_kw=mc_kw_,
        mcstat_kw=mcstat_kw_, proc_kw=proc_kw,
    )

    # Data ratio - bottom panel
    if not blind:
        kwargs = dict(data_kw)
        kwargs["label"] = ""
        df_data_ratio = df_data.copy()
        df_data_ratio.loc[:,"count"] = df_data["count"]
        df_data_ratio.loc[:,"sum_w"] = df_data["sum_w"]/df_mc_sum["sum_w"].values
        df_data_ratio.loc[:,"sum_ww"] = df_data["sum_ww"]/df_mc_sum["sum_w"].values**2
        data(df_data_ratio, ax[1], bins, data_kw=kwargs)

    if legend:
        offaxis = legend_kw.pop("offaxis", True)
        kwargs = dict(labelspacing=0.05)
        kwargs.update(legend_kw)
        legend_data_mc(ax, df_data, df_mc, offaxis=offaxis, legend_kw=kwargs)

    return fig, ax
