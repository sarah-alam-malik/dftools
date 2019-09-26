import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from .stats import poisson_interval

__all__ = [
    "cms_label", "legend_data_mc", "data_mc", "data", "mc", "heatmap",
    "annotate_heatmap",
    "process_names", "process_colours",
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

    down, up = poisson_interval(df["count"], scale=df["sum_w"]/df["count"])
    ax.errorbar(
        bin_cents, df["sum_w"], yerr=[df["sum_w"]-down, up-df["sum_w"]],
        **kwargs,
    )

def mc(ax, df, label, bins, mcstat=False, mc_kw={}, mcstat_kw={}, proc_kw={}):
    bin_edges, bin_cents = bin_lows_to_edges_cents(bins)

    # preprocess mc
    tdf = pd.pivot_table(
        df, index=label, columns="parent",
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
            df, index=label, columns="parent",
            values="sum_ww", aggfunc=np.sum,
        )
        neff = tdf**2 / tdf_ww
        down, up = poisson_interval(neff, scale=tdf/neff)
        kwargs = dict(color='black', alpha=0.2)
        kwargs.update(mcstat_kw)
        up = up[:,0]
        down = down[:,0]
        ax.fill_between(
            bin_edges, list(up)+[up[-1]], list(down)+[down[-1]],
            step='post', **kwargs
        )

def data_mc(
    ax, df_data, df_mc, label, bins, blind=False, log=True, legend=True,
    ratio=True, sm_total=True, mcstat_top=False, add_ratios=True, mc_kw={},
    mcstat_kw={}, sm_kw={}, data_kw={}, proc_kw={}, legend_kw={}, cms_kw={},
):
    if blind:
        df_data = df_data*0.

    # preprocessing
    df_mc_sum = df_mc.groupby(label).sum()
    df_mc_sum.loc[:,"parent"] = "SMTotal"
    df_mc_sum = df_mc_sum.groupby(["parent", label]).sum()

    # draw
    if log:
        ax[0].set_yscale('log')

    bin_edges, _ = bin_lows_to_edges_cents(bins)
    ax[0].set_xlim(bin_edges.min(), bin_edges.max())

    # MC - top panel
    mc_kw_ = dict(stacked=True)
    mc_kw_.update(mc_kw)
    mc(
        ax[0], df_mc, label, bins, mcstat=mcstat_top, mc_kw=mc_kw_,
        proc_kw=proc_kw,
    )

    # SM total - top panel
    if sm_total:
        mc_kw_ = dict(histtype='step')
        mc_kw_.update(sm_kw)
        mcstat_kw_ = dict(label="", color="black", alpha=0.2)
        mcstat_kw_.update(mcstat_kw)
        mc(
            ax[0], df_mc_sum, label, bins, mcstat=False, mc_kw=mc_kw_,
            mcstat_kw=mcstat_kw_, proc_kw=proc_kw,
        )

    # Data - top panel
    data(ax[0], df_data, label, bins, data_kw=data_kw)

    # CMS label - top panel
    kwargs = dict(label="Preliminary", lumi=35.9, energy=13)
    kwargs.update(cms_kw)
    #cms_label(ax[0], **kwargs)

    # SM total ratio - bottom panel
    df_mc_sum_ratio = df_mc_sum.copy()
    #df_mc_sum_ratio.loc[:,"count"] = df_mc_sum["count"]
    df_mc_sum_ratio.loc[:,"sum_w"] = 1.
    df_mc_sum_ratio.loc[:,"sum_ww"] = df_mc_sum["sum_ww"]/df_mc_sum["sum_w"]**2

    if ratio:
        mc_kw_ = dict(label="", histtype='step')
        mc_kw_.update(sm_kw)
        mcstat_kw_ = dict(label="MC stat. unc.", color="black", alpha=0.2)
        mcstat_kw_.update(mcstat_kw)
        mc(
            ax[1], df_mc_sum_ratio, label, bins, mcstat=True, mc_kw=mc_kw_,
            mcstat_kw=mcstat_kw_, proc_kw=proc_kw,
        )

        # Data ratio - bottom panel
        if not blind:
            kwargs = dict(data_kw)
            kwargs["label"] = ""
            df_data_ratio = df_data.copy()
            #df_data_ratio.loc[:,"count"] = df_data["count"]
            df_data_ratio.loc[:,"sum_w"] = df_data["sum_w"]/df_mc_sum["sum_w"].values
            df_data_ratio.loc[:,"sum_ww"] = df_data["sum_ww"]/df_mc_sum["sum_w"].values**2
            data(ax[1], df_data_ratio, label, bins, data_kw=kwargs)

        if legend:
            offaxis = legend_kw.pop("offaxis", True)
            kwargs = dict(labelspacing=0.05)
            kwargs.update(legend_kw)
            legend_data_mc(
                ax, df_data, df_mc, label, add_ratios=add_ratios,
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
    "Minor":            "Minors",
}
