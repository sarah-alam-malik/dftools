import numpy as np
import pandas as pd

__all__ = [
    "rebin", "merge",
    "analysis_aggcols",
    "analysis_pivot_merge",
    "analysis_pivot_variations",
]

def rebin(df, bins, label="binvar0", underflow=True, overflow=True):
    """
    Take a dataframe with the 'label' as an index and rebin/densify into the
    input bins returning a new dataframe.

    Parameters
    ----------
    df : DataFrame-like
        Dataframe with index `label` which will be rebinned (and densified)
        into a new set of bins.

    bins : ndarray-like
        The bins to rebin into.

    label : str (default, binvar0)
        The index label to rebin. Other indexes are left alone.

    underflow : bool (default, True)
        Include underflow entires in the lowest bin

    overflow : bool (default, True)
        Include overflow entries in the highest bin
    """
    index_names = df.index.names
    tdf = df.reset_index()
    #index_arr = bins[np.minimum(bins.searchsorted(tdf[label]), len(bins)-1)]
    tidx = bins.searchsorted(tdf[label], side='right')
    if underflow:
        tidx = np.maximum(tidx, 1)
    if overflow:
        tidx = np.minimum(tidx, len(bins))
    index_arr = bins[tidx-1]
    tdf.loc[:,label] = index_arr
    tdf = tdf.groupby(index_names).sum()
    index_names = [idx for idx in tdf.index.names if idx!=label]
    if len(index_names)==0:
        ret = tdf.reindex(bins).fillna(0.).groupby(label).sum()
    else:
        ret = (
            tdf.groupby(index_names)
            .apply(
                lambda dfgrp: (
                    dfgrp.reset_index(index_names, drop=True)
                    .reindex(bins).fillna(0.)
                )
            )
        )
    return ret

def merge(df, merge_cfg, index_level="parent"):
    """
    Change certain names in an index_level and merge

    Parameters
    ----------
    df : DataFrame-like
        Dataframe input to change some index levels

    merge_cfg : dict
        Dictionary of new labels with values corresponding to a list of
        index values to change into the new label

    index_level : str (default, parent)
        Index to change
    """
    index_names = df.index.names
    tdf = df.reset_index()
    for label, match_labels in merge_cfg.items():
        tdf.loc[tdf[index_level].isin(match_labels), index_level] = label
    return tdf.groupby(index_names).sum()

def merge_query(df, label_queries, index_level="parent"):
    index_names = df.index.names
    tdf = df.reset_index()
    for label, query in label_queries.items():
        mask = tdf.eval(query)
        tdf.loc[mask, index_level] = label
    return tdf.groupby(index_names).sum()

def analysis_aggcols(
    df, variation="lhePdfWeight", nominal="sum_w_{}", variance="sum_ww_{}",
    aggs=["std"], evals=[
        "sum_w_lhePdfWeightUp = sum_w + agg_std",
        "sum_w_lhePdfWeightDown = sum_w - agg_std",
        "sum_ww_lhePdfWeightUp = sum_ww * (1 + agg_std/sum_w)**2",
        "sum_ww_lhePdfWeightDown = sum_ww * (1 - agg_std/sum_w)**2",
    ], copy=True,
):
    """
    Aggregate across a set of columns and apply some function

    Parameters
    ----------
    df : DataFrame-like
        Dataframe input to process

    variation : str (default, "lhePdfWeight")
        Variation which appears in the columns

    nominal : str (default, "sum_w_{}")
        How the nominal weight column is defined

    variance: str (default, "sum_ww_{}")
        How the variance weight column is defined

    aggs : list (default, ["std"])
        List of aggregation functions to apply. Will add columns for these

    evals : list (default, [...])
        List of evals to generate new columns

    copy : bool (default, True)
        Copy the input dataframe
    """
    columns = [c for c in df.columns if nominal.format(variation) in c]

    tdf = df.copy(deep=True) if copy else df
    for agg in aggs:
        tdf["agg_{}".format(agg)] = tdf.loc[:, columns].agg(agg, axis=1)

    columns.extend([c for c in tdf.columns if variance.format(variation) in c])
    tdf = tdf.drop(columns, axis=1)

    tdf.eval("\n".join(evals), inplace=True)
    return tdf.drop(["agg_{}".format(agg) for agg in aggs], axis=1)

def analysis_pivot_merge(df, df_alt, label='table', new_value="Event"):
    columns = df_alt.columns
    index = df_alt.index.names

    df_piv = pd.pivot_table(
        df_alt, values=columns, index=[c for c in index if c!=label],
        columns=[label], fill_value=0.,
    )
    df_piv.columns = ["_".join(c) for c in df_piv.columns]
    df_piv[label] = new_value
    df_piv = df_piv.set_index("table", append=True)
    df_piv = df_piv.reorder_levels(index)
    return pd.concat([df, df_piv], axis='columns')

def analysis_pivot_variations(df, label="variation"):
    columns = df.columns
    index = df.index.names

    # Change column names to column values
    df.columns.name = label
    df = df.stack().reset_index(label)

    # Add quantity for sum_w/sum_ww and remove from variation
    df["quantity"] = "sum_w"
    df.loc[df[label].str.startswith("sum_ww"), "quantity"] = "sum_ww"
    df.loc[:,label] = (
        df[label]
        .str.replace("sum_ww_", "")
        .str.replace("sum_w_", "")
        .str.replace("sum_ww", "")
        .str.replace("sum_w", "")
    )

    # Make quantity column values in column labels and label into an index
    df = df.set_index([label, "quantity"], append=True).unstack()
    df.columns = [c[1] for c in df.columns]
    return df

def analysis_flat_variations(df, label="variation"):
    def check_inputs(data):
        bad_cols = not all(c in data.columns for c in ["sum_w", "sum_ww"])
        bad_index = not (list(data.index.names[-3:]) == [label, "bin_min", "bin_max"])
        bad_dups = data.index.duplicated().any()

        if bad_cols:
            raise KeyError(
                "('sum_w', 'sum_ww') need to be in the columns. Found "
                "{}".format(data.columns)
            )
        if bad_index:
            raise IndexError(
                "('{}', 'bin_min', 'bin_max') should appear as the last 3 "
                "indexes. Found {}".format(label, data.index.names)
            )
        if bad_dups:
            raise ValueError(
                "Found duplicate indexes: "
                "{}".format(df.loc[df.index.duplicated(),:])
            )
    check_inputs(df)

    # ["region", "process", "variation", "bin_min", "bin_max"]
    index = df.index.names
    idx_variation = index.index(label)

    mask_var = (df.index.get_level_values(label)=="")
    dfn = df.loc[mask_var,:].copy()
    dfv = df.loc[~mask_var,:].copy()

    dfn_sum = dfn.groupby(index[:idx_variation]).sum()
    dfv_sum = dfv.groupby(index[:idx_variation+1]).sum()
    dfv_sum = dfv_sum.divide(dfn_sum)
    dfv_sum.columns = ["scale_w", "scale_ww"]

    dfn = dfn.reset_index(label, drop=True)

    dfv = pd.merge(
        dfv_sum["scale_w"],
        dfn[["sum_w", "sum_ww"]],
        right_index=True, left_index=True,
    ).reorder_levels(index).sort_index()
    dfv.loc[:,"sum_w"] = dfv.eval("scale_w*sum_w")
    dfv.loc[:,"sum_ww"] = dfv.eval("(scale_w**2)*sum_ww")
    dfv = dfv.drop("scale_w", axis=1)

    dfn[label] = ""
    dfn = dfn.set_index(label, append=True).reorder_levels(index).sort_index()
    return pd.concat([dfn, dfv], axis=0, sort=True)

def analysis_smooth_variations(df, smoother, label="variation"):
    def check_inputs(data):
        bad_cols = not all(c in data.columns for c in ["sum_w", "sum_ww"])
        bad_index = not (list(data.index.names[-3:]) == [label, "bin_min", "bin_max"])
        bad_dups = data.index.duplicated().any()

        if bad_cols:
            raise KeyError(
                "('sum_w', 'sum_ww') need to be in the columns. Found "
                "{}".format(data.columns)
            )
        if bad_index:
            raise IndexError(
                "('{}', 'bin_min', 'bin_max') should appear as the last 3 "
                "indexes. Found {}".format(label, data.index.names)
            )
        if bad_dups:
            raise ValueError(
                "Found duplicate indexes: "
                "{}".format(df.loc[df.index.duplicated(),:])
            )
    check_inputs(df)

    # ["region", "process", "variation", "bin_min", "bin_max"]
    index = df.index.names
    idx_variation = index.index(label)

    mask_var = (df.index.get_level_values(label)=="")
    dfv = df[~mask_var].reset_index(index[idx_variation:])
    dfv["updown"] = "Up"
    dfv.loc[dfv[label].str.endswith("Down"), "updown"] = "Down"
    dfv.loc[:, label] = np.where(
        dfv[label].str.endswith("Down"),
        dfv[label].str.slice(0, -4),
        dfv[label].str.slice(0, -2),
    )
    dfv = (
        dfv.set_index(list(index[idx_variation:])+["updown"], append=True)
        .unstack()
    )
    dfv.columns = ["_".join(c) for c in dfv.columns]

    # add nominal
    dfv = pd.merge(
        dfv, df.loc[mask_var,:].reset_index(label, drop=True),
        right_index=True, left_index=True,
    )
    dfv["bin_cent"] = 0.5*(
        dfv.index.get_level_values(index[-2])
        + dfv.index.get_level_values(index[-1])
    )
    dfv["up_norm"] = dfv.eval("sum_w_Up/sum_w")
    dfv["do_norm"] = dfv.eval("sum_w_Down/sum_w")
    dfv["up_err"] = dfv.eval("sqrt(abs(sum_ww_Up - sum_ww_Down))/sum_w")
    dfv["do_err"] = dfv.eval("sqrt(abs(sum_ww_Down - sum_ww))/sum_w")

    # swap values
    mask_nonzero = (dfv["do_err"]==0.)
    dfv.loc[mask_nonzero, "do_err"] = dfv.loc[mask_nonzero, "up_err"].copy()
    dfv.loc[:, "up_err"] = np.maximum(1e-4, np.minimum(1., dfv["up_err"]))
    dfv.loc[:, "do_err"] = np.maximum(1e-4, np.minimum(1., dfv["do_err"]))

    # smooth
    dfv = pd.concat([
        dfv, dfv.groupby(index[:idx_variation+1]).apply(smoother),
    ], axis=1)

    # normalise
    dfv["sum_w_upsmooth"] = dfv.eval("sum_w*upsmooth")
    dfv["sum_w_dosmooth"] = dfv.eval("sum_w*dosmooth")
    dfv_sum = dfv[[
        "sum_w", "sum_w_Up", "sum_w_Down", "sum_w_upsmooth", "sum_w_dosmooth",
    ]].groupby(index[:idx_variation+1]).sum()
    dfv = dfv.reorder_levels(index).sort_index()

    sum_w_up = dfv.eval("sum_w*upsmooth").mul(dfv_sum.eval("sum_w_Up/sum_w_upsmooth"))
    sum_w_do = dfv.eval("sum_w*dosmooth").mul(dfv_sum.eval("sum_w_Down/sum_w_dosmooth"))

    dfv = pd.DataFrame({
        "sum_wUp": sum_w_up.values,
        "sum_wwUp": dfv["sum_ww_Up"].values,
        "sum_wDown": sum_w_do.values,
        "sum_wwDown": dfv["sum_ww_Down"].values,
    }, index=dfv.index)

    dfv.columns.name = "type"
    dfv = dfv.stack().reset_index([label, "type"])
    dfv.loc[:, label] = np.where(
        dfv["type"].str.endswith("Up"), dfv[label]+"Up", dfv[label]+"Down",
    )
    dfv.loc[:, "type"] = np.where(
        dfv["type"].str.endswith("Up"),
        dfv["type"].str.slice(0, -2),
        dfv["type"].str.slice(0, -4),
    )
    dfv = (
        dfv.set_index([label, "type"], append=True)
        .unstack().reorder_levels(index).sort_index()
    )
    dfv.columns = [c[1] for c in dfv.columns]

    return dfv
