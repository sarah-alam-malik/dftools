import numpy as np

__all__ = ["rebin", "merge"]

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
