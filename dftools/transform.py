import numpy as np

__all__ = ["rebin", "merge"]

def rebin(df, bins, label="binvar0"):
    """
    Take a dataframe with the 'label' as an index and rebin/densify into the
    input bins returning a new dataframe.
    """
    index_names = df.index.names
    tdf = df.reset_index()
    index_arr = bins[np.minimum(bins.searchsorted(tdf[label]), len(bins)-1)]
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
    """Change certain names in an index_level and merge"""
    index_names = df.index.names
    tdf = df.reset_index()
    for label, match_labels in merge_cfg.items():
        tdf.loc[tdf[index_level].isin(match_labels), index_level] = label
    return tdf.groupby(index_names).sum()
