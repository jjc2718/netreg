"""
Module to perform fuzzy mapping of symbols to Entrez IDs.

Uses MyGene for search of gene symbol aliases:
http://docs.mygene.info/projects/mygene-py/en/latest/

"""
import time
import mygene
import numpy as np
import pandas as pd

def filter_query_result(df, entrezgene=False):
    """Get the total number of result genes from a MyGene query."""
    unmatched_genes = df.index.values

    # first filter for notfound
    try:
        matched_df = df[df['notfound'].isnull()]
    except KeyError:
        matched_df = df

    # then filter for not null entrezgene, if applicable, since some symbols
    # may have a NaN entrezgene
    # (why the DB distinguishes this case from symbols that are not found,
    # I don't know)
    if entrezgene:
        try:
            matched_df = matched_df[matched_df['entrezgene'].notnull()]
        except KeyError:
            # this shouldn't ever happen, but if it does nothing matches
            matched_df = pd.DataFrame(columns=df.columns)

    num_matched_genes = len(matched_df)
    matched_genes = np.unique(matched_df.index.values)
    unmatched_genes = list(set(unmatched_genes) - set(matched_genes))

    return matched_df, matched_genes, unmatched_genes

def query_to_map(df_query, target_name, map_to_lists=False):
    """Convert results of a MyGene query to a dict."""
    query_map = {}
    if 'notfound' not in df_query:
        df_query['notfound'] = np.nan
    for query, row in df_query[df_query['notfound'].isnull()].iterrows():
        if map_to_lists:
            if query in query_map:
                query_map[query].append(row[target_name])
            else:
                query_map[query] = [row[target_name]]
        else:
            query_map[query] = row[target_name]
    return query_map

def invert_list_map(orig_map):
    """Invert a dict mapping keys to lists.

    The result is a dict as follows:

    For each key: [v1, v2, ..., vN] pair in the original dict, the inverted
    dict will have elements
    {v1: key, v2: key, ..., vN: key}.

    Warning: may cause unexpected behavior on a dict containing non-list
    values.
    """
    inverse_map = {}
    for k, v in orig_map.items():
        for vi in v:
            inverse_map[vi] = k
    return inverse_map


def map_loc_genes(gene_list):
    """Map gene names beginning with 'LOC'.

    See https://www.biostars.org/p/129299/ : these are genes with no
    published symbol, and thus have the format 'LOC' + Entrez ID.
    """
    gene_map = {}
    unmatched = []
    for gene in gene_list:
        if gene.startswith('LOC'):
            gene_map[gene] = gene.replace('LOC', '')
        else:
            unmatched.append(gene)
    return gene_map, unmatched

def fill_na(symbols_map, symbols_list):
    """Fill symbol map with 'N/A' for unmapped symbols."""
    filled_map = symbols_map.copy()
    for s in symbols_list:
        if s not in filled_map:
            filled_map[s] = 'N/A'
    return filled_map

def get_list_duplicates(in_list):
    """Identify duplicates in a list."""
    seen = set()
    duplicates = set()
    for item in in_list:
        if item in seen and item != 'N/A':
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)

def symbol_to_entrez_id(symbols_list, verbose=False, sleep_time=5):
    """Map a list of gene symbols to Entrez IDs.

    Uses the MyGene API to query first for exact symbol/Entrez ID mappings,
    then queries the same API for aliases of unmatched symbols and finds
    mappings for the aliases.

    Parameters
    ----------
    symbols_list : list of str
        List of symbols to map.

    verbose : bool, default=False
        Whether or not to print information about progress/output.

    sleep_time : int, default=5
        How many seconds to sleep between calls to the MyGene API.

    Returns
    -------
    symbol_map : (dict of str: str)
        Maps symbols to Entrez IDs. Unidentified symbols will map
        to the string 'N/A'.
    """
    mg = mygene.MyGeneInfo()

    if verbose:
        print('Querying for exact matches:')

    # first query for exact matches
    df_exact = mg.querymany(symbols_list,
                            scopes='symbol',
                            fields='entrezgene',
                            species='human',
                            verbose=False,
                            as_dataframe=True)

    # some symbols may have higher confidence results for
    # Ensembl keys, so sorting by entrezgene and dropping
    # duplicates keeps the result with an Entrez id
    try:
        df_exact.sort_values(by='entrezgene', inplace=True)
        df_exact = df_exact.loc[~df_exact.index.duplicated(keep='first')]
        df_exact, matched, unmatched = filter_query_result(df_exact,
                                                           entrezgene=True)
        symbol_map = query_to_map(df_exact, 'entrezgene')
    except KeyError:
        symbol_map = {}


    if verbose:
        print('-- Matched {} of {} genes'.format(
            len(matched), len(matched) + len(unmatched)))

    if len(unmatched) == 0:
        return symbol_map

    if verbose:
        print('Trying to manually map unmapped genes:')

    unmatched_before = len(unmatched)
    loc_map, unmatched = map_loc_genes(unmatched)
    symbol_map = {**symbol_map, **loc_map}

    if verbose:
        print('-- Matched {} of {} genes'.format(
            unmatched_before - len(unmatched), unmatched_before))

    if len(unmatched) == 0:
        return symbol_map

    time.sleep(sleep_time)

    if verbose:
        print('Querying MyGene for aliases of {} unmatched genes:'.format(
            len(unmatched)))

    # then query for aliases of unmatched symbols
    df_alias = mg.querymany(unmatched,
                            scopes='alias',
                            fields='symbol',
                            species='human',
                            verbose=False,
                            as_dataframe=True)

    # get rid of rows where the alias has already been matched
    df_alias = df_alias.loc[~df_alias['symbol'].isin(matched)]
    df_alias = df_alias.loc[~df_alias['_id'].isin(list(symbol_map.values()))]

    # duplicates are sorted in order of MyGene confidence score,
    # so keep the most confident and drop others
    #
    # TODO: maybe revisit this and try to keep genes that match
    # with TCGA data?
    df_alias = df_alias.loc[~df_alias.index.duplicated(keep='first')]
    df_alias = df_alias.loc[~df_alias['symbol'].duplicated(keep='first')]

    df_alias, matched, _ = filter_query_result(df_alias)

    if verbose:
        print('-- Found aliases for {} of {} genes'.format(
            len(matched), len(unmatched)))

    alias_map = query_to_map(df_alias, 'symbol', map_to_lists=True)
    inverse_alias_map = invert_list_map(alias_map)

    time.sleep(sleep_time)

    if verbose:
        print('Querying for alias entrez IDs:')

    # and get entrez IDs of aliases
    flat_aliases = [i for sl in alias_map.values() for i in sl]
    df_inexact = mg.querymany(flat_aliases,
                              scopes='symbol',
                              fields='entrezgene',
                              species='human',
                              verbose=False,
                              as_dataframe=True)
    try:
        df_inexact, matched, unmatched = filter_query_result(df_inexact,
                                                             entrezgene=True)
        inexact_map = query_to_map(df_inexact, 'entrezgene')
        inexact_map = {inverse_alias_map[k]: v
                        for k, v in inexact_map.items()}
        symbol_map = fill_na({**symbol_map, **inexact_map}, symbols_list)
    except KeyError:
        # keep symbol map the same if no entrez genes found
        pass

    if verbose:
        print('-- Matched {} of {} genes'.format(
            len(matched), len(matched) + len(unmatched)))

    if verbose:
        eids = list(symbol_map.values())
        total_count = len(eids)
        na_count = eids.count('N/A')
        duplicates = get_list_duplicates(eids)
        print('RESULTS: matched {} of {} genes ({} duplicate Entrez IDs)'.format(
            total_count - na_count, total_count, len(duplicates)))

    return symbol_map

if __name__ == '__main__':
    print(symbol_to_entrez_id(['BRAF', 'ILT10', 'LILRP2'], verbose=True))


