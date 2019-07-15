"""
Module to perform fuzzy mapping of symbols to Entrez IDs.

Uses MyGene for fuzzy symbol search:
http://docs.mygene.info/projects/mygene-py/en/latest/

"""
import time
import mygene
import numpy as np
import pandas as pd

def query_to_map(df_query, target_name, map_to_lists=False):
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

def invert_map(orig_map):
    inverse_map = {}
    for k, v in orig_map.items():
        for vi in v:
            inverse_map[vi] = k
    return inverse_map

def get_num_genes(df):
    num_genes = len(df)
    try:
        num_matched_genes = len(df[df['notfound'].isnull()])
    except KeyError:
        num_matched_genes = num_genes
    return num_genes, num_matched_genes

def map_loc_genes(gene_list):
    gene_map = {}
    unmatched = []
    for gene in gene_list:
        if gene.startswith('LOC'):
            gene_map[gene] = gene.replace('LOC', '')
        else:
            unmatched.append(gene)
    return gene_map, unmatched

def fill_na(symbols_map, symbols_list):
    filled_map = symbols_map.copy()
    for s in symbols_list:
        if s not in filled_map:
            filled_map[s] = 'N/A'
    return filled_map

def symbol_to_eid(symbols_list, verbose=False, sleep_time=5):
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

    df_exact = df_exact.loc[~df_exact.index.duplicated(keep='first')]
    symbol_map = query_to_map(df_exact, 'entrezgene')

    matched = df_exact[df_exact['notfound'].isnull()]
    unmatched = df_exact[df_exact['notfound'].notnull()].index.values

    loc_map, unmatched = map_loc_genes(unmatched)
    symbol_map = {**symbol_map, **loc_map}

    if verbose:
        num_genes, num_matched_genes = get_num_genes(df_exact)
        print('-- Matched {} of {} genes'.format(
            num_matched_genes + len(list(loc_map.keys())), num_genes))

    if len(unmatched) == 0:
        return symbol_map

    time.sleep(sleep_time)

    if verbose:
        print('Querying for aliases of {} unmatched genes:'.format(
            len(unmatched)))

    # then query for aliases of unmatched symbols
    df_alias = mg.querymany(unmatched,
                            scopes='alias',
                            fields='symbol',
                            species='human',
                            verbose=False,
                            as_dataframe=True)

    # duplicates are sorted in order of MyGene confidence score,
    # so keep the most confident and drop others
    # TODO: maybe revisit this and try to keep genes that match
    # with TCGA data? or may not be worth it
    df_alias = df_alias.loc[~df_alias.index.duplicated(keep='first')]
    df_alias = df_alias.loc[~df_alias['symbol'].duplicated(keep='first')]

    if verbose:
        num_genes, num_matched_genes = get_num_genes(df_alias)
        print('-- Found aliases for {} of {} genes'.format(
            num_matched_genes, len(unmatched)))

    alias_map = query_to_map(df_alias, 'symbol', map_to_lists=True)
    inverse_alias_map = invert_map(alias_map)

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
    df_inexact = df_inexact[df_inexact['entrezgene'].notnull()]

    if verbose:
        num_genes, num_matched_genes = get_num_genes(df_inexact)
        print('-- Matched {} of {} genes'.format(
            num_matched_genes, num_genes))

    inexact_map = query_to_map(df_inexact, 'entrezgene')
    inexact_map = {inverse_alias_map[k]: v
                    for k, v in inexact_map.items()}

    return fill_na({**symbol_map, **inexact_map}, symbols_list)

if __name__ == '__main__':
    # CCDC83 has an exact match, DDX26B has an inexact match
    # test_symbols = ['DDX26B', 'CCDC83']
    df = pd.read_csv('./data/pathway_data/canonical_pathways.tsv',
                     sep='\t')
    # test_symbols = df.index.values[0:500]
    test_symbols = df.index.values
    gene_map = symbol_to_eid(test_symbols, verbose=True)
    for k, v in gene_map.items():
        if v == 'N/A':
            print('{}\t{}'.format(k, v))
    # print(test_symbols)
    # print(symbol_to_eid(test_symbols, verbose=True))


