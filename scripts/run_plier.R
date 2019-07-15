suppressPackageStartupMessages(library(devtools))
suppressPackageStartupMessages(library(argparse))

# install PLIER from GitHub if not already installed (PLIER
# is not available through Conda)
suppressPackageStartupMessages(install_github('wgmao/PLIER'))
suppressPackageStartupMessages(library(PLIER))

process_data <- function(data) {
    # adapted from code at:
    # https://github.com/gitter-lab/prmf/blob/devel/script/PLIER/PLIER_wrapper.R#L61

    # assume data has more features than observations, and
    # features belong on rows
    data_dim <- dim(data)
    if (data_dim[1] < data_dim[2]) {
        data <- t(data)
        data_dim <- dim(data)
    }

    # PLIER requires that all genes have a measurement, so
    # filter for rows that meet this requirement
    data_rowsums <- rowSums(data)
    inds <- which(data_rowsums != 0)
    if (length(inds) < data_dim[1]) {
        warning(paste0("Subsetting data matrix to remove rows without measurements: ", data_dim[1], " -> ", length(inds)))
        data <- data[inds, ]
        data_dim <- dim(data)
    }
    data
}

resolve_gene_names <- function(canonicalPathways, entrez_mapping) {
    mapping = read.csv(entrez_mapping, header=T, sep='\t')
    gene_symbols = rownames(canonicalPathways)
    gene_eids = sapply(gene_symbols,
                       function(x) mapping[mapping$symbol == x, 1])
    pathways <- setattr(copy(canonicalPathways, 'row.names', gene_eids)
    pathways
}

run_plier <- function(args) {

    if (args$verbose) {
        cat('Loading and preprocessing data...\n')
    }
    data <- read.csv(args$data, header=T, sep='\t', row.names=1)
    processed_data <- process_data(data)

    if (args$verbose) {
        cat('Running PLIER...\n')
    }

    # MSigDB canonical pathways data (see PLIER paper)
    data(canonicalPathways)
    # convert gene symbols to Entrez gene names in pathway data
    pathways = resolve_gene_names(canonicalPathways, args$entrez_mapping)
    plierResult <- PLIER(data, pathways)
}

main <- function() {
    parser <- ArgumentParser(description='Script to run PLIER on TCGA data')
    parser$add_argument('--data', required=T)
    parser$add_argument('--k', type='integer', required=T)
    parser$add_argument('--entrez_mapping', default='data/entrez_mapping.tsv')
    parser$add_argument('--seed', type='integer', required=T)
    parser$add_argument('--verbose', action='store_true')
    args <- parser$parse_args()
    run_plier(args)
}

main()
