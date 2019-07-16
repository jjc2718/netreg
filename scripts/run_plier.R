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
        warning(paste0("Subsetting data matrix to remove rows without measurements: ",
                       data_dim[1], " -> ", length(inds)))
        data <- data[inds, ]
        data_dim <- dim(data)
    }
    data
}

run_plier <- function(args) {

    if (args$verbose) {
        cat('Loading and preprocessing data...\n')
    }
    data <- read.csv(args$data, sep='\t', header=T, row.names=1, check.names=F)
    processed_data <- process_data(data)
    processed_data <- as.matrix(processed_data)
    pathways <- read.csv(args$pathway_file, sep='\t', header=T, row.names=1)
    pathways <- as.matrix(pathways)

    if (args$verbose) {
        cat('Running PLIER...\n')
    }

    # MSigDB canonical pathways data (see PLIER paper), with gene
    # symbols mapped to Entrez IDs (see preprocessing notebook)
    plierResult <- PLIER(processed_data, pathways, k=args$k, seed=args$seed)
    write.table(plierResult$Z, file=args$output_file, quote=F, sep='\t')
}

main <- function() {
    parser <- ArgumentParser(description='Script to run PLIER on TCGA data')
    parser$add_argument('--data', required=T)
    parser$add_argument('--k', type='integer', required=T)
    parser$add_argument('--output_file', required=T)
    parser$add_argument('--pathway_file',
                        default='data/pathway_data/canonical_mapped.tsv')
    parser$add_argument('--seed', type='integer', required=T)
    parser$add_argument('--verbose', action='store_true')
    args <- parser$parse_args()
    run_plier(args)
}

main()
