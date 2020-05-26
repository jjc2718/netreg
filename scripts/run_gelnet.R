suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(gelnet))

run_gelnet <- function(args) {

    # read data/labels from input files
    # TODO: no row names/col names for now (if this ends up getting
    # extensive use, should implement to resolve network edges)
    X_train <- as.matrix(read.csv(args$train_data, sep='\t', header=F))
    X_test <- as.matrix(read.csv(args$test_data, sep='\t', header=F))
    y_train <- as.matrix(read.csv(args$train_labels, header=F))
    y_test <- as.matrix(read.csv(args$test_labels, header=F))

    # convert edge list to adjacency matrix
    G <- read_graph(args$network_file, format="ncol")
    G.X <- adj2nlapl(as_adjacency_matrix(G, sparse=F))
    # copy feature names from training data to adjacency matrix
    rownames(G.X) <- colnames(X_train)
    colnames(G.X) <- colnames(X_train)

    # train on training data
    fit <- gelnet(X=X_train,
                  y=y_train,
                  P=G.X,
                  l1=args$l1_penalty,
                  l2=args$network_penalty,
                  max.iter=args$num_epochs)

    # make predictions on test data
    y_pred_train <- X_train %*% fit$w + fit$b
    y_pred_test <- X_test %*% fit$w + fit$b
    coefs = c(fit$b, fit$w)

    # write predictions and coefficients to results_dir
    write.table(format(round(y_pred_train, 5), nsmall=5),
                file=paste0(args$results_dir, '/r_preds_train_n',
                            args$num_samples, '_p', args$num_features,
                            '_e', args$noise_stdev, '_u', args$uncorr_frac,
                            '_s', args$seed, '.txt'),
                quote=F, sep='\t', row.names=F, col.names=F)
    write.table(format(round(y_pred_test, 5), nsmall=5),
                file=paste0(args$results_dir, '/r_preds_test_n',
                            args$num_samples, '_p', args$num_features,
                            '_e', args$noise_stdev, '_u', args$uncorr_frac,
                            '_s', args$seed, '.txt'),
                quote=F, sep='\t', row.names=F, col.names=F)
    write.table(format(round(coefs, 5), nsmall=5),
                file=paste0(args$results_dir, '/r_coefs_n',
                            args$num_samples, '_p', args$num_features,
                            '_e', args$noise_stdev, '_u', args$uncorr_frac,
                            '_s', args$seed, '.txt'),
                quote=F, sep='\t', row.names=F, col.names=F)
}

main <- function() {
    parser <- ArgumentParser(description='Script to run gelnet R package')

    # input data
    parser$add_argument('--train_data', required=T)
    parser$add_argument('--train_labels', required=T)
    parser$add_argument('--test_data', required=T)
    parser$add_argument('--test_labels', required=T)

    # input parameters, only used to construct output filename
    parser$add_argument('--network_file', required=T)
    parser$add_argument('--num_samples', required=T)
    parser$add_argument('--num_features', required=T)
    parser$add_argument('--noise_stdev', required=T)
    parser$add_argument('--uncorr_frac', required=T)
    parser$add_argument('--results_dir', required=T)
    parser$add_argument('--seed', type='integer', required=T)
    parser$add_argument('--verbose', action='store_true')

    # parameters for network-regularized regression model
    parser$add_argument('--l1_penalty', type='double', default=1)
    parser$add_argument('--network_penalty', type='double', default=1)
    parser$add_argument('--num_epochs', type='integer', default=100)

    args <- parser$parse_args()
    run_gelnet(args)
}

main()
