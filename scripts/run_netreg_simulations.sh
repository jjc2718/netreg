n_subsets=( "100" "200" "500" "1000" "2000" "5000" )
num_feats=10

for n in "${n_subsets[@]}"; do
    cmd="python netreg_benchmark.py --gene TP53 --sample_num_genes ${n} --expression_file data/tcga_train_sim_subset_p${num_feats}.tsv --results_dir testing/p${num_feats}_n${n} --param_search --network_file data/networks/tcga_top_genes_w_p${num_feats}.tsv"
    echo "Running: $cmd"
    eval $cmd
    cmd="python netreg_benchmark.py --gene TP53 --sample_num_genes ${n} --expression_file data/tcga_train_sim_subset_p${num_feats}.tsv --results_dir testing/p${num_feats}_n${n}_random --param_search --network_file data/networks/tcga_top_genes_random_p${num_feats}.tsv"
    echo "Running: $cmd"
    eval $cmd
    cmd="python netreg_benchmark.py --gene TP53 --sample_num_genes ${n} --expression_file data/tcga_train_sim_subset_p${num_feats}.tsv --results_dir testing/p${num_feats}_n${n}_nn --param_search"
    echo "Running: $cmd"
    eval $cmd
done

cmd="python netreg_benchmark.py --gene TP53 --expression_file data/tcga_train_sim_subset_p${num_feats}.tsv --results_dir testing/p${num_feats} --param_search --network_file data/networks/tcga_top_genes_w_p${num_feats}.tsv"
echo "Running: $cmd"
eval $cmd
cmd="python netreg_benchmark.py --gene TP53 --expression_file data/tcga_train_sim_subset_p${num_feats}.tsv --results_dir testing/p${num_feats}_random --param_search --network_file data/networks/tcga_top_genes_random_p${num_feats}.tsv"
echo "Running: $cmd"
eval $cmd
cmd="python netreg_benchmark.py --gene TP53 --sample_num_genes ${n} --expression_file data/tcga_train_sim_subset_p${num_feats}.tsv --results_dir testing/p${num_feats}_n${n} --param_search"
echo "Running: $cmd"
eval $cmd

