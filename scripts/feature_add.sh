ns=( "10" "50" "100" "200" "500" "1000" )

OUTPUT_DIR="feature_add_no_l1"
mkdir -p $OUTPUT_DIR

for n in ${ns[@]}; do
    cmd="python feature_add.py --gene TP53 --num_features ${n} --batch_size 20 --l1_penalty 0.0 --learning_rate 0.001 --num_epochs 100 --verbose --results_dir $OUTPUT_DIR"
    echo "Running: $cmd"
    eval $cmd
    cmd="python feature_add.py --gene TP53 --num_features ${n} --batch_size 20 --l1_penalty 0.0 --learning_rate 0.001 --num_epochs 100 --random --verbose --results_dir $OUTPUT_DIR"
    echo "Running: $cmd"
    eval $cmd
done
