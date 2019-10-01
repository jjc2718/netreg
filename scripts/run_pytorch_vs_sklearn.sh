seeds=( "1" "2" "3" "4" "5" )

for seed in ${seeds[@]}; do
    cmd="python pytorch_vs_sklearn.py --param_search --ge TP53 --v --gpu --s $seed"
    echo $cmd
    eval $cmd
done
