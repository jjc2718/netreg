l1_pens=( "0" "0.01" "0.05" "0.1" "0.2" "0.5" )

for l1 in ${l1_pens[@]}; do
    cmd="python pytorch_vs_sklearn.py --gene TP53 --v --gpu --seed 1 --b 50 --learning_rate 0.00005 --num_epochs 200 --l1_penalty $l1"
    echo $cmd
    eval $cmd
done
