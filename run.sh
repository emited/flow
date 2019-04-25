run="
    find -name "*.pyc" -delete &&
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$2
    python $1 ${@:3}
    "

printf "$run \n"
eval $run    