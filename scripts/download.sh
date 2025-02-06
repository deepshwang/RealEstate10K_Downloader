num_splits=${1:-8}
split_idx=${2:-0}
num_threads=${3:-8}

python generate_dataset.py train --split_idx $split_idx --num_splits $num_splits --num_threads $num_threads
python generate_dataset.py test --split_idx $split_idx --num_splits $num_splits --num_threads $num_threads