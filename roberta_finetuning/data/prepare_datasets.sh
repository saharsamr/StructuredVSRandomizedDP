wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar

echo "*** Use GLUE-SST-2 as default SST-2 ***"
mv original/SST-2 original/SST-2-original
mv original/GLUE-SST-2 original/SST-2

echo "*** Done with downloading datasets ***"

cd ..

current_dir=$(pwd)
data_dir="$current_dir/data/original"
output_dir="$current_dir/data"

for K in 16 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python3 roberta_utils/tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K --data_dir $data_dir  --output_dir $output_dir
done
