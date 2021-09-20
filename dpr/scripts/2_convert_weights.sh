model_path=$1

model_dir=$(dirname $model_path)

echo "Processing $model_path"
python tools/convert_weights.py -q $model_path -c $model_path -o "$model_dir/hf_format"

# example:
# python dpr/tools/convert_weights.py -q GC-DPR/outputs/dpr_biencoder.2.16721 -c GC-DPR/outputs/dpr_biencoder.2.16721 -o GC-DPR/outputs/hf_format
