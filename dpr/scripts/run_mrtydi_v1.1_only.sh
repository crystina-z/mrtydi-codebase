set -e

base_model="mbert-cased"  # or mbert-uncaed

# dpr_output_dir="models/dpr_outputs/nq"
dpr_output_dir="models/dpr_outputs/mytydi-v1.1.mbert-cased.validation"

echo "Converting to dpr format to hf format"
sh scripts/2_convert_weights.sh models/dpr_outputs/mytydi-v1.1.mbert-cased.validation/dpr_biencoder.2.17735 && hf_format_model_dir="$dpr_output_dir/hf_format" && exit
hf_format_model_dir="$dpr_output_dir/hf_format"

# link the base model file to $hf_format_model_dir
template_dir=$(realpath "models/hf-models/${base_model}-template")
echo $template_dir
if [ ! -d $template_dir ]; then
    echo "Cannot find ${template_dir}. Exit"
    exit
fi

for encoder in "context" "question"; do
    subdir="mdpr-${encoder}-encoder"
    for fn in "config.json" "tokenizer.json" "tokenizer_config.json" "vocab.txt"; do
        if [ ! -f "${hf_format_model_dir}/${subdir}/$fn" ]; then
	    echo "${hf_format_model_dir}/${subdir}..."
            ln -s ${template_dir}/${subdir}/* "${hf_format_model_dir}/${subdir}"
        fi
    done
done


echo "Start index and searching" 
sh scripts/3_dindex_and_dsearch.sh $hf_format_model_dir $@
