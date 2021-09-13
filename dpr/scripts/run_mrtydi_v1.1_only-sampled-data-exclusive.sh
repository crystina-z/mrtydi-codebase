set -e

base_model="mbert-cased"  # or mbert-uncaed

# dpr_output_dir="models/dpr_outputs/nq"
# dpr_output_dir="models/dpr_outputs/mytydi-v1.1.mbert-cased.validation"
# dpr_output_dir="models/dpr_outputs/mytydi-v1.1.mbert-cased.validation.sampled-data"
exc_lang=arabic
# exc_lang=bengali
# exc_lang=english
# exc_lang=finnish
# exc_lang=indonesian
# exc_lang=japanese
# exc_lang=korean
# exc_lang=russian
# exc_lang=swahili
# exc_lang=telugu
# exc_lang=thai

dpr_output_dir="models/dpr_outputs/mytydi-v1.1.mbert-cased.validation.sampled-data.no-$exc_lang"

echo "Converting to dpr format to hf format"
# sh scripts/2_convert_weights.sh $dpr_output_dir/dpr_biencoder.2.4000 && exit 
hf_format_model_dir="$dpr_output_dir/hf_format"
mkdir -p $hf_format_model_dir

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
	    echo $encoder ": ${hf_format_model_dir}/${subdir}..."
	    mkdir -p "${hf_format_model_dir}/${subdir}"
            ln -s ${template_dir}/${subdir}/* "${hf_format_model_dir}/${subdir}"
        fi
    done
done


echo "Start index and searching" 
sh scripts/3_dindex_and_dsearch.sh $hf_format_model_dir $@
