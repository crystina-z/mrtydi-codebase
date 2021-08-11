mr_tydi_dir="../../dataset/open-retrieval"
output_dir="$mr_tydi_dir/../dpr_inputs"
mkdir -p $output_dir

for lang in arabic bengali english finnish indonesian japanese korean russian swahili telugu thai 
do
        echo $lang
        python tools/generate_dpr_json.py $mr_tydi_dir $output_dir $lang
done
