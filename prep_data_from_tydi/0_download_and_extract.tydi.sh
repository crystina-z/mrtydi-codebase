tydi_dir=$1
if [ -z $tydi_dir ]; then
	echo "Requried to provide directory path to store the TyDi dataset"
	echo "Example: sh 0_download_and_extract.tydi.sh /path/to/tydi_dir"
	exit
fi

mkdir -p $tydi_dir

for set_name in "train" "dev"
do
    gz_name="tydiqa-v1.0-${set_name}.jsonl.gz"
    gz_fn="${tydi_dir}/${gz_name}"
    json_fn="${tydi_dir}/tydiqa-v1.0-${set_name}.jsonl"

    if [ ! -f "$json_fn" ]; then
        if [ ! -f "$gz_fn" ]; then
            wget "https://storage.googleapis.com/tydiqa/v1.0/${gz_name}" -P $tydi_dir
        fi
        gzip -cd $gz_fn > $json_fn
    fi
done
