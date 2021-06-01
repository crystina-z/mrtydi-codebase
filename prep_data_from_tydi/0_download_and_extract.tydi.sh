tydi_dir=$1
if [ -z $tydi_dir ]; then
	echo "Requried to provide directory path to store the TyDi dataset"
	echo "Example: sh 0_download_and_extract.tydi.sh /path/to/tydi_dir"
	exit
fi

mkdir -p $tydi_dir

for set_name in "train" "dev"
do
    gz_fn="tydiqa-v1.0-${set_name}.jsonl.gz"
    json_fn="tydiqa-v1.0-${set_name}.jsonl.gz"
    wget "https://storage.googleapis.com/tydiqa/v1.0/${gz_fn}" -P $tydi_dir 
    gzip -cd $gz_fn > $json_fn 
done