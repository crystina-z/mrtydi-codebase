hf_model_dir=$1
command=$2

if [ ! -d $hf_model_dir ]; then
	echo "Cannot find $hf_model_dir"
fi

echo $hf_model_dir

tmp_sbatch_dir="$hf_model_dir/logs"
mkdir -p $tmp_sbatch_dir
tmp_sbatch_script="$tmp_sbatch_dir/sbatch_script.sh"

hf_model_line="hf_model_dir=$hf_model_dir"

# prepare temporary
if [ "$command" = "sbatch" ]; then
    cat scripts/slurm-header.sh > "$tmp_sbatch_script"
    log_fn="$tmp_sbatch_dir/dindex-and-dsearch.output.log"
    echo $log_fn
    echo "#SBATCH --output $log_fn" >> "$tmp_sbatch_script"
    echo $hf_model_line >> "$tmp_sbatch_script"
    cat scripts/_dindex_and_dsearch.sh >> "$tmp_sbatch_script"
    sbatch "$tmp_sbatch_script"
else
    echo "" > "$tmp_sbatch_script"
    echo $hf_model_line >> "$tmp_sbatch_script"
    cat scripts/_dindex_and_dsearch.sh >> "$tmp_sbatch_script"
    sh "$tmp_sbatch_script"
fi
