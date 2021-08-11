tydi_dir=$1
tydi_has_pos_label_dir="${tydi_dir}/with_answer"
tydi_has_no_pos_label_dir="${tydi_dir}/without_answer"

if [ ! -d $tydi_dir/with_answer ]; then 
	python get_lines_with_answers.py $tydi_dir 
fi

for set_name in 'train' 'dev'
do	
	for lang in thai swahili telugu finnish bengali russian japanese arabic indonesian korean english
	do
		if [ ! -f "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.${lang}.jsonl" ]; then
			echo "Extracting ${lang}..." 
			cat "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.jsonl" | grep  "\"language\":\"${lang}\"" > "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.${lang}.jsonl" 
		fi
		if [ ! -f "${tydi_has_no_pos_label_dir}/tydiqa-v1.0-${set_name}.${lang}.jsonl" ]; then
			echo "Extracting ${lang}..." 
			cat "${tydi_has_no_pos_label_dir}/tydiqa-v1.0-${set_name}.jsonl" | grep  "\"language\":\"${lang}\"" > "${tydi_has_no_pos_label_dir}/tydiqa-v1.0-${set_name}.${lang}.jsonl" 
		fi
	done
done
