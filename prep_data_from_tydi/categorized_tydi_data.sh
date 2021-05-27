tydi_dir=$1
tydi_has_pos_label_dir="${tydi_dir}/has_pos_label"
tydi_has_no_pos_label_dir="${tydi_dir}/no_pos_label"

mkdir -p $tydi_has_pos_label_dir
mkdir -p $tydi_has_no_pos_label_dir

for set_name in 'train' 'dev'
do	
	
	cat "${tydi_dir}/tydiqa-v1.0-${set_name}.jsonl" | grep -v '"passage_answer":{"candidate_index":-1},"yes_no_answer":"NONE"}' > "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.jsonl"
	cat "${tydi_dir}/tydiqa-v1.0-${set_name}.jsonl" | grep '"passage_answer":{"candidate_index":-1},"yes_no_answer":"NONE"}' > "${tydi_has_no_pos_label_dir}/tydiqa-v1.0-${set_name}.jsonl"

	wc -l "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.jsonl"
	wc -l "${tydi_has_no_pos_label_dir}/tydiqa-v1.0-${set_name}.jsonl"

	for lang in thai swahili telugu finnish bengali russian japanese arabic indonesian korean english
	do
		echo ">> $lang <<"
		cat "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.jsonl" | grep  "\"language\":\"${lang}\"" > "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.${lang}.jsonl" 
		wc -l "${tydi_has_pos_label_dir}/tydiqa-v1.0-${set_name}.${lang}.jsonl" 
	done
done
