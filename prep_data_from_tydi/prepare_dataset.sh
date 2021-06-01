wiki_dir="/scratch/czhang/xling/data/tydi/raw"
# wiki_dir="/home/czhang/src/xling/data/raw" 


# for lang in ar bn en fr id ja ko ru te tl sw th
# for lang in id ja ko ru te tl sw th
for lang in th ## fr 
do
	wiki_xml="${wiki_dir}/${lang}wiki-20190201-pages-articles-multistream.xml"
	wiki_json="${wiki_dir}/${lang}wiki.20190201.json"
	# url="https://archive.org/download/thwiki-20190101/thwiki-20190101-pages-articles-multistream.xml.bz2
	if [ ! -f $wiki_json ]; then
		echo "preparing $wiki_json"
		python wikIR/wikiextractor/WikiExtractor.py ${wiki_xml} --links --quiet --json \
			--output - \
			--bytes 10G > "${wiki_json}" 
	fi
done

python passage_level_dataset.py \
	--tydi_dir "/scratch/czhang/xling/data/tydi/tydi_official/has_pos_label" \
	--wiki_dir $wiki_dir \
	--output_dir "/scratch/czhang/xling/data/tydi/open-retrieval"
	# --output_dir "/scratch/czhang/xling/data/tydi/open-retrieval"
exit


python main.py \
	--tydi_dir "/scratch/czhang/xling/data/tydi/tydi_official" \
	--wiki_dir $wiki_dir \
	--output_dir "/scratch/czhang/xling/data/tydi/open-retrieval"
