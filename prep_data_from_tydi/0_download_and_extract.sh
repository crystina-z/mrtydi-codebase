for lang in th sw te 'fi' bn ru ja ar id ko en 
do
	if [ $lang = "th" ]; then
		date="20190101"
	else
		date="20190201"
	fi

	bz_fn="${lang}wiki-$date-pages-articles-multistream.xml.bz2" 
	unbz_fn="${lang}wiki-$date-pages-articles-multistream.xml" 
	wiki_json="${wiki_dir}/${lang}wiki.$date.json"

	if [ ! -f $wiki_json ]; then
		echo "preparing $wiki_json"

		if [ ! -f $unbz_fn ]; then
			if [ ! -f $bz_fn ]; then
				wget "https://archive.org/download/${lang}wiki-$date/$bz_fn"
			fi
			bzip2 -d $bz_fn 
		fi

		python wikIR/wikiextractor/WikiExtractor.py ${wiki_xml} --links --quiet --json \
			--output - --bytes 10G > "${wiki_json}" 
	fi
done
