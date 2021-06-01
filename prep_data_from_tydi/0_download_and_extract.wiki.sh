wiki_dir=$1
if [ -z $wiki_dir ]; then
	echo "Requried to provide directory path to store the Wikipedia dataset"
	echo "Example: sh 0_download_and_extract.wiki.sh /path/to/wiki_dir"
	exit
fi

mkdir -p $wiki_dir

for lang in th sw te 'fi' bn ru ja ar id ko en 
do
	if [ $lang = "th" ]; then
		date="20190101"
	else
		date="20190201"
	fi

	bz_name="${lang}wiki-$date-pages-articles-multistream.xml.bz2"
	bz_fn="${wiki_dir}/$bz_name"
	unbz_fn="${wiki_dir}/${lang}wiki-$date-pages-articles-multistream.xml"
	wiki_json="${wiki_dir}/${lang}wiki.$date.json"

	if [ ! -f $wiki_json ]; then
		echo "preparing $wiki_json"

		if [ ! -f $unbz_fn ]; then
			if [ ! -f $bz_fn ]; then
				wget "https://archive.org/download/${lang}wiki-$date/$bz_name" -P $wiki_dir
			fi
			bzip2 -ckd $bz_fn > $unbz_fn
		fi

		python wikIR/wikiextractor/WikiExtractor.py ${unbz_fn} --links --quiet --json \
			--output - --bytes 10G > "${wiki_json}" 
	fi
done
