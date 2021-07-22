for lang in thai finnish bengali russian japanese arabic indonesian korean english 
do
	echo $lang
	python tools/generate_dpr_json.py ../../dataset/open-retrieval/ $lang
done
