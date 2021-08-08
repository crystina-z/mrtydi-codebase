open_retrieval_dir=$1
lang=$2
# anserini_dir="/path/to/anserini"
root_dir="${open_retrieval_dir}/${lang}"
output_dir="${open_retrieval_dir}/.."

alias index="${ANSERINI_DIR}/target/appassembler/bin/IndexCollection"
alias search="${ANSERINI_DIR}/target/appassembler/bin/SearchCollection"

if [ ! -d $root_dir ]; then
	echo "unfound directory $root_dir"
	exit
fi

# convert lang into the anserini version
lang_abbr=""
case $lang in 
    "english")
        lang_abbr="en"
    ;;
    "finnish")
        lang_abbr="fi"
    ;;
    "japanese")
        lang_abbr="ja"
    ;;
    "thai")
        lang_abbr="th"
    ;;
    "russian")
        lang_abbr="ru"
    ;;
    "arabic")
        lang_abbr="ar"
    ;;
    "bengali")
        lang_abbr="bn"
    ;;
    "indonesian")
        lang_abbr="id"
    ;;
    "korean")
        lang_abbr="ko"
    ;;
    "telugu")
        lang_abbr="te"
    ;;
    "swahili")
        lang_abbr="sw"
    ;;
   *)
        echo "Unknown language: $lang"
        exit
    ;;
esac 
echo $lang $lang_abbr

# output directories
collection_dir="${root_dir}/collection"
index_path="${output_dir}/bm25-indexes/${lang}/lucene-index.pos+docvectors+raw"
runfile_dir="${output_dir}/bm25-runfiles/${lang}"

for dir in $collection_dir $(dirname $index_path) $runfile_dir 
do
    mkdir -p $dir
done


# index 
collection_type=JsonCollection
if [ ! -d $index_path ]; then
    cmd="-collection $collection_type -input $collection_dir -index $index_path -generator DefaultLuceneDocumentGenerator -threads 16 -storePositions -storeDocvectors -storeRaw"

    if [ "$lang_abbr" = "te" ] || [ "$lang_abbr" = "sw" ]; then
        index $cmd -pretokenized
    else
        index $cmd -language $lang_abbr
    fi
fi


# search (train and dev) 
hits=1000
topicreader="TsvString"

for set_name in "train" "dev" "test"
do
    # bm25
    topic_fn="${root_dir}/topic.${set_name}.tsv"
    runfile="${runfile_dir}/bm25.${set_name}.default.txt"
    if [ ! -f $runfile ]; then
        cmd="-index $index_path -topics $topic_fn -topicreader $topicreader -output $runfile -bm25 -threads 16 -hits $hits"
	echo "$cmd -language $lang_abbr"

        if [ "$lang_abbr" = "te" ] || [ "$lang_abbr" = "sw" ]; then
            search $cmd -pretokenized
        else
            search $cmd -language $lang_abbr
        fi
    fi

done

