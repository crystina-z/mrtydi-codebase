lang=$1
anserini_dir="/home/x978zhan/task-xling/anserini" 

alias index="${anserini_dir}/target/appassembler/bin/IndexCollection"
alias search="${anserini_dir}/target/appassembler/bin/SearchCollection"

# root_dir="/scratch/czhang/xling/data/tydi/open-retrieval/${lang}"
root_dir="/home/x978zhan/task-xling/dataset/tydi/open-retrieval/${lang}"

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
    "bengali")
        lang_abbr="bn"
    ;;
    "indonesian")
        lang_abbr="in"
    ;;
    "korean")
        lang_abbr="ko"
    ;;
    *)
        echo "Unknown language: $lang"
        exit
    ;;
esac 
echo $lang $lang_abbr

# output directories
collection_dir="${root_dir}/collection"
collection_flie="${root_dir}/collection/collection.txt"
index_path="${root_dir}/index/lucene-index.pos+docvectors+raw"
runfile_dir="${root_dir}/runfiles"

for dir in $collection_dir $(dirname $index_path) $runfile_dir 
do
    mkdir -p $dir
done

# prepare collection
if [ ! -f $collection_flie ]; then    
    cp "${root_dir}/collection.txt" $collection_flie 
fi


# index 
collection_type=TrecCollection
if [ ! -d $index_path ]; then
    index -collection $collection_type \
    -input $collection_dir \
    -index $index_path \
    -generator DefaultLuceneDocumentGenerator \
    -threads 16 -storePositions -storeDocvectors -storeRaw -language $lang_abbr
fi


# search (train and dev) 
k1=0.9
b=0.4
hits=1000
topicreader="TsvString"

for set_name in "train" "dev"
do
    topic_fnG="${root_dir}/topics.${set_name}.tsv"
    runfile="${runfile_dir}/bm25.${set_name}.k1=$k1.b=$b.txt"

    if [ ! -f $runfile ]; then
        search -index $index_path \
            -topics $topic_fn -topicreader $topicreader \
            -output $runfile -bm25 -threads 16 -language $lang_abbr -hits $hits
    fi

    # evaluate
    qrels_fn="${root_dir}/qrels.${set_name}.txt"
    trec_eval $qrels_fn $runfile
done

Beyourself0_!@#$%^&*()
