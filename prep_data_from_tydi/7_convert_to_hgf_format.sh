open_retrieval_dir=$1

for lang in  arabic bengali english finnish indonesian japanese korean russian telugu thai swahili 
do
    echo $lang
    python 7_convert_to_hgf_format.py $open_retrieval_dir  $lang
done
