1. To prepare open-retrieval data from the TyDi primary task data and Wikipedia dump:
```
data_dir=/path/to/data 
cd prep_data_from_tydi && sh prepare_dataset.sh $data_dir && cd ..
```

2. To prepare Anserini Index and run BM25
```
open_retrieval_dataset_dir="$data_dir/open-retrieval"
# index and search
sh bm25/run_all.sh $open_retrieval_dataset_dir 

# evaluation
sh bm25/run_all.sh $open_retrieval_dataset_dir eval     # trec_eval
sh bm25/run_all.sh $open_retrieval_dataset_dir eval -J  # trec_eval -J      
```