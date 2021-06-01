The processed dataset could be downloaded [here](https://uofwaterloo-my.sharepoint.com/:f:/g/personal/x978zhan_uwaterloo_ca/Egpbu0I7gIRMlLyIOlI3dyQB02_S5s5iLDIMW3n0jTX_Qg?e=lvp3Ns).
Each `.zip` contains directories: `arabic   english  indonesian  korean   swahili  thai bengali  finnish  japanese    russian  telugu`.

## Prepare open-retrieval data from the TyDi primary task data and Wikipedia dump
Heads-up: the data downloading and processing could take over 20 hours, where the English Wikipedia is the most time-consuming one.
```
data_dir=/path/to/data
cd prep_data_from_tydi && sh prepare_dataset.sh $data_dir && cd ..
```

## To prepare Anserini Index and run BM25
1. Set up Anserini and `trec_eval`, and add them to enviroment variables.
Require the [latest commit](https://github.com/castorini/anserini/commit/a72b65268f54a4cfe63a36918f4ab7ca09b2e7e8) of Anserini.
```
# copied from anserini repo
git clone https://github.com/castorini/anserini.git --recurse-submodules
cd anserini && mvn clean package appassembler:assemble
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../..
cd ..

export ANSERINI_DIR="$pwd/anserini"
trec_eval="$ANSERINI_DIR/tools/eval/trec_eval.9.0.4"
```

2. Then we are ready to run Index, Searching and Evaluation
```
open_retrieval_dataset_dir="$data_dir/open-retrieval"

# index and search
sh bm25/run_all.sh $open_retrieval_dataset_dir 

# evaluation
sh bm25/run_all.sh $open_retrieval_dataset_dir eval     # trec_eval
sh bm25/run_all.sh $open_retrieval_dataset_dir eval -J  # trec_eval -J      
```
