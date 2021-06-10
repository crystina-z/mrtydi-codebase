download collections/indexes/topics/qrels:
```bash
wget https://www.dropbox.com/s/p5o1s9acgxfn77o/xling-data.tar.gz
tar -czvf xling-data.tar.gz
```

### Thai

```bash
DATA_DIR="/<path to data_dir>/data/thai"
```
BM25 Retrieval
```bash
$ python -m pyserini.search --topics ${DATA_DIR}/topics-and-qrels/topic.dev.tsv \
                            --index ${DATA_DIR}/index/lucene-index.pos+docvectors+raw \
                            --output runs/run.thai.dev.bm25.txt \
                            --language th
```

```bash
$ python -m pyserini.eval.trec_eval -mrecip_rank -mmap -mrecall.1000 ${DATA_DIR}/topics-and-qrels/qrels.dev.txt runs/run.thai.dev.bm25.txt
map                     all     0.2883
recip_rank              all     0.3015
recall_1000             all     0.8469
```

DPR Retrieval
```bash
MODEL_DIR="/home/xueguang/chia_temp/xling/models/mdpr-question-encoder/"
```
```bash
$ python -m pyserini.dsearch --topics ${DATA_DIR}/topics-and-qrels/topic.dev.tsv \
                             --index ${DATA_DIR}/dindex/ \
                             --encoder ${MODEL_DIR} \
                             --batch-size 36 \
                             --threads 12 \
                             --output runs/run.thai.dev.mdpr.txt
```

```bash
$ python -m pyserini.eval.trec_eval -mrecip_rank -mmap -mrecall.1000 ${DATA_DIR}/topics-and-qrels/qrels.dev.txt runs/run.thai.dev.mdpr.txt
map                     all     0.0609
recip_rank              all     0.0651
recall_1000             all     0.3898
```

Hybrid Retrieval
```bash
$ python -m pyserini.hsearch dense  --index ${DATA_DIR}/dindex/ \
                                    --encoder ${MODEL_DIR} \
                             sparse --index ${DATA_DIR}/index/lucene-index.pos+docvectors+raw \
                                    --language th \
                             fusion --alpha 3 \
                             run    --topics ${DATA_DIR}/topics-and-qrels/topic.dev.tsv \
                                    --batch-size 36 --threads 12 \
                                    --output runs/run.thai.dev.hybrid.txt
```

```bash
$ python -m pyserini.eval.trec_eval -mrecip_rank -mmap -mrecall.1000 ${DATA_DIR}/topics-and-qrels/qrels.dev.txt runs/run.thai.dev.hybrid.txt
map                     all     0.3272
recip_rank              all     0.3434
recall_1000             all     0.8427
```