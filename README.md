# Mr. TyDi

[**Download**](#download) |
[**Baselines and Evaluation**](#baselines-and-evaluation) | 
[**Paper**](https://arxiv.org/abs/2108.08787)

## Introduction
Mr. TyDi is a multi-lingual benchmark dataset built on [TyDi](https://arxiv.org/abs/2003.05002), covering 11 typologically diverse languages.
It is designed for mono-lingual retrieval, specifically to evaluate ranking with learned dense representations.

## Download 

1. Dataset (topic, qrels, folds, collections)

      [Arabic](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-arabic.tar.gz)
    | [Bengali](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-bengali.tar.gz)
    | [English](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-english.tar.gz)
    | [Finnish](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-finnish.tar.gz)
    | [Indonesian](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-indonesian.tar.gz)
    | [Japanese](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-japanese.tar.gz)
    | [Korean](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-korean.tar.gz)
    | [Russian](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-russian.tar.gz)
    | [Swahili](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-swahili.tar.gz)
    | [Telugu](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-telugu.tar.gz)
    | [Thai](https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-thai.tar.gz)

2. Pre-build sparse index (for BM25)

      [Arabic]()
    | [Bengali]() 
    | [English]() 
    | [Finnish]()
    | [Indonesian]()
    | [Japanese]() 
    | [Korean]()
    | [Russian]() 
    | [Kiswahili]()
    | [Telugu]()
    | [Thai]()

3. Pre-build dense index (for mDPR)

      [Arabic]()
    | [Bengali]() 
    | [English]() 
    | [Finnish]()
    | [Indonesian]()
    | [Japanese]() 
    | [Korean]()
    | [Russian]() 
    | [Kiswahili]()
    | [Telugu]()
    | [Thai]()

4. Checkpoints

    [mDPR (trained on NQ)]()


## Baselines and Evaluation
1. BM25  (pointer to pyserini)
1. mDPR (pointer to pyserini)


## Citation
If you find our paper useful or use the dataset in your work, please cite our paper and the TyDi paper:
```
@article{mrtydi,
      title={{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval}, 
      author={Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
      year={2021},
      journal={arXiv:2108.08787},
}}
```
```
@article{tydiqa,
    title={{TyDi QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
    author={Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
    year={2020},
    journal={Transactions of the Association for Computational Linguistics}
}
```

## Contact us
If you have any question or suggestions regarding the dataset, code or publication, 
please contact Xinyu Zhang (x978zhan[at]uwaterloo.ca)