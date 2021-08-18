<!-- This files contains the BM25 scores (`k1=0.9 b=0.4`) of each language. (on dataset `v0.5`) -->

To obtain the BM25 scores in the following Figure:

The following script would run the default BM25 / tuned BM25 / evaluation.
```
sh run_all.sh $data_output_dir/open-retrieval {default | tune | eval}
```

The results should match the value in the following Table:
![image](https://user-images.githubusercontent.com/31640436/129927124-78acf1eb-f269-485b-bc8b-0763c5cfcb9f.png)
