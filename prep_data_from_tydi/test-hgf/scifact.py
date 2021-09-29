# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SciFact Dataset (Retrieval Only)"""

import json

import datasets

_CITATION = """
@inproceedings{Wadden2020FactOF,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  booktitle={EMNLP},
  year={2020},
}
"""

_DESCRIPTION = "dataset load script for SciFact"

_DATASET_URLS = {
    'train': "https://huggingface.co/datasets/Tevatron/scifact/resolve/main/train.jsonl.gz",
    'dev': "https://huggingface.co/datasets/Tevatron/scifact/resolve/main/dev.jsonl.gz",
    'test': "https://huggingface.co/datasets/Tevatron/scifact/resolve/main/test.jsonl.gz",
}


class Scifact(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(version=VERSION,
                               description="SciFact train/dev/test datasets"),
    ]

    def _info(self):
        features = datasets.Features({
            'query_id': datasets.Value('string'),
            'query': datasets.Value('string'),
            'positive_passages': [
                {'docid': datasets.Value('string'), 'text': datasets.Value('string'),
                 'title': datasets.Value('string')}
            ],
            'negative_passages': [
                {'docid': datasets.Value('string'), 'text': datasets.Value('string'),
                 'title': datasets.Value('string')}
            ],
        })

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="",
            # License for the dataset if available
            license="",
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_DATASET_URLS)
        splits = [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name='dev',
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                },
            ),
            datasets.SplitGenerator(
                name='test',
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                },
            ),
        ]
        return splits

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get('negative_passages') is None:
                    data['negative_passages'] = []
                if data.get('positive_passages') is None:
                    data['positive_passages'] = []
                yield data['query_id'], data

