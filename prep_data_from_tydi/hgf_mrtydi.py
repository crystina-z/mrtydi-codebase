# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import json

import datasets
from dataclasses import dataclass

_CITATION = '''
@article{mrtydi,
      title={{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval}, 
      author={Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
      year={2021},
      journal={arXiv:2108.08787},
}
'''

languages = [
    'arabic',
    'bengali',
    'english',
    'indonesian',
    'finnish',
    'korean',
    'russian',
    'swahili',
    'telugu',
    'thai',
    'japanese',
]

_DESCRIPTION = 'dataset load script for Mr. TyDi'

_DATASET_URLS = {
    lang: {
        # 'train': f'https://huggingface.co/datasets/tevatron/mrtydi/resolve/main/{lang}/train.jsonl.gz',
        # 'dev': f'https://huggingface.co/datasets/tevatron/mrtydi/resolve/main/{lang}/dev.jsonl.gz',
        # 'test': f'https://huggingface.co/datasets/tevatron/mrtydi/resolve/main/{lang}/test.jsonl.gz',
        'train': f'/store/scratch/x978zhan/mr-tydi/v1.1/hgf-format-dataset/mrtydi-v1.1-{lang}/train.jsonl.gz',
        'dev': f'/store/scratch/x978zhan/mr-tydi/v1.1/hgf-format-dataset/mrtydi-v1.1-{lang}/dev.jsonl.gz',
        'test': f'/store/scratch/x978zhan/mr-tydi/v1.1/hgf-format-dataset/mrtydi-v1.1-{lang}/test.jsonl.gz',
    } for lang in languages
}


class MrTyDi(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            version=datasets.Version('1.1.0'),
            name=lang, description=f'Mr TyDi dataset in language {lang}.'
        ) for lang in languages
    ]

    def _info(self):
        features = datasets.Features({
            'query_id': datasets.Value('string'),
            'query': datasets.Value('string'),

            'positive_passages': [{
                'docid': datasets.Value('string'),
                'text': datasets.Value('string'), 'title': datasets.Value('string')
            }],
            'negative_passages': [{
                'docid': datasets.Value('string'),
                'text': datasets.Value('string'), 'title': datasets.Value('string'),
            }],
        })

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage='https://github.com/castorini/mr.tydi',
            # License for the dataset if available
            license='',
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        lang = self.config.name
        downloaded_files = dl_manager.download_and_extract(_DATASET_URLS[lang])

        splits = [
            datasets.SplitGenerator(
                name='train',
                gen_kwargs={
                    'filepath': downloaded_files['train'],
                },
            ),
            datasets.SplitGenerator(
                name='dev',
                gen_kwargs={
                    'filepath': downloaded_files['dev'],
                },
            ),
            datasets.SplitGenerator(
                name='test',
                gen_kwargs={
                    'filepath': downloaded_files['test'],
                },
            ),
        ]
        return splits

    def _generate_examples(self, filepath):
        with open(filepath) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                for feature in ['negative_passages', 'positive_passages']:
                    if data.get(feature) is None:
                        data[feature] = []

                yield data['query_id'], data
