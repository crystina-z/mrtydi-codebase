import sys
from datasets import load_dataset

lang = sys.argv[1]

dataset = load_dataset(
    # 'hgf_mrtydi.py',
    'hgf_mrtydi_corpus.py',
    lang
)
# pdb.set_trace()
