import sys
# import pdb
from datasets import load_dataset

lang = sys.argv[1]

dataset = load_dataset(
    'hgf_mrtydi.py', 
    # 'arabic'
    # 'english'
    lang
)
# pdb.set_trace()
