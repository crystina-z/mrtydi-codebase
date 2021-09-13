import os
import numpy as np

ANSERINI_DIR = os.environ["ANSERINI_DIR"]
index = f"{ANSERINI_DIR}/target/appassembler/bin/IndexCollection"
search = f"{ANSERINI_DIR}/target/appassembler/bin/SearchCollection"


lang2abbr = {
    "english": "en",
    "finnish": "fi",
    "japanese": "ja",
    "thai": "th",
    "russian": "ru",
    "arabic": "ar",
    "bengali": "bn",
    "indonesian": "id",
    "korean": "ko",
    "swahili": "sw",
    "telugu": "te",
}
optimize = "recip_rank"

collection_type = "JsonCollection"
topicreader = "TsvString"

# parameters
hits = 1000
k1_s = [float("%.2f" % v) for v in np.arange(0.1, 1.6, 0.1)]
b_s = [float("%.2f" % v) for v in np.arange(0.1, 1.0, 0.1)]
fb_terms = [5, 10, 20, 40] 
fb_docs = [1, 2, 5, 10] 
ori_weights = [float("%.2f" % v) for v in np.arange(0.1, 1.0, 0.2)]
