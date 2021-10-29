import json


for lang in "arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai".split():
    fn = f"/GW/carpet/nobackup/czhang/dpr/data/mrtydi/v1.1-delimiter-nn/dpr_inputs/{lang}/train.gcdpr.json"
    lines = json.load(open(fn))
    print(lang, len(lines))
