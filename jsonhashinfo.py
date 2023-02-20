# get lora model info from a pile of jsons saved from Civitai
# can be tweaked to do more, like ckpts, TIs, etc, and list more hashes...
# currently is bare minimum to populate list hopefully for Model Keyword to use

import re
from pprint import pprint
import sys
import os
from pathlib import Path
import json

for file in Path('.').iterdir():
    # Check if the file is a regular file
    if file.is_file():
        # if it's a json
        if file.suffix in [".json"]:
            #print(f"opening {file.stem}")
            with open(file) as model_file:
                 model = json.load(model_file)
                 #pprint(model)
                 model_type = model["type"]
                 if model_type == "LORA":
                     for items in model["modelVersions"]:
                        #pprint(items)
                        twords = []
                        for word in items['trainedWords']:
                            if re.match("<lora.*>$",word):
                               # it's not a trigger word
                               pass
                            else:
                               twords.append(word.strip().strip(",").strip())
                        trainedwords = re.sub("\|\|","|", re.sub(", ?","|","|".join(twords)).rstrip("|"))
                        if trainedwords == "":
                           break
                        for fileinfo in items['files']:
                           #if 'AutoV2' in fileinfo['hashes'].keys():
                           #   hash = fileinfo['hashes']['AutoV2']
                           #   print(f"{hash}, {trainedwords}, {file.stem}")
                           if 'AutoV1' in fileinfo['hashes'].keys():
                              hash = fileinfo['hashes']['AutoV1']
                              print(f"{hash}, {trainedwords}, {file.stem}")
