# get lora info decide if it's 2.x or 1.x lora, rename if it's 2x to 2x-
# This code is based off code thanks to cloneofsimo and kohya

# run in directory, and renames/adds 1x or 2x to filename
 
import torch
from safetensors.torch import load_file, safe_open
from library import model_util
import re
import pprint
import sys
import os
from pathlib import Path

def load_state_dict(file_name, dtype):
  if model_util.is_safetensors(file_name):
    sd = load_file(file_name)
    with safe_open(file_name, framework="pt") as f:
      metadata = f.metadata()
  else:
    sd = torch.load(file_name, map_location='cpu')
    metadata = None

  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)

  return sd, metadata

def get_info(filepath):

  def str_to_dtype(p):
    if p == 'float':
      return torch.float
    if p == 'fp16':
      return torch.float16
    if p == 'bf16':
      return torch.bfloat16
    return None

  dtype = str_to_dtype('float')  # matmul method above only seems to work in float32

  lora_sd, metadata = load_state_dict(filepath, dtype)
  count1 = 0
  count2 = 0
  for key, value in lora_sd.items():
      dim = str(value.size())
      if re.search('1024', dim):
            #print("2.x model")
            count2 += 1
      if re.search('768', dim):
            #print("1.x model")
            count1 += 1
            return 1   # quicky shortcut
  if count2 and not count1:
      return 2
  elif count1 > count2 and count2 < 10:
      return 1
  else:
      print(f"This is weird, unsure, total count of 1024 = {count2} vs count of 768 = {count1}")
      return 1

for file in Path('.').iterdir():
    # Check if the file is a regular file
    if file.is_file():
        # if it's a textual inversion embedding
        if file.suffix in [".ckpt", ".pt", ".safetensors"]:
            if re.match("1x", file.stem) or re.match("2x", file.stem):
                  print(f"{file.stem} is already named correctly")
                  continue
            print(f"opening {file.stem}")
            model_type = get_info(file)
            if model_type == 1:
               if re.match("1x", file.stem):
                  print(f"{file.stem} is a Stable Diffusion 1.x Lora - named correctly")
               else:
                  print(f"Stable Diffusion 1.x Lora - will rename to 1x{file.stem}")
                  newname = file.with_stem("1x" + file.stem)
                  # Move the file to the new name
                  file.rename(newname)
            if model_type == 2:
               if re.match("2x", file.stem):
                  print(f"{file.stem} is a Stable Diffusion 2.x Lora - named correctly")
               else:
                  print(f"Stable Diffusion 2.x Lora - will rename to 2x{file.stem}")
                  newname = file.with_stem("2x" + file.stem)
                  # Move the file to the new name
                  file.rename(newname)
