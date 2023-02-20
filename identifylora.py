# get lora info only, decide if it's 2.x or 1.x lora
# This code is based off code thanks to cloneofsimo and kohya

# barebone code, proof of concept - uses awkward single argument.  Much faster to batch, due to torch load time.
# see the rename script for a better use.

import argparse
import os
import torch
from safetensors.torch import load_file, save_file, safe_open
from library import model_util
import re

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

def getinfo(args):

  def str_to_dtype(p):
    if p == 'float':
      return torch.float
    if p == 'fp16':
      return torch.float16
    if p == 'bf16':
      return torch.bfloat16
    return None

  dtype = str_to_dtype('float')  # matmul method above only seems to work in float32

  #print("loading Model...")
  lora_sd, metadata = load_state_dict(args.model, dtype)

  #print(f"Metadata if any: {metadata}")
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
  if count2 and not count1:
      print(f"This is a 2.x lora")
  elif count1 and count2 < 10:
      print(f"This is a 1.x lora")
  else:
      print(f"This is weird, total count of 1024 = {count2} vs count of 768 = {count1}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--model", type=str, default=None,
                      help="LoRA model to get info: ckpt or safetensors file")

  args = parser.parse_args()
  getinfo(args)
