

import torch
import safetensors
import pprint
import sys
import os
from pathlib import Path

# this code mostly lifted from invoke-ai, thanks for heavy lifting, guys.
def parse_embedding_pt(embedding_file):
        embedding_ckpt = torch.load(embedding_file, map_location='cpu')
        embedding_info = {}

        # Check if valid embedding file
        if 'string_to_token' and 'string_to_param' in embedding_ckpt:

            # Catch variants that do not have the expected keys or values.
            try:
                embedding_info['name'] = embedding_ckpt['name'] or os.path.basename(os.path.splitext(embedding_file)[0])

                # Check num of embeddings and warn user only the first will be used
                embedding_info['num_of_embeddings'] = len(embedding_ckpt["string_to_token"])
                if embedding_info['num_of_embeddings'] > 1:
                    print('>> More than 1 embedding found. Will use the first one')

                embedding = list(embedding_ckpt['string_to_param'].values())[0]
            except (AttributeError,KeyError):
                return handle_broken_pt_variants(embedding_ckpt, embedding_file)

            embedding_info['embedding'] = embedding
            embedding_info['num_vectors_per_token'] = embedding.size()[0]
            embedding_info['token_dim'] = embedding.size()[1]

            try:
                embedding_info['trained_steps'] = embedding_ckpt['step']
                embedding_info['trained_model_name'] = embedding_ckpt['sd_checkpoint_name']
                embedding_info['trained_model_checksum'] = embedding_ckpt['sd_checkpoint']
            except AttributeError:
                print(">> No Training Details Found. Passing ...")

        # .pt files found at https://cyberes.github.io/stable-diffusion-textual-inversion-models/
        # They are actually .bin files
        elif len(embedding_ckpt.keys())==1:
            print('>> Detected .bin file masquerading as .pt file')
            embedding_info = parse_embedding_bin(embedding_file)

        else:
            print('>> Invalid embedding format')
            embedding_info = None

        return embedding_info

def parse_embedding_bin(embedding_file):
        embedding_ckpt = torch.load(embedding_file, map_location='cpu')
        embedding_info = {}

        if list(embedding_ckpt.keys()) == 0:
            print(">> Invalid concepts file")
            embedding_info = None
        else:
            for token in list(embedding_ckpt.keys()):
                embedding_info['name'] = token or os.path.basename(os.path.splitext(embedding_file)[0])
                embedding_info['embedding'] = embedding_ckpt[token]
                embedding_info['num_vectors_per_token'] = 1 # All Concepts seem to default to 1
                embedding_info['token_dim'] = embedding_info['embedding'].size()[0]

        return embedding_info

def handle_broken_pt_variants(embedding_ckpt:dict, embedding_file:str)->dict:
        '''
        This handles the broken .pt file variants. We only know of one at present.
        '''
        embedding_info = {}
        if isinstance(list(embedding_ckpt['string_to_token'].values())[0],torch.Tensor):
            print(f'>> Variant Embedding Detected. Parsing: {embedding_file}') 
            # example at https://github.com/invoke-ai/InvokeAI/issues/1829
            token = list(embedding_ckpt['string_to_token'].keys())[0]
            embedding_info['name'] = os.path.basename(os.path.splitext(embedding_file)[0])
            embedding_info['embedding'] = embedding_ckpt['string_to_param'].state_dict()[token]
            embedding_info['num_vectors_per_token'] = embedding_info['embedding'].shape[0]
            embedding_info['token_dim'] = embedding_info['embedding'].size()[0]
        else:
            print('>> Invalid embedding format')
            embedding_info = None

        return embedding_info

# You could make this more single use, but I found that awkward... but left for clarity
# if len(sys.argv) < 2:
#    print("Usage: python determine_stable_diffusion.py <filename>")
#    sys.exit()
# Load the model parameters from the specified .pt file
# filename = sys.argv[1]

# this is the bulk method....
# Loop through all files in the current directory
for file in Path('.').iterdir():
    # Check if the file is a regular file
    if file.is_file():
        # if it's a textual inversion embedding
        if file.suffix in [".pt", ".bin", ".safetensors"]:
            print(f"opening {file.stem}")
            model_params = parse_embedding_pt(file)

            # Pretty print the entire contents of the model_params dictionary
            # pprint.pprint(model_params)

            if model_params["name"] != file.stem:
               print(f"{file.stem} has original name: {model_params['name']}")
            if model_params["token_dim"] == 768:
               print("{file.stem} is a Stable Diffusion 1.x embedding")
            if model_params["token_dim"] == 1024:
               print("{file.stem} is a Stable Diffusion 2.x embedding")
               #rename is intentionally commented out, uncomment and adjust for your needs
               # newlocation = file.with_stem("2x" + file.stem)
               # Get the last word of the filename, excluding extension
               # last_word = file.stem.split()[-1]
               # Create the subfolder for 2x TIs
               # if not Path("2xTIs").exists():
               #   Path("2xTIs").mkdir()
               # Move the file to the appropriate subfolder
               # file.rename(newlocation)
