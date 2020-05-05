''' 
Python script to predict the class of flower image

usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu]
                  image_path model_checkpoint

Python script to predict the class of a flower image

positional arguments:
  image_path            a path/to/image
  model_checkpoint      name of a model checkpoint

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         number of top-K classes to be returned
  --category_names CATEGORY_NAMES
                        mapping of categories to real names (json file)
  --gpu                 Use GPU for inference
'''

###############
# import packages
import numpy as np
import torch

import argparse

import predict_support_functions as pred

###############################
# parse input arguments

parser = argparse.ArgumentParser(
    description='Python script to predict the class of a flower image',
)

parser.add_argument('image_path',
                    help='a path/to/image')

parser.add_argument('model_checkpoint', 
                    help='name of a model checkpoint')

parser.add_argument('--top_k', type = int, default = 1,
                    help='number of top-K classes to be returned')

parser.add_argument('--category_names', default = "",
                    help='mapping of categories to real names (json file)')

parser.add_argument('--gpu', action='store_true',
                    help='Use GPU for inference')

args = parser.parse_args()

## define  device:
device = "cpu"

if args.gpu:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("Warning! GPU requested but not available. Using CPU")

#########################################
# get predictions

probs, classes, class_names = pred.predict(args.image_path, args.model_checkpoint, args.top_k, device, args.category_names)

#########################################
# Print header output
print("")
if args.top_k == 1:
    print(f'Top class')
else:
    print(f'Top {args.top_k} classes')


#########################################
# Print classes and names if existing, and probs:
for this_class, this_prob in zip(class_names, probs):
    print(f' {this_class}: {round(this_prob*100,2)}%')

