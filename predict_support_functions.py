'''
Supporting functions for the predict.py scripts

'''
#### import packages

import torch
from torchvision import models

from PIL import Image
from torchvision import transforms

from create_classifier import model_classifier, initialize_model

import json

#### functions:

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # load image
    im = Image.open(image)
    
    # transforms:
    process_transforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    
    tensor_image = process_transforms(im)
    
    return tensor_image  


def load_checkpoint(filepath, device):
    
    # load checkpoint:
    if device == "cuda":
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location = device)
        
    # load pre-trained model:
    #model = eval("models.{}(pretrained=True)".format(checkpoint['base_model']))
    model, first_classifier_input = initialize_model(checkpoint['base_model'])
    
    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False
    
    # re-build untrained classifier
    model.classifier = model_classifier(first_classifier_input = first_classifier_input, 
                                    hidden_units = checkpoint['hidden_units'],
                                    output_classes = checkpoint['n_output_classes'])
   
    # load model state:
    model.load_state_dict(checkpoint['state_dict'])
    model.classes_dictionary = checkpoint['classes_dictionary']

    return model


def predict(image_path, model, topk, device, category_names):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # process the image:
    processed_image = process_image(image_path)
    
    ## run a forward pass in the model:
    
    # load the model from checkpoint
    model = load_checkpoint(model, device)
    
    # Put elements in the device
    model.to(device)
    processed_image =  processed_image.to(device)
    # processed image is a tensor 3 x 224 x 224, but model expects:
    # 1 x 3 x 224 x 224, 
    # because it used to be 64 x 3 224 x 224 where 64 is the batch size:
    processed_image.unsqueeze_(0)

    # Get the class probabilities
    model.eval()
    with torch.no_grad():
        logps = model(processed_image)
        ps = torch.exp(logps)

    top_p, top_class = ps.topk(topk, dim=1)
    
    # move to cpu
    top_p = top_p.cpu()
    top_class = top_class.cpu()
    
    # convert to numpy:
    top_p = top_p.squeeze_(0).numpy()
    top_class = top_class.squeeze_(0).numpy()
    
    ## convert classes to strings:
    if category_names == "":
        top_class_names = [str(i) + ": (unknown class name)" for i in top_class]
        
    else:
        full_dictionary = parse_dictionary(category_names, model.classes_dictionary)
        top_class_names = [full_dictionary[i] for i in top_class]
        
        
    
    return top_p, top_class, top_class_names

def parse_dictionary(json_dictionary, classes_dictionary):
    

    with open(json_dictionary, 'r') as f:
        cat_to_name = json.load(f)
    
    ## read and prep mapping from classes to idx:
    inv_classes_dictionary = inv_map = {v: k for k, v in classes_dictionary.items()}
    full_dictionary = inv_classes_dictionary.copy()

    for item in inv_classes_dictionary.keys():
        full_dictionary[item] = cat_to_name[inv_classes_dictionary[item]]
        
    return full_dictionary