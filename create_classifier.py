'''
This python script shares the creation of the classifier to ensure that
both train and predict use the same classifier architecture
'''

from torch import nn
from torchvision import models

# the function below is adapted from:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def initialize_model(model_name, use_pretrained=True):
    # Initialize model and get parameter needed for the (untrained classifier)

    model_ft = None
    num_ftrs = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features


    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, num_ftrs

def model_classifier(first_classifier_input, hidden_units, output_classes):
    classifier = nn.Sequential(nn.Linear(first_classifier_input, hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 256),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(256, output_classes),
                             nn.LogSoftmax(dim=1))
    
    return classifier