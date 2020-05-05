'''
This python script shares the creation of the classifier to ensure that
both train and predict use the same classifier architecture
'''

from torch import nn

def model_classifier(first_classifier_input, hidden_units):
    classifier = nn.Sequential(nn.Linear(first_classifier_input, hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 256),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(256, 102),
                             nn.LogSoftmax(dim=1))
    
    return classifier