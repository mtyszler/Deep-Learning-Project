'''
Python script to train a neural network to predict the class of a flower image

usage: train.py [-h] [--save_directory SAVE_DIRECTORY] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu]
                data_directory

Python script to train a neural network to predict the class of a flower image

positional arguments:
  data_directory        root folder for the data directory, split into train,
                        valid and test folders

optional arguments:
  -h, --help            show this help message and exit
  --save_directory SAVE_DIRECTORY
                        folder to save the model checkpoints
                        (train_checkpoint.pth). Default current folder
  --arch ARCH           Base model architecture. Default is densenet121. Full
                        list at https://pytorch.org/docs/master/torchvision/mo
                        dels.html
  --learning_rate LEARNING_RATE
                        Learning rate. Default = 0.003
  --hidden_units HIDDEN_UNITS
                        Number of hidden_units of the classifier. Default =
                        512
  --epochs EPOCHS       Number of training epochs. Default = 3
  --gpu                 Use GPU for inference
'''
#### import packages

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse

from workspace_utils import keep_awake
from create_classifier import model_classifier

###############################
# parse input arguments

parser = argparse.ArgumentParser(
    description='Python script to train a neural network to predict the class of a flower image',
)

parser.add_argument('data_directory',
                    help='root folder for the data directory, split into train, valid and test folders')

parser.add_argument('--save_directory', default = "", 
                    help='folder to save the model checkpoints (train_checkpoint.pth). Default current folder')


parser.add_argument('--arch', default = 'densenet121',
                    help='Base model architecture. Default is densenet121. Full list at https://pytorch.org/docs/master/torchvision/models.html')

parser.add_argument('--learning_rate', default = 0.003, type = float,
                    help='Learning rate. Default = 0.003')

parser.add_argument('--hidden_units', default = 512, type = int,
                    help='Number of hidden_units of the classifier. Default = 512')

parser.add_argument('--epochs', default = 3, type = int,
                    help='Number of training epochs. Default = 3')

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

# define saving name:
filename = "training_checkpoint.pth"
full_filename = args.save_directory + filename
####################################
# load data:

# directories:
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

# transforms:
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

valid_transforms = test_transforms

# map each folder:
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data =  datasets.ImageFolder(valid_dir, transform=valid_transforms)

# dataloaders:
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

####################################
# load pre-trained model:
model = eval("models.{}(pretrained=True)".format(args.arch))

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False


####################################
# create a new classifier
n_output_classes = len(train_data.class_to_idx)
model.classifier = model_classifier(first_classifier_input = model.classifier.in_features, 
                                    hidden_units = args.hidden_units,
                                    output_classes = n_output_classes)
 
####################################
# Prep Training

# Train the classifier layers using backpropagation using the pre-trained network to get the features
# Track the loss and accuracy on the validation set to determine the best hyperparameters

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# move model to selected device
model.to(device);

##################################################################################
# Train:

# Train the classifier layers using backpropagation using the pre-trained network to get the features
# Track the loss and accuracy on the validation set to determine the best hyperparameters

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
save_every = 50

print("Training started")

for epoch in keep_awake(range(epochs)):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                   
            print(f"Step {steps}.. "
                  f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader)*100}%")
            
            running_loss = 0
            model.train()
            
        # save current checkpoint:    
        if steps % save_every == 0:
            
            checkpoint = {
              'base_model': args.arch,
              'hidden_units': args.hidden_units,
              'n_output_classes': n_output_classes,
              'steps': steps, 
              'validation accuracy': accuracy,              
              'optimizer': optimizer.state_dict(),
              'criterion': criterion,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'classes_dictionary': train_data.class_to_idx
            }
            
            torch.save(checkpoint, full_filename)
    
    print("This traiing epoch is finished")
    
print("Training is finished")

## Save after training is finished
checkpoint = {
  'base_model': args.arch,
  'hidden_units': args.hidden_units,
  'n_output_classes': n_output_classes,
  'steps': steps, 
  'validation accuracy': accuracy,              
  'optimizer': optimizer.state_dict(),
  'criterion': criterion,
  'state_dict': model.state_dict(),
  'epochs': epochs,
  'classes_dictionary': train_data.class_to_idx
}

torch.save(checkpoint, full_filename)

###############################################################
## Test accuracy:

test_loss = 0
accuracy = 0
model.eval()

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print("")
print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader)*100}%")