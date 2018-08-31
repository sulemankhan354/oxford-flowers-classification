# performing necessary imports for task

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from collections import OrderedDict
import time
import copy
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')



args, _ = parser.parse_known_args()

# This method loads and tunes in a model
def load_model(arch='vgg16', num_labels=102, hidden_units=1024):
    # Load a pre-trained model
    if arch=='vgg16':
        # Load a pre-trained model of vgg16
        model = models.vgg16(pretrained=True)
        # extending classifier for training it
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, num_labels)),
                              ]))
        
    elif arch=='alexnet':
        # Load a pre-trained model of alexnet
        model = models.alexnet(pretrained=True)
        # extending classifier for training it
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(9216, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, num_labels)),
                              ]))
        
        
    else:
        raise ValueError('Unexpected network architecture', arch)
    #   Freezing weights of the model 
    for param in model.parameters():
            param.requires_grad = False
    model.classifier = classifier

    return model



# This method trains a model
def train_model(image_datasets, arch='vgg16', hidden_units=1024, epochs=50, learning_rate=0.0001, gpu=False, checkpoint=''):
    # Use command line values when specified
    if args.arch:
        arch = args.arch     
        
    if args.hidden_units:
        hidden_units = args.hidden_units

    if args.epochs:
        epochs = args.epochs
            
    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.gpu:
        gpu = args.gpu

    if args.checkpoint:
        checkpoint = args.checkpoint        
        
    # Using the image datasets, define the dataloaders for whole data i.e. train,valid,test
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=20, shuffle=True, num_workers=2)
        for x in list(image_datasets.keys())
    }
 
    # Calculate dataset sizes for whole data i.e. train,valid,test
    dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(image_datasets.keys())
    }    

        
    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    # Load the model     
    num_labels = len(image_datasets['train'].classes)
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")     

                
    # Defining criterion, optimizer and scheduler
    # Observe that only parameters that require gradients are optimized
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)    
        
    since = time.time()
    # loading best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]    # total loss for an epoch
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  # total accuracy for an epoch

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Store class_to_idx into a model property
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    # Save checkpoint if requested
    if checkpoint:
        print ('Saving checkpoint to:', checkpoint) 
        checkpoint_dict = {
            'architecture': arch,
            'class_to_idx': model.class_to_idx, 
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units
        }
        
        torch.save(checkpoint_dict, checkpoint)
    
    # Return the model
    return model


# Train model if invoked from command line
if args.data_dir:    
    # Default transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }
    # calling the function for training the model      
    train_model(image_datasets)
