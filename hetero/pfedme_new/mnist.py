from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import FashionMNIST

import random
import numpy as np

import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist

DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"

def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load FashionMNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859), (0.3530))]
    )
    # Load the FashionMNIST dataset
    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    selected_classes = [9, 0, 1, 2, 3]  # Replace with your selected classes

    # Convert selected_classes list to a tensor
    selected_classes_tensor = torch.tensor(selected_classes)

    # Filter the dataset to include only the selected classes
    indices = torch.where(torch.isin(trainset.targets, selected_classes_tensor))[0]


   
    indices=indices.numpy()
    np.random.shuffle(indices)
    num_samples= random.randint(4000,6000)
    indices=indices[:num_samples]
    subset_indices=torch.from_numpy(indices)
    subset_dataset = torch.utils.data.Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=True)

    
    selected_targets = trainset.targets[indices]

    class_counts = {}
    for class_idx in selected_classes:
        count = (selected_targets == class_idx).sum().item()
        class_counts[class_idx] = count

    # Print the class counts
    for class_idx, count in class_counts.items():
        print(f"Class {class_idx}: {count}")

    # Filter the dataset to include only the selected classes
    indices = torch.where(torch.isin(testset.targets, selected_classes_tensor))[0]
  
  
    indices=indices.numpy()
    np.random.shuffle(indices)
    num_samples= int(num_samples*0.1)
    indices=indices[:num_samples]
    subset_indices=torch.from_numpy(indices)
    subset_dataset = torch.utils.data.Subset(testset, subset_indices)
    testloader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=True)
    
    num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}

    return trainloader, testloader, testset, num_examples


#training the personalized network which, infacts trains the one that goes to the global
def train(
    model: general_mnist.Net,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    local_epochs: int,
    local_iterations: int,
    local_rounds: int,
    device: torch.device,
    eta: float,
    lambda_reg: float  
) -> None:
    """Train the network."""
       
    # Define the personalized objective function using the Moreau envelope algorithm
    global_params = [val.detach().clone() for val in model.parameters()]
    model.train()
  
    for r in range(local_epochs):
        # Local update on client 
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for r in range(local_rounds):
            for i in range(local_iterations):
                data_iterator = iter(trainloader)
                data, target = next(data_iterator)
                data, target = data.to(device), target.to(device) #sample a batch
                optimizer.zero_grad()
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                loss = criterion(model(data), target) + (lambda_reg/2) * proximal_term
                loss.backward()
                optimizer.step()


                #update the model
                
            
            with torch.no_grad():
                for param, global_param in zip(model.parameters(), global_params):
                    global_param.data = global_param.data-eta*lambda_reg*(global_param.data-param.data)
            

    loss_person, accuracy_person= general_mnist.test_local(local_model=model, testloader=testloader, device=device)
    with torch.no_grad():     
        for param, global_param in zip(model.parameters(), global_params):
            param.data = global_param.data
    loss_global, accuracy_global= general_mnist.test_global(net=model, testloader=testloader, device=device)

    print("Accuracy_personalized: ", accuracy_person)
    print("Accuracy_global: ", accuracy_global)

    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, testset, _ = load_data()
    net = general_mnist.Net().to(device)

    print("Start training")
    net= train(net=net, trainloader=trainloader, testloader=testloader, local_epochs=1, local_iterations=10, local_rounds=120, device=device, eta=0.005, lambda_reg=15)
    print("Evaluate model")
  
    loss_person, accuracy_person= general_mnist.test_local(local_model=net, testloader=testloader, device=device)
    print("Loss_personalized: ", loss_person)
    print("Accuracy_personalized: ", accuracy_person)
   

if __name__ == "__main__":
    main()
