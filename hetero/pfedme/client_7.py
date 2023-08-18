"""Flower client example using PyTorch for MNIST image classification."""

import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import FashionMNIST

import random
import flwr as fl
import numpy as np
import torch
import torchvision
import copy
import mnist
from server import lambda_reg
from server import local_epochs
DATA_ROOT = "./dataset"
Benchmark=False
FED_BN=False

def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859), (0.3530))]
    )
    # Load the MNIST dataset
    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    selected_classes=[2,3]

    #train
    # Filter the dataset to include only the selected classes
    indices = torch.where(torch.logical_or(trainset.targets == selected_classes[0],
                                        trainset.targets == selected_classes[1]))[0]
    indices=indices.numpy()
    np.random.shuffle(indices)
    num_samples= random.randint(4000,6000)
    indices=indices[:num_samples]
    subset_indices=torch.from_numpy(indices)
    subset_dataset = torch.utils.data.Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=True)
    
    #test
    # Filter the dataset to include only the selected classes
    indices = torch.where(torch.logical_or(testset.targets == selected_classes[0],
                                        testset.targets == selected_classes[1]))[0]
    indices=indices.numpy()
    np.random.shuffle(indices)
    num_samples= int(num_samples*0.1)
    indices=indices[:num_samples]
    subset_indices=torch.from_numpy(indices)
    subset_dataset = torch.utils.data.Subset(testset, subset_indices)
    testloader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=True)
    
    num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}

    return trainloader, testloader, testset, num_examples

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def objective_function(local_model, global_model, lambda_reg, data, target):
    output = local_model(data)
    loss = F.cross_entropy(output, target)

    local_params = torch.cat([param.view(-1) for param in local_model.parameters()])
    global_params = torch.cat([param.view(-1) for param in global_model.parameters()])

    proba=torch.norm(local_params - global_params)**2

    proba_leaf = torch.tensor(proba, requires_grad=True)
    objective = loss + (lambda_reg/2) * proba_leaf

    return objective, loss, output, target
def gradient_norm_stop_callback(threshold=1e-5):
    """
    Callback function to monitor the optimization process and stop it once the gradient norm falls below a certain
    threshold.

    Args:
        threshold (float): Gradient norm threshold. Default is 1e-5.
    """

    def callback_function(optimizer):
        gradient_norm = 0.0
        for group in optimizer.param_groups:
          
            for param in group['params']:
                
                if isinstance(param.grad, torch.Tensor):
                    gradient_norm += torch.norm(param.grad)**2
        gradient_norm = gradient_norm.sqrt().item()
        if gradient_norm < threshold:
            print(f'Gradient norm ({gradient_norm:.6f}) is below the threshold ({threshold:.6f}). Stopping optimization.')
            return True

    return callback_function


# Flower Client
class MnistClient(fl.client.NumPyClient):
    """Flower client implementing mnist image classification using
    PyTorch."""

    def __init__(
        self,
        model: mnist.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if FED_BN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if FED_BN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]  

    def fit(self, parameters, config):
        
        self.set_parameters(parameters)
        # Define the personalized objective function using the Moreau envelope algorithm
        global_params = [val.detach().clone() for val in self.model.parameters()]
        self.model.train()
  
        for r in range(local_epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            # Local update on client 
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            counter=0
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                for i in range(10):
                    optimizer.zero_grad()
                    proximal_term = 0.0
                    for local_weights, global_weights in zip(self.model.parameters(), global_params):
                        proximal_term += (local_weights - global_weights).norm(2)**2
                    loss = criterion(self.model(data), target) + (lambda_reg/2) * proximal_term
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss
                    total += target.size(0)
                    correct += (torch.max(self.model(data).data, 1)[1] == target).sum().item()

                    # Check if the gradient norm is below a threshold
                    
                
                with torch.no_grad():
                    for param, global_param in zip(self.model.parameters(), global_params):
                        global_param.data = global_param.data-0.005*lambda_reg*(global_param.data-param.data)
                counter+=1
                if counter==100:
                   break
            epoch_loss /= len(self.trainloader.dataset)
            epoch_acc = correct / total
            print(f"Epoch {r+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        loss_person, accuracy_person= mnist.test_local(local_model=self.model, testloader=self.testloader, device=DEVICE)
        with torch.no_grad():     
          for param, global_param in zip(self.model.parameters(), global_params):
                param.data = global_param.data
        loss_global, accuracy_global= mnist.test_global(net=self.model, testloader=self.testloader, device=DEVICE)
        
        return self.get_parameters(self.model), self.num_examples["trainset"], {"accuracy_global": float(accuracy_global),"accuracy_person": float(accuracy_person)}
        
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = mnist.test_global(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start CifarClient."""

    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data
    if Benchmark==True:
        data = torch.load('/home/s124m21/projekat_DDU/hetero/fedavg/data_7.pth')
        # Retrieve the variables
        trainloader = data['trainloader']
        num_examples = data['num_examples']
        testloader = data['testloader']
    else:    
        trainloader, testloader, _, num_examples = load_data()
    # Load model
    model = mnist.Net().to(DEVICE)

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = MnistClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client)


if __name__ == "__main__":
    main()