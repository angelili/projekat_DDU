"""Flower client using PyTorch for FashionMNIST image classification."""


import os

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

import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist


DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"
Benchmark=True
FED_BN=False

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower Client
class MnistClient(fl.client.NumPyClient):
    """Flower client implementing FashionMNIST image classification using
    PyTorch."""

    def __init__(
        self,
        model: general_mnist.Net,
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
        new: bool = config['new']
        lambda_reg: int = config["lambda_reg"]
        local_rounds: int = config["local_rounds"]
        local_iterations: int= config["local_iterations"]
        lr: float = config["learning_rate"]
        mu: float = config["global_learning_rate"]
    
        self.set_parameters(parameters)
        global_params = general_mnist.train_pfedme(model=self.model, trainloader=self.trainloader, new=new, device=self.device, local_rounds=local_rounds, local_iterations=local_iterations, lambda_reg=lambda_reg, lr=lr, mu=mu)

        loss_person, accuracy_person = general_mnist.test(model=self.model, testloader=self.testloader, device=self.device)
        with torch.no_grad():     
            for param, global_param in zip(self.model.parameters(), global_params):
                param = global_param
        loss_local, accuracy_local = general_mnist.test(model=self.model, testloader=self.testloader, device=self.device)
        
        return self.get_parameters(self.model), len(self.testloader.dataset), {"accuracy_local": float(accuracy_local),"accuracy_person": float(accuracy_person)}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = general_mnist.test_global(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start MnistClient."""

    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data
    if Benchmark==True:
        data = torch.load('/home/s124m21/projekat_DDU/hetero/fedavg/data_10.pth')
        # Retrieve the variables
        trainloader = data['trainloader']
        num_examples = data['num_examples']
        testloader = data['testloader']
    else:    
        trainloader, testloader, _, num_examples = load_data()
    
    # Load model
    model = general_mnist.Net().to(DEVICE)
   
    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))

    # Start client
    client = MnistClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client)


if __name__ == "__main__":
    main()