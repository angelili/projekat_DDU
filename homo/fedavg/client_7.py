
"""Flower client example using PyTorch for Fashion_MNIST image classification."""


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
from server import local_epochs
from mnist import trainloaders, testloaders
DATA_ROOT = "./dataset"
Benchmark=True
Non_uniform_cardinality=True

def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859), (0.3530))]
    )
    # Load the MNIST dataset
    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    if Non_uniform_cardinality==True:
        sample_size_train = random.randint(4000, 6000)
        sample_size_test =  int(sample_size_train*0.1)
    else:
        sample_size_train=5000
        sample_size_test=500

    indices_train = random.sample(range(len(trainset)), sample_size_train)
    sampler_train= torch.utils.data.SubsetRandomSampler(indices_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, sampler=sampler_train)
    indices_test = random.sample(range(len(testset)), sample_size_test)
    sampler_test = torch.utils.data.SubsetRandomSampler(indices_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, sampler=sampler_test)
    num_examples = {"trainset": sample_size_train, "testset": sample_size_test}

    return trainloader, testloader, testset, num_examples


# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


# Flower Client
class MnistClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
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

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
       
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        mnist.train(self.model, self.trainloader, epochs=local_epochs, device=DEVICE)
        loss, accuracy = mnist.test(net=self.model, testloader=self.testloader, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {"accuracy": float(accuracy)}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = mnist.test(self.model, self.testloader, device=DEVICE)
        print(type(accuracy), accuracy)
        print(type({"accuracy": float(accuracy)}))
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start MnistClient."""

    fedl_no_proxy=True


    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data
    #trainloader, testloader, _, num_examples = load_data()
    trainloader=trainloaders[6]
    testloader=testloaders[6]    
    num_examples={"trainset": 5400, "testset": 600}

    # Load model
    model = mnist.Net().to(DEVICE).train()
    
    if Benchmark==True:
        data_7 = {
            'trainloader': trainloader,
            'testloader': testloader,
            'num_examples': num_examples,
        }
        torch.save(data_7, 'data_7.pt')

    # Start client
    client = MnistClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client)
    
    

if __name__ == "__main__":
    main()