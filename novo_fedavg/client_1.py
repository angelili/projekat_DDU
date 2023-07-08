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

DATA_ROOT = "./dataset"

def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    # Load the MNIST dataset
    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    selected_classes=[0,1]

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
    trainloader = torch.utils.data.DataLoader(subset_dataset, batch_size=16, shuffle=True)
    
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
    testloader = torch.utils.data.DataLoader(subset_dataset, batch_size=16, shuffle=True)
    
    num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}

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
        self.model.train()
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        mnist.train(self.model, self.trainloader, epochs=1, device=DEVICE)
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
    """Load data, start CifarClient."""

    fedl_no_proxy=True


    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data
    trainloader, testloader, _, num_examples = mnist.load_data()

    # Load model
    model = mnist.Net().to(DEVICE).train()


    # Start client
    client = MnistClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client)


if __name__ == "__main__":
    main()