"""Flower client example using PyTorch for FashionMNIST image classification."""


import os

from collections import OrderedDict
from typing import Dict, List, Tuple


import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import random
import flwr as fl
import numpy as np
import torch
import mnist



DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"
Benchmark=False
FED_BN=False

Non_uniform_cardinality=False

def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load FashionMNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859,), (0.3530,))]
    )
    # Load the FashionMNIST dataset
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower Client
class MnistClient(fl.client.NumPyClient):
    """Flower client implementing FashionMNIST image classification using
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
        lambda_reg: int = config["lambda_reg"]
        local_rounds: int = config["local_rounds"]
        local_epochs: int = config["local_epochs"]
        local_iterations: int= config["local_iterations"]
    
        self.set_parameters(parameters)
        # Define the personalized objective function using the Moreau envelope algorithm
        global_params = [val.detach().clone() for val in self.model.parameters()]
        self.model.train()
  
        for r in range(local_epochs):
            # Local update on client 
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            for r in range(local_rounds):
                data_iterator = iter(self.trainloader)
                data, target = next(data_iterator)
                data, target = data.to(DEVICE), target.to(DEVICE) #sample a batch
                for i in range(local_iterations):
                    optimizer.zero_grad()
                    proximal_term = 0.0
                    for local_weights, global_weights in zip(self.model.parameters(), global_params):
                        proximal_term += (local_weights - global_weights).norm(2)**2
                    loss = criterion(self.model(data), target) + (lambda_reg/2) * proximal_term
                    loss.backward()
                    optimizer.step()

                   #update the model
                    
                
                with torch.no_grad():
                    for param, global_param in zip(self.model.parameters(), global_params):
                        global_param.data = global_param.data-0.005*lambda_reg*(global_param.data-param.data)

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
    """Load data, start MnistClient."""

    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data
    if Benchmark==True:
        data = torch.load('/home/s124m21/projekat_DDU/homo/fedavg/data_9.pt')
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