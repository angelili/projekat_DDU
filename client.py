import general_mnist
from general_mnist import FED_BN
import torch
import flwr as fl

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np
import random

DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"


#loading data for heterogeneous clients
def load_data(selected_classes:List) -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]):
    """Load FashionMNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859), (0.3530))]
    )
    # Load the MNIST dataset
    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

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
    
    #checkup
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
    
  

    return trainloader, testloader

# Flower Client in all three algorithms
class MnistClient(fl.client.NumPyClient):
    """Flower client implementing FashionMNIST image classification using
    PyTorch."""

    def __init__(
        self,
        model: general_mnist.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
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

        pfedme: bool = config['pfedme']

        if pfedme == True:
            new: bool = config['new']
            lambda_reg: int = config["lambda_reg"]
            local_rounds: int = config["local_rounds"]
            local_iterations: int= config["local_iterations"]
            lr: float= config["learning_rate"]
            mu: float= config["global_learning_rate"]

            self.set_parameters(parameters)
            
            global_params=general_mnist.train_pfedme(model=self.model, trainloader=self.trainloader,
            new=new, device=self.device, local_rounds=local_rounds, local_iterations=local_iterations, lambda_reg=lambda_reg,
            lr=lr, mu=mu)

            loss_person, accuracy_person = general_mnist.test(model=self.model, testloader=self.testloader, device=self.device)
            with torch.no_grad():     
             for param, global_param in zip(self.model.parameters(), global_params):
                    param = global_param
            loss_local, accuracy_local = general_mnist.test(model=self.model, testloader=self.testloader, device=self.device)
            
            return self.get_parameters(self.model), len(self.testloader.dataset), {"accuracy_local": float(accuracy_local),"accuracy_person": float(accuracy_person)}
        else:
            local_epochs: int = config['local_epochs']
            lr: float = config["learning_rate"]


            self.set_parameters(parameters)
            general_mnist.train_fedavg(model=self.model, trainloader=self.trainloader, local_epochs=local_epochs, device=self.device, lr=lr)
            loss_local, accuracy_local = general_mnist.test(model=self.model, testloader=self.testloader, device=self.device)
            
            return self.get_parameters(self.model), len(self.testloader.dataset), {"accuracy_local": float(accuracy_local)}
    
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = general_mnist.test(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}
    



