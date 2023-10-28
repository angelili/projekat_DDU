import general_mnist

import torch
import flwr as fl

from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np

FED_BN=False


# Flower Client in pfedme, pfedme_new
class MnistClient_pfedme(fl.client.NumPyClient):
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
        loss, accuracy = general_mnist.test(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}
    

# Flower Client in fedavg
class MnistClient_fedavg(fl.client.NumPyClient):
    """Flower client implementing image classification using
    PyTorch."""

    def __init__(
        self,
        model: general_mnist.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device=device

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
        local_epochs: int = config["local_epochs"]
        self.set_parameters(parameters)
        general_mnist.train(self.model, self.trainloader, epochs=local_epochs, device=self.device)
        loss, accuracy = general_mnist.test(net=self.model, testloader=self.testloader, device=self.device)
        return self.get_parameters(config={}), self.num_examples["trainset"], {"accuracy": float(accuracy)}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = general_mnist.test(self.model, self.testloader, device=self.device)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
