"""Flower client example using PyTorch for CIFAR-10 image classification."""


import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision
import copy
import mnist


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
class CifarClient(fl.client.NumPyClient):
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

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
       
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
       
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Define the personalized objective function using the Moreau envelope algorithm
       
        local_model=copy.deepcopy(self.model).to(DEVICE)
        local_model.train()
        for r in range(2):
            correct, total, epoch_loss = 0, 0, 0.0
            # Local update on client 
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)
            for batch_idx, (data, target) in enumerate(self.trainloader):
                optimizer.zero_grad()
                objective, loss, output, target = objective_function(local_model, self.model, 15, data.to(DEVICE), target.to(DEVICE))
                #objective.requires_grad = True
                objective.backward()
                optimizer.step()
                # Metrics

                epoch_loss += loss
                total += target.size(0)
                correct += (torch.max(output.data, 1)[1] == target).sum().item()

                # Check if the gradient norm is below a threshold
                if gradient_norm_stop_callback(threshold=1e-5)(optimizer):
                      break
            # Compute Moreau envelope of local model
            with torch.no_grad():
              for param, global_param in zip(local_model.parameters(), self.model.parameters()):
                  global_param.data=global_param.data-0.05*15*(global_param.data-param.data)

            epoch_loss /= len(self.trainloader.dataset)
            epoch_acc = correct / total
            print(f"Epoch {r+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test_global(self.model, self.testloader, device=DEVICE)
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
    client = CifarClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client)


if __name__ == "__main__":
    main()
