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

import mnist


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
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
        set_parameters(self.net, parameters)
        local_model=copy.deepcopy(self.net).to(DEVICE)
        local_model.train()
        for r in range(R):
            correct, total, epoch_loss = 0, 0, 0.0
            # Local update on client i
            optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            for batch_idx, (data, target) in enumerate(self.trainloader):
                optimizer.zero_grad()
                objective, loss, output, target = objective_function(local_model, self.net, lambda_reg, data.to(DEVICE), target.to(DEVICE))
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
              for param, global_param in zip(local_model.parameters(), self.net.parameters()):
                  global_param.data=global_param.data-eta*lambda_reg*(global_param.data-param.data)

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
    trainloader, testloader, _, num_examples = cifar.load_data()
    
    # Load model
    model = cifar.Net().to(DEVICE).train()


    # Start client
    client = CifarClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client)


if __name__ == "__main__":
    main()
