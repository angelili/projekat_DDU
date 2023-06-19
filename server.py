"""Flower server example."""

from collections import OrderedDict
import flwr as fl
from flwr.common import Metrics
import os
import numpy as np
import torch
import torchvision
from typing import Callable, Optional, Tuple, Dict, Union, List
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch import Tensor
import mnist

def load_data()
    """Load MNIST test set."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    
    testset = MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    return testset
    
def get_evaluate_fn(
    testset: torchvision.datasets.MNIST,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = mnist.Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
       
       

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = mnist.test_global(model, testloader, device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
    
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    testset=load_data()
    strategy = fl.server.strategy.FedAvgM(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(testset),  #centralised evaluation of global model
        evaluate_metrics_aggregation_fn=weighted_average
    )
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
