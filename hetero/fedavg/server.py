"""Flower server example."""

from collections import OrderedDict
import flwr as fl
from flwr.common import Metrics
import os
import numpy as np
import torch
import torchvision
from typing import Callable, Optional, Tuple, Dict, Union, List
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch import Tensor
import matplotlib.pyplot as plt
import json
import mnist
DATA_ROOT = "./dataset"

local_epochs=1



def plot_training_history(training_history, path):
    plt.figure()
    # Iterate over each metric in the training history dictionary
    for metric, values in training_history.items():
        # Create a line plot for the metric
        plt.plot(values, label=metric)

    # Add labels, title, and legend to the plot
    plt.xlabel('Training Round')
    plt.ylabel('Metric Value')
    plt.title('Training History')
    plt.legend()
    plt.savefig(path)
    # Show the plot
    plt.show()

def load_data_server():
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
    testset_server = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True)
 

    
    
    #selected_classes = [0, 1, 2, 3]  # Replace with  selected classes

    # Convert selected_classes list to a tensor
    #selected_classes_tensor = torch.tensor(selected_classes)

    # Filter the dataset to include only the selected classes
    #indices = torch.where(torch.isin(testset.targets, selected_classes_tensor))[0]

    #indices=indices.numpy()
    #subset_indices=torch.from_numpy(indices)
    #subset_dataset = torch.utils.data.Subset(testset, subset_indices)
    #testset_server = torch.utils.data.DataLoader(subset_dataset, batch_size=50, shuffle=True)
    

    return testset_server


training_history_acc_dist={"accuracy_global": [], "accuracy_local": []}
training_history_acc_cent={'accuracy_centralized': []}
training_history_loss_dist={"loss_distributed": []}
training_history_loss_cent={"loss_centralized": []}



def get_evaluate_fn(
    testset: torchvision.datasets.MNIST,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire MNIST test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = mnist.Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
       
       

        
        loss, accuracy = mnist.test(model, testset, device)
        training_history_acc_cent["accuracy_centralized"].append(accuracy)
        training_history_loss_cent["loss_centralized"].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
   
    examples = [num_examples for num_examples, _ in metrics]

    training_history_acc_dist["accuracy_global"].append(sum(accuracies)/sum(examples))
    # Aggregate and return custom metric (weighted average)
    return {"accuracy_global": sum(accuracies) / sum(examples)}



def agg_metrics_train(metrics: List[Tuple[int, Metrics]]) -> Metrics:
   # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    training_history_acc_dist["accuracy_local"].append(sum(accuracies)/sum(examples))

    # Aggregate and return custom metric (weighted average)
    return {"accuracy_local": sum(accuracies) / sum(examples)}
    
   


    
if __name__ == "__main__":
    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
      testset= load_data_server()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=9,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(testset),  #centralised evaluation of global model
        fit_metrics_aggregation_fn=agg_metrics_train,
        evaluate_metrics_aggregation_fn=weighted_average
    )
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy
    )

    plot_training_history(training_history_acc_dist,'photo_1.png')
    plot_training_history(training_history_acc_cent,'photo_2.png')
    plot_training_history(training_history_loss_cent,'photo_3.png')

    with open("training_history_acc_dist_fed_avg.json", "w") as json_file:
        json.dump(training_history_acc_dist, json_file)

    with open("training_history_acc_cent_fed_avg.json", "w") as json_file:
        json.dump(training_history_acc_cent, json_file)

    with open("training_history_loss_cent_fed_avg.json", "w") as json_file:
        json.dump(training_history_loss_cent, json_file)