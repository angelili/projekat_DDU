import torch

from collections import OrderedDict
import flwr as fl
from flwr.common import Metrics

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from typing import Callable, Optional, Tuple, Dict, Union, List

import matplotlib.pyplot as plt
import numpy as np

import general_mnist

DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"
FED_BN=False




def load_data_server():
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    testset_server = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True)
    

    return testset_server    

#The following metrics are the same; the difference is in keys from the dictionaries we are filling in _fedavg_pfedme_pfedme_new

    
def get_evaluate_fn_pfedme(
    testset: FashionMNIST, dict_acc: Dict, dict_loss: Dict
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire FashionMNIST test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = general_mnist.Net()
        if FED_BN==True:
            keys = [k for k in model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        model.to(device)
       
       
        
        loss, accuracy = general_mnist.test_global(model, testset, device)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        #loss, accuracy = mnist.test_global(model, testloader, device)

        dict_acc["accuracy_centralized_pfedme"].append(accuracy)
        dict_loss["loss_centralized_pfedme"].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def get_evaluate_fn_pfedme(
    testset: FashionMNIST, dict_acc: Dict, dict_loss: Dict
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire MNIST test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = general_mnist.Net()
        if FED_BN==True:
            keys = [k for k in model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        model.to(device)
       
       
        
        loss, accuracy = general_mnist.test_global(model, testset, device)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        #loss, accuracy = mnist.test_global(model, testloader, device)

        dict_acc["accuracy_centralized_pfedme_new"].append(accuracy)
        dict_loss["loss_centralized_pfedme_new"].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def get_evaluate_fn_fedavg(
    testset: FashionMNIST, dict_acc: Dict, dict_loss: Dict
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire MNIST test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = general_mnist.Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
       
       

        
        loss, accuracy = general_mnist.test(model, testset, device)
        dict_acc["accuracy_centralized_fedavg"].append(accuracy)
        dict_loss["loss_centralized_fedavg"].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def weighted_average_pfedme(dict: Dict) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        dict["accuracy_global_pfedme"].append(sum(accuracies)/sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {"accuracy_global": sum(accuracies) / sum(examples)}
    return evaluate

def weighted_average_pfedme_new(dict: Dict) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        dict["accuracy_global_pfedme_new"].append(sum(accuracies)/sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {"accuracy_global": sum(accuracies) / sum(examples)}
    return evaluate

def weighted_average_fedavg(dict: Dict) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        dict["accuracy_global_fedavg"].append(sum(accuracies)/sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {"accuracy_global": sum(accuracies) / sum(examples)}
    return evaluate

#function used to evaluate at the end of training

def agg_metrics_train_pfedme(dict: Dict) -> Metrics:    
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies_person = [num_examples * m["accuracy_person"] for num_examples, m in metrics]
        accuracies_global = [num_examples * m["accuracy_global"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        dict["accuracy_personalized_pfedme"].append(sum(accuracies_person)/sum(examples))
        dict["accuracy_local_pfedme"].append(sum(accuracies_global)/sum(examples))


        # Aggregate and return custom metric (weighted average)
        return {"accuracy_personalized": sum(accuracies_person)/sum(examples), "accuracy_local": sum(accuracies_global)/sum(examples)}
    return evaluate

def agg_metrics_train_pfedme_new(dict: Dict) -> Metrics:    
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies_person = [num_examples * m["accuracy_person"] for num_examples, m in metrics]
        accuracies_global = [num_examples * m["accuracy_global"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        dict["accuracy_personalized_pfedme_new"].append(sum(accuracies_person)/sum(examples))
        dict["accuracy_local_pfedme_new"].append(sum(accuracies_global)/sum(examples))


        # Aggregate and return custom metric (weighted average)
        return {"accuracy_personalized": sum(accuracies_person)/sum(examples), "accuracy_local": sum(accuracies_global)/sum(examples)}
    return evaluate
    
def agg_metrics_train_fedavg(dict: Dict) -> Metrics:  
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        dict["accuracy_local"].append(sum(accuracies)/sum(examples))

        # Aggregate and return custom metric (weighted average)
        return {"accuracy_local": sum(accuracies) / sum(examples)}
    return evaluate






    # Aggregate and return custom metric (weighted average)
    return {"accuracy_personalized": sum(accuracies_person)/sum(examples), "accuracy_local": sum(accuracies_global)/sum(examples)}

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

#this plot is for comparing two histories, pfedme and fedavg
def plot_training_comparison(training_history,data,path,lambda_reg=15):
    plt.figure()
    # Iterate over each metric in the training history dictionary
    for metric, values in training_history.items():
        # Create a line plot for the metric
        plt.plot(values, label=metric)
    for metric, values in data.items():
        # Create a line plot for the metric
        plt.plot(values, label=metric)
    # Add labels, title, and legend to the plot
    plt.xlabel('Training Round')
    plt.ylabel('Metric Value')
    plt.title('Training History: lambda='+str(lambda_reg))
    plt.legend()
    plt.savefig(path)
    # Show the plot
    plt.show()

def plot_training_comparison_thin(training_history,data,path, lambda_reg=15):
    plt.figure()
    # Iterate over each metric in the training history dictionary
    for metric, values in training_history.items():
        # Create a line plot for the metric
        plt.plot(values, label=metric, linewidth=0.5)
    for metric, values in data.items():
        # Create a line plot for the metric
        plt.plot(values, label=metric)
    # Add labels, title, and legend to the plot
    plt.xlabel('Training Round')
    plt.ylabel('Metric Value')
    plt.title('Training History: lambda='+str(lambda_reg))
    plt.legend()
    plt.savefig(path)
    # Show the plot
    plt.show()