"""Flower server"""


import flwr as fl

import os

import matplotlib.pyplot as plt

from typing import Dict,  List

import json

import sys
sys.path.append('/home/s124m21/projekat_DDU')
import flwr as fl
from flwr.common import Metrics
import os
import numpy as np
import torch
import torchvision
from typing import Callable, Optional, Tuple, Dict, Union, List
from collections import OrderedDict
import mnist
import json
import matplotlib.pyplot as plt
# import your module without specifying the full path
import general_server
import general_mnist

lambda_reg=15
FED_BN=False

# Load each dictionary from the JSON files
with open("/home/s124m21/projekat_DDU/hetero/fedavg/training_history_acc_cent_fed_avg.json", "r") as json_file:
    data1 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/hetero/fedavg/training_history_acc_dist_fed_avg.json", "r") as json_file:
    data2 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/hetero/fedavg/training_history_loss_cent_fed_avg.json", "r") as json_file:
    data3 = json.load(json_file)


with open("/home/s124m21/projekat_DDU/hetero/pfedme/training_history_acc_cent_pfedme.json", "r") as json_file:
    data4 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/hetero/pfedme/training_history_acc_dist_pfedme.json", "r") as json_file:
    data5 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/hetero/pfedme/training_history_loss_cent_pfedme.json", "r") as json_file:
    data6 = json.load(json_file)




def plot_key_differences_local(fedavg,pfedme,pfedme_new,path):
    plt.figure()
    # Extract local metrics from fedavg
    values=fedavg["accuracy_local"]
    # Create a line plot for the metric
    plt.plot(values, label="accuracy_local_fedavg")
    # Extract local metrics from pfedme
    values=pfedme["accuracy_local_pfedme"]
    # Create a line plot for the metric
    plt.plot(values, label="accuracy_local_pfedme")
    # Extract local metrics from pfedme_new
    values=pfedme_new["accuracy_local_pfedme_new"]
    # Create a line plot for the metric
    plt.plot(values, label="accuracy_local_pfedme_new")

 
    # Add labels, title, and legend to the plot
    plt.xlabel('Training Round')
    plt.ylabel('Metric Value')
    plt.title('Comparison of local models')
    plt.legend()
    plt.savefig(path)
    # Show the plot
    plt.show()

def plot_key_differences_centralized(fedavg,pfedme,pfedme_new,path):
    plt.figure()
    # Extract local metrics from fedavg
    values=fedavg["accuracy_centralized"]
    # Create a line plot for the metric
    plt.plot(values, label="accuracy_centralized_fedavg")
    # Extract local metrics from pfedme
    values=pfedme["accuracy_centralized_pfedme"]
    # Create a line plot for the metric
    plt.plot(values, label="accuracy_centralized_pfedme")
    # Extract local metrics from pfedme_new
    values=pfedme_new["accuracy_centralized_pfedme_new"]
    # Create a line plot for the metric
    plt.plot(values, label="accuracy_centralized_pfedme_new")

 
    # Add labels, title, and legend to the plot
    plt.xlabel('Training Round')
    plt.ylabel('Metric Value')
    plt.title('Comparison of global models')
    plt.legend()
    plt.savefig(path)
    # Show the plot
    plt.show()

training_history_acc_dist={"accuracy_global_pfedme_new": [], "accuracy_local_pfedme_new": [], "accuracy_personalized_pfedme_new":[]}
training_history_acc_cent={'accuracy_centralized_pfedme_new': []}
training_history_loss_dist={"loss_distributed_pfedme_new": []}
training_history_loss_cent={"loss_centralized_pfedme_new": []}



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

        training_history_acc_cent["accuracy_centralized_pfedme_new"].append(accuracy)
        training_history_loss_cent["loss_centralized_pfedme_new"].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    training_history_acc_dist["accuracy_global_pfedme_new"].append(sum(accuracies)/sum(examples))
    # Aggregate and return custom metric (weighted average)
    return {"accuracy_global": sum(accuracies) / sum(examples)}
    
def agg_metrics_train(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies_person = [num_examples * m["accuracy_person"] for num_examples, m in metrics]
    accuracies_global = [num_examples * m["accuracy_global"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    training_history_acc_dist["accuracy_personalized_pfedme_new"].append(sum(accuracies_person)/sum(examples))
    training_history_acc_dist["accuracy_local_pfedme_new"].append(sum(accuracies_global)/sum(examples))


    # Aggregate and return custom metric (weighted average)
    return {"accuracy_personalized": sum(accuracies_person)/sum(examples), "accuracy_local": sum(accuracies_global)/sum(examples)}

def fit_config(server_round: int):
    """Return training configuration dict for each round."""

    config = {
        "lambda_reg":15,
        "local_epochs":1,
        "local_rounds":120,
        "local_iterations":10
    }
    return config

if __name__ == "__main__":
    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    testset=general_server.load_data_server()
    
   
    strategy = fl.server.strategy.FedAvgM(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=9,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(testset),#centralised evaluation of global model
     
        fit_metrics_aggregation_fn=agg_metrics_train,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
       
       )
    
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy)
    
    #detalied comparison with FedAvg, same as for pFedMe
    general_server.plot_training_comparison(training_history_acc_dist, data2,'photo_1.png')
    general_server.plot_training_comparison(training_history_acc_cent, data1,'photo_2.png')
    general_server.plot_training_comparison(training_history_loss_cent, data3, 'photo_3.png')

    #key comparison of FedAvg, pFedMe, pFedMe_new local models
    plot_key_differences_local(data2, data5,training_history_acc_dist, "key_differences_local.png")

    #key comparison of FedAvg, pFedMe, pFedMe_new centralized models
    plot_key_differences_centralized(data1,data4,training_history_acc_cent,"key_differences_centralized")
