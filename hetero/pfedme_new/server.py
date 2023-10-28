"""Flower server"""
import os
import json


import flwr as fl
from flwr.common import Metrics
import torch
import torchvision
from typing import Callable, Optional, Tuple, Dict, Union, List
from collections import OrderedDict

import matplotlib.pyplot as plt

import sys
sys.path.append('/home/s124m21/projekat_DDU')
import general_server
import general_mnist

lambda_reg=15
FED_BN=False

# Load each dictionary from the JSON files of FedAvg
with open("/home/s124m21/projekat_DDU/hetero/fedavg/training_history_acc_cent_fed_avg.json", "r") as json_file:
    data1 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/hetero/fedavg/training_history_acc_dist_fed_avg.json", "r") as json_file:
    data2 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/hetero/fedavg/training_history_loss_cent_fed_avg.json", "r") as json_file:
    data3 = json.load(json_file)


# Load each dictionary from the JSON files of pFedMe
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




def fit_config(server_round: int):
    """Return training configuration dict for each round."""

    config = {'pfedme':True,
            'new': False,
            "lambda_reg":15,
            "local_rounds":120,
            "local_iterations":10,
            "learning_rate": 0.1,
            "global_learning_rate": 0.005 }
    return config

if __name__ == "__main__":
    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    testset=general_server.load_data_server()
    
   
    
    strategy = fl.server.strategy.FedAvgM(
    min_fit_clients=9,
    min_evaluate_clients=10,
    min_available_clients=10,
    evaluate_fn=general_server.get_evaluate_fn(testset,training_history_acc_cent, training_history_loss_cent,
                                                      'accuracy_centralized_pfedme_new','loss_centralized_pfedme_new'),
                                
    fit_metrics_aggregation_fn=general_server.agg_metrics_train_both_pfedme(training_history_acc_dist,'accuracy_local_pfedme_new','accuracy_person_pfedme_new'),
    evaluate_metrics_aggregation_fn=general_server.weighted_average(training_history_acc_dist,'accuracy_global_pfedme_new'),
    on_fit_config_fn=fit_config,
        )
    
    
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy)
    
    #detalied comparison with FedAvg, same as for pFedMe
    general_server.plot_training_comparison(training_history_acc_dist, data2,'accuracies_clients.png')
    general_server.plot_training_comparison(training_history_acc_cent, data1,'accuracies_server.png')
    general_server.plot_training_comparison(training_history_loss_cent, data3, 'losses_server.png')

    #key comparison of FedAvg, pFedMe, pFedMe_new local models
    plot_key_differences_local(data2, data5,training_history_acc_dist, "key_differences_local.png")

    #key comparison of FedAvg, pFedMe, pFedMe_new centralized models
    plot_key_differences_centralized(data1,data4,training_history_acc_cent,"key_differences_centralized")
