"""Flower server"""
"The global variables for Benchmarking, and FED_BN are defined in the general_mnist"

import flwr as fl

import os

import torch

from typing import Dict,  List

import json

import sys
sys.path.append('/home/s124m21/projekat_DDU')
import general_server
from general_mnist import Benchmark

lambda_reg=15

# Load each dictionary from the JSON files, metrics from FedAvg
with open("/home/s124m21/projekat_DDU/homo/fedavg/training_history_acc_cent_fed_avg.json", "r") as json_file:
    data1 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/homo/fedavg/training_history_acc_dist_fed_avg.json", "r") as json_file:
    data2 = json.load(json_file)

with open("/home/s124m21/projekat_DDU/homo/fedavg/training_history_loss_cent_fed_avg.json", "r") as json_file:
    data3 = json.load(json_file)


#preparing empty dictionaries for metrics in pfedme, to be filled in each global round
training_history_acc_dist={"accuracy_global_pfedme": [], "accuracy_local_pfedme": [], "accuracy_personalized_pfedme":[]}
training_history_acc_cent={'accuracy_centralized_pfedme': []}
training_history_loss_dist={"loss_distributed_pfedme": []}
training_history_loss_cent={"loss_centralized_pfedme": []}

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
    # since the dataset is PARTITIONED, the partitioning is done here at the server, to make sure everyone gets a different portion,
    #ofcourse this is not possible in reality, here if we want new data not the same as for fedavg, loading new!
    if Benchmark==False:
  
        trainloaders, testloaders = general_server.load_datasets()
    
        # Save each data partition to separate files with client IDs from 1 to 10
        for client_id, (trainloader, testloader) in enumerate(zip(trainloaders, testloaders), start=1):
                data_dict = {
                    'trainloader': trainloader,
                    'testloader': testloader
                }
                file_name = f'data_{client_id}.pt'
                torch.save(data_dict, file_name)
   
    strategy = fl.server.strategy.FedAvgM(
    min_fit_clients=9,
    min_evaluate_clients=10,
    min_available_clients=10,
    evaluate_fn=general_server.get_evaluate_fn(testset,training_history_acc_cent, training_history_loss_cent,
                                                      "accuracy_centralized_pfedme","loss_centralized_pfedme"),
                                
    fit_metrics_aggregation_fn=general_server.agg_metrics_train_both_pfedme(training_history_acc_dist,
                                                    'accuracy_local_pfedme','accuracy_personalized_pfedme'),
    evaluate_metrics_aggregation_fn=general_server.weighted_average(training_history_acc_dist,'accuracy_global_pfedme'),
    on_fit_config_fn=fit_config)
    
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy)

    if Benchmark==True:
        
        # Load each dictionary from the JSON files, metrics from FedAvg
        with open("/home/s124m21/projekat_DDU/homo/fedavg/training_history_acc_cent_fed_avg.json", "r") as json_file:
            data1 = json.load(json_file)

        with open("/home/s124m21/projekat_DDU/homo/fedavg/training_history_acc_dist_fed_avg.json", "r") as json_file:
            data2 = json.load(json_file)

        with open("/home/s124m21/projekat_DDU/homo/fedavg/training_history_loss_cent_fed_avg.json", "r") as json_file:
            data3 = json.load(json_file)

        general_server.plot_training_comparison(training_history_acc_dist, data2,'accuracies_clients.png')
        general_server.plot_training_comparison(training_history_acc_cent, data1,'accuracies_server.png')
        general_server.plot_training_comparison(training_history_loss_cent, data3, 'losses_server.png')
    else:
        general_server.plot_training_history(training_history_acc_dist,'accuracies_clients.png')
        general_server.plot_training_history(training_history_acc_cent,'accuracies_server.png')
        general_server.plot_training_history(training_history_loss_cent,'loss_server.png')
