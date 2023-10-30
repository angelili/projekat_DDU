"""Flower server"""

import flwr as fl
import torch
import os

from typing import  Dict ,List

import json

import sys
sys.path.append('/home/s124m21/projekat_DDU')
import general_server


training_history_acc_dist={"accuracy_global_fedavg": [], "accuracy_local_fedavg": []}
training_history_acc_cent={'accuracy_centralized_fedavg': []}
training_history_loss_dist={"loss_distributed_fedavg": []}
training_history_loss_cent={"loss_centralized_fedavg": []}



def fit_config(server_round: int):
        """Return training configuration dict for each round."""

        config = {"pfedme": False,
            "local_epochs": 2,
            "learning_rate": 0.1, }
        return config



if __name__ == "__main__":
    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    testset=general_server.load_data_server()
    # since the dataset is PARTITIONED, the partitioning is done here at the server, to make sure everyone gets a different portion,
    #ofcourse this is not possible in reality
    trainloaders, testloaders = general_server.load_datasets()
  
    # Save each data partition to separate files with client IDs from 1 to 10
    for client_id, (trainloader, testloader) in enumerate(zip(trainloaders, testloaders), start=1):
            data_dict = {
                'trainloader': trainloader,
                'testloader': testloader
            }
            file_name = f'data_{client_id}.pt'
            torch.save(data_dict, file_name)


    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=9,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_fn=general_server.get_evaluate_fn(testset,training_history_acc_cent, training_history_loss_cent, 
                                                   'accuracy_centralized_fedavg','loss_centralized_fedavg'),#centralised evaluation of global model
        fit_metrics_aggregation_fn=general_server.agg_metrics_train_fedavg(training_history_acc_dist, 'accuracy_local_fedavg'),
        evaluate_metrics_aggregation_fn=general_server.weighted_average(training_history_acc_dist, 'accuracy_global_fedavg'),
        on_fit_config_fn=fit_config)
    
    fl.server.start_server(
    server_address= "10.30.0.254:9000",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy)

  
    general_server.plot_training_history(training_history_acc_dist,'accuracies_clients.png')
    general_server.plot_training_history(training_history_acc_cent,'accuracies_server.png')
    general_server.plot_training_history(training_history_loss_cent,'loss_server.png')

    with open("training_history_acc_dist_fed_avg.json", "w") as json_file:
        json.dump(training_history_acc_dist, json_file)

    with open("training_history_acc_cent_fed_avg.json", "w") as json_file:
        json.dump(training_history_acc_cent, json_file)

    with open("training_history_loss_cent_fed_avg.json", "w") as json_file:
        json.dump(training_history_loss_cent, json_file)