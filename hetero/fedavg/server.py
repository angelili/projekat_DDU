"""Flower server"""


import flwr as fl

import os 

from typing import  Dict, List

import json

import sys
sys.path.append('/home/s124m21/projekat_DDU')
import general_server


training_history_acc_dist={"accuracy_global": [], "accuracy_local": []}
training_history_acc_cent={'accuracy_centralized': []}
training_history_loss_dist={"loss_distributed": []}
training_history_loss_cent={"loss_centralized": []}


def fit_config(server_round: int):
    """Return training configuration dict for each round."""

    config = {
     "local_epochs":1,
    }
    return config


    
if __name__ == "__main__":
    fedl_no_proxy=True
    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
      testset= general_server.load_data_server()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=9,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_fn=general_server.get_evaluate_fn_fedavg(testset,training_history_acc_cent, training_history_loss_cent),#centralised evaluation of global model
        fit_metrics_aggregation_fn=general_server.agg_metrics_train_fedavg(training_history_acc_dist),
        evaluate_metrics_aggregation_fn=general_server.weighted_average_fedavg(training_history_acc_dist),
        on_fit_config_fn=fit_config)
    
    fl.server.start_server(
        server_address= "10.30.0.254:9000",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy
    )

    general_server.plot_training_history(training_history_acc_dist,'photo_1.png')
    general_server.plot_training_history(training_history_acc_cent,'photo_2.png')
    general_server.plot_training_history(training_history_loss_cent,'photo_3.png')

    with open("training_history_acc_dist_fed_avg.json", "w") as json_file:
        json.dump(training_history_acc_dist, json_file)

    with open("training_history_acc_cent_fed_avg.json", "w") as json_file:
        json.dump(training_history_acc_cent, json_file)

    with open("training_history_loss_cent_fed_avg.json", "w") as json_file:
        json.dump(training_history_loss_cent, json_file)