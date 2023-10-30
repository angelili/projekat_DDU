import torch

from collections import OrderedDict
from typing import Callable, Optional, Tuple, Dict, Union, List

import flwr as fl
from flwr.common import Metrics


import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split


import matplotlib.pyplot as plt


import general_mnist
from general_mnist import FED_BN

DATA_ROOT = "/home/s124m21/projekat_DDU/dataset"



#server has always the whole test set for the evaluation

def load_data_server():
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    testset_server = DataLoader(testset, batch_size=50, shuffle=True)
    

    return testset_server    


#THIS SECTION IS ONLY FOR THE HOMOGENEOUS CASE
#partition both the training and test set on 10 clients!

def load_datasets():
  

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859), (0.3530))]
    )

    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

    # Split the training set into 10 partitions
    size_train = len(trainset) // 10

    partition_len = [size_train] * 10
    partitions_train= random_split(
        trainset, partition_len, torch.Generator().manual_seed(42))

    # Split the test set into 10 partitions in the same way
    size_test = len(testset) // 10

    partition_len = [size_test] * 10
    partitions_test= random_split(
        testset, partition_len, torch.Generator().manual_seed(42))


    # Create DataLoader for each partition
    trainloaders=[]
    testloaders = []
    for partition_train, partition_test in zip(partitions_train, partitions_test):
        trainloaders.append(
            DataLoader(partition_train, batch_size=32, shuffle=True)
        )
        testloaders.append(
            DataLoader(partition_test, batch_size=32, shuffle=False)
        )

    return trainloaders, testloaders

#this is the function I used in my master thesis, the server has the whole test set which is not seen by the clients. 
#so the test set is left intact and wil be used by the central server to asses the performance of the global model.
# The FashionMNIST trainset is partitioned among 10 clients,
#  the partition in each clientis then partitioned as trainset and testset of the client.
#  This old version of partitioning produced better accuracy results on the global model on the server??

def load_datasets_old():
    
   
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2859), (0.3530))]
    )

    trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)

 

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // 10

    partition_len = [num_images] * 10

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    testloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(0.1 * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(42)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=32, shuffle=True)
        )
        testloaders.append(
            DataLoader(for_val, batch_size=32, shuffle=False)
        )


    return trainloaders, testloaders
#The following functions are evaluation functions.

#function which tests the global model on the server
#we out the empty dictionaries to be filled in, and their keys
def get_evaluate_fn(
    testset: FashionMNIST, dict_acc: Dict, dict_loss: Dict, key_acc: str, key_loss: str
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Union[int, float, complex]]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire FashionMNIST test set for evaluation."""

        # determine device for the server, and call the model on it, place the parameters
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
       
       
        
        loss, accuracy = general_mnist.test(model, testset, device)
        #fill the dictionaries
        dict_acc[key_acc].append(accuracy)
        dict_loss[key_loss].append(loss)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate



#same for every algorithm, this function takes the returned values from evaluation method in MnistClient
def weighted_average(dict: Dict, key: str) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        dict[key].append(sum(accuracies)/sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {key : sum(accuracies) / sum(examples)}
    return evaluate



#for pfedme, pfedme_new because they have local, personalized model tto return from fit method in MnistClient
def agg_metrics_train_both_pfedme(dict: Dict, key_local:str, key_person:str ) -> Metrics:    
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies_person = [num_examples * m["accuracy_person"] for num_examples, m in metrics]
        accuracies_local = [num_examples * m["accuracy_local"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        dict[key_person].append(sum(accuracies_person)/sum(examples))
        dict[key_local].append(sum(accuracies_local)/sum(examples))


        # Aggregate and return custom metric (weighted average)
        return {key_person: sum(accuracies_person)/sum(examples), key_local: sum(accuracies_local)/sum(examples)}
    return evaluate

#for pfedme, pfedme_new because it has ave local model results to return from fit method in MnistClient
def agg_metrics_train_fedavg(dict: Dict, key:str ) -> Metrics:  
    def evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy_local"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        dict["accuracy_local_fedavg"].append(sum(accuracies)/sum(examples))

        # Aggregate and return custom metric (weighted average)
        return {"accuracy_local_fedavg": sum(accuracies) / sum(examples)}
    
    return evaluate

#same for every algorithm, this function takes the returned values from evaluation method in MnistClient
def weighted_average(dict: Dict, key: str) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    def evaluate (metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        dict[key].append(sum(accuracies)/sum(examples))
        # Aggregate and return custom metric (weighted average)
        return {key : sum(accuracies) / sum(examples)}
    return evaluate

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