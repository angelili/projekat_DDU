"""Flower client using PyTorch for FashionMNIST image classification."""

import os

import torch

import flwr as fl

import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist
import client

Benchmark=True


def main() -> None:
    """Load data, start Client."""

    fedl_no_proxy=True


    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data
    trainloader, testloader = client.load_data([8, 9, 0, 1, 2])
    # Set device
    DEVICE= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model
    model = general_mnist.Net().to(DEVICE).train()

    # Save the variables to a file
    if Benchmark==True:
        data_9 = {
            'trainloader': trainloader,
            'testloader': testloader,

        }
        torch.save(data_9, 'data_9.pth')

    # Start client
    client_9 = client.MnistClient_fedavg(model, trainloader, testloader, DEVICE)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client_9)

if __name__ == "__main__":
    main()