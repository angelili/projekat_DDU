"""Flower client using PyTorch for FashionMNIST image classification."""

import os

import torch

import flwr as fl

import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist
import client
from general_mnist import Benchmark

def main() -> None:
    """Load data, start Client."""

    fedl_no_proxy=True


    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    # Load data
    trainloader, testloader = client.load_data([0, 1, 2, 3, 4])
    # Set device
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Load model
    model = general_mnist.Net().to(DEVICE).train()
    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))


    # Save the variables to a file
    if Benchmark==True:
        data_1 = {
            'trainloader': trainloader,
            'testloader': testloader
        }
        torch.save(data_1, 'data_1.pth')

    # Start client
    client_1 = client.MnistClient(model, trainloader, testloader, DEVICE)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client_1)



if __name__ == "__main__":
    main()