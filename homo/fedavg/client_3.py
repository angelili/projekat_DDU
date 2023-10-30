"""Flower client using PyTorch for FashionMNIST image classification."""
import os

import torch

import flwr as fl

import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist
import client



def main() -> None:
    """Load data, start MnistClient."""

    fedl_no_proxy=True


    if fedl_no_proxy:
      os.environ["http_proxy"] = ""
      os.environ["https_proxy"] = ""
    
    # Load the variables as data
    data = torch.load('/home/s124m21/projekat_DDU/homo/fedavg/data_3.pt')
    # Retrieve the variables
    trainloader = data['trainloader']
    testloader = data['testloader']

    #Set up the device
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load model
    model = general_mnist.Net().to(DEVICE)

    
    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))


    # Start client
    client_3 = client.MnistClient(model, trainloader, testloader, DEVICE)
    fl.client.start_numpy_client(server_address="10.30.0.254:9000",
    client=client_3)



if __name__ == "__main__":
    main()