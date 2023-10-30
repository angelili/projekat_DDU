from typing import Tuple, Dict

import random


import torch
import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist
import client





def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    
    # Generate a random integer between 1 and 10 (inclusive)
    random_integer = random.randint(0, 9)
    # Load the variables as data, using the random integer to construct the file path
    data = f'/home/s124m21/projekat_DDU/homo/fedavg/data_{random_integer}.pt'
    # Retrieve the variables
    trainloader = data['trainloader']
    testloader = data['testloader']
   
    trainloader, testloader = client.load_data()
    net = general_mnist.Net().to(DEVICE)
    net.train()
    print("Start training")
    general_mnist.train_fedavg(net=net, trainloader=trainloader, local_epochs=10, device=DEVICE, lr=0.1)
    net.eval()
    print("Evaluate model")
    loss, accuracy = general_mnist.test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()

