from typing import Tuple, Dict

import random


import torch
import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist
import general_server






def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    
    # Generate a random integer between 0 and 9
    random_integer = random.randint(0, 9)
    trainloaders, testloaders = general_server.load_datasets()
    trainloader, testloader = trainloaders[random_integer], testloaders[random_integer]

    net = general_mnist.Net().to(DEVICE)
    net.train()
    print("Start training")
    general_mnist.train_fedavg(model=net, trainloader=trainloader, local_epochs=10, device=DEVICE, lr=0.1)
    net.eval()
    print("Evaluate model")
    loss, accuracy = general_mnist.test(model=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()

