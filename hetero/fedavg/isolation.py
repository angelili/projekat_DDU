from typing import Tuple, Dict

import torch


import matplotlib.pyplot as plt

import sys
sys.path.append('/home/s124m21/projekat_DDU')

# import your module without specifying the full path
import general_mnist
import client




def ploting(losses,accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('photo.png')


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader  = client.load_data([0, 1, 2, 3 ,4])
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
