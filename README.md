# projekat_DDU
Implementation of Personalized FL w Moreau Envelopes in Flower

This directory contains the code for the Master's thesis that addresses the problem of Personalized Federated Learning.
The main algorithm is the pFedMe, from the following paper : [Personalized Federated Learning with Moreau
Envelopes](https://arxiv.org/pdf/2006.08848.pdf). The pracitcal novelty of this thesis is its implementation in [Flower](https://flower.dev/).
This framework simplifies the development and deployment of FL(Federated Learning) systems. The thesis also provides implementations of FedAvg, and of a modified version of pFedMe, pFedMe_new.
The focus is on the heterogeneity among data distributions of clients, since that is the key issue that Personalized FL tackles. 

In this version, there are 10 different clients. Each client has its dataset, and its client.py. Since the experimentation took place within the faculty’s ”Axiom” computer cluster infrastructure. Each client was trained on a CUDA partition node, this is specified in client.sh while the main,access node of the cluster hosted the server. Clients are then grouped into run.sh, which is then used for invoking them.
So, in every experiment scenario with FL here, we start the server in a terminal as follows:
```
Look! You can see my backticks.
```
