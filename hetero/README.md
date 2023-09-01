This folder has everything needed for implementing a heterogeneity scenario.In order to model a heterogeneous setting in terms of local data sizes and
classes, each client is allocated a different training set with the size in the range
of [4000, 6000], and 5 classes. Testing set is 10% of the training set size. They
are distributed among 4 available CUDA nodes. The distribution of classes
among clients is modeled as: Client 1 has [0, 1, 2, 3, 4], Client 2 has [1, 2, 3, 4,
5], Client 3 has [2, 3, 4, 5, 6], and so on. Client 10 has [9, 0, 1, 2, 3]. However, this could be easily changed into a different setting
