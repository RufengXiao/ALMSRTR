This code is the implementation of the algorithms and experiments in our paper "Riemannian Trust Region Methods for SC $^1$ Minimization".

* `algorithm/`: the source code for the other algorithms.

### Introduction
This code can be run in Matlab R2021b. 

In order to run the comparison experiment properly, we recommend referring to `algorithm/almssn-master/README.md`.

* `CMRTR/`: the implementation of our algorithms for the compressed modes (CM) problem experiments.
* `SPCARTR/`: the implementation of our algorithms for the  Sparse Principal Component Analysis (SPCA) problem experiments.

### How to get the results
* To get the results of the Table 2 in our paper, please run `synthetic_compare_CM.m`. The results obtained will be stored in the `log` folder.
* To get the results of the Table 3 in our paper, please run `synthetic_compare_SPCA.m`. The results obtained will be stored in the `log` folder.
* To get the results of the Table 4 in our paper, please run `real_compare_SPCA.m`. The results obtained will be stored in the `log` folder.
* To get the results of the Figure 1 in our paper, please run `plot_CMs.m`. The results obtained will be stored in the `figure` folder.
* To get the results of the Figure 2 in our paper, please run `plot_SPCA.m`. The results obtained will be stored in the `figure` folder.

### Citation
If you found the provided code useful, please cite our work.

If you have any questions, please contact us.
