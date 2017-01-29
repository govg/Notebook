---
title: Project discussion
---

-----------

###Paper 1 : Online and Stochastic Gradient Methods for Non-Decomposoable Loss Functions


[link](file:///mnt/Data/Academics/Semester%202/CS773/Papers/nondecomp.pdf)



####Abstract
Online framework for optimizing non-decomposable measures, like
precision @k, F-measure.


####Introduction

Measures are frequently non-decomposable, they can't be expressed as sum of 
individual losses. 

#####Contributions : 

1. Framework for Online learning
2. Efficient FTRL algorithm
3. Sublinear regret for Precision@k, pAUC
4. Experiments

#### Formulation 

$x_{1:t} = \{ x_1, \dots, x_t \} \in \mathbb{R}^d$ - Observed data points

$y_{1:t} = \{ y_1, \dots, y_t \}, \in \{ -1, 1\}$ - true binary labels

$y$ 





