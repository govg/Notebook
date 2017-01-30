---
title: Project discussion
author : Govind
---

-----------

###Paper 1 : Online and Stochastic Gradient Methods for Non-Decomposable Loss Functions


[link](http://www.cse.iitk.ac.in/users/purushot/papers/nondecomp.pdf)



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

#### Main ideas


Use a convex surrogate for the given penalty functions.

Mainly adapt from structural SVM penalty function given as : 

$l_{P}(\textbf{w}) = max \sum^T_{i=1} (\Delta y_i) x^T_iw - \mathcal{P}(y',y)$

Precision at K and Partial AUC are given corresponding surrogate losses.


For surrogate losses, use the notion of instantaneous regret as $\Delta \mathcal{L}$ 
when you add a new data point. Now minimize cumulative regret with respect to best
possible $w$ vector you can find.

Update equation remains simple : FTRL - update to solution of objective function, 
where our objective is minimizing total loss till time t and regularizer. 

Note that total regret till time t, 

$\frac{1}{T} \sum^T \mathcal{L}_t(w_t) = \frac{1}{T} \sum^T l_{1:t}(w_t) - l_{1:t-1}(w_t)$


This is telescoping, and thus can be written as: $ \frac{1}{T} l_{1:t}(w_t)$

Proof of regret follows this.


Online to batch conversion follows : Consider the data to be split into batches, define
regret and loss similarly over batches instead of over a single datapoint. 

The algorithms above require solving an optimization problem, for which now SGD
type methosds are being promoted. Uniform convergence is when given a sample of $s$
from a total population of $n$ points, the loss is below a certain threshold 
$\alpha$ with probability $1-\delta$. The $\alpha$ is poly$(s, \delta)$. This
is a property solely of the loss function, with respect to a set of predictors.

Given are two algorithms 1PMB and 2PMB, requiring 1,2 passes over the stream
respectively. 2PMB manages to exploit relations across epochs, by subsampling
the relevant labels (this is useful in case of label imbalance).

Regret bounds follow for the 1PMB algorithm, which is in terms of the above 
regret, as well as some extra terms poly in $\alpha$.

Further followed by $\alpha$ for pAUC as well as p@K rates. 

Ends with experiments. 


Refer to appendix for the proofs given. 

-------


###Paper 2 : Optimizing Performance Measures - Tale of two classes. 


[link](http://www.cse.iitk.ac.in/users/purushot/papers/sgd-tpr-tnr.pdf)


####Abstract
Performance measures that can be expressed as combinations of TPR and
TNR. Provide algorithms for the case where they are convex combinations,
as well as psuedo-linear functions. 


####Introduction

Lots of measures like F-measure, etc are expressed as convex combinations
of the TPR and the TNR rate. Other methods (like the paper above) require
maintaining a large buffer, or surrogate losses. 

#####Contributions 

1. Truly point wise updates, whereas the older method relied on batch
	errors and the like.
2. Works for combinations that are convex, as well as psuedo-linear
3. Efficient in terms of memory.
4. Extremely fast convergence

#### Main ideas

##### Convex combinations

Deal with the case where the loss functions are convex combinations of
TPR and TNR. Stochastic Primal Dual Ascent method (SPADE) enables identical
regret bounds, no batch bias in the algorithm, and faster rates of 
convergence. 

Define stability of a performance measure - sensitivity to change in
the data. Also relates to Lipschitzness of the link funciton.


Define sufficient dual region - Accounts for preserving in the projection
step. Somehow relates the Lipschitzness of the function with the location
it is projected onto. 

Theorem follows that the stability of a performance measure is Lipschitz
if its sufficient region is bounded in a ball of radius marked by the
Lipschitz measure. 

Theorem follows that with probability $1-\delta$, the algorithm (SPADE)
outputs an averaged weight vector that has close to optimal performance.
Proof is given in the appendix. 

[ ] Cover this correctly 

Consider when the link function is not Lipschitz. The notion of a 
sufficient dual region can't be used correctly. In this case, the authors
propose assigning non-zero reward at every step. 



##### Pseudo-linear measures


F-measure is one such measure.

Authors propose an alternating minimization procedure (STAMP) for such 
measures. Define a *valuation function* for the performance measure in this
case, it provides for a lower bound on the performance measure. AMP is the
non-stochastic version for which they have proved regret bounds, STAMP is
the stochastic version.

Proofs of regret bounds follow later. 
