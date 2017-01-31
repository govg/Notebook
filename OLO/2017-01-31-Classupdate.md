---
title : Class update 31st January
author : Govind
---

[Scribed notes (old session)](http://web.cse.iitk.ac.in/users/purushot/courses/olo/2015-16-w/material/scribes/lec6.pdf)


## Perceptron 

Perceptron has a very simple update equation, on mistake do the following update: 

$w_{t+1} = w_t + y^t {\bf x}^t$

We analyze it in the hard margin case, as well as the soft margin case. Note 
the soft margin case is slightly different from that of the SVM formulation,
we do not allow for slack, rather we allow some misclassification within the
margin for the benchmark classifier.

### Mistake bound

For the realizable setting with a margin $\gamma$, we get a bound on the
total number of perceptron updates to be made. This is trivial, it 
follows from the fact that : 
1. Weights of the classifier are bounded by some value (assumption)
2. All data points present in our training data are bounded by a ball

Using this and a potential function that denotes the cosine similarity of
the target weight vector and the learned vector, we can show that the updates
will be bounded above.

### Extension to non-realizable setting 

For the perceptron in the non-realizable setting, we choose to use a surrogate
loss that penalizes the perceptron whenever it makes a mistake. This is similar
to a truncated version of the hinge loss. Our potential now becomes : 

$\Phi_{t+1} \geq \Phi_t + \gamma - \left[ \gamma - y^t \langle x^t, w^o\rangle \right]$

Here the second term denotes a hinge loss like term. We then bound this by
finding a quadratic in the number of mistakes (note that our earlier bound for
increase in the weight vector still holds). 

One remarkable thing is that the perceptron is optimal in the realizable 
setting.

### Margintrons

Also discussed was the effort to learn a perceptron with a hard margin, aptly
termed a margintron. The only difference is that we update the perceptron
weights whenever there's a mistake **and** whenever there is a margin of
less than $\frac{\gamma}{2}$ in our prediction. 

**Claim** : If there exists a $\gamma$ margin classifier with $\| w^o \| =1$, 
then we can bound the number of mistakes by : $M_T \leq \frac{4.5}{\gamma^2}$.

Further reading : 

1. [Perceptron](https://en.wikipedia.org/wiki/Perceptron)
2. [Convergence of Perceptrons](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-867-machine-learning-fall-2006/lecture-notes/lec2.pdf)
3. [Margin Perceptrons](https://www.cs.cmu.edu/~avrim/ML10/lect0125.pdf)
