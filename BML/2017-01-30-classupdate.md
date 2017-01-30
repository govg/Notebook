---
title : Class update 30th January
author : Govind
---

[Slides](http://web.cse.iitk.ac.in/users/piyush/tmp/bml_17/bml_lec7_slides.pdf)

## The exponential family and Generalized Linear Models


The class covers the basics of the exponential family of probability
distributions. 

General form of the exponential family is : 

$P(D|\theta) =  h(D) \exp\{\theta^T\phi(D) - A(\theta)\}$

Here, we refer to $Z(\theta)$ as the cumulant, or partition function, 
$\theta$ is the parameter of the distribution, $\phi(D)$ is termed
the sufficient statistic of the distribution, $h(D)$ is a constant
w.r.t. $\theta$. 


Derivatives of $A(\theta)$ give us the cumulants of the distribution.

The sufficient statistics of the parametrized distribution match the 
empirical distribution for the MLE estimates. 

If we have a prior on our distributions, we instead match the sufficient
statistics with empirical + pseudo estimates (MAP).

The conjugancy of distributions arise when we do the prior x likelihood
multiplication. If the resulting distribution is naturally in the 
exponential form, it is easy(maybe) to do the integral.


Some mathematical manipulation can give us the following form: 

$P(\mathcal{D}' | \mathcal{D}) = \frac{Z_c(\nu + N + N', \tau + \phi(\mathcal{D}) + \phi(\mathcal{D}'))}{Z_c(\nu + N, \tau + \phi(\mathcal{D}))} * \left[ \prod^{\mathcal{D}'} h(x_i) \right]$


This is the ratio of the marginal under the new data and the marginal under the
training + prior data. A high probability would reflect the fact that the two
distributions match pretty well, and vice versa. 


### Further reading : 

1. [Sufficient Statistics](https://en.wikipedia.org/wiki/Sufficient_statistic)
2. [Exponential Family](https://en.wikipedia.org/wiki/Exponential_family)
3. [Moment Matching](https://en.wikipedia.org/wiki/Method_of_moments_(statistics))
4. [Marginal](https://en.wikipedia.org/wiki/Marginal_distribution)
