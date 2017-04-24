---
title : Project Notes
author : Govind
---

This is a list of all the papers to be studied as well as their abstracts. As time goes by, 
I will update this post with comments on each paper, how relevant it is, and also try and
organize this set into a coherent collection.


---

### 2003

####[Generalized Mean Field algorithm for Variational Inference](https://arxiv.org/abs/1212.2512)

UAI 2003

The mean field methods, which entail approximating intractable probability distributions variationally with distributions from a tractable family, enjoy high efficiency, guaranteed convergence, and provide lower bounds on the true likelihood. But due to requirement for model-specific derivation of the optimization equations and unclear inference quality in various models, it is not widely used as a generic approximate inference algorithm. In this paper, we discuss a generalized mean field theory on variational approximation to a broad class of intractable distributions using a rich set of tractable distributions via constrained optimization over distribution spaces. We present a class of generalized mean field (GMF) algorithms for approximate inference in complex exponential family models, which entails limiting the optimization over the class of cluster-factorizable distributions. GMF is a generic method requiring no model-specific derivations. It factors a complex model into a set of disjoint variable clusters, and uses a set of canonical fix-point equations to iteratively update the cluster distributions, and converge to locally optimal cluster marginals that preserve the original dependency structure within each cluster, hence, fully decomposed the overall inference problem. We empirically analyzed the effect of different tractable family (clusters of different granularity) on inference quality, and compared GMF with BP on several canonical models. Possible extension to higher-order MF approximation is also discussed.

---

### 2012

####[Fast Variational Inference in the conjugate exponential family](http://papers.nips.cc/paper/4766-fast-variational-inference-in-the-conjugate-exponential-family)

NIPS 2012

We present a general method for deriving collapsed variational inference algorithms for probabilistic models in the conjugate exponential family. Our method unifies many existing approaches to collapsed variational inference. Our collapsed variational inference leads to a new lower bound on the marginal likelihood. We exploit the information geometry of the bound to derive much faster optimization methods based on conjugate gradients for these models. Our approach is very general and is easily applied to any model where the mean field update equations have been derived. Empirically we show significant speed-ups for probabilistic models optimized using our bound.


####[Distributed Vartiational Inference in Sparse GP Regression and Latent Variable Models](http://papers.nips.cc/paper/5593-distributed-variational-inference-in-sparse-gaussian-process-regression-and-latent-variable-models)

NIPS 2012
Gaussian processes (GPs) are a powerful tool for probabilistic inference over functions. They have been applied to both regression and non-linear dimensionality reduction, and offer desirable properties such as uncertainty estimates, robustness to over-fitting, and principled ways for tuning hyper-parameters. However the scalability of these models to big datasets remains an active topic of research. We introduce a novel re-parametrisation of variational inference for sparse GP regression and latent variable models that allows for an efficient distributed algorithm. This is done by exploiting the decoupling of the data given the inducing points to re-formulate the evidence lower bound in a Map-Reduce setting. We show that the inference scales well with data and computational resources, while preserving a balanced distribution of the load among the nodes. We further demonstrate the utility in scaling Gaussian processes to big data. We show that GP performance improves with increasing amounts of data in regression (on flight data with 2 million records) and latent variable modelling (on MNIST). The results show that GPs perform better than many common models often used for big data.

####[Truncation-free Online Variational Inference for Bayesian Nonparametric Models](http://papers.nips.cc/paper/4534-truncation-free-online-variational-inference-for-bayesian-nonparametric-models)

NIPS 2012

We present a truncation-free online variational inference algorithm for Bayesian nonparametric models. Unlike traditional (online) variational inference algorithms that require truncations for the model or the variational distribution, our method adapts model complexity on the fly. Our experiments for Dirichlet process mixture models and hierarchical Dirichlet process topic models on two large-scale data sets show better performance than previous online variational inference algorithms.

####[Stochastic Variational Inference](https://arxiv.org/abs/1206.7051)

2012

We develop stochastic variational inference, a scalable algorithm for approximating posterior distributions. We develop this technique for a large class of probabilistic models and we demonstrate it with two probabilistic topic models, latent Dirichlet allocation and the hierarchical Dirichlet process topic model. Using stochastic variational inference, we analyze several large collections of documents: 300K articles from Nature, 1.8M articles from The New York Times, and 3.8M articles from Wikipedia. Stochastic inference can easily handle data sets of this size and outperforms traditional variational inference, which can only handle a smaller subset. (We also show that the Bayesian nonparametric topic model outperforms its parametric counterpart.) Stochastic variational inference lets us apply complex Bayesian models to massive data sets.

####[Variational Inference in Nonconjugate Models](https://arxiv.org/abs/1209.4360)

2012

Mean-field variational methods are widely used for approximate posterior inference in many probabilistic models. In a typical application, mean-field methods approximately compute the posterior with a coordinate-ascent optimization algorithm. When the model is conditionally conjugate, the coordinate updates are easily derived and in closed form. However, many models of interest---like the correlated topic model and Bayesian logistic regression---are nonconjuate. In these models, mean-field methods cannot be directly applied and practitioners have had to develop variational algorithms on a case-by-case basis. In this paper, we develop two generic methods for nonconjugate models, Laplace variational inference and delta method variational inference. Our methods have several advantages: they allow for easily derived variational algorithms with a wide class of nonconjugate models; they extend and unify some of the existing algorithms that have been derived for specific models; and they work well on real-world datasets. We studied our methods on the correlated topic model, Bayesian logistic regression, and hierarchical Bayesian logistic regression.

####[Variational Bayesian Inference with Stochastic Search](https://arxiv.org/abs/1206.6430)

ICML 2012

Mean-field variational inference is a method for approximate Bayesian posterior inference. It approximates a full posterior distribution with a factorized set of distributions by maximizing a lower bound on the marginal likelihood. This requires the ability to integrate a sum of terms in the log joint likelihood using this factorized distribution. Often not all integrals are in closed form, which is typically handled by using a lower bound. We present an alternative algorithm based on stochastic optimization that allows for direct optimization of the variational lower bound. This method uses control variates to reduce the variance of the stochastic search gradient, in which existing lower bounds can play an important role. We demonstrate the approach on two non-conjugate models: logistic regression and an approximation to the HDP.

####[Nonparametric Variational Inference](https://arxiv.org/abs/1206.4665)

ICML 2012

Variational methods are widely used for approximate posterior inference. However, their use is typically limited to families of distributions that enjoy particular conjugacy properties. To circumvent this limitation, we propose a family of variational approximations inspired by nonparametric kernel density estimation. The locations of these kernels and their bandwidth are treated as variational parameters and optimized to improve an approximate lower bound on the marginal likelihood of the data. Using multiple kernels allows the approximation to capture multiple modes of the posterior, unlike most other variational approximations. We demonstrate the efficacy of the nonparametric approximation with a hierarchical logistic regression model and a nonlinear matrix factorization model. We obtain predictive performance as good as or better than more specialized variational methods and sample-based approximations. The method is easy to apply to more general graphical models for which standard variational methods are difficult to derive.


---

### 2013

####[Adaptive Learning Rate for Stochastic Variational Inference](http://www.jmlr.org/proceedings/papers/v28/ranganath13.pdf)

ICML 2013

Stochastic  variational  inference  finds  good
posterior approximations of probabilistic mod-
els with very large data sets.  It optimizes the
variational objective with stochastic optimiza-
tion, following noisy estimates of the natural
gradient.  Operationally, stochastic inference
iteratively subsamples from the data, analyzes
the subsample, and updates parameters with
a decreasing learning rate.  However, the algo-
rithm is sensitive to that rate, which usually
requires hand-tuning to each application.  We
solve  this  problem  by  developing  an  adap-
tive  learning  rate  for  stochastic  variational
inference.   Our  method  requires  no  tuning
and is easily implemented with computations
already made in the algorithm.  We demon-
strate our approach with latent Dirichlet al-
location applied to three large text corpora.
Inference with the adaptive learning rate con-
verges faster and to a better approximation
than the best settings of hand-tuned rates.

####[Variance Reduction for Stochastic Gradient Optimization](http://papers.nips.cc/paper/5034-variance-reduction-for-stochastic-gradient-optimization)

NIPS 2013

Stochastic gradient optimization is a class of widely used algorithms for training machine learning models. To optimize an objective, it uses the noisy gradient computed from the random data samples instead of the true gradient computed from the entire dataset. However, when the variance of the noisy gradient is large, the algorithm might spend much time bouncing around, leading to slower convergence and worse performance. In this paper, we develop a general approach of using control variate for variance reduction in stochastic gradient. Data statistics such as low-order moments (pre-computed or estimated online) is used to form the control variate. We demonstrate how to construct the control variate for two practical problems using stochastic gradient optimization. One is convex---the MAP estimation for logistic regression, and the other is non-convex---stochastic variational inference for latent Dirichlet allocation. On both problems, our approach shows faster convergence and better performance than the classical approach.

####[Memoized Online Variational Inference for Dirichlet Process Mixture Models](http://papers.nips.cc/paper/4969-memoized-online-variational-inference-for-dirichlet-process-mixture-models)

NIPS 2013

Variational inference algorithms provide the most effective framework for large-scale training of Bayesian nonparametric models. Stochastic online approaches are promising, but are sensitive to the chosen learning rate and often converge to poor local optima. We present a new algorithm, memoized online variational inference, which scales to very large (yet finite) datasets while avoiding the complexities of stochastic gradient. Our algorithm maintains finite-dimensional sufficient statistics from batches of the full dataset, requiring some additional memory but still scaling to millions of examples. Exploiting nested families of variational bounds for infinite nonparametric models, we develop principled birth and merge moves allowing non-local optimization. Births adaptively add components to the model to escape local optima, while merges remove redundancy and improve speed. Using Dirichlet process mixture models for image clustering and denoising, we demonstrate major improvements in robustness and accuracy.

####[Black Box Variational Inference](https://arxiv.org/abs/1401.0118)

2013

Variational inference has become a widely used method to approximate posteriors in complex latent variables models. However, deriving a variational inference algorithm generally requires significant model-specific analysis, and these efforts can hinder and deter us from quickly developing and exploring a variety of models for a problem at hand. In this paper, we present a "black box" variational inference algorithm, one that can be quickly applied to many models with little additional derivation. Our method is based on a stochastic optimization of the variational objective where the noisy gradient is computed from Monte Carlo samples from the variational distribution. We develop a number of methods to reduce the variance of the gradient, always maintaining the criterion that we want to avoid difficult model-based derivations. We evaluate our method against the corresponding black box sampling based methods. We find that our method reaches better predictive likelihoods much faster than sampling methods. Finally, we demonstrate that Black Box Variational Inference lets us easily explore a wide space of models by quickly constructing and evaluating several models of longitudinal healthcare data.

####[Stochastic Collapsed Variational Bayesian Inference for LDA](https://arxiv.org/abs/1305.2452)

2013

In the internet era there has been an explosion in the amount of digital text information available, leading to difficulties of scale for traditional inference algorithms for topic models. Recent advances in stochastic variational inference algorithms for latent Dirichlet allocation (LDA) have made it feasible to learn topic models on large-scale corpora, but these methods do not currently take full advantage of the collapsed representation of the model. We propose a stochastic algorithm for collapsed variational Bayesian inference for LDA, which is simpler and more efficient than the state of the art method. We show connections between collapsed variational Bayesian inference and MAP estimation for LDA, and leverage these connections to prove convergence properties of the proposed algorithm. In experiments on large-scale text corpora, the algorithm was found to converge faster and often to a better solution than the previous method. Human-subject experiments also demonstrated that the method can learn coherent topics in seconds on small corpora, facilitating the use of topic models in interactive document analysis software.


---

### 2014

####[Bayesian Nonparametric Poisson Factorization for Recommendation Systems](http://www.jmlr.org/proceedings/papers/v33/gopalan14.pdf)

AISTATS 2013

We develop a Bayesian nonparametric Pois-
son factorization model for recommendation
systems.
Poisson   factorization   implicitly
models  each  user’s  limited  budget  of  atten-
tion (or money) that allows consumption of
only  a  small  subset  of  the  available  items.
In  our  Bayesian  nonparametric  variant,  the
number of latent components is theoretically
unbounded  and  effectively  estimated  when
computing a posterior with observed user be-
havior  data.   To  approximate  the  posterior,
we  develop  an  efficient  variational  inference
algorithm.   It  adapts  the  dimensionality  of
the latent components to the data,  only re-
quires iteration over the user/item pairs that
have been rated, and has computational com-
plexity on the same order as for a parametric
model  with  fixed  dimensionality.   We  stud-
ied our model and algorithm with large real-
world  data  sets  of  user-movie  preferences.
Our  model  eases  the  computational  burden
of searching for the number of latent compo-
nents and gives better predictive performance
than its parametric counterpart.

####[On convergence of Stochastic Variational Inference in Bayesian Networks](https://arxiv.org/abs/1507.04505)

NIPS 2014

We highlight a pitfall when applying stochastic variational inference to general Bayesian networks. For global random variables approximated by an exponential family distribution, natural gradient steps, commonly starting from a unit length step size, are averaged to convergence. This useful insight into the scaling of initial step sizes is lost when the approximation factorizes across a general Bayesian network, and care must be taken to ensure practical convergence. We experimentally investigate how much of the baby (well-scaled steps) is thrown out with the bath water (exact gradients). 

####[Beta Process non-negative matrix factorization with Stochastic Mean-field Variational Inference](https://arxiv.org/abs/1411.1804)

2014

Beta process is the standard nonparametric Bayesian prior for latent factor model. In this paper, we derive a structured mean-field variational inference algorithm for a beta process non-negative matrix factorization (NMF) model with Poisson likelihood. Unlike the linear Gaussian model, which is well-studied in the nonparametric Bayesian literature, NMF model with beta process prior does not enjoy the conjugacy. We leverage the recently developed stochastic structured mean-field variational inference to relax the conjugacy constraint and restore the dependencies among the latent variables in the approximating variational distribution. Preliminary results on both synthetic and real examples demonstrate that the proposed inference algorithm can reasonably recover the hidden structure of the data.

####[Structured Stochastic Variational Inference](https://arxiv.org/abs/1404.4114)

2014

Stochastic variational inference makes it possible to approximate posterior distributions induced by large datasets quickly using stochastic optimization. The algorithm relies on the use of fully factorized variational distributions. However, this "mean-field" independence approximation limits the fidelity of the posterior approximation, and introduces local optima. We show how to relax the mean-field approximation to allow arbitrary dependencies between global parameters and local hidden variables, producing better parameter estimates by reducing bias, sensitivity to local optima, and sensitivity to hyperparameters.

####[Smoothed Gradient for Stochastic Variational Inference](https://arxiv.org/abs/1406.3650)

NIPS 2014

Stochastic variational inference (SVI) lets us scale up Bayesian computation to massive data. It uses stochastic optimization to fit a variational distribution, following easy-to-compute noisy natural gradients. As with most traditional stochastic optimization methods, SVI takes precautions to use unbiased stochastic gradients whose expectations are equal to the true gradients. In this paper, we explore the idea of following biased stochastic gradients in SVI. Our method replaces the natural gradient with a similarly constructed vector that uses a fixed-window moving average of some of its previous terms. We will demonstrate the many advantages of this technique. First, its computational cost is the same as for SVI and storage requirements only multiply by a constant factor. Second, it enjoys significant variance reduction over the unbiased estimates, smaller bias than averaged gradients, and leads to smaller mean-squared error against the full gradient. We test our method on latent Dirichlet allocation with three large corpora.

####[Markov Chain Monte Carlo and Variational Inference](https://arxiv.org/abs/1410.6460)

2014

Recent advances in stochastic gradient variational inference have made it possible to perform variational Bayesian inference with posterior approximations containing auxiliary random variables. This enables us to explore a new synthesis of variational inference and Monte Carlo methods where we incorporate one or more steps of MCMC into our variational approximation. By doing so we obtain a rich class of inference algorithms bridging the gap between variational methods and MCMC, and offering the best of both worlds: fast posterior approximation through the maximization of an explicit objective, with the option of trading off additional computation for additional accuracy. We describe the theoretical foundations that make this possible and show some promising first results.

####[Stochastic Variational Inference for Hidden Markov Models](https://arxiv.org/abs/1411.1670)

NIPS 2014

Variational inference algorithms have proven successful for Bayesian analysis in large data settings, with recent advances using stochastic variational inference (SVI). However, such methods have largely been studied in independent or exchangeable data settings. We develop an SVI algorithm to learn the parameters of hidden Markov models (HMMs) in a time-dependent data setting. The challenge in applying stochastic optimization in this setting arises from dependencies in the chain, which must be broken to consider minibatches of observations. We propose an algorithm that harnesses the memory decay of the chain to adaptively bound errors arising from edge effects. We demonstrate the effectiveness of our algorithm on synthetic experiments and a large genomics dataset where a batch algorithm is computationally infeasible.

####[Stochastic Variational Inference for Bayesian Time Series Models](http://www.jmlr.org/proceedings/papers/v32/johnson14.pdf)

ICML 2014

Bayesian models provide powerful tools for an-
alyzing  complex  time  series  data,  but  perform-
ing inference with large datasets is a challenge.
Stochastic variational inference (SVI) provides a
new framework for approximating model poste-
riors with only a small number of passes through
the data, enabling such models to be fit at scale.
However,  its  application  to  time  series  models
has not been studied.
In  this  paper  we  develop  SVI  algorithms  for
several  common  Bayesian  time  series  models,
namely the hidden Markov model (HMM), hid-
den semi-Markov model (HSMM), and the non-
parametric HDP-HMM and HDP-HSMM. In ad-
dition, because HSMM inference can be expen-
sive even in the minibatch setting of SVI, we de-
velop fast approximate updates for HSMMs with
durations distributions that are negative binomi-
als or mixtures of negative binomials.

####[Scalable and Robust Bayesian Inference via the Median Posterior](http://www.jmlr.org/proceedings/papers/v32/minsker14.pdf)

ICML 2014

Many  Bayesian  learning  methods  for  massive
data benefit from working with small subsets of
observations.   In particular,  significant progress
has been made in scalable Bayesian learning via
stochastic  approximation.    However,  Bayesian
learning  methods  in  distributed  computing  en-
vironments  are  often  problem-  or  distribution-
specific  and  use  ad  hoc  techniques.    We  pro-
pose  a  novel  general  approach  to  Bayesian  in-
ference that is scalable and robust to corruption
in the data.  Our technique is based on the idea
of splitting the data into several non-overlapping
subgroups,  evaluating  the  posterior  distribution
given each independent subgroup, and then com-
bining the results.  Our main contribution is the
proposed  aggregation  step  which  is  based  on
finding  the  geometric  median  of  subset  poste-
rior distributions.  Presented theoretical and nu-
merical results confirm the advantages of our ap-
proach

####[Distributed Bayesian Posterior Sampling via Moment Sharing](http://papers.nips.cc/paper/5596-spectral-methods-for-indian-buffet-process-inference.pdf)

NIPS 2014

We propose a distributed Markov chain Monte Carlo (MCMC) inference algo-
rithm for large scale Bayesian posterior simulation.  We assume that the dataset
is partitioned and stored across nodes of a cluster.  Our procedure involves an in-
dependent MCMC posterior sampler at each node based on its local partition of
the data. Moment statistics of the local posteriors are collected from each sampler
and propagated across the cluster using expectation propagation message passing
with low communication costs.  The moment sharing scheme improves posterior
estimation quality by enforcing agreement among the samplers.  We demonstrate
the speed and inference quality of our method with empirical studies on Bayesian
logistic regression and sparse linear regression with a spike-and-slab prior.

####[Variational Gaussian Process State-Space Models](http://papers.nips.cc/paper/5375-variational-gaussian-process-state-space-models)

NIPS 2014

State-space models have been successfully used for more than fifty years in different areas of science and engineering. We present a procedure for efficient variational Bayesian learning of nonlinear state-space models based on sparse Gaussian processes. The result of learning is a tractable posterior over nonlinear dynamical systems. In comparison to conventional parametric models, we offer the possibility to straightforwardly trade off model capacity and computational cost whilst avoiding overfitting. Our main algorithm uses a hybrid inference approach combining variational Bayes and sequential Monte Carlo. We also present stochastic variational inference and online learning approaches for fast learning with long time series.

####[Doubly Stochastic Variational Bayes for non-conjugate Inference](http://www.jmlr.org/proceedings/papers/v32/titsias14.pdf)

ICML 14

We  propose  a  simple  and  effective  variational
inference algorithm based on stochastic optimi-
sation  that  can  be  widely  applied  for  Bayesian
non-conjugate inference in continuous parameter
spaces. This algorithm is based on stochastic ap-
proximation and allows for efficient use of gra-
dient information from the model joint density.
We  demonstrate  these  properties  using  illustra-
tive examples as well as in challenging and di-
verse Bayesian inference problems such as vari-
able  selection  in  logistic  regression  and  fully
Bayesian inference over kernel hyperparameters
in Gaussian process regression.

####[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)

ICML 2014

We marry ideas from deep neural networks and approximate Bayesian inference to derive a generalised class of deep, directed generative models, endowed with a new algorithm for scalable inference and learning. Our algorithm introduces a recognition model to represent approximate posterior distributions, and that acts as a stochastic encoder of the data. We develop stochastic back-propagation -- rules for back-propagation through stochastic variables -- and use this to develop an algorithm that allows for joint optimisation of the parameters of both the generative and recognition model. We demonstrate on several real-world data sets that the model generates realistic samples, provides accurate imputations of missing data and is a useful tool for high-dimensional data visualisation.


####[Streaming Variational Inference for Bayesin Nonparametric Mixture Models](https://arxiv.org/abs/1412.0694)

2014

 In theory, Bayesian nonparametric (BNP) models are well suited to streaming data scenarios due to their ability to adapt model complexity with the observed data. Unfortunately, such benefits have not been fully realized in practice; existing inference algorithms are either not applicable to streaming applications or not extensible to BNP models. For the special case of Dirichlet processes, streaming inference has been considered. However, there is growing interest in more flexible BNP models building on the class of normalized random measures (NRMs). We work within this general framework and present a streaming variational inference algorithm for NRM mixture models. Our algorithm is based on assumed density filtering (ADF), leading straightforwardly to expectation propagation (EP) for large-scale batch inference as well. We demonstrate the efficacy of the algorithm on clustering documents in large, streaming text corpora.

---

### 2015

####[Nested Hierarchical Dirichlet Processes](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6802355)

PAMI 2015

We develop a nested hierarchical Dirichlet process (nHDP) for hierarchical topic modeling. The nHDP generalizes the
nested Chinese restaurant process (nCRP) to allow each word to follow its own path to a topic node according to a per-document
distribution over the paths on a shared tree. This alleviates the rigid, single-path formulation assumed by the nCRP, allowing
documents to easily express complex thematic borrowings. We derive a stochastic variational inference algorithm for the model, which
enables efficient inference for massive collections of text documents. We demonstrate our algorithm on 1.8 million documents from
The
New York Times
and 2.7 million documents from
Wikipedia


####[Stochastic Expectation Propagation](http://papers.nips.cc/paper/5760-stochastic-expectation-propagation)

NIPS 2015

Expectation propagation (EP) is a deterministic approximation algorithm that is often used to perform approximate Bayesian parameter learning. EP approximates the full intractable posterior distribution through a set of local-approximations that are iteratively refined for each datapoint. EP can offer analytic and computational advantages over other approximations, such as Variational Inference (VI), and is the method of choice for a number of models. The local nature of EP appears to make it an ideal candidate for performing Bayesian learning on large models in large-scale datasets settings. However, EP has a crucial limitation in this context: the number approximating factors needs to increase with the number of data-points, N, which often entails a prohibitively large memory overhead. This paper presents an extension to EP, called stochastic expectation propagation (SEP), that maintains a global posterior approximation (like VI) but updates it in a local way (like EP). Experiments on a number of canonical learning problems using synthetic and real-world datasets indicate that SEP performs almost as well as full EP, but reduces the memory consumption by a factor of N. SEP is therefore ideally suited to performing approximate Bayesian learning in the large model, large dataset setting.

####[Automatic Variational Inference in Stan](https://arxiv.org/abs/1506.03431)

2015

Variational inference is a scalable technique for approximate Bayesian inference. Deriving variational inference algorithms requires tedious model-specific calculations; this makes it difficult to automate. We propose an automatic variational inference algorithm, automatic differentiation variational inference (ADVI). The user only provides a Bayesian model and a dataset; nothing else. We make no conjugacy assumptions and support a broad class of models. The algorithm automatically determines an appropriate variational family and optimizes the variational objective. We implement ADVI in Stan (code available now), a probabilistic programming framework. We compare ADVI to MCMC sampling across hierarchical generalized linear models, nonconjugate matrix factorization, and a mixture model. We train the mixture model on a quarter million images. With ADVI we can use variational inference on any model we write in Stan.

####[Early Stopping is Nonparametric Variational Inference](https://arxiv.org/abs/1504.01344)

2015

We show that unconverged stochastic gradient descent can be interpreted as a procedure that samples from a nonparametric variational approximate posterior distribution. This distribution is implicitly defined as the transformation of an initial distribution by a sequence of optimization updates. By tracking the change in entropy over this sequence of transformations during optimization, we form a scalable, unbiased estimate of the variational lower bound on the log marginal likelihood. We can use this bound to optimize hyperparameters instead of using cross-validation. This Bayesian interpretation of SGD suggests improved, overfitting-resistant optimization procedures, and gives a theoretical foundation for popular tricks such as early stopping and ensembling. We investigate the properties of this marginal likelihood estimator on neural network models.

####[Variational Inference with normalizing flows](https://arxiv.org/abs/1505.05770)

ICML 2015

The choice of approximate posterior distribution is one of the core problems in variational inference. Most applications of variational inference employ simple families of posterior approximations in order to allow for efficient inference, focusing on mean-field or other simple structured approximations. This restriction has a significant impact on the quality of inferences made using variational methods. We introduce a new approach for specifying flexible, arbitrarily complex and scalable approximate posterior distributions. Our approximations are distributions constructed through a normalizing flow, whereby a simple initial density is transformed into a more complex one by applying a sequence of invertible transformations until a desired level of complexity is attained. We use this view of normalizing flows to develop categories of finite and infinitesimal flows and provide a unified view of approaches for constructing rich posterior approximations. We demonstrate that the theoretical advantages of having posteriors that better match the true posterior, combined with the scalability of amortized variational approaches, provides a clear improvement in performance and applicability of variational inference.

####[Trust region method for Stochastic Variational Inference](https://arxiv.org/abs/1505.07649)

ICML 2015

Stochastic variational inference allows for fast posterior inference in complex Bayesian models. However, the algorithm is prone to local optima which can make the quality of the posterior approximation sensitive to the choice of hyperparameters and initialization. We address this problem by replacing the natural gradient step of stochastic varitional inference with a trust-region update. We show that this leads to generally better results and reduced sensitivity to hyperparameters. We also describe a new strategy for variational inference on streaming data and show that here our trust-region method is crucial for getting good performance.


####[Copula Variational Inference](https://arxiv.org/abs/1506.03159)

NIPS 2015

We develop a general variational inference method that preserves dependency among the latent variables. Our method uses copulas to augment the families of distributions used in mean-field and structured approximations. Copulas model the dependency that is not captured by the original variational distribution, and thus the augmented variational family guarantees better approximations to the posterior. With stochastic optimization, inference on the augmented distribution is scalable. Furthermore, our strategy is generic: it can be applied to any inference procedure that currently uses the mean-field or structured approach. Copula variational inference has many advantages: it reduces bias; it is less sensitive to local optima; it is less sensitive to hyperparameters; and it helps characterize and interpret the dependency among the latent variables.

####[Fast Second Order Stochastic Backpropogation for Variational Inference](https://arxiv.org/abs/1509.02866)

NIPS 2015

We propose a second-order (Hessian or Hessian-free) based optimization method for variational inference inspired by Gaussian backpropagation, and argue that quasi-Newton optimization can be developed as well. This is accomplished by generalizing the gradient computation in stochastic backpropagation via a reparametrization trick with lower complexity. As an illustrative example, we apply this approach to the problems of Bayesian logistic regression and variational auto-encoder (VAE). Additionally, we compute bounds on the estimator variance of intractable expectations for the family of Lipschitz continuous function. Our method is practical, scalable and model free. We demonstrate our method on several real-world datasets and provide comparisons with other stochastic gradient methods to show substantial enhancement in convergence rates.


####[Stochastic Collapsed Variational Inference for Sequential Data](https://arxiv.org/abs/1512.01666)

NIPS 2015

Stochastic variational inference for collapsed models has recently been successfully applied to large scale topic modelling. In this paper, we propose a stochastic collapsed variational inference algorithm in the sequential data setting. Our algorithm is applicable to both finite hidden Markov models and hierarchical Dirichlet process hidden Markov models, and to any datasets generated by emission distributions in the exponential family. Our experiment results on two discrete datasets show that our inference is both more efficient and more accurate than its uncollapsed version, stochastic variational inference.

####[Robust Inference with Variational Bayes](https://arxiv.org/abs/1512.02578)

2015

In Bayesian analysis, the posterior follows from the data and a choice of a prior and a likelihood. One hopes that the posterior is robust to reasonable variation in the choice of prior and likelihood, since this choice is made by the modeler and is necessarily somewhat subjective. Despite the fundamental importance of the problem and a considerable body of literature, the tools of robust Bayes are not commonly used in practice. This is in large part due to the difficulty of calculating robustness measures from MCMC draws. Although methods for computing robustness measures from MCMC draws exist, they lack generality and often require additional coding or computation.

In contrast to MCMC, variational Bayes (VB) techniques are readily amenable to robustness analysis. The derivative of a posterior expectation with respect to a prior or data perturbation is a measure of local robustness to the prior or likelihood. Because VB casts posterior inference as an optimization problem, its methodology is built on the ability to calculate derivatives of posterior quantities with respect to model parameters, even in very complex models. In the present work, we develop local prior robustness measures for mean-field variational Bayes(MFVB), a VB technique which imposes a particular factorization assumption on the variational posterior approximation. We start by outlining existing local prior measures of robustness. Next, we use these results to derive closed-form measures of the sensitivity of mean-field variational posterior approximation to prior specification. We demonstrate our method on a meta-analysis of randomized controlled interventions in access to microcredit in developing countries.

####[Local Expectation Gradients for Black Box Variational Inference](http://papers.nips.cc/paper/5678-local-expectation-gradients-for-black-box-variational-inference)

NIPS 2015

We introduce local expectation gradients which is a general purpose stochastic variational inference algorithm for constructing stochastic gradients by sampling from the variational distribution. This algorithm divides the problem of estimating the stochastic gradients over multiple variational parameters into smaller sub-tasks so that each sub-task explores intelligently the most relevant part of the variational distribution. This is achieved by performing an exact expectation over the single random variable that most correlates with the variational parameter of interest resulting in a Rao-Blackwellized estimate that has low variance. Our method works efficiently for both continuous and discrete random variables. Furthermore, the proposed algorithm has interesting similarities with Gibbs sampling but at the same time, unlike Gibbs sampling, can be trivially parallelized.

####[Scalable Bayesian Non-negative Tensor Factorization for Massive Count Data](http://link.springer.com/chapter/10.1007/978-3-319-23525-7_4)

ECML 2015

We present a Bayesian non-negative tensor factorization model for count-valued tensor data, and develop scalable inference algorithms (both batch and online) for dealing with massive tensors. Our generative model can handle overdispersed counts as well as infer the rank of the decomposition. Moreover, leveraging a reparameterization of the Poisson distribution as a multinomial facilitates conjugacy in the model and enables simple and efficient Gibbs sampling and variational Bayes (VB) inference updates, with a computational cost that only depends on the number of nonzeros in the tensor. The model also provides a nice interpretability for the factors; in our model, each factor corresponds to a “topic”. We develop a set of online inference algorithms that allow further scaling up the model to massive tensors, for which batch inference methods may be infeasible. We apply our framework on diverse real-world applications, such as multiway topic modeling on a scientific publications database, analyzing a political science data set, and analyzing a massive household transactions data set.

####[ On the properties of variational approximations of Gibbs posteriors](https://arxiv.org/abs/1506.04091)

The PAC-Bayesian approach is a powerful set of techniques to derive non- asymptotic risk bounds for random estimators. The corresponding optimal distribution of estimators, usually called the Gibbs posterior, is unfortunately intractable. One may sample from it using Markov chain Monte Carlo, but this is often too slow for big datasets. We consider instead variational approximations of the Gibbs posterior, which are fast to compute. We undertake a general study of the properties of such approximations. Our main finding is that such a variational approximation has often the same rate of convergence as the original PAC-Bayesian procedure it approximates. We specialise our results to several learning tasks (classification, ranking, matrix completion),discuss how to implement a variational approximation in each case, and illustrate the good properties of said approximation on real datasets. 

---

### 2016

####[Variational Inference A review for statisticians](https://arxiv.org/abs/1601.00670)

2016

One of the core problems of modern statistics is to approximate difficult-to-compute probability densities. This problem is especially important in Bayesian statistics, which frames all inference about unknown quantities as a calculation involving the posterior density. In this paper, we review variational inference (VI), a method from machine learning that approximates probability densities through optimization. VI has been used in many applications and tends to be faster than classical methods, such as Markov chain Monte Carlo sampling. The idea behind VI is to first posit a family of densities and then to find the member of that family which is close to the target. Closeness is measured by Kullback-Leibler divergence. We review the ideas behind mean-field variational inference, discuss the special case of VI applied to exponential family models, present a full example with a Bayesian mixture of Gaussians, and derive a variant that uses stochastic optimization to scale up to massive data. We discuss modern research in VI and highlight important open problems. VI is powerful, but it is not yet well understood. Our hope in writing this paper is to catalyze statistical research on this class of algorithms.

####[Automatic Differentiation Variational Inference](https://arxiv.org/abs/1603.00788)

2016

Probabilistic modeling is iterative. A scientist posits a simple model, fits it to her data, refines it according to her analysis, and repeats. However, fitting complex models to large data is a bottleneck in this process. Deriving algorithms for new models can be both mathematically and computationally challenging, which makes it difficult to efficiently cycle through the steps. To this end, we develop automatic differentiation variational inference (ADVI). Using our method, the scientist only provides a probabilistic model and a dataset, nothing else. ADVI automatically derives an efficient variational inference algorithm, freeing the scientist to refine and explore many models. ADVI supports a broad class of models-no conjugacy assumptions are required. We study ADVI across ten different models and apply it to a dataset with millions of observations. ADVI is integrated into Stan, a probabilistic programming system; it is available for immediate use.


####[Overdispersed Black Box Variational Inference](https://arxiv.org/abs/1603.01140)

2016

 We introduce overdispersed black-box variational inference, a method to reduce the variance of the Monte Carlo estimator of the gradient in black-box variational inference. Instead of taking samples from the variational distribution, we use importance sampling to take samples from an overdispersed distribution in the same exponential family as the variational approximation. Our approach is general since it can be readily applied to any exponential family distribution, which is the typical choice for the variational approximation. We run experiments on two non-conjugate probabilistic models to show that our method effectively reduces the variance, and the overhead introduced by the computation of the proposal parameters and the importance weights is negligible. We find that our overdispersed importance sampling scheme provides lower variance than black-box variational inference, even when the latter uses twice the number of samples. This results in faster convergence of the black-box inference procedure.


####[Rejection Sampling Variational Inference](https://arxiv.org/abs/1610.05683)

2016

Variational inference using the reparameterization trick has enabled large-scale approximate Bayesian inference in complex probabilistic models, leveraging stochastic optimization to sidestep intractable expectations. The reparameterization trick is applicable when we can simulate a random variable by applying a (differentiable) deterministic function on an auxiliary random variable whose distribution is fixed. For many distributions of interest (such as the gamma or Dirichlet), simulation of random variables relies on rejection sampling. The discontinuity introduced by the accept--reject step means that standard reparameterization tricks are not applicable. We propose a new method that lets us leverage reparameterization gradients even when variables are outputs of a rejection sampling algorithm. Our approach enables reparameterization on a larger class of variational distributions. In several studies of real and synthetic data, we show that the variance of the estimator of the gradient is significantly lower than other state-of-the-art methods. This leads to faster convergence of stochastic optimization variational inference.

















