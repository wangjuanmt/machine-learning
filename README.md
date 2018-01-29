# machine-learning

This repository is to recall basic concepts, and small practices for algorithms.

### Concepts recalling ###

n - # feature number </br>

m - # example number </br>

#### Linear Regression
hypothesis h(x) = sum(θ<sub>i</sub>x<sub>i</sub>) = θ<sup>T</sup>x, i from 1 to n. </br>

cost function J(θ) least-squares cost function.
J(θ) = 1/2sum((h<sub>θ</sub>(x<sup>(i)</sup>) − y<sup>(i)</sup>)<sup>2</sup>), i from 1 to m.

Steps to do a linear regression
1. Fit θ to minimize J(θ).
2. Output θ<sup>T</sup>x.

##### LMS(Least Mean Squares) algorithm
__gradient descent__ </br>

For a single training example, this gives the update rule:
θ<sub>j</sub> := θ<sub>j</sub> + α(y<sup>(i)</sup> − h<sub>θ</sub>(x<sup>(i)</sup>))x<sup>(i)</sup><sub>j</sub>  </br>

__batch gradient descent__ </br>

Repeat until convergence {
    θ<sub>j</sub> := θ<sub>j</sub> + αsum((y<sup>(i)</sup> − h<sub>θ</sub>(x<sup>(i)</sup>))x<sup>(i)</sup><sub>j</sub>)}

__stochastic gradient descent (also incremental
gradient descent)__

Loop {
  for i=1 to m, {
    θ<sub>j</sub> := θ<sub>j</sub> + α(y<sup>(i)</sup> − h<sub>θ</sub>(x<sup>(i)</sup>))x<sup>(i)</sup><sub>j</sub>  (for every j).
  }
}

##### Normal equations
∇<sub>θ</sub>J(θ) = X<sup>T</sup>Xθ − X<sup>T</sup> ⃗y

to minimize J(θ), we have:

X<sup>T</sup>Xθ = X<sup>T</sup> ⃗y

θ = (XTX)<sup>−1</sup>X<sup>T</sup> ⃗y.

##### Underfitting
##### Overfitting

##### Weighted linear regression
Steps to do a weighted linear regression
1. Fit θ to minimize sum(w<sup>(i)</sup>(h<sub>θ</sub>(x<sup>(i)</sup>) − y<sup>(i)</sup>)<sup>2</sup>).
2. Output θ<sup>T</sup>x.

w<sup>(i)</sup>s are __wights__. Commonly, w<sup>(i)</sup> is took:

w<sup>(i)</sup> = exp(-(x<sup>(i)</sup>-x)<sup>2</sup>/(2τ)<sup>2</sup>)

τ is called the __bandwidth__ parameter.

#### Logistic regression

Logistic regression is __not__ a __regression__ problem, but a binary __classification__ problem.

y ∈ {0, 1}. 0 is also called the negative class, and 1 the positive class.

hypotheses h<sub>θ</sub>(x) = g(θ<sup>T</sup>x) = 1/(1 + e<sup>−θT</sup>x)

where g(z) = 1/(1 + e<sup>-z</sup>) is called the __logistic function__ or the __sigmoid function__.

g′(z) = g(z)(1 − g(z)).

Assume:

P(y = 1 | x; θ) = h<sub>θ</sub>(x)

P(y = 0 | x; θ) = 1 − h<sub>θ</sub>(x)

This also can be written as:
p(y | x; θ) = (h<sub>θ</sub>(x))<sup>y</sup>(1 − h<sub>θ</sub>(x))<sup>1−y</sup>

The likelihood L(θ) = p(⃗y | X; θ)

Take l(θ) = logL(θ) = sum(y<sup>(i)</sup> log h(x<sup>(i)</sup>) + (1 − y<sup>(i)</sup>) log(1 − h(x<sup>(i)</sup>))),  i from 1 to m.

To maximize L(θ) is the same with maximize l(θ).


##### The perceptron learning algorithm

Using __stochastic gradient ascent__ to maximize l(θ)

∂l(θ)/∂θ<sub>j</sub> = (y − h<sub>θ</sub>(x)) x<sub>j</sub>

θ<sub>j</sub> := θ<sub>j</sub> + α(y<sup>(i)</sup> − h<sub>θ</sub>(x<sup>(i)</sup>))x<sup>(i)</sup><sub>j</sub>

This is called perceptron learning algorithm.

##### Newton's method

θ := θ − l′(θ)/l′′(θ) = H<sup>−1</sup>∇<sub>θ</sub>l(θ)

H is an n-by-n matrix (actually, n + 1-by-n + 1, assuming
that we include the intercept term) called the __Hessian matrix__.

H<sub>ij</sub> = ∂<sup>2</sup>l(θ) / (∂θ<sub>i</sub>∂θ<sub>j</sub>)

