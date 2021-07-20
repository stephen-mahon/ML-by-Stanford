# Machine Learning
Coursera Notes for the Stanford Course by Andrew Ng

# Week 1
# Week 2
## Multivariate Linear Regression

**Multiple Features**
Linear regression with multiple variables is also known as “multivariate linear regression”.
We now introduce notation for the equations where we can have any number of input variables.

| $$x_j^{\left(i\right)}=$$ value of feature $$j$$ in the $$i^{th}$$ training example<br>$$x^{\left(i\right)}=$$ the input (features) of the $$i^{th}$$ training example<br>$$m=$$ the number of training examples<br>$$n=$$ the number of features |

The multivariate form of the hypothesis function accommodating these multiple features is as follows:
$$h_\theta(x) = \theta_0+\theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$$
In order to develop intuition about this function, we can think about $$\theta_0$$ as the basic price of a house, $$\theta_1$$ as the price per square meter, $$\theta_2$$ as the price per floor, etc. $$x_1$$ will be the number of square meters in the house. $$x_2$$ the number of floors, etc.
Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

| $$h_\theta (x) = \begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_n\end{bmatrix}\begin{bmatrix} x_0\\ x_1\\ \vdots \\ x_n \end{bmatrix}=\theta^Tx$$ |

This is a vectorization of our hypothesis function for one training example; see lessons on vectorization to learn more.
Remark: Note that for convenience reasons in this course we assume $$x_0^{\left(i\right)} = 1$$ for $$(i \in 1, \ldots, m)$$. This allows us to do matrix operations with $$\theta$$ and $$x$$.  Hence making the two vectors $$\theta$$ and $$x^{\left(i\right)}$$ match each other element-wise (that is, have the same number of elements: $$n+1$$.

**Gradient Descent for Multiple Variables**
The gradient descent equation itself is generally the same form; we just have to repeat if for $$n$$ features.

| repeat until convergence: $$\{$$<br>$$\theta_0 := \theta_0-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_0^{\left(i\right)}$$<br>$$\theta_1 := \theta_1-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_1^{\left(i\right)}$$<br>$$\theta_2 := \theta_2-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_2^{\left(i\right)}$$<br>$$\vdots$$<br>$$\}$$ |

In other words:

| repeat until convergence: $$\{$$<br>$$\theta_j := \theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_j^{\left(i\right)}$$ for $$j:=0\ldots n$$ |


**Gradient Decent in Practice 1 — Feature Scaling**
We can speed up gradient descent by having each of our input values in roughly the same range. This is because $$\theta$$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variable s are very uneven.
The way to prevent this is to modify the ranges of out input variables so that the are all roughly the same. Ideally:
$$-1 \leq x_{\left(i\right)}\leq 1$$
or 
$$-0.5 \leq x_{\left(i\right)}\leq 0.5$$
These aren’t exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.
Two techniques to help with this are **feature scaling** and **mean normalization**. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just $$1$$. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:
$$x_i := \frac{x_i-\mu_i}{s_i}$$
Where $$\mu_i$$ is the **average** of all the values for feature $$(i)$$ and $$s_i$$ is the range of values (max - min), of $$s_i$$ is the standard deviation.
Note that dividing by the range, or dividing by the standard deviation, give different results.

# Week 3
## Classification and Representation

**Representation**
To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as *One* and all less than 0.5 as *Zero*. However, this method doesn’t work well because classification is not actually a linear function.
The classification problem is just like the linear regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the **binary classification problem** in which $$y$$ can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multi-class case.) For instance, if we are trying to build a spam classifier for email, then $$x^{(i)}$$ may be some feature of a piece of email, and $$y$$ maybe 1 if it is a piece of spam email, and 0 otherwise. Hence $$y\in \{0,1\}$$. 0 is also called the negative, and 1 the positive class, and they are sometime denoted by the symbols “$$-$$” and “$$+$$”. Given $$x^{(i)}$$, the corresponding $$y^{(i)}$$ is also called the label for the training example.

**Hypothesis Representation**
We could approach the classification problem ignoring the fact that $$y$$ is discrete-valued, and use our old #linear-regression algorithm to try and predict $$y$$ given $$x$$. However, it is easy to construct examples where this method performs poorly. Intuitively, it also doesn’t make sense for $$h_\theta(x)$$
to take values larger than 1 or smaller than 0 when we know that $$y\in\{0,1\}$$. To fix this, let’s change the form for our hypothesis $$h_\theta(x)$$ to satisfy $$0 \leq h_\theta(x) \leq 1$$. This is accomplished by plugging $$\theta^Tx$$ into the logistics function.
Out new for uses the #sigmoid-function, also called the #logistic-function.

| $$h_\theta(x)=g(\theta^Tx)$$<br>$$z = \theta^Tx$$<br>$$g(z)=\frac{1}{1-e^{-z}}$$ |

The following image shows us what the #sigmoid-function looks like:

![Sigmoid function - Wikipedia](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png)


The function $$g(z)$$, shown here, maps any real number to the $$(0,1)$$ interval, making it very useful for transforming an arbitrary-valued function into a function better suited fro classification.
$$h_\theta(x)$$ will give us the **probability** that our output is 1. For example, $$h_\theta(x)=0.7$$ will give us a probability of $$70\%$$ that our output is $$1$$. Our probability that our output is $$0$$ is just the complement of our probability that it is $$1$$ (e.g. if probability that it is $$1$$ is $$70\%$$, then the probability that it is $$0$$ is $$30\%$$).

| $$h_\theta(x)=P(y=1|x;\theta)=1-P(y=0|x;\theta)$$<br>$$P(y=1|x;\theta)+P(y=0|x;\theta)=1$$ |

**Decision Boundary**
In order to get out discrete $$0$$ or $$1$$ classification, we can translate the output of the hypothesis function as follows:

| $$h_\theta(x)\geq 0.5 \rightarrow y = 1$$<br>$$h_\theta(x)< 0.5 \rightarrow y = 0$$ |

The way our #logistic-function $$g$$ behaves is that when its input is greater than or equal to $$0$$, its output is greater than or equal to $$0.5$$:

| $$g(z)\geq 0.5$$<br>when $$z\geq0$$ |

Remember:

| $$z=0, e^0=1, g(z)=0.5$$<br>$$z\rightarrow\infty, e^\infty=0, g(z)=1$$<br>$$z\rightarrow-\infty, e^{-\infty}=\infty, g(z)=0$$ |

So if our input to $$g$$ is $$\theta^Tx$$, then it means:

| $$h_\theta(x)=g(\theta^Tx)\geq0.5$$<br>when $$\theta^Tx\geq0$$ |

From these statements we can now say:

| $$\theta^Tx\geq0, y=1$$<br>$$\theta^Tx<0, y=0$$ |

The #decision-boundary is the line that separates the area where $$y=0$$ and where $$y=1$$. It is created by our hypothesis function.
Example

| $$\theta = \begin{bmatrix}5\\-1\\0\end{bmatrix}$$, $$X=\begin{bmatrix}0\\x_1\\x_2\end{bmatrix}$$, $$h_\theta(x)=\theta^TX$$<br>$$y=1$$ if $$\theta_0+\theta_1x_1+\theta_2x_2\geq0$$<br>$$5-x_1\geq0$$<br>$$-x_1\geq-5$$<br>$$x_1\leq5$$ |

In this case, our decision boundary is a straight vertical line place on the graph where $$x_1=5$$ and everything to the left of that denotes $$y=1$$, while everything to the right denotes $$y=0$$.
Again, the inputs to the #sigmoid-function $$g(z)$$ (e.g $$\theta^TX$$) doesn’t need to be linear, and could be a function that describes a circle (e.g. $$z=\theta_0+\theta_1x_1^2+\theta_2x_2^2$$) or any shape to fit our data.


## Logistic Regression Model

**Cost Function**
We cannot use the cost function we used for linear regression because the #logistic-function will cause the output to be wavy, causing many local optima. In other words, it will no be a #convex function.
Instead, our cost function for linear regression look like:

| $$J(\theta)= \frac{1}{m}\sum_{i=1}^{m} \frac{1}{2}\mathrm{Cost}(h_{\theta}(x^{(i)}),(y^{(i)}))$$<br>$$\mathrm{Cost}(h_{\theta}(x^{(i)}),(y^{(i)}))=-\log(h_{\theta}(x))$$ if $$y=1$$ <br>$$\mathrm{Cost}(h_{\theta}(x^{(i)}),(y^{(i)}))=-\log(1-h_{\theta}(x))$$ if $$y=0$$ |

When $$y=1$$, we get the following plot for $$J(\theta)$$ versus $$h_{\theta}(x)$$

![](https://paper-attachments.dropbox.com/s_86BD3108660EB24CA60F3467729E5296DBB07968BDDF6917739AB5B609B2106E_1612288341566_image.png)


Similarly, when $$y=0$$, we get the following plot for $$J(\theta)$$ versus $$h_{\theta}(x)$$

![](https://paper-attachments.dropbox.com/s_86BD3108660EB24CA60F3467729E5296DBB07968BDDF6917739AB5B609B2106E_1612288400581_image.png)

| $$\mathrm{Cost}(h_{\theta}(x),y)=0$$ if $$h_{\theta}(x)=y$$ <br>$$\mathrm{Cost}(h_{\theta}(x),y)\rightarrow\infty$$ if $$y=0$$ and $$h_{\theta}(x)\rightarrow 1$$ <br>$$\mathrm{Cost}(h_{\theta}(x),y)\rightarrow\infty$$ if $$y=1$$ and $$h_{\theta}(x)\rightarrow 0$$ |

If our correct answer $$y$$ is $$0$$, then the #cost-function will be $$0$$ if our hypothesis function also outputs $$0$$. If our hypothesis approaches $$1$$, then the cost function will approach infinity.
If our correct answer $$y$$ is $$1$$, then the cost function will be $$0$$ if our hypothesis function outputs $$1$$. If our hypothesis approaches $$0$$, then the cost function will approach infinity.
Note that writing the cost function in this way guarantees that $$J(\theta)$$ is convex for logistic regression.

**Simplified Cost Function and Gradient Decent**
We can compress out #cost-function’s two conditional cases into one case:
$$\mathrm{Cost}(h_{\theta}(x),y)=-y\log(h_{\theta}(x))-(1-y)\log(1-h_{\theta}(x))$$
Notice that when $$y=1$$, $$1-y=0$$ and the second term will not affect the result. If $$y=0$$, the first term equals zero and will not affect the result.
We can fully write out the entire cost function as follows:

| $$J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(h_{\theta}(x^{(i)}))+(1-y^{(i)})\log(1-h_{\theta}(x^{(i)}))\right]$$ |

A vectorized implementation is:

| $$h=g(X\theta)$$<br>$$J(\theta)=\frac{1}{m}\cdot\left(-y^T\log(h)-(1-y^T)\log(h)\right)$$ |

**Gradient Decent**
Remember the general form of #gradient-decent is:

| Repeat $$\{$$<br>$$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta}J(\theta)$$<br>$$\}$$ |

We can work out the derivative part using calculus to get:

| Repeat $$\{$$<br>$$\theta_j:=\theta_j-\frac{\alpha}{m}\sum_{i=1}^{m}\left(h_{\theta}(x^{(i)})-y^{(i)}\right)x_j^{(i)}$$<br>$$\}$$ |

Notice that this algorithm is identical to the one we used in linear regression. We sill have to simultaneously update all values in $$\theta$$.
A vectorized implementation is:
$$\theta:=\theta-\frac{\alpha}{m}X^T\left(g(X^T\theta)-y\right)$$

**Advanced optimization**
*Conjugate gradient*, *BFGS*, and *L-BFGS* are more sophisticated, faster ways to optimize $$\theta$$ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.
We first need to provide a function that evaluates the following two functions for a given input value $$\theta$$:
$$J(\theta)$$
$$\frac{\partial}{\partial\theta}J(\theta)$$
We can a single function that returns both of these:

    function [jVals, gradient] = costFunction(theta)
      jVals = [...code to compute J(theta)...];
      gradient = [...code to compute derivative of J(theta)...];
    end

Then we can use octave's `fminunc()` optimization algorithm along with the `optimset()` function that creates an object containing the options we want to send to `fminunc()`.

    options = optimset('GradObj', 'on', 'MaxIter', 100);
    initialTheta = zeros(2,1);
       [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options); 

We give to the function `fminunc()` our cost function, our initial vector of theta values, and the `options` object that we created beforehand.


## Multiclass Classification

**One-vs-all**
Now we will approach the classification of data when we have more than two categories. Instead of $$y=\{0,1\}$$ we will expand our definition so that $$y=\{0,1,\ldots,n\}$$.
Since $$y=\{0,1,\ldots,n\}$$, we divide our problem into $$n+1$$ (because the index in Octave starts at $$0$$) binary classification problems; in each one we predict the probability that $$y$$ is a member of one of our classes.

| $$y=\{0,1,\ldots,n\}$$<br>$$h_{\theta}^{(0)}(x)=P(y=0|x:\theta)$$<br>$$h_{\theta}^{(1)}(x)=P(y=1|x:\theta)$$<br>$$\vdots$$<br>$$h_{\theta}^{(n)}(x)=P(y=n|x:\theta)$$<br>$$\mathrm{prediction}=\max_i\left(h_{\theta}^{(i)}(x)\right)$$ |

We are basically choosing one class and lumping all the others into a single second class. We do this repeatedly, applying binary regression to each case, and then use the hypothesis that returned the highest value as our prediction.
The following three images shows how we could classify 3 classes:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/cqmPjanSEeawbAp5ByfpEg_299fcfbd527b6b5a7440825628339c54_Screenshot-2016-11-13-10.52.29.png?expiry=1612483200000&hmac=N0djoJTTkHtGSIifuNLv7Jxv2ggB5Fk5i9mYlfqSGkM)


The summarize, train a logistic regression classifier $$h_{\theta}(x)$$ for each class to predict the probability that $$y=i$$. To make a prediction on a new $$x$$, pick the class that maximizes $$h_{\theta}(x)$$.


![](https://paper-attachments.dropbox.com/s_86BD3108660EB24CA60F3467729E5296DBB07968BDDF6917739AB5B609B2106E_1612466648910_image.png)

![](https://paper-attachments.dropbox.com/s_86BD3108660EB24CA60F3467729E5296DBB07968BDDF6917739AB5B609B2106E_1612466656687_image.png)



![](https://paper-attachments.dropbox.com/s_86BD3108660EB24CA60F3467729E5296DBB07968BDDF6917739AB5B609B2106E_1612466666141_image.png)



![](https://paper-attachments.dropbox.com/s_86BD3108660EB24CA60F3467729E5296DBB07968BDDF6917739AB5B609B2106E_1612466673590_image.png)


