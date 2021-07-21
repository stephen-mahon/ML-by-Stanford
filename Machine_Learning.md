# Machine Learning
Coursera Notes for the Stanford Course by Andrew Ng

# Week 1
## Introduction

**What is Machine Learning**
Two definitions of Machine Learning are offered. Arthur Samuel described it as: “the field of study that gives computers the ability to learn without being explicitly programmed.” This is an older, informal definition.
Tom Mitchell provides a more modern definition: “A computer program is said to learn from experience $$E$$ with respect to some class of tasks $$T$$ and performance measure $$P$$, if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$.
Example: playing checkers.
$$E =$$ the experience of playing many games of checkers
$$T=$$ the task of playing checkers.
$$P =$$ the probability that the program will win the next game.
In general, any machine learning problem can be assigned to one of two broad classifications: *Supervised Learning* and *Unsupervised Learning*

**Supervised Learning**
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
Supervised learning problems are categorized into “regressions” and “classification” problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.
*Example 1* — given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.
We could turn this example into a classification problem by instead making out output about whether the house “sells for more or less than the asking price.” Here we are classifying the houses based on price into two discrete categories.
*Example 2* — (a) Regression — Given a picture of a person, we have to predict their age on the basis of the given picture. (b) Classification — Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

**Unsupervised Learning**
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don’t necessarily know the effect of the variables.
We can derive this structure by clustering the data based on relationships among the variables in the data.
With unsupervised learning there is no feedback based on the prediction results.
*Example* — Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on. Non-clustering: The “Cocktail Party Algorithm”, allows you to find structure in a chaotic environment (i.e. identifying individual voices and music from a mesh of sounds at a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect)).


## Model and Cost function

**Model representation**
To establish notation for future use, we’ll use $$x^{\left(i\right)}$$ to denote the “input” variables (living area in this example), also called input features, and $$y^{\left(i\right)}$$ to denote the “output” or target variables that we are trying to predict (price). A pair $$\left(x^{\left(i\right)}, y^{\left(i\right)}\right)$$ is called a training example, and the dataset that we’ll be using to learn—a list of $$m$$ training examples $$\left(x^{\left(i\right)}, y^{\left(i\right)}\right);i=1,\ldots, m$$—is called a training set. Note that the superscript $$(i)$$ in the notation is a simply an index into the training set, and has nothing to do with exponentiation. We will also use $$X$$ to denote the space if input values, and $$Y$$ to denote the space of output values. In this example, $$X=Y=\Re$$.
To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $$h:X \rightarrow Y$$ so that $$h(x)$$ is a “good” predictor for the corresponding value of $$y$$. For historic reasons, the function $$h$$ is called a hypothesis. Seen pictorially, the process is therefor like this:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58.png?expiry=1626912000000&hmac=LrARCcGu1w0pr1_jdLlz3w0YNFwLwgoMH7QAygvwXwM)


When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When $$y$$ can take only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call is a classification problem.

**Cost Function**
We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from $$x$$’s and the actual outputs $$y$$’s.
$$J(\theta_0, \theta_1)=\frac{1}{2m}\sum^{m}_{i=1}(\hat{y}_i-y_i)^2=\frac{1}{2m}\sum^{m}_{i=1}(h_\theta(x_i)-y_i)^2$$
To break it apart, it is $$\frac{1}{2}\bar{x}$$ where $$\bar{x}$$ is the mean of the squares of $$h_\theta(x_i)-y_i$$, or the difference between the predicted value and the actual value.
This function is otherwise called the “Squared error function”, or “Mean squared error”. This mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $$\frac{1}{2}$$ term The following image summarizes what the cost function does:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/R2YF5Lj3EeajLxLfjQiSjg_110c901f58043f995a35b31431935290_Screen-Shot-2016-12-02-at-5.23.31-PM.png?expiry=1626912000000&hmac=WoZwpTTDjJUYTIUy3zbH23jCZpGwW3nMVabpVoksayI)


**Cost function - Intuition I**
If we try to think of it in visual terms, out training data set is scattered on the $$x$$-$$y$$ plane. We are trying to make a straight line (defined by $$h_\theta(x)$$) which passed through these scattered data points.
Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the values of $$J(\theta_0, \theta_1)$$ will be $$0$$. The following examples show the ideal situation where we have a cost function of $$0$$.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56.png?expiry=1626912000000&hmac=qlOjLd3EjyLZCazdv1vBJybmGyJVgrl7Arve9xWHlok)


When $$\theta_1 = 1$$, we get a slope of 1 which goes through every single data point in out model. Conversely, when $$\theta_1=0.5$$, we see the vertical distance from out fit to the data points increase.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07.png?expiry=1626912000000&hmac=F5OEPWFeT856g91_1-ml26SL4eHkzogodAb1Ms3-Eek)


This increases our cost function to $$0.58$$. Plotting several other points yields to the following graph:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05.png?expiry=1626912000000&hmac=S9yTufTQdLhzIMrz7K5SlQjIR7F3wwM2rksd6FjvuOE)


Thus as a goal, we should try to minimize the cost function. In this case, $$\theta_1=1$$ is out global minimum.

**Cost Function - Intuition II**
A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37.png?expiry=1626912000000&hmac=LxtUyj0movw0jfyHTIjgZrKdtRZCyo4AcaMZH7r5Rvw)


Taking any color and going along the “circle”, one would expect to get he same value of the cost function. For example, the three green points found on the green line above have the same value for $$J(\theta_0, \theta_1)$$ and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when $$\theta_0=800$$ and $$\theta_1=-0.15$$. Taking another $$h(x)$$ and plotting its contour plot, one gets the following graphs:

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57.png?expiry=1626912000000&hmac=iG3uNqkWWesWAwbCKDZgSNwnhgT2aJh8iVX8Wj1MSpw)


When $$\theta_0=360$$ and $$\theta_1=0$$, the values of $$J(\theta_0, \theta_1)$$ in the contour plot gets closer to the center thus reducing the cost function error. Now giving out hypothesis function a slightly positive slope results in a better fit of the data.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41.png?expiry=1626912000000&hmac=iVHXm-cShWalvmWumf5Oi1Euqc1KhFCUGmX7gyk7piU)


The graph above minimizes the cost function as much as possible and consequently, the results of $$\theta_1$$ and $$\theta_0$$ tend to be around $$0.12$$ and $$250$$ respectively. Plotting those values on our graph to the right seems to put our point on the inner most “circle”.


## Parameter Learning

**Gradient Decent**
So we have our hypothesis and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That’s where gradient descent comes in.
Imagine that we graph our hypothesis function based on its fields $$\theta_0$$ and $$\theta_1$$ (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing $$x$$ and $$y$$ itself, but the parameter range of our hypothesis function and the cost resulting from selecting  particular set of parameters.
We put $$\theta_0$$ on the $$x$$-axis and $$\theta_1$$ on the $$y$$-axis, with the cost function on the vertical $$z$$ axis. The points on our graph will be the result of the cost function using out hypothesis with those specific $$\theta$$ parameters. The graph below depicts such a setup.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1626912000000&hmac=aKibmmMJS1x08UbPq2d2ZYU6dunz-L63kbibBzjZeSo)


**Gradient Decent Intuition**
Here explore the scenario with one parameter $$\theta_1$$ and plot its cost function to implement a gradient descent. Out formula for a single parameter is
$$\theta_1:=\theta_1-\alpha\frac{d}{d\theta_1}J(\theta_1)$$
Regardless of the slope sign for $$\frac{d}{d\theta_1}J(\theta_1)$$, $$\theta_1$$ eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of $$\theta_1$$ decreases.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/SMSIxKGUEeav5QpTGIv-Pg_ad3404010579ac16068105cfdc8e950a_Screenshot-2016-11-03-00.05.06.png?expiry=1626998400000&hmac=SAITLnfcWWt_NrDHK8Q7jLph4n10h75PmI5gaECg_p0)


On a side note, we should adjust our parameter $$\alpha$$ to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value implies that our step size is wrong.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/UJpiD6GWEeai9RKvXdDYag_3c3ad6625a2a4ec8456f421a2f4daf2e_Screenshot-2016-11-03-00.05.27.png?expiry=1626998400000&hmac=RXPwxPXVMGJQvZAbSt03pts_FjfRjGAHthF6sos1R8c)


*How does gradient descent converge with a fixed step size* $$\alpha$$?
The intuition behind the convergence is that $$\frac{d}{d\theta_1}J(\theta_1)$$ approaches $$0$$ as we approach the bottom of our convex function. At the minimum, the derivative will always be $$0$$ and thus we get
$$\theta_0:=\theta_1-\alpha \times 0$$

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/RDcJ-KGXEeaVChLw2Vaaug_cb782d34d272321e88f202940c36afe9_Screenshot-2016-11-03-00.06.00.png?expiry=1626998400000&hmac=wUvOr0r2ohKlk27bjlMYUMDhIo7-6-5PneYA6_YgBBM)


**Gradient Descent for Linear Regression**
When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to
$$\textrm{Repeat until convergence:}\{$$

    $$\theta_0:=\theta_0-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x_i\right)-y_i\right)$$
    $$\theta_1:=\theta_1-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x_i\right)-y_i\right)$$

$$\}$$
where $$m$$ is the size of the training set, $$\theta_0$$ a constant that will be changing simultaneously with $$\theta_1$$ and $$x_i, y_i$$ are values of the given training set (data).
Note that we have separated out the two cases for $$\theta_j$$ into separate equations for $$\theta_0$$ and $$\theta_1$$; and that for $$\theta_1$$ we are multiplying $$x_i$$ at the end due to the derivative. The following is a derivation of $$\frac{\partial}{\partial\theta_j}J(\theta)$$ for a single example:
$$\frac{\partial}{\partial\theta_j}J(\theta) = \frac{\partial}{\partial\theta_j}\frac{1}{2}\left(h_\theta(x)-y\right)^2$$
$$= 2\frac{1}{2}\left(h_\theta(x)-y\right)\cdot\frac{\partial}{\partial\theta_j}\left(h_\theta(x)-y\right)$$
$$= \left(h_\theta(x)-y\right)\cdot\frac{\partial}{\partial\theta_j}\left(\sum_{i=0}^n\theta_ix_i-y\right)$$
$$= \left(h_\theta(x)-y\right)x_j$$
The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, out hypothesis will become more and more accurate.
So, this is simply gradient descent on the original cost function $$J$$. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local optima; thus gradient descent always converges (assuming the learning rate $$\alpha$$ is not too large) to the global minimum. Indeed, $$J$$ is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49.png?expiry=1626998400000&hmac=gL47MAE4VYYOBRvZpzl1v3I-YEa5VcCpVelCU-EA7CE)


The ellipses shown above are contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at $$(48, 30)$$. The *x*’s in the figure (joined by straight lines) mark successive values of $$\theta$$ that gradient descent went through as it converged to its minimum.

----------
# Week 2
## Multivariate Linear Regression

**Multiple Features**
Linear regression with multiple variables is also known as “multivariate linear regression”.
We now introduce notation for the equations where we can have any number of input variables.
$$x_j^{\left(i\right)}=$$ value of feature $$j$$ in the $$i^{th}$$ training example
$$x^{\left(i\right)}=$$ the input (features) of the $$i^{th}$$ training example
$$m=$$ the number of training examples
$$n=$$ the number of features
The multivariate form of the hypothesis function accommodating these multiple features is as follows:
$$h_\theta(x) = \theta_0+\theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$$
In order to develop intuition about this function, we can think about $$\theta_0$$ as the basic price of a house, $$\theta_1$$ as the price per square meter, $$\theta_2$$ as the price per floor, etc. $$x_1$$ will be the number of square meters in the house. $$x_2$$ the number of floors, etc.
Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:
$$h_\theta (x) = \begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_n\end{bmatrix}\begin{bmatrix} x_0\\ x_1\\ \vdots \\ x_n \end{bmatrix}=\theta^Tx$$
This is a vectorization of our hypothesis function for one training example; see lessons on vectorization to learn more.
Remark: Note that for convenience reasons in this course we assume $$x_0^{\left(i\right)} = 1$$ for $$(i \in 1, \ldots, m)$$. This allows us to do matrix operations with $$\theta$$ and $$x$$.  Hence making the two vectors $$\theta$$ and $$x^{\left(i\right)}$$ match each other element-wise (that is, have the same number of elements: $$n+1$$.

**Gradient Descent for Multiple Variables**
The gradient descent equation itself is generally the same form; we just have to repeat if for $$n$$ features.
repeat until convergence: $$\{$$
$$\theta_0 := \theta_0-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_0^{\left(i\right)}$$
$$\theta_1 := \theta_1-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_1^{\left(i\right)}$$
$$\theta_2 := \theta_2-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_2^{\left(i\right)}$$
$$\vdots$$
$$\}$$
In other words:
repeat until convergence: $$\{$$
$$\theta_j := \theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta\left(x^{\left(i\right)}\right)-y^{\left(i\right)}\right)x_j^{\left(i\right)}$$ for $$j:=0\ldots n$$

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


## Gradient Descent in Practice II — Learning Rate

**Debugging gradient descent:** Make a plot with *number of iterations* of the x-axis. Now plot the cost function $$J(\theta)$$ of the number of iterations of gradient descent. If $$J(\theta)$$ ever increases, then you probably need to decrease $$\alpha$$.
**Automatic convergence test:** Declare convergence if $$J(\theta)$$ decreases by less than $$E$$ in one iterations, where $$E$$ is some small value such as $$10^{-3}$$. However in practice it’s difficult to choose this threshold value.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/FEfS3aajEea3qApInhZCFg_6be025f7ad145eb0974b244a7f5b3f59_Screenshot-2016-11-09-09.35.59.png?expiry=1626912000000&hmac=8Xuq4LfepHS7jNWxP4zvlFVqSnuzkAeiSSlkOpdlJzo)


It has been proven that if learning rate $$\alpha$$ is sufficiently small, then $$J(\theta)$$ will decrease on every iteration.

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/rC2jGKgvEeamBAoLccicqA_ec9e40a58588382f5b6df60637b69470_Screenshot-2016-11-11-08.55.21.png?expiry=1626912000000&hmac=OjjdSNKDdEc_7lrFvSf6oVir2usXOLOh77eJS5HHsBY)


To summarize:

- If $$\alpha$$ is too small: slow convergence.
- If $$\alpha$$ is too large $$J(\theta)$$ may not decrease on every iteration and this may not converge.

**Features and Polynomial Regression**
We can improve our features and the form of our hypothesis function in a couple different ways. We can **combine** multiple features into one. For example, we can combine $$x_1$$ and $$x_2$$ into a new feature $$x_3$$ by taking $$x_1 \times x_2$$

- Polynomial Regression

Our hypothesis function need not be linear (a straight line) if that does not fit the data well. We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic, or square root function (or any other form).
For example, if our hypothesis function is $$h(\theta) = \theta_0 + \theta_1 x_1$$ then we can create addition features based on $$x_1$$, to get the quadratic function $$h(\theta) = \theta_0 + \theta_1 x_1+\theta_2x_1^2$$ or the cubic function $$h(\theta) = \theta_0 + \theta_1 x_1+\theta_2x_1^2+\theta_3x_1^3$$.
In the cubic version, we have created new features $$x_2$$ and $$x_3$$ where $$x_2 = x_1^2$$ and $$x_3=x_1^3$$.
To make it a square root function, we could do: $$h(\theta) = \theta_0 + \theta_1 x_1+\theta_2\sqrt{x_1}$$
One important thing to keep in mind is if you choose your features this way then feature scaling becomes very important.
e.g. if $$x_1$$ has a range of 1-1000 then the range of $$x_2$$ becomes 1-1000000 and that of $$x_1^3$$ becomes 1-1000000000.


## Computing Parameters Analytically

**Normal Equation**
Gradient descent gives one way of minimizing $$J$$. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. The the *Normal Equation* method, we will minimize $$J$$ by explicitly taking its derivatives with respect to $$\theta_j$$’s and setting them to zero. This allows us to find the optimum $$\theta$$ without iterations. The normal equation formula is given below:
$$\theta = \left(X^TX\right)^{-1}X^Ty$$

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dykma6dwEea3qApInhZCFg_333df5f11086fee19c4fb81bc34d5125_Screenshot-2016-11-10-10.06.16.png?expiry=1626912000000&hmac=GeAI5NXfNVbD1HnT--8i4sRCHqsnz9eALfvJMeLyNUM)


There is **no need** to do feature scaling with the normal equation.
The following is a comparison of gradient descent and the normal equation:

| Gradient Descent               | Normal Equation                                                  |
| ------------------------------ | ---------------------------------------------------------------- |
| Need to choose $$\alpha$$      | No need to choose $$\alpha$$                                     |
| Needs many iterations          | No need to iterate                                               |
| $$O\left(kn^2\right)$$         | $$O\left(n^3\right)$$, need to calculate the inverse of $$X^TX$$ |
| Works well when $$n$$ is large | Slow if $$n$$ is very large                                      |

With the normal equation, computing the inversion has complexity $$O\left(n^3\right)$$. So if we have a very large number of features, the normal equation will be slow. In practice, when $$n$$ exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

**Normal Equation Noninvertibility**
When implementing the normal equation in octave we want the use the `pinv` function rather than `inv`. The `pinv` function will give you a value of $$\theta$$ even if $$X^TX$$ is not invertible.
 If $$X^TX$$is **noninvertible**, the common causes might be having:

- Redundant features, where two features are very closely related (i.e. the are linearly dependent)
- Too many features (e.g. $$m\leq n$$). In this case, delete some features or use *regularization* (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

----------
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


