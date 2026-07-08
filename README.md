# 2-polynomial_regressionnn
# **Multivariate Regression**
- **Multivariate linear regression, also known as multiple linear regression, extends the concept of simple linear regression to include multiple independent variables to predict a single dependent variable.** 
- **It's a statistical technique used to model the relationship between a dependent variable and two or more independent variables.**
- **The general form of a multivariate linear regression model with $n$ independent variables is:**

    **$$ℎ_θ (𝑥)=θ_0+θ_1 𝑋_1+…+θ_𝑛 𝑋_𝑛+ε$$**

- **Where:**
    - **$y$ is the dependent variable.**
    - **$𝑋_1, 𝑋_2, …, 𝑋_𝑛$** are the independent variables.
    - **$θ_0$ is the intercept (the value of $y$ when all independent variables are 0).**
    - **$θ_1, θ_2, …, θ_𝑛$ are the coefficients associated with each independent variable.**
    - **$ε$ is the error term, representing the difference between the predicted and actual values of $y$.**

- **The goal of multivariate linear regression is to estimate the coefficients ($θ_0, θ_1, θ_2, …, θ_𝑛$) that minimize the difference between the observed values of the dependent variable and the values predicted by the model.**

- **Similar to simple linear regression, multivariate linear regression assumes that there is a linear relationship between the dependent variable and each independent variable.**
### **Multivariate Linear Regression**
Suppose we have the following dataset:

$X1$ | $X2$ | $X3$ |  $Y$ |
---|----|----|----|
5  | 20 | 6  | 114|
5  | 35 | 6  | 120|
6  | 38 | 8  | 123|
7  | 40 | 8  | 121|
7  | 46 | 10 | 135|

- **Each row represents a data point (or a house, for instance), where:**
    - $X1$ represent the size of the house.
    - $X2$ represent the distance from the city center.
    - $X3$ represent the number of rooms.
    - $Y$ represents the price of the house.

- **We want to build a multivariate linear regression model to predict the price of the house based on these features.**
- **So, the model equation would be:**
    
    **$$Y=θ_0+θ_1 𝑋_1+θ_1 𝑋_2+θ_2+𝑋_3+ε$$**
- **Using this dataset, we aim to estimate the coefficients $(θ_0, θ_1, θ_2, θ_3)$ that minimize the difference between the actual prices($Y$) and the predicted prices.**

- **Once we have the estimated coefficients, we can use them to predict the price of a house given its features(𝑋_1, 𝑋_2, 𝑋_3)**


- **Hypothesis: $Y=θ_0+θ_1 𝑋_1+θ_1 𝑋_1+θ_2+𝑋_3+ε$**
- **Parameters: $θ_0, θ_1, θ_2, θ_3$**
- **Cost function: $𝐽(𝑥) = \frac{1}{2𝑚} ∑_i[(ℎ_𝜃 (𝑥_𝑖)−𝑦_𝑖)]^2 $**
- **Goal: minimize  $𝐽(𝑥)$**
    **$$\ 𝜃_𝑗≔𝜃_𝑗−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥^𝑖 )−𝑦^𝑖) 𝑥_𝑗^𝑖)]$$**
    **$$\ 𝜃_0≔𝜃_0−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥^𝑖 )−𝑦^𝑖))]$$**
    **$$\ 𝜃_1≔𝜃_1−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥^𝑖 )−𝑦^𝑖) 𝑥_1^𝑖)]$$**
    **$$\ 𝜃_2≔𝜃_2−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥^𝑖 )−𝑦^𝑖) 𝑥_2^𝑖)]$$**
    **$$\ 𝜃_3≔𝜃_3−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥^𝑖 )−𝑦^𝑖) 𝑥_3^𝑖)]$$**
- **First, let's initialize the parameter vector $θ$ with zeros: $θ=[0,0,0,0]$**
- **Then, we define the learning rate $(α)$ and the number of iterations for gradient descent.**
- **Next, for each iteration of gradient descent, we compute the new values of $θ$ using the update rule:**
**$$\ 𝜃_𝑗≔𝜃_𝑗−𝛼 \frac{1}{2𝑚} ∑_i[((ℎ_𝜃 (𝑥^𝑖 )−𝑦^𝑖) 𝑥_𝑗^𝑖)]$$**
- **$X$ as the design matrix containing the features $𝑋_0$(for the intercept term), $𝑋_1, 𝑋_2$, and $𝑋_3$**
- **$Y$ as the vector containing the target values.**
- Our design matrix $X$ and target vector $Y$ will be:

$$X = \begin{bmatrix} 1 & 5 & 20 & 6 \\ 1 & 5 & 35 & 6 \\ 1 & 6 & 38 & 8 \\ 1 & 7 & 40 & 8 \\ 1 & 7 & 46 & 10 \end{bmatrix}$$

$$ Y = \begin{bmatrix} 114 \\ 120 \\ 123 \\ 121 \\ 135 \end{bmatrix}$$

- **Now, let's initialize our parameter vector $θ$ with zeros:**
$$ θ = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

- **Let's set the learning rate $α=0.01$ and the number of iterations to be $1000$. Then, we'll perform gradient descent to update the parameters. We'll iterate until convergence, updating $θ$ in each iteration using the update rule.**
# **Polynomial Regression**
- **Polynomial regression is a form of regression analysis in which the relationship between the independent variable $x$ and the dependent variable $y$ is modeled as an $n-th$ degree polynomial in $x$.**

- **The general form of a polynomial regression equation of degree $n$ is:**
    **$$Y=θ_0 + θ_1 𝑋_1 + θ_2 𝑋_2^2 + θ_3 𝑋_3^3 + … + θ_𝑛 𝑋_𝑛^n + ε$$**
