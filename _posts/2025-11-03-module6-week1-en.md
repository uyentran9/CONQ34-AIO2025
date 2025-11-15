---
layout: post
title: Module 6 – Week 1
date: 06-11-2025
categories: AIO2025 Module6 Deep Learning
use_math:  true
image: /assets/module6-week1/m6w1.jpg
---
<div id="lang-switch-static">
  <span>🌐</span>
  <a class="active" href="#">EN</a>
  <a href="{{ page.url | replace:'-en','-vi' }}">VI</a>
</div>

{% include mathjax.html %}


## Summary of the Wednesday lecture on 11/05/2025

<p align="center">
  <img src="{{ '/assets/module6-week1/m6w1.jpg' | relative_url }}" alt="Logistic Regression vs Linear Regression illustration" width="80%">
</p>



## 📈 Part 1 – Review Linear Regression

**Linear Regression** is a model used to **predict continuous values** from input variables (*features*).  
The goal is to find a line (or hyperplane in higher dimensions) that minimizes the total error between predictions and true data.

General equation:
 
$$
\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \mathbf{X}\boldsymbol{\theta}
$$

Where:

- $\mathbf{X}\in\mathbb{R}^{m\times n}$: input data matrix with **m samples** and **n features**
  (each row is a data sample, each column is a feature)
- $\boldsymbol{\theta}=[w_1, w_2, \ldots, w_n, b]^T$: weight vector and bias
- $\hat{y}$: predicted output value
- $y$: ground-truth value of the training data
- $b$: bias term that shifts the regression line/hyperplane
 


---  

$$
L(\hat{y}, y) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

Where:

- $y_i$: true value  
- $\hat{y}_i$: predicted value  
- $m$: number of samples  


---

The objective of **Linear Regression** is to **find optimal parameters** — specifically the **weight vector** $\mathbf{w}$ and **bias** $b$ — such that $L_{\text{MSE}}$ is **as small as possible**.

<p align="center">
  <img src="{{ '/assets/module6-week1/loss-function.png' | relative_url }}" alt="Loss Function" width="600">
</p>  

- Each pair $(\hat{y}^{(i)}, y^{(i)})$ represents the **deviation** between prediction and truth.  
- Squaring the error $(\hat{y}^{(i)} - y^{(i)})^2$ **removes negative signs** and **penalizes large mistakes more**.  
- Averaging over all samples yields the **overall mean error of the model**.  
- By adjusting $\mathbf{w}$ and $b$, the model “rotates” or “shifts” the regression line to **reduce the total error**.

---

**Overall pipeline of Linear Regression**

After understanding how the model learns to reduce loss, we can outline the training pipeline (which we’ll map to Logistic Regression later):

<p align="center">
  <img src="{{ '/assets/module6-week1/pipeline.png' | relative_url }}" alt="Pipeline Diagram" width="600">
</p>  

---

**⚠️ Limitations of Linear Regression for classification**

Although Linear Regression is foundational, intuitive, and effective for **continuous prediction**,  
it has several **serious drawbacks** for **binary classification (0/1)**:

**No output bound in [0, 1]**
- The linear model has the form $\hat{y} = \mathbf{w}^T\mathbf{x} + b$.  
- The output $\hat{y}$ can take **any real value** in $(-\infty, +\infty)$.  
> This **cannot be interpreted as a probability**, making the **classification threshold** arbitrary.

**Linear decision boundary only**
- In many real problems, the two classes (0 and 1) **cannot be separated by a straight line**.  
- Linear Regression only captures linear relationships, so it **cannot model nonlinear boundaries**.

**MSE is not ideal for classification**
- Linear Regression uses **Mean Squared Error (MSE)**, designed for regression.  
- In classification, it:
  - Does not properly reflect **confidence** of probability predictions.  
  - Combined with the sigmoid, MSE leads to **slow convergence** and **local minima traps**.

**No stability guarantee**
- Because $\hat{y}$ is unbounded, the model can **grow unbounded** trying to reduce MSE.  
- As a result:
  - Gradient descent may **oscillate or diverge** (not converge).  
  - The model is **very sensitive to noise and outliers**.

👉 The solution is **Logistic Regression**, which extends Linear Regression with the **sigmoid activation** to map outputs into **(0, 1)** — enabling **probability predictions**.


## 🔀 Part 2 – Logistic Regression

**1. Basic concept**

**Logistic Regression** is a **supervised learning** model for **binary classification**.

⚖️ Quick comparison:

| Model                | Output type                      | Example                        |
|----------------------|----------------------------------|---------------------------------|
| **Linear Regression**  | Continuous                      | Predict temperature, house price |
| **Logistic Regression**| Discrete (0 or 1)              | Predict disease vs. no disease  |


**2. Start from a linear model**

Like **Linear Regression**, **Logistic Regression** starts with a **linear model** to combine input features:

$$
z = wx + b
$$

Where:

- $x$: input feature  
- $w$: weight to be learned  
- $b$: bias term  
- $z$: linear score (*logit*)

At this step, $z$ can be any real value from $-\infty$ to $+\infty$.  
However, our goal is to **predict a probability** that a sample belongs to class 1 (e.g., “diseased”, “pass”, “spam”), so the output must be in **(0, 1)**.

👉 To achieve this, we **do not use** $z$ directly as the prediction; we need a **transfer (activation) function**.

**3. Sigmoid function**

If we directly used:

$$
\hat{y} = z = \mathbf{w}x + b
$$

for **classification**, we immediately face problems:

- $z$ can take **any value** from $-\infty$ to $+\infty$
- But a **probability** must lie in $[0, 1]$

Therefore, the **sigmoid** (or *logistic*) function transforms the linear value \(z\) into a **valid probability**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

<p align="center">
  <img src="{{ '/assets/module6-week1/sigmoid-function.png' | relative_url }}" alt="Sigmoid Function" width="500">
</p>

**🧠 Explanation:**

- When $z \to +\infty \Rightarrow e^{-z} \to 0 \Rightarrow \sigma(z) \to 1$  
- When $z = 0 \Rightarrow \sigma(0) = \frac{1}{2} = 0.5$  
- When $z \to -\infty \Rightarrow e^{-z} \to +\infty \Rightarrow \sigma(z) \to 0$

Thus, **sigmoid maps all real $z$ to (0, 1)** — exactly the range we need for probabilities.

## 🎯 Part 3 – Loss: MSE or BCE?

**1. Try MSE (Mean Squared Error)**

This is the **familiar regression loss**:

$$
L_{\text{MSE}}(\hat{y}, y) = (\hat{y} - y)^2
$$

Applied to **Logistic Regression**:

$$
\hat{y} = \sigma(w x + b) = \frac{1}{1 + e^{-(w x + b)}}
$$

→ We can compute derivatives and update parameters via **Gradient Descent**.

**⚠️ However, MSE is *not suitable* for Logistic Regression because:**

**Nonlinearity of sigmoid → complex curvature**

- Combining **sigmoid + MSE** makes the loss **non-convex**.  
- **Gradients** can be very small (*vanishing gradient*) when $\hat{y}$ is near 0 or 1.  
  → The model learns **very slowly** and can **get stuck in local minima**.

<p align="center">
  <img src="{{ 'assets/module6-week1/MSE-Sigmoid.png' | absolute_url }}"
       alt="Sigmoid combined with MSE" width="500">
</p>

👉 Because of this issue, a loss designed specifically for probabilistic classification is used: **Binary Cross-Entropy (BCE)**, also known as **Log Loss**.

<p align="center">
  <img src="{{ 'assets/module6-week1/BCE-Sigmoid.png' | absolute_url }}"
       alt="Sigmoid combined with BCE" width="500">
</p>


**2. BCE (Binary Cross-Entropy)**

**a) General formula**

$$
L_{\text{BCE}}(\hat{y}, y) = -\Big[\, y \log(\hat{y}) + (1 - y)\, \log(1 - \hat{y}) \,\Big]
$$

**b) Explanation of each term**

| Symbol | Meaning | Role |
|----------|----------|----------|
| $y$ | **Ground-truth label**, either 0 or 1 | Indicates the true class (positive = 1, negative = 0). |
| $\hat{y}$ | **Predicted probability** from logistic regression via sigmoid:  $\hat{y} = \frac{1}{1 + e^{-(wx + b)}}$ | Represents model confidence that the sample is class 1. |
| $\log(\hat{y})$ | Natural log of the probability of class 1 | Measures **“surprise”** (information content) when predicting class 1. |
| $\log(1 - \hat{y})$ | Natural log of the probability of class 0 | Measures **“surprise”** when predicting class 0. |
| Minus sign outside | Negates the (negative) log-likelihood | Turns loss into a positive value to **minimize**. |

**c) Intuition**

The formula can be read as:

- **If \( y = 1 \)**, then:

$$
L = -\log(\hat{y})
$$

→ The model is **heavily penalized** when $\hat{y}$ is small (since $\log(\hat{y})$ is very negative near 0).  
→ **Encourages pushing $\hat{y}$ close to 1** for positive samples.

---

- **If \( y = 0 \)**, then:

$$
L = -\log(1 - \hat{y})
$$

→ The model is **heavily penalized** when $\hat{y}$ is large (predicting class 1 incorrectly).  
→ **Encourages pushing $\hat{y}$ close to 0** for negative samples.


## 🛠️ Part 4 – Learning Parameters with Gradient Descent

After defining the **loss (Binary Cross-Entropy)**, the next step is **optimizing the parameters**  
$$
\boldsymbol{\theta} = [\mathbf{w}, b]
$$  
to improve predictions.

The most common method is **Gradient Descent** — updating parameters **opposite to the gradient** of the loss.


**⚙️ 1️⃣ Loss function**

The loss for **Logistic Regression** is:

$$
L(\boldsymbol{\theta}) = -\sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big]
$$

with:

$$
\hat{y}^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}, \quad z^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)} + b
$$


**🧾 2️⃣ Gradient Descent rule**

Update:

$$
\boldsymbol{\theta} := \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})
$$

Where:

- $\eta$: learning rate  
- $\nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})$: gradient of the loss w.r.t. each parameter

**🧮 3️⃣ Derivatives of BCE**

- **W.r.t. weights:**

$$
\frac{\partial L}{\partial w} = X^T(\hat{y} - y)
$$

- **W.r.t. bias:**

$$
\frac{\partial L}{\partial b} = \sum_{i=1}^{m} \big(\hat{y}^{(i)} - y^{(i)}\big)
$$

These formulas hold for **Batch**, **Mini-Batch**, and **Stochastic** cases.

For a **single sample**, they reduce to:

$$
\frac{\partial L}{\partial w} = x^{(i)} \big(\hat{y}^{(i)} - y^{(i)}\big), \quad
\frac{\partial L}{\partial b} = \hat{y}^{(i)} - y^{(i)}
$$


**🔁 4️⃣ Parameter updates *(General Form)***

Update parameters via Gradient Descent:

$$
w := w - \eta \frac{\partial L}{\partial w}
$$

$$
b := b - \eta \frac{\partial L}{\partial b}
$$

Where:

- $\eta$: learning rate  
- $\frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}$: gradients of the loss w.r.t. each parameter

---

Training **Logistic Regression** is **very similar to Linear Regression**,  
since it also uses **Batch Gradient Descent** to optimize $w$ and $b$.

<p align="center">
  <img src="{{ 'assets/module6-week1/LogisticR_training.png' | absolute_url }}"
       alt="Training Logistic Regression" width="800">
</p>

## 🧭 Part 5 – Decision Boundary & Probabilistic View


**1️⃣ Decision boundary**

In **Logistic Regression**, the predicted probability of class 1 is:

$$
\hat{p} = \sigma(wx + b) = \frac{1}{1 + e^{-(wx + b)}}
$$

To classify, set a **threshold**:

$$
\hat{p} = 0.5
$$

This is equivalent to:

$$
\sigma(wx + b) = 0.5 \quad \Longleftrightarrow \quad wx + b = 0
$$

---

> 🔹 **The boundary $wx + b = 0$** is a **linear hyperplane**,  
> splitting the feature space into two halves:

- If $wx + b > 0 \Rightarrow \hat{p} > 0.5$ ⇒ Predict **class 1**  
- If $wx + b < 0 \Rightarrow \hat{p} < 0.5$ ⇒ Predict **class 0**

> 💬 **Conclusion:**  
> Logistic Regression yields a **linear decision boundary**, similar to Linear Regression,  
> but the output is passed through **sigmoid** to represent probability.

**2️⃣ Probabilistic (Log-Odds) view**

We have:

$$
\hat{p} = \frac{1}{1 + e^{-(wx + b)}}
$$

Then the **odds** (probability ratio of class 1 vs. 0) is:

$$
\text{odds} = \frac{\hat{p}}{1 - \hat{p}}
$$

Taking natural log gives the **log-odds** (a.k.a. *logit*):

$$
\log \frac{\hat{p}}{1 - \hat{p}} = wx + b
$$

> 🔸 This is **linear in $x$**!  
> In other words, Logistic Regression is **a linear regression model on log-odds**.

---

## 📚 Resources
- [Code for Dr. Quang-Vinh Dinh’s lecture (Google Drive)](https://…)  
- [Wikipedia – Logistic Function](https://en.wikipedia.org/wiki/Logistic_function)
- **Further reading:** Logistic Regression & Vectorization  

<br>

<div style="border:1px solid #e5e7eb; border-radius:14px; padding:18px 22px; background:#fafafa; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
  <h4 style="margin-top:0; font-weight:700;">📄 Report <code>MD06W01_v1.pdf</code></h4>
  <p style="margin:6px 0; line-height:1.6;">
    A 4-page A4 summary for <b>Week 1</b> — <b>Logistic Regression</b>, 
    <i>Vectorization and Application</i>.
  </p>
  <p style="margin:10px 0;">
    <a href="{{ site.baseurl }}/assets/module6-week1/Report_MD06W01_v1.pdf" 
       target="_blank" rel="noopener"
       style="display:inline-block; background:#eef4ff; padding:8px 16px; border-radius:8px; text-decoration:none; font-weight:600; color:#0044cc;">
      🔗 View or download PDF
    </a>
  </p>
</div>

---

💡 *See you in the next Week 1 lecture — where we dive deeper into **Multi-feature Logistic Regression**!* 🚀

---
layout: post
title: "💜 Advanced Logistic Regression — Vectorization & Application (AIO 2025)"
date: 2025-11-08
---

---

## Summary of the Friday lecture on 11/07/2025

## 🌸 Introduction

This week we move on to **Advanced Logistic Regression**, with two focuses:
1. Mastering **vectorization** — converting all computations into matrix form to speed up training.
2. Understanding **gradient descent & binary cross-entropy loss** from a linear-algebra viewpoint.

Ultimate goal: rewrite the **entire Logistic Regression pipeline** in a **vectorized** way — faster, shorter, and easier to extend to neural networks. 💻

---

## 🌿 Part 1 – Core Logistic Regression

**💎 Model essence**

Despite the name **“Regression”**, Logistic Regression is actually a **binary classification** model, not ordinary linear regression.  
Instead of predicting continuous values, it predicts the **probability of class 1**:

$$
P(y=1|x) = \hat{y} = \sigma(z)
$$

with:
$$
z = w^T x + b
$$

Where:  
- $w$: weight vector  
- $b$: bias  
- $x$: input feature vector  
- $\sigma(z)$: **sigmoid** mapping any real number to (0, 1), suitable for probability.

---

**🧭 Sigmoid — the bridge from linear to probabilistic**

Definition:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Properties:
- $z \to +\infty \Rightarrow \sigma(z) \to 1$
- $z \to -\infty \Rightarrow \sigma(z) \to 0$
- $z = 0 \Rightarrow \sigma(0) = 0.5$

→ This is the **midpoint** representing the decision boundary between the two classes.

Sigmoid **turns linear outputs into probabilities** — crucial for any binary classifier.

---

**📈 Classification rule**

Given \(\hat{y}\):

- If $\hat{y} \ge 0.5 \Rightarrow y_{\text{pred}} = 1$
- If $\hat{y} < 0.5 \Rightarrow y_{\text{pred}} = 0$

Example:  
If predicted probability is 0.83 → assign label “1”.  
If 0.24 → assign label “0”.

---

**⚙️ Loss — Binary Cross Entropy (Log Loss)**

To train Logistic Regression, use a loss measuring the gap between $\hat{y}$ and $y$:

$$
L(\hat{y}, y) = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]
$$

Explanation:
- If $y = 1$: the loss reduces to $-\log(\hat{y})$ → heavy penalty for low probabilities.  
- If $y = 0$: the loss reduces to $-\log(1 - \hat{y})$ → heavy penalty for high probabilities.

Across $N$ samples:
$$
J(w, b) = \frac{1}{N} \sum_{i=1}^{N} L(\hat{y}^{(i)}, y^{(i)})
$$

---

**🔧 Parameter updates — basic Gradient Descent**

Goal: find $w, b$ minimizing $J(w,b)$.  
Use **Gradient Descent**:

$$
w := w - \eta \frac{\partial J}{\partial w}, \qquad b := b - \eta \frac{\partial J}{\partial b}
$$

with \( \eta \) the **learning rate**.

Per-sample derivatives:

$$
\frac{\partial L}{\partial w} = x(\hat{y} - y), \qquad
\frac{\partial L}{\partial b} = (\hat{y} - y)
$$

→ Core formulas for training on a single sample.

---

**💡 One-sample learning flow**

1. **Pick one sample** $(x, y)$   
2. **Predict:**  
   $\hat{y} = \sigma(w^T x + b)$
3. **Loss:**  
   $L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$  
4. **Gradients:**  
   $dW = x(\hat{y} - y), \quad db = (\hat{y} - y)$  
5. **Update:**  
   $w := w - \eta dW, \quad b := b - \eta db$  

Repeat until convergence.

---

**🧩 Geometric meaning**

- The **decision boundary** is $w^T x + b = 0$.  
- On the boundary, $\hat{y} = 0.5$.  
- Each update of $w, b$ “rotates/shift” the boundary to separate data better.

---

**⚛️ Relation to Linear Regression loss**

Linear Regression uses **MSE**; Logistic Regression uses **BCE (Log Loss)**.  
They share the same objective (reduce prediction error) but differ because:
- MSE assumes real-valued targets, Log Loss assumes probabilities.  
- MSE is unsuitable when \(y\) is binary, as predictions can go outside [0, 1].

---

### 🌈 Summary of Part 1

- Logistic Regression is a **binary probabilistic classifier**.  
- Uses **sigmoid** to map linear outputs to probabilities.  
- Uses **Binary Cross-Entropy loss**.  
- Trained with **Gradient Descent** updating \(w, b\).  
- Easily extended to **vectorization**, **mini-batch**, and **deep layers**.

---

📘 *Source: Advanced Logistic Regression Slides – Quang-Vinh Dinh, PhD (2025)* :contentReference[oaicite:1]{index=1}

---

## 🌸 Part 2 – Vectorization for single sample, mini-batch, and full dataset

**🌿 1️⃣ From one sample → vectorized**

For **one sample** $(x, y)$:

$$
z = w^T x + b, \qquad 
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\nabla_w L = x(\hat{y} - y), \qquad 
\nabla_b L = \hat{y} - y
$$

---

Vectorization merges multiplications/additions/derivatives into **matrix operations** for speed and fewer bugs.  

Absorb $b$ into the parameter vector:

$$
\theta = [b, w_1, \dots, w_n]^T
$$

and extend features:

$$
x \to x' = [1, x_1, \dots, x_n]^T
$$

---

Then:

$$
z = \theta^T x', \qquad 
\hat{y} = \sigma(z)
$$

$$
\nabla_\theta L = x'(\hat{y} - y)
$$

$$
\theta := \theta - \eta \nabla_\theta L
$$

This formula **holds for every sample** and is the **core** for extending  
from **1 sample → m samples** in vectorized training.

---

**🪷 2️⃣ Vectorization for *m samples* (mini-batch)**

For a mini-batch $(x^{(1)},…,x^{(m)})$, form:

$$
X = 
\begin{bmatrix}
1 & x^{(1)}_1 & … & x^{(1)}_n \\
1 & x^{(2)}_1 & … & x^{(2)}_n \\
⋮ & ⋮ & ⋱ & ⋮ \\
1 & x^{(m)}_1 & … & x^{(m)}_n
\end{bmatrix}_{m\times(n+1)},\quad 
\theta = [b,w_1,…,w_n]^T
$$  

#### 🔹 Step 1 – Outputs
$$
z = X \theta,\qquad 
\hat y = \sigma(z)
$$

#### 🔹 Step 2 – Mini-batch loss
$$
L(\hat y,y)
 = -\frac1m\sum_{i=1}^m 
\big[y^{(i)}\log\hat y^{(i)}+(1-y^{(i)})\log(1-\hat y^{(i)})\big]
$$

#### 🔹 Step 3 – Vectorized gradient
$$
\nabla_\theta L = \frac1m X^T(\hat y - y)
$$

#### 🔹 Step 4 – Update
$$
\theta := \theta - \eta\nabla_\theta L
$$

---

**🌼 3️⃣ Geometry & optimization**

- Each row of $X$ is one sample’s feature vector.  
- $X \theta$ computes linear scores for all samples at once.  
- $\frac1m X^T(\hat y - y)$ averages the error directions → reduces batch loss.  

---

**🪞 4️⃣ Concrete example (from slides)**

With $m = 2$:
$$
X = 
\begin{bmatrix}
1 & 1.5 & 0.2\\
1 & 4.1 & 1.3
\end{bmatrix},
\quad 
y = \begin{bmatrix}0\\1\end{bmatrix},
\quad 
\theta = 
\begin{bmatrix}
0.1\\0.5\\-0.1
\end{bmatrix}
$$

Compute:
$$
z = X\theta = 
\begin{bmatrix}
0.83\\2.02
\end{bmatrix},\quad 
\hat y = \sigma(z) = 
\begin{bmatrix}
0.6963\\0.8828
\end{bmatrix}
$$

Loss:
$$
L = -\frac12
\big[\log(1-0.6963)+\log(0.8828)\big]
≈ 0.65815
$$

Gradient:
$$
\nabla_\theta L = \frac12 X^T(\hat y - y)
=
\begin{bmatrix}
0.2896\\0.2822\\-0.0064
\end{bmatrix}
$$

Update:
$$
\theta_{new} = \theta - \eta\nabla_\theta L
\Rightarrow 
\begin{bmatrix}
0.0971\\0.4971\\-0.099
\end{bmatrix}
\quad(\eta = 0.01)
$$  
📊 This is the **mini-batch (m = 2)** vectorized version.

---

**🌺 5️⃣ Full-batch vectorization (Batch GD)**

For $N$ samples:

$$
Z = X\theta, \qquad 
\hat{Y} = \sigma(Z)
$$

$$
L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}
\Big[ y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)}) \Big]
$$

$$
\nabla_\theta L = \frac{1}{N} X^T(\hat{Y} - Y)
$$

$$
\theta := \theta - \eta \nabla_\theta L
$$

---

### 📊 Example from slides:

$$
X =
\begin{bmatrix}
1 & 1.4 & 0.2\\
1 & 1.5 & 0.2\\
1 & 3.0 & 1.1\\
1 & 4.1 & 1.3
\end{bmatrix},
\quad 
y =
\begin{bmatrix}
0\\
0\\
1\\
1
\end{bmatrix}
$$

$$
Z = X\theta =
\begin{bmatrix}
0.78\\
0.83\\
1.49\\
2.02
\end{bmatrix}
\Rightarrow 
\hat{y} =
\begin{bmatrix}
0.6856\\
0.6963\\
0.8160\\
0.8828
\end{bmatrix}
$$

$$
L \approx 0.6691, \qquad 
\nabla_\theta L =
\begin{bmatrix}
0.2702\\
0.2431\\
-0.019
\end{bmatrix}
\Rightarrow 
\theta_{\text{new}} =
\begin{bmatrix}
0.0971\\
0.4971\\
-0.099
\end{bmatrix}
$$  

---

📘 **Processing all samples at once** helps the model:
- Stabilize gradients  
- Converge faster  
- Reduce numerical errors during training


---

**🌸 6️⃣ Summary of the three vectorization levels**

| 🌿 **Level** | ✏️ **Notation** | 📏 **Scale** | 🧮 **Gradient** | ⚙️ **Traits** |
|:--:|:--|:--|:--|:--|
| 1️⃣ | One sample | 1 sample | $ \nabla_\theta L = x(\hat{y} - y) $ | Slow updates, high noise |
| 2️⃣ | Mini-batch | m samples | $ \nabla_\theta L = \frac{1}{m} X^T(\hat{Y} - Y) $ | Balance speed/accuracy |
| 3️⃣ | Full batch | N samples | $ \nabla_\theta L = \frac{1}{N} X^T(\hat{Y} - Y) $ | Most stable, more memory |

---

**🪶 7️⃣ Optimization notes & practice**

- **Batch GD** fits small datasets, whole-epoch updates.  
- **Mini-batch** is standard in DL — efficient on GPUs with stable convergence.  
- **Stochastic (one sample)** suits streaming/online learning.

---

**💎 8️⃣ Vectorized Python (NumPy demo)**

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def vectorized_gradient(X, y, theta):
    m = X.shape[0]
    y_hat = sigmoid(X @ theta)
    grad = (1/m) * (X.T @ (y_hat - y))
    return grad

def train(X, y, lr=0.1, epochs=1000):
    theta = np.zeros(X.shape[1])
    for i in range(epochs):
        grad = vectorized_gradient(X, y, theta)
        theta -= lr * grad
        if i % 200 == 0:
            loss = -np.mean(y*np.log(sigmoid(X@theta)) + (1-y)*np.log(1-sigmoid(X@theta)))
            print(f"Iter {i:4d} | Loss={loss:.6f}")
    return theta
