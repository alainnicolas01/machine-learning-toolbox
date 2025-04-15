# Notation Guide for Machine Learning Toolbox

This table provides an overview of the mathematical notation used in the project and their corresponding Python representations.

---

## **General Notation**
| Notation | Description | Python (if applicable) |
|----------|------------|------------------------|
| *a*      | Scalar, non-bold | |
| **a**    | Vector, bold | |
| **A**    | Matrix, bold capital | |

---

## **Regression Notation**
| Notation | Description | Python (if applicable) |
|----------|------------|------------------------|
| **X**    | Training example matrix | `X_train` |
| **y**    | Training example targets | `y_train` |
| *x*^(i), *y*^(i) | *iᵗʰ* Training Example | `X[i]`, `y[i]` |
| *m*      | Number of training examples | `m` |
| *n*      | Number of features in each example | `n` |
| **w**    | Parameter: weights | `w` |
| *b*      | Parameter: bias | `b` |
| *f*_(w,b)_(*x*^(i)) | The result of the model evaluation at *x*^(i) parameterized by **w**, *b*:<br> *f*_(w,b)_(*x*^(i)) = **w** · *x*^(i) + *b* | `f_wb` |
| ∂J(w,b)/∂wⱼ | The gradient or partial derivative of cost w.r.t. a parameter *wⱼ* | `dj_dw[j]` |
| ∂J(w,b)/∂b | The gradient or partial derivative of cost w.r.t. a parameter *b* | `dj_db` |

---

This notation will be useful for understanding how data and parameters are structured in this project.
