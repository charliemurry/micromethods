# Derivation of the Log-Likelihood for the Logit Model

## 1. Model Setup and Notation

We consider a standard discrete choice framework where decision-makers choose a single alternative from a finite set of mutually exclusive options to maximize their utility.

### Dimensions
* Let $N$ be the number of individuals, indexed by $i = 1, \dots, N$.
* Let $J$ be the number of alternatives (choices), indexed by $j = 1, \dots, J$.

### Data Components
The data we have has the following features:
1.  **Individual Choices:** Let $y_{i}$ denote the choice observed for individual $i$, where $y_{i} \in \{1, \dots, J\}$. It is often helpful to represent this with dummy variables $d_{ij}$:
    $$
    d_{ij} = \begin{cases} 1 & \text{if individual } i \text{ chooses alternative } j \\ 0 & \text{otherwise} \end{cases}
    $$
2.  **Individual Characteristics:** Let $\mathbf{z}_i$ be a vector of characteristics specific to individual $i$ (e.g., income, age). Note that these do not vary across alternatives.
3.  **Features of the Choices:** Let $\mathbf{x}_{ij}$ be a vector of attributes for alternative $j$ as faced by individual $i$ (e.g., price, travel time).

## 2. Utility Specification

We assume a Random Utility Model (RUM) structure. The utility $U_{ij}$ that individual $i$ derives from alternative $j$ is composed of a deterministic component ($V_{ij}$) and a stochastic component ($\varepsilon_{ij}$):

$$U_{ij} = V_{ij} + \varepsilon_{ij}$$

### Parameterizing the Deterministic Utility ($V_{ij}$)
The deterministic part $V_{ij}$ is typically a linear function of parameters $\boldsymbol{\beta}$. To incorporate both choice features and individual characteristics, we must ensure identification:
* **Choice Features ($\mathbf{x}_{ij}$):** These enter directly as they vary by $j$.
* **Individual Characteristics ($\mathbf{z}_i$):** Since $\mathbf{z}_i$ is constant across all $j$, it falls out of the probability calculation (differences in utility) unless it is **interacted** with alternative-specific constants or attributes.

A common general specification is:
$$V_{ij} = \mathbf{z}_i*\mathbf{x}_{j}'\boldsymbol{\gamma} + \mathbf{x}_j'\boldsymbol{\beta}$$

* $\boldsymbol{\beta}$: Coefficients for choice attributes (generic across alternatives).
* $\boldsymbol{\beta}$: Coefficients for individual characteristics interacted with choice features. 
* Note: In the original mutlinomial specification, McFadden allowed choice specific coefficients, (normalized such that one $\boldsymbol{\gamma}_k = 0$ for identification). This is not the typical way we think about indirect utility in IO.

For the sake of this derivation, we can simplify notation by stacking all relevant covariates for alternative $j$ and individual $i$ into a single vector $\mathbf{w}_{ij}$ with coefficient vector $\boldsymbol{\theta}$:

$$V_{ij} = \mathbf{w}_{ij}'\boldsymbol{\theta}$$

* Note that the innovation in Berry (1994) and BLP 1995 is to add an error term so that $V_{ij} = \mathbf{w}_{ij}'\boldsymbol{\theta} + \xi_j$, which makes clear the source of price endogeneity when price is a component of $\mathbf{x}_j$. 

## 3. Probability Derivation (The Logit Formula)

The probability that individual $i$ chooses alternative $j$ is the probability that $U_{ij}$ is greater than the utility of any other alternative $k$:

$$P_{ij} = P(U_{ij} > U_{ik} \quad \forall k \neq j)$$
$$P_{ij} = P(\varepsilon_{ik} - \varepsilon_{ij} < V_{ij} - V_{ik} \quad \forall k \neq j)$$

### The Logit Assumption
We assume the error terms $\varepsilon_{ij}$ are independently and identically distributed (i.i.d.) according to a **Type I Extreme Value (Gumbel)** distribution.

The Cumulative Distribution Function (CDF) for Gumbel is $F(\varepsilon) = e^{-e^{-\varepsilon}}$. This assumption leads to the closed-form Logit probability:

$$P_{ij}(\boldsymbol{\theta}) = \frac{e^{V_{ij}}}{\sum_{k=1}^{J} e^{V_{ik}}} = \frac{e^{\mathbf{w}_{ij}'\boldsymbol{\theta}}}{\sum_{k=1}^{J} e^{\mathbf{w}_{ik}'\boldsymbol{\theta}}}$$

## 4. The Likelihood Function

The likelihood function $L(\boldsymbol{\theta})$ represents the joint probability of observing the sample data given the parameters. Assuming independent observations across individuals, the likelihood is the product of the probabilities of the chosen alternatives.

For a single individual $i$, the contribution to the likelihood is simply the probability of the choice they actually made. Using the indicator $d_{ij}$ defined earlier:

$$L_i(\boldsymbol{\theta}) = \prod_{j=1}^{J} (P_{ij})^{d_{ij}}$$

Note that since only one $d_{ij}$ is 1 and the rest are 0, this product simply selects the probability of the observed choice.

For the full sample of $N$ individuals:

$$L(\boldsymbol{\theta}) = \prod_{i=1}^{N} L_i(\boldsymbol{\theta}) = \prod_{i=1}^{N} \prod_{j=1}^{J} (P_{ij})^{d_{ij}}$$

## 5. The Log-Likelihood Function

Maximizing the likelihood is equivalent to maximizing the natural logarithm of the likelihood (log-likelihood), which is computationally more stable and easier to manipulate (converting products to sums).

$$\mathcal{LL}(\boldsymbol{\theta}) = \ln(L(\boldsymbol{\theta})) = \ln \left( \prod_{i=1}^{N} \prod_{j=1}^{J} (P_{ij})^{d_{ij}} \right)$$

Using the properties of logarithms ($\ln(\prod) = \sum \ln$ and $\ln(x^y) = y \ln x$):

$$\mathcal{LL}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \sum_{j=1}^{J} d_{ij} \ln(P_{ij})$$

### Substituting the Logit Probability
Substitute the explicit formula for $P_{ij}$:

$$\ln(P_{ij}) = \ln \left( \frac{e^{V_{ij}}}{\sum_{k=1}^{J} e^{V_{ik}}} \right) = V_{ij} - \ln \left( \sum_{k=1}^{J} e^{V_{ik}} \right)$$

Therefore, the final log-likelihood function is:

$$\mathcal{LL}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \sum_{j=1}^{J} d_{ij} \left( \mathbf{w}_{ij}'\boldsymbol{\theta} - \ln \left( \sum_{k=1}^{J} e^{\mathbf{w}_{ik}'\boldsymbol{\theta}} \right) \right)$$

Since $\sum_{j=1}^J d_{ij} = 1$ (the individual must choose exactly one option), the second term simplifies, as the log-sum constant does not depend on the specific $j$ being summed over:

$$\mathcal{LL}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \left( \sum_{j=1}^{J} d_{ij} (\mathbf{w}_{ij}'\boldsymbol{\theta}) - \ln \left( \sum_{k=1}^{J} e^{\mathbf{w}_{ik}'\boldsymbol{\theta}} \right) \right)$$

This is the objective function typically maximized via numerical methods (e.g., Newton-Raphson) to obtain the maximum likelihood estimates $\hat{\boldsymbol{\theta}}$.