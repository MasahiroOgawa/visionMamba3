---
title: Derivation of All Mamba-3 Equations
tags: Mamba, SSM, DeepLearning, MachineLearning, Math
---

This article derives the key equations (1),(2) in the [Mamba-3 paper](https://arxiv.org/abs/2603.15569) with one goal: **every single step should be easy to follow**. No steps are skipped. I will post the other proof later. Each equation transforms into the next with a clear reason. If you can follow basic calculus and linear algebra, you can follow this entire derivation.

# 1. Definitions: Dimensions and Parameters

## 1.1 Basic Dimensions

- $T$: Sequence Length
- $D$: Number of Input/Output Channels (Model Dimension)
- $N$: State Dimension (per channel)

## 1.2 Variable Dimension Definitions

- $\mathbf{X} \in \mathbb{R}^{T \times D}$: Input sequence
- $\mathbf{Y} \in \mathbb{R}^{T \times D}$: Output sequence
- $\mathbf{h}(t) \in \mathbb{C}^{D \times N}$: State matrix. Each column $\mathbf{h}\_d(t) \in \mathbb{C}^{N \times 1}$ is the state vector of channel $d$.
- $\mathbf{A}(t)$: Collection of diagonal components.
    - Storage dimension: $\mathbb{C}^{D \times N}$. Each element $A\_{d,n}(t)$ is the complex coefficient for the $n$-th state of channel $d$.
    - Operational dimension: $\mathbb{C}^{D \times N \times N}$. For each channel $d$, it acts as an $N \times N$ diagonal matrix $\mathbf{A}\_d(t) = \mathrm{diag}(A\_{d,1}(t), \dots, A\_{d,N}(t))$.
- $\mathbf{B}(t) \in \mathbb{R}^{T \times N}$, $\mathbf{C}(t) \in \mathbb{R}^{T \times N}$: Projection vectors shared across all channels.

# 2. Derivation of Equation (1): The Discrete SSM Recurrence

The paper's equation (1) is the discrete-time recurrence:

$$\mathbf{h}\_t = \alpha\_t \mathbf{h}\_{t-1} + \gamma\_t \mathbf{B}\_t \mathbf{x}\_t, \qquad \mathbf{y}\_t = \mathbf{C}\_t^\top \mathbf{h}\_t \tag{paper 1}$$

where:

- $A\_t < 0$ is a scalar (the *scalar SSM* parameterization: $\mathbf{A}\_t = A\_t \mathbf{I}\_N$). The constraint $A\_t < 0$ ensures stability: in the continuous-time ordinary differential equation (ODE), the homogeneous solution is $\mathbf{h}(t) \propto e^{At}$, which decays only when $A < 0$. If $A > 0$, the state would grow exponentially.
- $\Delta\_t > 0$ is the time step size.
- $\alpha\_t := e^{\Delta\_t A\_t} \in (0,1)$ is the scalar state-transition.
- $\gamma\_t := \Delta\_t$ is the discretization factor.
- $\mathbf{B}\_t \in \mathbb{R}^{N}$: input projection, $\mathbf{C}\_t \in \mathbb{R}^{N}$: output projection, $\mathbf{h}\_t \in \mathbb{R}^{N}$: hidden state.

We derive this from the continuous-time state space model (SSM).

## 2.1 Continuous-Time SSM

The underlying continuous-time ODE is:

$$\dot{\mathbf{h}}(t) = \mathbf{A}(t) \mathbf{h}(t) + \mathbf{B}(t) x(t), \qquad y(t) = \mathbf{C}(t)^\top \mathbf{h}(t) \tag{1}$$

where $\mathbf{h}(t) \in \mathbb{R}^{N}$, $\mathbf{A}(t) \in \mathbb{R}^{N \times N}$, $\mathbf{B}(t), \mathbf{C}(t) \in \mathbb{R}^{N}$, and $x(t), y(t) \in \mathbb{R}$.
This is for a single channel; the multi-channel (multi-input, multi-output (MIMO)) extension is discussed in Section 3.3.

## 2.2 Continuous-Time Solution (Integrating Factor Method)

To solve the first-order linear non-homogeneous ODE (1), we multiply both sides by the integrating factor $\exp\left(-\int\_{0}^{t} \mathbf{A}(\tau) d\tau\right)$:

$$\exp\left(-\int\_{0}^{t}\mathbf{A}(\tau) d\tau\right) \left[\dot{\mathbf{h}}(t) - \mathbf{A}(t) \mathbf{h}(t) \right] = \exp\left(-\int\_{0}^{t}\mathbf{A}(\tau) d\tau\right) \mathbf{B}(t) x(t) \tag{2}$$

We apply the reverse of the product rule $\frac{d}{dt}(fg) = f'g + fg'$ to the left-hand side:

$$\frac{d}{dt}\left[\exp\left(-\int\_{0}^{t}\mathbf{A}(\tau) d\tau\right) \mathbf{h}(t) \right] = \exp\left(-\int\_{0}^{t}\mathbf{A}(\tau) d\tau\right) \mathbf{B}(t) x(t) \tag{3}$$

Setting the initial state $\mathbf{h}(0)=\mathbf{0}$ and integrating both sides from $0$ to $t$:

$$\exp\left(-\int\_{0}^{t}\mathbf{A}(\tau) d\tau\right) \mathbf{h}(t) - \underbrace{\exp(0) \mathbf{h}(0)}\_{= \mathbf{0}} = \int\_{0}^{t}\exp\left(-\int\_{0}^{s}\mathbf{A}(\tau) d\tau\right) \mathbf{B}(s) x(s) ds \tag{4}$$

Multiplying both sides from the left by $\exp\left(\int\_{0}^{t}\mathbf{A}(\tau) d\tau\right)$:

$$\mathbf{h}(t) = \exp\left(\int\_{0}^{t}\mathbf{A}(\tau) d\tau\right) \int\_{0}^{t}\exp\left(-\int\_{0}^{s}\mathbf{A}(\tau) d\tau\right) \mathbf{B}(s) x(s) ds \tag{5}$$

Since $\exp\left(\int\_{0}^{t}\mathbf{A} d\tau\right)$ does not depend on $s$, we bring it inside the integral by linearity. Then, because $\mathbf{A}$ is diagonal, $\int\_{0}^{t}\mathbf{A} d\tau$ and $\int\_{0}^{s}\mathbf{A} d\tau$ commute, so we can combine the two matrix exponentials via $\exp(\mathbf{M}\_1)\exp(\mathbf{M}\_2) = \exp(\mathbf{M}\_1 + \mathbf{M}\_2)$:

$$\mathbf{h}(t) = \int\_{0}^{t}\exp\left(\int\_{0}^{t}\mathbf{A}(\tau) d\tau - \int\_{0}^{s}\mathbf{A}(\tau) d\tau\right) \mathbf{B}(s) x(s) ds \tag{6}$$

By the linearity of integration $\int\_{0}^{t} - \int\_{0}^{s} = \int\_{s}^{t}$, we obtain the continuous-time general solution:

$$\mathbf{h}(t) = \int\_{0}^{t} \underbrace{\exp\left(\int\_{s}^{t}\mathbf{A}(\tau) d\tau\right)}\_{\Phi(t,s)} \mathbf{B}(s) x(s) ds \qquad[\mathbf{h}(t)\in\mathbb{R}^{N}] \tag{7}$$

## 2.3 Discretization

We derive the state $\mathbf{h}\_t$ at time $\tau\_t$ by splitting the integration range $[0, \tau\_t]$ into $[0, \tau\_{t-1}]$ and $[\tau\_{t-1}, \tau\_t]$:

$$\mathbf{h}\_t = \int\_{0}^{\tau\_{t-1}}\Phi(\tau\_t,s) \mathbf{B}(s) x(s) ds + \int\_{\tau\_{t-1}}^{\tau\_t}\Phi(\tau\_t,s) \mathbf{B}(s) x(s) ds \tag{8}$$

We apply the semigroup property $\Phi(t,s)=\Phi(t,u)\Phi(u,s)$ to the first term:

$$\mathbf{h}\_t = \Phi(\tau\_t,\tau\_{t-1}) \underbrace{\int\_{0}^{\tau\_{t-1}}\Phi(\tau\_{t-1},s) \mathbf{B}(s) x(s) ds}\_{\mathbf{h}\_{t-1}} + \int\_{\tau\_{t-1}}^{\tau\_t}\Phi(\tau\_t,s) \mathbf{B}(s) x(s) ds \tag{9}$$

**Right-hand approximation.** Following the paper (Section 3.1.1), we approximate $\mathbf{A}(s) \approx \mathbf{A}(\tau\_t) =: \mathbf{A}\_t$ for all $s \in [\tau\_{t-1},\tau\_t]$, with $\Delta\_t := \tau\_t - \tau\_{t-1}$.

**State-transition.** From the definition of $\Phi$ in (7):

$$\Phi(\tau\_t,\tau\_{t-1}) = \exp\left(\int\_{\tau\_{t-1}}^{\tau\_t}\mathbf{A}(s) ds\right) \approx \exp\left(\int\_{\tau\_{t-1}}^{\tau\_t}\mathbf{A}\_t ds\right) = \exp\left(\mathbf{A}\_t(\tau\_t - \tau\_{t-1})\right) = \exp(\Delta\_t \mathbf{A}\_t) \tag{10}$$

**State-input integral.** Starting from the second term in (9), we expand $\Phi(\tau\_t,s)$ under the right-hand approximation $\mathbf{A}(s)\approx\mathbf{A}\_t$:

$$\int\_{\tau\_{t-1}}^{\tau\_t}\Phi(\tau\_t,s) \mathbf{B}(s) x(s) ds = \int\_{\tau\_{t-1}}^{\tau\_t}\exp\left(\int\_{s}^{\tau\_t}\mathbf{A}(\sigma)d\sigma\right) \mathbf{B}(s) x(s) ds \approx \int\_{\tau\_{t-1}}^{\tau\_t}\exp\left((\tau\_t-s)\mathbf{A}\_t\right) \mathbf{B}(s) x(s) ds \tag{11}$$

Under exponential-Euler, we further hold $\mathbf{B}(s)\approx\mathbf{B}\_t$ and $x(s)\approx x\_t$:

$$\approx \int\_{\tau\_{t-1}}^{\tau\_t}\exp\left((\tau\_t-s)\mathbf{A}\_t\right) ds \; \mathbf{B}\_t x\_t \approx \Delta\_t \mathbf{B}\_t x\_t \tag{12}$$

This yields the general discrete update:

$$\mathbf{h}\_t \approx \exp(\Delta\_t \mathbf{A}\_t)\mathbf{h}\_{t-1} + \Delta\_t \mathbf{B}\_t x\_t \tag{13}$$

**Scalar SSM parameterization.** Mamba-2/3 parameterizes $\mathbf{A}\_t = A\_t \mathbf{I}\_N$ with $A\_t < 0$ (scalar times identity). Defining

$$\alpha\_t := e^{\Delta\_t A\_t} \in (0,1), \qquad \gamma\_t := \Delta\_t \tag{14}$$

the update (13) becomes:

$$\mathbf{h}\_t = \alpha\_t \mathbf{h}\_{t-1} + \gamma\_t \mathbf{B}\_t x\_t, \qquad y\_t = \mathbf{C}\_t^\top \mathbf{h}\_t \tag{15}$$

which is the paper's equation (1). $\blacksquare$

## 2.4 Recursive Unrolling

We unroll the recurrence (15) starting from $\mathbf{h}\_0 = \mathbf{0}$.

**Base cases.**

$$\begin{aligned} \mathbf{h}\_0 &= \mathbf{0} \\\\ \mathbf{h}\_1 &= \alpha\_1\mathbf{h}\_0 + \gamma\_1\mathbf{B}\_1 x\_1 = \gamma\_1\mathbf{B}\_1 x\_1 \\\\ \mathbf{h}\_2 &= \alpha\_2\mathbf{h}\_1 + \gamma\_2\mathbf{B}\_2 x\_2 = \alpha\_2\gamma\_1\mathbf{B}\_1 x\_1 + \gamma\_2\mathbf{B}\_2 x\_2 \\\\ \mathbf{h}\_3 &= \alpha\_3\mathbf{h}\_2 + \gamma\_3\mathbf{B}\_3 x\_3 = \alpha\_3\alpha\_2\gamma\_1\mathbf{B}\_1 x\_1 + \alpha\_3\gamma\_2\mathbf{B}\_2 x\_2 + \gamma\_3\mathbf{B}\_3 x\_3 \end{aligned}$$

**Pattern.** Each term indexed by $s$ carries the input $\gamma\_s \mathbf{B}\_s x\_s$ propagated forward through the product of transition scalars from step $s+1$ to $t$. Using the paper's cumulative product notation $\alpha\_{t:s}^\times := \prod\_{j=s}^{t}\alpha\_j$ (with $\alpha\_{t:t+1}^\times = 1$ when the range is empty):

$$\mathbf{h}\_t = \sum\_{s=1}^{t}\alpha\_{t:s+1}^\times \gamma\_s \mathbf{B}\_s x\_s \tag{16}$$

# 3. Derivation of Equation (2): The Matrix Output Form

Using the discrete SSM derived in Section 2, we now derive the matrix output form $\mathbf{Y} = (\mathbf{L}\odot\mathbf{C}\mathbf{B}^\top)\mathbf{X}$ (paper equation 2).

## 3.1 Output Equation

The output at time step $t$ is:

$$y\_t = \mathbf{C}\_t^\top \mathbf{h}\_t \tag{17}$$

Substituting the unrolled state (16):

$$y\_t = \mathbf{C}\_t^\top \sum\_{s=1}^{t}\alpha\_{t:s+1}^\times \gamma\_s \mathbf{B}\_s x\_s = \sum\_{s=1}^{t}\alpha\_{t:s+1}^\times \gamma\_s (\mathbf{C}\_t^\top\mathbf{B}\_s) x\_s \tag{18}$$

Since $\alpha\_t$ is scalar, the cumulative product $\alpha\_{t:s+1}^\times$ commutes freely with the vectors.

## 3.2 Kernel Factorization

Define (for $t,s = 1,\dots,T$):

- **Structured mask** $\mathbf{L}\in\mathbb{R}^{T\times T}$: $(t,s)$-element $L\_{ts}=\alpha\_{t:s+1}^\times \gamma\_s = \left(\prod\_{j=s+1}^{t}\alpha\_j\right)\gamma\_s$ (lower triangular: $L\_{ts}=0$ for $s>t$, because the recurrence is causal --- the unrolled sum (16) only includes inputs up to time $t$, so future inputs $x\_s$ with $s>t$ have no contribution)
- **Projection kernel** $\mathbf{C}\mathbf{B}^\top\in\mathbb{R}^{T\times T}$: $(t,s)$-element $(\mathbf{C}\mathbf{B}^\top)\_{ts}=\mathbf{C}\_t^\top\mathbf{B}\_s = \sum\_{n=1}^{N}C\_{t,n}B\_{s,n}$

Note that $\mathbf{L}$ absorbs both the cumulative transition $\alpha\_{t:s+1}^\times$ and the discretization factor $\gamma\_s$. Both $\mathbf{L}$ and $\mathbf{C}\mathbf{B}^\top$ are $T\times T$ matrices (shared across channels).

The output becomes:

$$y\_t = \sum\_{s=1}^{T}L\_{ts}\cdot(\mathbf{C}\mathbf{B}^\top)\_{ts}\cdot x\_s = \sum\_{s=1}^{T}(\mathbf{L}\odot\mathbf{C}\mathbf{B}^\top)\_{ts} x\_s \tag{19}$$

## 3.3 Multi-Channel (MIMO) Extension

The recurrence (15) is applied independently to each of $D$ channels with the same $\mathbf{B}\_t$, $\mathbf{C}\_t$, and $\mathbf{L}$, but potentially different inputs $x\_t^{(d)}$. Collecting all channels into matrices $\mathbf{X},\mathbf{Y}\in\mathbb{R}^{T\times D}$, the per-channel outputs stack into:

$$\mathbf{Y} = (\mathbf{L}\odot\mathbf{C}\mathbf{B}^\top) \mathbf{X} \tag{20}$$

where $(\mathbf{L}\odot\mathbf{C}\mathbf{B}^\top)\in\mathbb{R}^{T\times T}$ is a single matrix applied to all $D$ columns of $\mathbf{X}$. $\blacksquare$
