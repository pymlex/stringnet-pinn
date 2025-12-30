# PINN for Nonlinear Transverse Vibrations of an Ideal String (StringNet)

![wave_3d](https://github.com/user-attachments/assets/5faa302e-38d8-452a-b13c-00107103905f)

## Introduction

In recent years, physics-informed neural networks (PINNs) have been increasingly applied to solving partial differential equations (PDEs). Of particular interest is solving problems in the [nonlinear](https://www.researchgate.net/publication/383078380_Physics-informed_neural_network_for_nonlinear_dynamics_of_self-trapped_necklace_beams) regime, where the equation contains combinations of products and powers of the function and its higher-order partial derivatives.

PINNs have previously been used to solve the wave equation in the linear regime in [one-dimensional](https://github.com/okada39/pinn_wave/tree/master) and [multi-dimensional](https://arxiv.org/pdf/2006.11894) settings. In the case of modeling the dynamics of a string, this approach allows the investigation to be limited to computing second derivatives with respect to space and time, using only the linear mass density and tension from the physical parameters of the problem. Such a formulation permits obtaining an analytical solution.

As [shown](https://mtt.ipmnet.ru/files/1993/1993-4/mtt1993_n4_p87-92.pdf) by L. D. Akulenko and S. V. Nesterov, describing the dynamics of a string with rigidly fixed ends in the nonlinear regime for oscillations along two mutually orthogonal axes $u$ and $v$ leads to a coupling between $u$ and $v$. The authors proposed a single-mode analytical solution using a series of approximations.

We propose a solution of the system of partial differential equations describing the dynamics of a string with fixed ends for given initial conditions using a physics-informed neural network (PINN). The work employs modern [techniques](https://arxiv.org/pdf/2308.08468) to improve PINN training quality: nondimensionalization, weighting losses from different parts of the system, exponential weighting of losses from temporal segments at interior points, input representations using random Fourier features, and so on.

As a result of this work, a model called StringNet was trained to predict the displacements $(u, v)$ from equilibrium for input points. A 3D visualization of the string’s oscillations over time was produced. The solution shows good agreement with the initial and boundary conditions and exhibits smoothness.

---

## Governing equations (dimensional and nondimensional form)

Dimensional nonlinear PDEs (first nonlinear approximation, $\rho$ — linear mass density, $T$ — tension, $N=ES-T$):

$$\rho\ u_{tt} = \Bigl(T + \tfrac12 N (u_x^2 + v_x^2)\Bigr) u_{xx} + N (u_x u_{xx} + v_x v_{xx}) u_x,$$
$$\rho\ v_{tt} = \Bigl(T + \tfrac12 N (u_x^2 + v_x^2)\Bigr) v_{xx} + N (u_x u_{xx} + v_x v_{xx}) v_x.$$

Boundary conditions (clamped ends):

$$u(0,t)=u(L,t)=0,\qquad v(0,t)=v(L,t)=0.$$

Example initial conditions used (dimensional):

$$u(x,0)=A\sin\!\Big(\frac{2\pi x}{L}\Big),\quad v(x,0)=4A\frac{x}{L}\Big(1-\frac{x}{L}\Big),\quad u_t(x,0)=v_t(x,0)=0.$$

Nondimensionalisation:

$$x'=\frac{x}{L},\quad t'=\frac{c t}{L},\quad u'=\frac{u}{A},\quad v'=\frac{v}{A},\quad c=\sqrt{\frac{T}{\rho}}.$$

Define the small parameter $\beta$:

$$\beta=\frac{N}{T}\frac{A^2}{L^2}\ll 1.$$

Nondimensional PDEs (primes dropped):

$$u_{tt} = \Bigl(1 + \tfrac{1}{2}\beta (u_x^2+v_x^2)\Bigr) u_{xx} + \beta (u_x u_{xx} + v_x v_{xx}) u_x,$$
$$v_{tt} = \Bigl(1 + \tfrac{1}{2}\beta (u_x^2+v_x^2)\Bigr) v_{xx} + \beta (u_x u_{xx} + v_x v_{xx}) v_x.$$

Initial and boundary conditions in nondimensional form:

$$u(x,0)=\sin(2\pi x),\quad v(x,0)=4x(1-x),\quad u_t(x,0)=v_t(x,0)=0,$$

$$u(0,t)=u(1,t)=0,\quad v(0,t)=v(1,t)=0.$$

---

## PINN architecture and training

The `StringNet` network consists of:
- $(x',t')$ input, optionally passed through RFF embedding.
- 4 hidden layers, each 60 neurons, activation $\tanh$.
- 2 output values $[u',v']$.

The Random Fourier Features (RFF) extraction is also implemented:

$$\gamma(z) = [\cos(B z),\; \sin(B z)],\quad B_{ij}\sim\mathcal{N}(0,\sigma^2),$$

use it by setting `use_rff=True`. 

**Loss components** are the initial-condition loss, the boundary-condition loss, and the PDE-residual loss, calculated with L2 normalization. The total loss is weighted as follows:

$$\mathcal L_{\text{total}} = \lambda_{ic}\,\mathcal L_{ic} + \lambda_{bc}\,\mathcal L_{bc} + \lambda_r\,\frac{\mathcal L_r^{\text{weighted}}}{M},$$

where $\mathcal L_r^{\text{weighted}}$ is the temporally weighted sum of residual losses:

$$\mathcal{L}_r(\mathbf{\theta}) = \frac{1}{M} \sum_{i=1}^M w_i \mathcal L_r^i(\mathbf{\theta}),$$
$$w_i = \exp\left(- \epsilon \sum_{k=1}^{i-1}  \mathcal{L}_r^k( \mathbf{\theta})\right),$$

and $\lambda_{ic},\lambda_{bc},\lambda_r$ are adaptive global scaling weights.

The optimiser was chosen to be `Adam` with learning rate equal to $10^{-3}$.

The model was trained on a T4 GPU in Google Colab. The example training run typically takes on the order of 15–30 minutes for the configuration above. Actual time depends on whether RFF is used and on the number of collocation points.
