# PDCP
Primal Dual Algorithm for Conic Programming
- A Julia and CUDA implementation of the Primal-Dual algorithm for conic programming

### Overview
This project implements the algorithm for solving conic optimization problems of the form:

$$\min_{x=(x_1,x_2),x_1\in \mathbb{R}^{n_1}, x_2\in \mathbb{R}^{n_2}} c^{\top} x\ \  \text{s.t.} Gx-h\in \mathcal{K}_G, l\leq x_1\leq u, x_2 \in \mathcal{K}_x,$$

where $\mathcal{K}_G$ is a closed convex cone, and $\mathcal{K}_x$ is a closed convex cone.



### Features
- Supports multiple cone types:
    - Second-order cone (recommended but not the rsoc) and Rotated second-order cone
    - Exponential cone and dual exponential cone


### Installation
```bash
git clone https://github.com/zhenweilin/PDCP.git
cd PDCP
cd src/pdcp_gpu/cuda
make # compile the cuda code
cd ../../..
```


### Usage
- Given the example code in `./test/`, you can run the following commands to test the code.
```julia
julia ./test/env.jl # install the dependencies
julia ./test/test_exp.jl # test the exponential cone cpu
julia ./test/test_soc.jl # test the second-order cone cpu
julia ./test/test_rsoc.jl # test the rotated second-order cone cpu

julia ./test/test_exp_gpu.jl # test the exponential cone gpu
julia ./test/test_soc_gpu.jl # test the second-order cone gpu
julia ./test/test_rsoc_gpu.jl # test the rotated second-order cone gpu
```


### Performance
Three criteria are considered for the performance:
- Primal infeasibility: $\frac{\|(Gx - h) - \text{proj}_{\mathcal{K}_G}(Gx - h) \|_{\infty}}{1+\max(\|h\|_{\infty}, \|Gx\|_{\infty}, \|\text{proj}_{\mathcal{K}_G}(Gx - h) \|_{\infty})}$
- Dual infeasibility: $\frac{\max\{\|\lambda_1-\text{proj}_{\Lambda_1}(\lambda_1)\|_{\infty},\|\lambda_2-\text{proj}_{\mathcal{K}_x^*}(\lambda_2)\|_{\infty}\}}{1+\max\{\|c\|_{\infty},\|G^\top y\|_{\infty}\}}$, 
- Objective value accuracy: $\frac{|c^{\top}x-(y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-})|}{1+\max\{|c^{\top}x|, |y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-}|\}}$

where $\lambda=c-G^{\top}y=[\lambda_{1}^{\top},\lambda_{2}^{\top}]^{\top},\lambda_1\in \Lambda_1 \subseteq \mathbb{R}^{n_1}, \lambda_2\in \mathbb{R}^{n_2}$, and 
 $\Lambda_{1}=\begin{cases}
    \{0\} & l_{i}=-\infty,u_{i}=+\infty\\
    \mathbb{R}^{-} & l_{i}=-\infty,u_{i}\in\mathbb{R}\\
    \mathbb{R}^{+} & l_{i}\in\mathbb{R},u_{i}=+\infty\\
    \mathbb{R} & \text{otherwise}
\end{cases}$

##### CBLIB dataset
CBLIB is a classical [dataset collection](https://cblib.zib.de/download/all/) for testing conic programming, which includes some mixed-integer conic programming problems. Through a relaxation process, these mixed-integer conic constraints are transformed into continuous constraints, resulting in a refined dataset of 1943 problems without exponential cone constraints and 205 problems with such constraints. We employed the COPT solver to presolve all datasets.
Finally, the algorithm terminates when all three criteria are smaller than $10^{-6}$ and calculate the SGM(10).

**Table 1: Conic Programming without Exponential Cone**  
*(Columns show average SGM(10) time in seconds and number of problems solved, for each problem size category.) Use the nonzeros of the G matrix to classify the problem size.*

| **Method**      | **Small (1641)<br>SGM(10)** | **Small (1641)<br>solved** | **Medium (220)<br>SGM(10)** | **Medium (220)<br>solved** | **Large (82)<br>SGM(10)** | **Large (82)<br>solved** | **Total (1943)<br>SGM(10)** | **Total (1943)<br>solved** |
|-----------------|-----------------------------|----------------------------|-----------------------------|----------------------------|---------------------------|---------------------------|-----------------------------|----------------------------|
| **PDCP(CPU)**   | 2.26                       | 1641                       | 209.61                      | 219                        | 313.22                    | 80                        | 2.74                        | 1940                       |
| **PDCP(GPU)**   | 2.92                       | 1640                       | 164.62                      | 219                        | 311.30                    | 82                        | 2.27                        | 1941                       |


**Table 2: Conic Programming with Exponential Cone**  
*(The total number of cases is 157.)*

|                | **PDCP(CPU)** | **PDCP(GPU)** |
|----------------|---------------:|-----------:|
| **SGM(10)**    | 17.43          | 12.02       |
| **solved**     | 152           | 155        |


### Fisher Market Equilibrium
$$
\begin{aligned}
\min_{p\in \mathbb{R}^m, x\in \mathbb{R}^{m\times n}} & -\sum_{i\in[m]}w_i\log\left(\sum_{j\in[n]}u_{ij}x_{ij}\right) \\
\text{subject to} \quad & \sum_{i\in[m]}x_{ij}=b_j, \quad x_{ij}\geq 0
\end{aligned}
$$


*Note: The first three columns m, n, nnz are for matrix u. "f" means the solver fails by out of memory or return timeout flag.*

#### Results for tolerance $10^{-3}$

| m | n | nnz | PDCP(GPU) |
|---:|---:|---:|---:|
| 5.0E+02 | 5.0E+00 | 2.5E+03 | 5.7E-01 |
| 1.0E+03 | 5.0E+00 | 5.0E+03 | 5.9E-01 |
| 1.0E+04 | 5.0E+01 | 2.5E+05 | 3.5E+00 |
| 5.0E+04 | 5.0E+01 | 1.3E+06 | 1.2E+01 |
| 1.0E+05 | 5.0E+01 | 1.0E+06 | 1.4E+01 |
| 1.0E+05 | 5.0E+02 | 1.0E+07 | 4.2E+02 |
| 1.0E+05 | 1.0E+03 | 2.0E+07 | 3.8E+02 |
| 1.2E+05 | 1.0E+03 | 2.5E+07 | 5.9E+02 |
| 1.5E+05 | 1.0E+03 | 3.0E+07 | 6.6E+02 |
| 1.8E+05 | 1.0E+03 | 3.5E+07 | 1.1E+03 |
| 2.0E+05 | 1.0E+03 | 4.0E+07 | 1.0E+03 |
| 2.2E+05 | 1.0E+03 | 4.5E+07 | 1.3E+03 |
| 2.5E+05 | 1.0E+03 | 5.0E+07 | 1.8E+03 |
| 2.8E+05 | 1.0E+03 | 5.5E+07 | 1.6E+03 |

#### Results for tolerance $10^{-6}$

| m | n | nnz | PDCP(GPU) |
|---:|---:|---:|---:|
| 5.0E+02 | 5.0E+00 | 2.5E+03 | 4.7E+02 |
| 1.0E+03 | 5.0E+00 | 5.0E+03 | 4.5E+02 |
| 1.0E+04 | 5.0E+01 | 2.5E+05 | 3.7E+01 |
| 5.0E+04 | 5.0E+01 | 1.3E+06 | 9.1E+01 |
| 1.0E+05 | 5.0E+01 | 1.0E+06 | 1.6E+02 |
| 1.0E+05 | 5.0E+02 | 1.0E+07 | 2.9E+03 |
| 1.0E+05 | 1.0E+03 | 2.0E+07 | 3.3E+03 |
| 1.2E+05 | 1.0E+03 | 2.5E+07 | 5.0E+03 |
| 1.5E+05 | 1.0E+03 | 3.0E+07 | 1.0E+04 |
| 1.8E+05 | 1.0E+03 | 3.5E+07 | 7.3E+03 |
| 2.0E+05 | 1.0E+03 | 4.0E+07 | 1.6E+04 |
| 2.2E+05 | 1.0E+03 | 4.5E+07 | 9.4E+03 |
| 2.5E+05 | 1.0E+03 | 5.0E+07 | 1.2E+04 |
| 2.8E+05 | 1.0E+03 | 5.5E+07 | 1.2E+04 |

### Lasso Problem 
$$
\min_{x\in \mathbb{R}^n} \frac{1}{2}\|Ax-b\|^2+\lambda \|x\|_1
$$

### Lasso Problem Results

#### Results for tolerance $10^{-3}$
| m | n | nnz | PDCP(GPU) |
|---:|---:|---:|---:|
| 1.0E+04 | 1.0E+05 | 1.0E+05 | 7.1E-02 |
| 1.0E+04 | 4.0E+05 | 4.0E+05 | 1.1E-01 |
| 1.0E+04 | 7.0E+05 | 7.0E+05 | 1.2E-01 |
| 4.0E+04 | 1.0E+05 | 4.0E+05 | 7.8E-02 |
| 4.0E+04 | 4.0E+05 | 1.6E+06 | 1.7E-01 |
| 4.0E+04 | 7.0E+05 | 2.8E+06 | 3.0E-01 |
| 7.0E+04 | 1.0E+05 | 7.0E+05 | 1.3E-01 |
| 7.0E+04 | 4.0E+05 | 2.8E+06 | 2.6E-01 |
| 7.0E+04 | 7.0E+05 | 4.9E+06 | 2.9E-01 |
| 1.0E+05 | 1.0E+06 | 1.0E+07 | 5.5E-01 |
| 1.0E+05 | 4.0E+06 | 4.0E+07 | 9.1E+00 |
| 1.0E+05 | 7.0E+06 | 7.0E+07 | 1.9E+01 |
| 4.0E+05 | 1.0E+06 | 4.0E+07 | 3.3E+00 |
| 4.0E+05 | 4.0E+06 | 1.6E+08 | 3.6E+01 |
| 4.0E+05 | 7.0E+06 | 2.8E+08 | 1.2E+02 |
| 7.0E+05 | 1.0E+06 | 7.0E+07 | 5.6E+00 |
| 7.0E+05 | 4.0E+06 | 2.8E+08 | 6.2E+01 |
| 7.0E+05 | 7.0E+06 | 4.9E+08 | 3.2E+02 |
| 7.5E+05 | 7.5E+06 | 5.6E+08 | 4.7E+02 |

#### Results for tolerance $10^{-6}$

| m | n | nnz | PDCP(GPU) |
|---:|---:|---:|---:|
| 1.0E+04 | 1.0E+05 | 1.0E+05 | 1.1E-01 |
| 1.0E+04 | 4.0E+05 | 4.0E+05 | 2.4E-01 |
| 1.0E+04 | 7.0E+05 | 7.0E+05 | 5.3E-01 |
| 4.0E+04 | 1.0E+05 | 4.0E+05 | 1.5E-01 |
| 4.0E+04 | 4.0E+05 | 1.6E+06 | 3.6E-01 |
| 4.0E+04 | 7.0E+05 | 2.8E+06 | 6.1E-01 |
| 7.0E+04 | 1.0E+05 | 7.0E+05 | 2.1E-01 |
| 7.0E+04 | 4.0E+05 | 2.8E+06 | 5.9E-01 |
| 7.0E+04 | 7.0E+05 | 4.9E+06 | 8.4E-01 |
| 1.0E+05 | 1.0E+06 | 1.0E+07 | 1.4E+00 |
| 1.0E+05 | 4.0E+06 | 4.0E+07 | 2.6E+01 |
| 1.0E+05 | 7.0E+06 | 7.0E+07 | 5.6E+01 |
| 4.0E+05 | 1.0E+06 | 4.0E+07 | 7.5E+00 |
| 4.0E+05 | 4.0E+06 | 1.6E+08 | 8.0E+01 |
| 4.0E+05 | 7.0E+06 | 2.8E+08 | 2.6E+02 |
| 7.0E+05 | 1.0E+06 | 7.0E+07 | 1.3E+01 |
| 7.0E+05 | 4.0E+06 | 2.8E+08 | 2.6E+02 |
| 7.0E+05 | 7.0E+06 | 4.9E+08 | 5.2E+02 |
| 7.5E+05 | 7.5E+06 | 5.6E+08 | 6.0E+02 |

### Multi-peroid Portfolio Optimization
$$
\begin{aligned}
\max_{w_{\tau+1},\tau=0,\ldots,T-1} & \sum_{\tau=0}^{T-1}\hat{r}_{\tau}^\top w_{\tau+1} \\
\text{subject to} \quad & \mathbf{1}^\top(w_{\tau+1}-w_{\tau})=0, \quad \forall\tau=0,\ldots,T-1, \\
& w_{\tau}\geq0, \quad \forall\tau=0,\ldots,T-1, \\
& (\hat{w}_{\tau}^{m}\hat{\Sigma}_{\tau})(w_{\tau+1})=0, \quad \forall\tau=0,\ldots,T-1, \\
& \|{\hat{\Sigma}_{\tau+1}^{1/2}(w_{\tau+1}-w_{b})}\|\leq\gamma_{1\tau}, \quad \forall\tau=0,\ldots,T-1, \\
& -\gamma_{2\tau,i}\leq(w_{\tau+1}-w_{\tau})_{i}\leq\gamma_{2\tau,i}, \quad \forall\tau=0,\ldots,T-1,i=1,\ldots,N+1, \\
& -\gamma_{3\tau}\leq\sum_{i=1}^{N}\hat{\Sigma}_{ii}^{1/2}(w_{\tau+1}-w_{b})_{i}\leq\gamma_{3\tau}, \quad \forall\tau=0,\ldots,T-1
\end{aligned}
$$

### Multi-period Portfolio Optimization Results

#### Results for tolerance $10^{-3}$

| T | PDCP(GPU) |
|---:|---:|
| 3 | 1.9E+00 |
| 6 | 2.0E+00 |
| 48 | 7.2E+00 |
| 72 | 9.0E+00 |
| 96 | 1.1E+01 |
| 360 | 5.1E+01 |
| 720 | 7.1E+01 |
| 1440 | 4.9E+02 |
| 2160 | 5.8E+02 |
| 2880 | 9.3E+02 |
| 3600 | 1.1E+03 |

#### Results for tolerance $10^{-6}$

| T | PDCP(GPU) |
|---:|---:|
| 3 | 7.5E+00 |
| 6 | 6.4E+00 |
| 48 | 1.8E+01 |
| 72 | 1.0E+02 |
| 96 | 5.7E+01 |
| 360 | 4.2E+02 |
| 720 | 1.4E+03 |
| 1440 | 3.4E+03 |
| 2160 | 1.0E+03 |
| 2880 | 6.5E+03 |
| 3600 | 9.0E+03 |

### Citation
- Coming soon.
