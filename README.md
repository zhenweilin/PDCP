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
- Primal infeasibility: $\frac{\|(Gx - h) - \text{proj}\_{\mathcal{K}_G}(Gx - h)\|\_{\infty}}{1+\max(\|h\|\_{\infty}, \|Gx\|\_{\infty}, \|\text{proj}\_{\mathcal{K}_G}(Gx - h)\|\_{\infty})}$
- Dual infeasibility: $\frac{\max\\{\|\lambda_1-\text{proj}\_{\Lambda_1}(\lambda_1)\|\_{\infty},\|\lambda_2-\text{proj}\_{\mathcal{K}_x^*}(\lambda_2)\|\_{\infty}\\}}{1+\max\\{\|c\|\_{\infty},\|G^\top y\|\_{\infty}\\}}$, 
- Objective value accuracy: $\frac{|c^{\top}x-(y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-})|}{1+\max\{|c^{\top}x|, |y^{\top}h+l^{\top}\lambda_{1}^{+}+u^{\top}\lambda_{1}^{-}|\}}$

where $\lambda=c-G^{\top}y=[\lambda_{1}^{\top},\lambda_{2}^{\top}]^{\top},\lambda_1\in \Lambda_1 \subseteq \mathbb{R}^{n_1}, \lambda_2\in \mathbb{R}^{n_2}$.

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
|----------------|----------------|------------|
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

#### Results for tolerance $10^{-3}$ and $10^{-6}$
| m | 5.0E+02 | 1.0E+03 | 1.0E+04 | 5.0E+04 | 1.0E+05 | 1.0E+05 | 1.0E+05 | 1.2E+05 | 1.5E+05 | 1.8E+05 | 2.0E+05 | 2.2E+05 | 2.5E+05 | 2.8E+05 |
|---|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| n | 5.0E+00 | 5.0E+00 | 5.0E+01 | 5.0E+01 | 5.0E+01 | 5.0E+02 | 1.0E+03 | 1.0E+03 | 1.0E+03 | 1.0E+03 | 1.0E+03 | 1.0E+03 | 1.0E+03 | 1.0E+03 |
| nnz | 2.5E+03 | 5.0E+03 | 2.5E+05 | 1.3E+06 | 1.0E+06 | 1.0E+07 | 2.0E+07 | 2.5E+07 | 3.0E+07 | 3.5E+07 | 4.0E+07 | 4.5E+07 | 5.0E+07 | 5.5E+07 |
| PDCP(GPU) $10^{-3}$ | 5.7E-01 | 5.9E-01 | 3.5E+00 | 1.2E+01 | 1.4E+01 | 4.2E+02 | 3.8E+02 | 5.9E+02 | 6.6E+02 | 1.1E+03 | 1.0E+03 | 1.3E+03 | 1.8E+03 | 1.6E+03 |
| PDCP(GPU) $10^{-6}$ |  4.7E+02 | 4.5E+02 | 3.7E+01 | 9.1E+01 | 1.6E+02 |2.9E+03 | 3.3E+03 | 5.0E+03 | 1.0E+04  | 7.3E+03 | 1.6E+04 | 9.4E+03  | 1.2E+04 | 1.2E+04 |



### Lasso Problem 
$$
\min_{x\in \mathbb{R}^n} \frac{1}{2}\|Ax-b\|^2+\lambda \|x\|_1
$$

### Lasso Problem Results

#### Results for tolerance $10^{-3}$ and $10^{-6}$

| m | 1.0E+04 | 1.0E+04 | 1.0E+04 | 4.0E+04 | 4.0E+04 | 4.0E+04 | 7.0E+04 | 7.0E+04 | 7.0E+04 | 1.0E+05 | 1.0E+05 | 1.0E+05 | 4.0E+05 | 4.0E+05 | 4.0E+05 | 7.0E+05 | 7.0E+05 | 7.0E+05 | 7.5E+05 |
|---|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| n | 1.0E+05 | 4.0E+05 | 7.0E+05 | 1.0E+05 | 4.0E+05 | 7.0E+05 | 1.0E+05 | 4.0E+05 | 7.0E+05 | 1.0E+06 | 4.0E+06 | 7.0E+06 | 1.0E+06 | 4.0E+06 | 7.0E+06 | 1.0E+06 | 4.0E+06 | 7.0E+06 | 7.5E+06 |
| nnz | 1.0E+05 | 4.0E+05 | 7.0E+05 | 4.0E+05 | 1.6E+06 | 2.8E+06 | 7.0E+05 | 2.8E+06 | 4.9E+06 | 1.0E+07 | 4.0E+07 | 7.0E+07 | 4.0E+07 | 1.6E+08 | 2.8E+08 | 7.0E+07 | 2.8E+08 | 4.9E+08 | 5.6E+08 |
| PDCP(GPU) $10^{-3}$ | 7.1E-02 | 1.1E-01 | 1.2E-01 | 7.8E-02 | 1.7E-01 | 3.0E-01 | 1.3E-01 | 2.6E-01 | 2.9E-01 | 5.5E-01 | 9.1E+00 | 1.9E+01 | 3.3E+00 | 3.6E+01 | 1.2E+02 | 5.6E+00 | 6.2E+01 | 3.2E+02 | 4.7E+02 |
| PDCP(GPU) $10^{-6}$ | 1.1E-01 | 2.4E-01 | 5.3E-01 | 1.5E-01 | 3.6E-01 | 6.1E-01 | 2.1E-01 | 5.9E-01 | 8.4E-01 | 1.4E+00 | 2.6E+01 | 5.6E+01 | 7.5E+00 | 8.0E+01 | 2.6E+02 | 1.3E+01 | 2.6E+02 | 5.2E+02 | 6.0E+02 |

### Multi-peroid Portfolio Optimization

#### Results for tolerance $10^{-3}$ and $10^{-6}$

| T | 3.0E+00 | 6.0E+00 | 4.8E+01 | 7.2E+01 | 9.6E+01 | 3.6E+02 | 7.2E+02 | 1.4E+03 | 2.2E+03 | 2.9E+03 | 3.6E+03 |
|---|----|----|----|----|----|----|----|----|----|----|----|
| PDCP(GPU) $10^{-3}$ | 1.9E+00 | 2.0E+00 | 7.2E+00 | 9.0E+00 | 1.1E+01 | 5.1E+01 | 7.1E+01 | 4.9E+02 | 5.8E+02 | 9.3E+02 | 1.1E+03 |
| PDCP(GPU) $10^{-6}$ | 7.5E+00 | 6.4E+00 | 1.8E+01 | 1.0E+02 | 5.7E+01 | 4.2E+02 | 1.4E+03 | 3.4E+03 | 1.0E+03 | 6.5E+03 | 9.0E+03 |



### Citation
- Coming soon.
