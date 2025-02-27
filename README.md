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
- Primal infeasibility: $\frac{\| (Gx - h) - \text{proj}_{\mathcal{K}_G}(Gx - h) \|_{\infty}}{1+\max(\|h\|_{\infty}, \|Gx\|_{\infty}, \|\text{proj}_{\mathcal{K}_G}(Gx - h) \|_{\infty})}$
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


### Citation
- Coming soon.
