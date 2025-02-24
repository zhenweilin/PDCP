# PDCP
Primal Dual Algorithm for Conic Programming
- A Julia and CUDA implementation of the Primal-Dual algorithm for conic programming

### Overview
This project implements the algorithm for solving conic optimization problems of the form:

$$\min_{x=(x_1,x_2),x_1\in \mathbb{R}^{n_1}, x_2\in \mathbb{R}^{n_2}} c^{\top} x\ \  \text{s.t.} Gx-h\in \mathcal{K}_G, l\leq x_1\leq u, x_2 \in \mathcal{K}_x,$$

where $\mathcal{K}_G$ is a closed convex cone, and $\mathcal{K}_x$ is a closed convex set.



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
- Coming soon.


### Citation
- Coming soon.
