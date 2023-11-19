# The preformance prediction model

Given an input program $P$ and a target FPGA device $D$, the model searches for the design with the best performance without over-utilizing the on-chip resource. So the optimization problem is summarized as:

$$
\begin{aligned}
    \max_{x\in DF(P)} & performance(x) \\
    \text{subject to} \quad & resource_i(x) \leq b_i(D) \\
    & i = \text{Logic utils, DSP, SRAM}
\end{aligned}
$$

where $DF(P)$ includes all legal design factor choices of the program $P$, and $b_i(D)$ is the resource limit for different types of resource $i$ on-chip. $resource_i(x)$ is the resource usage of the design optimized with the design factor $x$.

## File structure

* `*.hpp`: Routines required for linear regression model training.

* `predict.py`: The linear programming model implementation.

* `build_resource_model.py`: Implementation of the linear regression model used for predicting the usage of logic utils and SRAM.

* `find_all_params.py`: A tool to help find all the parameters. 
