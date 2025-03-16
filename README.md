We appreciate your interest in our work. Code accompanying the paper.

[Brain-Inspired Binaural Sound Source Localization Method Based on Liquid State Machine](https://link.springer.com/chapter/10.1007/978-981-99-8067-3_15)

`Authors: Li, Yuan and Zhao, Jingyue and Xiao, Xun and Chen, Renzhi and Wang, Lei`

We feel so sorry that we have identified a few minor issues in the original paper. Below is the corrections:

> [!CAUTION]
>
> In Section 4.2, Equation (4) \& Equation (5):
>
> "We adopt a two-dimensional flat liquid structure and Euclidean Distance to measure the the distance of neuron *m* and neuron *n* in the liquid layer, which is mathematically expressed as: 
> $$
> D_{m,n} = \sqrt{\sum_{i=1}^{2}(m_{i}-n_{i})^{2}}
> $$
> A bio-inspired rule is: when two neurons are closer, they are more likely to interconnect and enhance synaptic density. Thus the connection probability can be further formulated as:
> $$
> P_{m,n} = W * e^{-(D_{m,n}/\lambda)}
> $$
> "

These contents require correction. We did not adopt the distance-aware LSM structure generation. We directly search the connection probability between different neuron groups, as shown in **Table 1** in our paper, and in **RC_generater.py** in our source code file.

> [!NOTE]
>
> The distance-aware LSM structure generation is just one of widely-used LSM structure generation methods. It not a component of our contributions. This errata is intended to clarify ambiguities in the paper, and you can still reproduce the  experimental results using the source codes provided.

Readers who are interested in our work are welcome to contact me via email liyuan22@nudt.edu.cn.
