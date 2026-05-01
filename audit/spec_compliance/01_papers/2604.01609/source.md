Swift-SVD: Theoretical Optimality Meets Practical Efficiency

in Low-Rank LLM Compression

Ruoling Qi * 1 2 Yirui Liu * 1 Xuaner Wu 1 Xiangyu Wang 1 Ming Li 3 Chen Chen 2
Jian Chen † 4 5 Yin Chen 1 Qizhen Weng 1

Abstract

This pressure arises from two sources. First, modern LLMs
contain a massive number of parameters that must reside
in memory. Second, auto-regressive decoding maintains
cached Key–Value (KV) states (Ott et al., 2019), introducing
additional runtime memory. Unlike model parameters,
these cached representations grow with sequence length, creating
a distinct challenge as memory usage and data movement
accumulate over time. While hardware advances can
help mitigate memory bandwidth limitations (Rhee et al.,
2025), its cost and deployment complexity make algorithmic
compression approaches an attractive alternative (Shi
et al., 2024).

The deployment of Large Language Models
is constrained by the memory and bandwidth
demands of static weights and dynamic KeyValue
cache. SVD-based compression provides a

arXiv:2604.01609v1 [cs.CL] 2 Apr 2026

hardware-friendly solution to reduce these costs.
However, existing methods suffer from two key
limitations: some are suboptimal in reconstruction
error, while others are theoretically optimal
but practically inefficient. In this paper, we
propose Swift-SVD, an activation-aware, closedform
compression framework that simultaneously
guarantees theoretical optimum, practical efficiency
and numerical stability. Swift-SVD incrementally
aggregates covariance of output activations
given a batch of inputs and performs a single
eigenvalue decomposition after aggregation, enabling
training-free, fast, and optimal layer-wise
low-rank approximation. We employ effective
rank to analyze local layer-wise compressibility
and design a dynamic rank allocation strategy that
jointly accounts for local reconstruction loss and
end-to-end layer importance. Extensive experiments
across six LLMs and eight datasets demonstrate
that Swift-SVD outperforms state-of-the-art
baselines, achieving optimal compression accuracy
while delivering 3-70× speedups in end-toend
compression time. Our code will be released
upon acceptance.

Among algorithmic solutions, post-training compression

provides a practical way to reduce resource usage without
retraining large models. Existing approaches include
quantization (Zhou et al., 2024) and pruning (Guo et al.,
2025; Ashkboos et al., 2024), which lower numerical precision
and remove parameters, respectively. In contrast,
low-rank compression (Ji et al., 2025; Chang et al., 2024)
reduces the intrinsic dimensionality of linear layers and can
be viewed as a matrix approximation problem that seeks
a lower-dimensional projection minimizing reconstruction
error under a given objective. This formulation preserves
dense operators, maintains compatibility with existing hardware
and software stacks, and remains orthogonal to quantization
and pruning approaches.

Low-rank compression typically relies on SVD to obtain
optimal projections under standard matrix approximation
objectives. Early methods directly approximate projection
weights for keys and values without explicitly minimizing
reconstruction error over data-dependent activations (Chang
et al., 2024), which limits their effectiveness under real
input distributions. More recent approaches incorporate
data dependence (or activation awareness), but often require
Cholesky decomposition and/or multiple SVD computations,
introducing numerical instability (Wang et al.,
2025b; Meyer, 2023; Chen et al., 2021b) and reducing efficiency
when scaling to large datasets (Qinsi et al., 2025).
Non-uniform compression across layers has also been explored
(Wang et al., 2025a; Qinsi et al., 2025); however,
the lack of efficient layer-wise loss estimation hinders ex1.
Introduction

The deployment of large language models (LLMs) is constrained
by memory resource requirements during inference.

*Equal contribution †Work done when Jian was at University
at Buffalo 1
Institute of Artificial Intelligence (TeleAI), China
Telecom 2
Shanghai Jiao Tong University 3University of Maryland
4University at Buffalo 5Dolby Laboratories. Correspondence
to: Jian Chen <Jian.Chen@dolby.com>, Qizhen Weng
<wengqzh@chinatelecom.cn>.

Preprint. April 3, 2026.

1
Original Weight Original Weight

• We enable fast layer-wise compression using the
closed-form formulation, facilitating grid search for
optimal dynamic rank allocation beyond uniform compression.Original
KV cache


Low-rank Weight

Low-rank Weight




decompose

• We reveal a negative correlation between layer importance
and compression loss, providing insight into the
design of dynamic compression strategies.

decompose









Reduced KV cache

• We conduct extensive experiments showing that SwiftSVD
outperforms existing low-rank compression baselines
on perplexity and QA tasks while maintaining
high efficiency across diverse LLMs and datasets.

Reduced Weight

1 2 3 4
Cache  =

Figure 1. Swift-SVD for static weights and KV cache reduction.

haustive searching for optimal rank allocation. As a result,
heuristic strategies are used, which can lead to suboptimal
performance that is sometimes worse than uniform rank
allocation.

2. Preliminary

2.1. Low-Rank Compression for LLMs
As shown in Figure 1, let X ∈ R
l×m denote a batch of
input activations and W ∈ R
m×n the corresponding weight
matrix in an LLM. Low-rank compression first computes
a rank-k approximation Wk ∈ R
m×n with rank(Wk) = k,
and then decompose it as Wk = AkBk, where Ak ∈ R
m×k
and Bk ∈ R
k×n. The original weight W is subsequently
replaced with AkBk. This reduces memory usage in two
ways: 1) For model weights, the size changes from m×n to
k(m+n), yielding compression whenever k(m+n) < m×
n; 2) For KV caches, instead of caching output activations
XW ∈ R
l×n, one can cache intermediate latents XAk ∈
R
l×k
, which requires less memory whenever k < n.

To address these challenges, we propose Swift-SVD, an
activation-aware, training-free low-rank compression framework
that jointly reduces the memory footprint of static
model weights and the KV cache, as shown in Figure 1.
Swift-SVD provides a direct spectral solution that avoids
repeated SVD operations. By performing once eigenvalue
decomposition, it obtains the optimal low-rank projection
in closed form, achieving low memory overhead, high efficiency,
and strong numerical stability independent of dataset
scale or sequence length. The resulting spectral representation
also enables fast layer-wise compression loss computation,
making grid searches over dynamic rank allocations
feasible without relying on heuristics.

2.2. Minimize layer-wise compression loss

To validate the effectiveness of Swift-SVD, we conduct extensive
experiments across six LLMs and eight datasets,
evaluating end-to-end performance after compression using
perplexity and QA accuracy. Swift-SVD consistently outperforms
existing low-rank compression baselines, including
gradient-based methods (Qinsi et al., 2025). We further
demonstrate the time and memory efficiency of Swift-SVD
across different compression ratios, as well as its robustness
to dataset scale. Experiment also confirms its numerical stability
advantages over methods that rely on repeated SVD.
Additionally, we observe that layer compressibility is not
solely determined by reconstruction loss. Specifically, we
find that the end-to-end layer importance score (Shi et al.,
2025) can exhibit a negative correlation with local layerwise
compressibility, which motivates our search strategy
for dynamic rank allocation.

For low-rank model compression, a natural question is how
to define the compression loss and how to find the optimal
Wk that minimizes it. A straightforward choice is to define
the loss as the Frobenius norm ||W − Wk||F between the
original matrix W and the low-rank approximation matrix
Wk. Under this definition, the optimal Wk is obtained via
the SVD of W, where truncating to the top k singular values
and their corresponding singular vectors yields the optimal
rank-k approximation.

The above loss definition ||W − Wk||F ignores the input
activations X entirely. As a result, direct SVD truncation of
W leads to significant performance degradation in practice.
To address this limitation, recent works adopt an activationaware
(or data-dependent) loss that aims to ensure the output
of the compressed LLM closely matches the original output
Y = XW. This leads to the following formulation of the
compression problem (Chen et al., 2021b; Yuan et al., 2024;
Wang et al., 2025b; Qinsi et al., 2025; Wang et al., 2025a):

Our main contributions are summarized as follows:

• We derive an optimal solution for activation-aware lowrank
compression of LLMs that requires only a single
eigenvalue decomposition, improving efficiency, flexibility,
and numerical stability.

W∗
k = arg min
Wk∈

||XW − XWk||F (1)

ϵ
∗
k = ||XW − XW∗
k

||F (2)

2
1
2


  . ∑ .

 =

rank


∗
=
=+1


2

Original LLM

Original Weight Original Weight


∗ =
=+1
rank

2

.


.

.


∗ =

Layer 1

. . .



hook

Layer 2
Layer N

 ∑
2  . .



once
eigenvalue decomposition compute
covariance

output activations
 =



a) Optimal Activation-Aware Low-Rank Compression

Low-rank Weight

Low-rank Weight

Compressed LLM

Frobenius Loss  Layer Importance

Layer 1





. . .

     g

Compressed
LLM

. . .

Layer 2
Layer N

decompose

Compressed . . .
LLM 1

Compressed
LLM 2

Compressed
LLM g

Compressed
LLM

W

Wk

Grid Optimum

b) Dynamic Compression

Figure 2. Overview of Swift-SVD. a) Optimal Activation-Aware Low-Rank Compression: At each transformer layer, Swift-SVD hooks
the output activation Y = XW and incrementally aggregates the covariance matrix Y
T Y . A single eigenvalue decomposition of this
covariance yields the singular values Σ and right singular vectors V, from which the optimal activation-aware compression matrix W∗
k and
minimal reconstruction loss ϵ
∗
k are derived; b) Dynamic Compression: Swift-SVD generates a set of candidate dynamic rank allocation
schemes that jointly consider local layer-wise Frobenius loss ϵ
∗
and end-to-end layer importance β. A lightweight grid search is then

performed over these candidates—each model is compressed using the optimal solution in a) and evaluated on a validation set—to select
the configuration that yields the best end-to-end performance.

where k = {Wk ∈ R
m×n|rank(Wk) = k} denote the
set of m × n matrices with rank equal to k, and W∗
k
and
ϵ
∗
k
are the optimal compression matrix and the corresponding
minimal reconstruction loss for the above optimization
problem.

present the formal statement as follows:

Theorem 3.1. Given input activations X and weight matrix
W, let V and Σ denote the right singular vectors and singular
values of Y = XW, respectively. For any k < rank(Y ),
the optimal solution to the problem defined in (1) and (2) is,

W∗
k = WVkV
T
k

3. Method

, ∀k < rank(Y ) (3)

ϵ
∗
k = (Xrank(Y )
j=k+1
σ
2
j
)
1
2 , ∀k < rank(Y ) (4)

Swift-SVD is a training-free, activation-aware low-rank
compression framework for large language models that combines
theoretical optimality with practical efficiency. As
illustrated in Figure 2, our method operates in two stages:
a) Optimal Activation-Aware Low-Rank Compression; and
b) Dynamic compression. Swift-SVD can be seamlessly
applied to all types of weight matrices–such as query, key,
value, and others–in the same manner. For simplicity, we
use the generic notation W throughout our exposition.

where Vk ∈ R
n×k
consists of the top-k right singular vectors
corresponding to the k largest singular values.

Proof. Firstly, to show W∗
k
is optimal, it suffices to show
that: 1) ||XW−XWk||F ≥ ||XW−XW∗
k
||F , ∀ Wk ∈ k
and 2) W∗
k ∈ k. For 1) observe that,

XW∗
k = XWVkV
T
k
T1= Y VkV
T
k
T2= UΣ(V
T Vk)V
T
k
T3= U

Σ
Ik0
V
T
k =

U
Σk
0
 V
T
k = UkΣkV
T
k
(5)

3.1. Optimal Activation-Aware Low-Rank Compression

This subsection presents the foundation of Swift-SVD: an
activation-aware low-rank compression method that computes
the optimal weight approximation for any target rank.
We first establish a closed-form spectral solution that characterizes
the optimal compressed weights and their minimal
reconstruction loss. Then, we describe an efficient incremental
algorithm to compute this solution from input activations.

Where Ik is a k by k identity matrix. T1 holds because
by definition Y = XW; T2 holds because UΣV
T
is the

SVD of Y ; T3 holds because singular vectors form an orthogonal
basis, thus V
T
k Vk yields a k by k identify matrix.
Thus, XW∗
k
is precisely the rank-k truncated SVD
of Y . By the Eckart-Young-Mirsky theorem (Eckart &
Young, 1936), it minimizes ||Y − Y
′
||F over all Y
′ with
rank(Y
′
) ≤ k. Moreover, for any Wk ∈ k, we have
rank(XWk) ≤ rank(Wk) = k. Hence,

3.1.1. A CLOSED FORM SPECTRAL SOLUTION

Theoretically, Swift-SVD establishes a new theorem that
fully characterizes the optimal solution to the activationaware
compression problem defined in (1) and (2). We

||XW − XWk||F ≥ ||XW − XW∗
k

||F , ∀ Wk ∈ k (6)

3
Algorithm 1 Incremental SVD Algorithm

For 2) note that,

Input: Input activations X = [x1; ...; xl
] ∈ R
l×m; LLM
model weights W ∈ R
m×n
Output: singular values Σ and right singular vectors V of
Y = XW ∈ R
l×n
1: C ← zero matrix of size (n × n)
2: for t = 1, . . . , l do
3: yt ← xtW
4: C ← C + y
T
t yt
5: end for
6: V, Σ
2
, V
T ← eigen-decomposition(C)
7: return Σ, V

T4= k (7)

rank(W∗
k

) = rank(WVkV
T
k
) ≤ rank(Vk)

T4 holds since Vk has k orthogonal singular vectors. On the
other hand,

T
k
)
T5= k (8)

rank(W∗
k
) ≥ rank(XW∗
k

) = rank(UkΣkV

where T5 holds because σk > 0 under the assumption k <
rank(Y ). Thus, rank(W∗
k
) = k, which implies W∗
k ∈ k.

Secondly, as XW∗
k
is the best k-rank approximation of
Y = XW, it follows direct from the Eckart-Young-Mirsky
theorem that the minimal loss is,

ϵ
∗
k = ||XW − XW∗
k
||F

||F = (Xrank(Y )
j=k+1
σ
2
j
)
1
2
(9)

T
k

= ||XW − UkΣkV

3.1.2. DECOMPOSITION VIA INCREMENTAL
AGGREGATION

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output gate up down Layer Importance

Theorem 3.1 paves a new pathway to the numerical computation
of W∗
k
and ϵ
∗
k
, as it suffices to compute the right singular
vectors V and singular values Σ of XW. Swift-SVD first
computes the covariance matrix of Y as C = Y
T Y , and
then perform an eigenvalue decomposition of C to obtain V
and Σ. To see why this works, consider the following:

Figure 3. Layer-wise NER across distinct modules and layer importance
in Mistral-7B with dataset C4.

works propose non-uniform strategies—e.g., SVD-LLM
v2 (Wang et al., 2025a) uses layer-wise reconstruction loss,
and Dobi-SVD (Qinsi et al., 2025) relies on end-to-end training
to determine per-layer ranks—they are not optimal and
can still under-perform uniform allocation sometimes. To
address this, Swift-SVD introduces a novel dynamic rank
allocation scheme that jointly considers local layer-wise
compressibility and end-to-end compressibility.

Y
T Y
T1= (UΣV
T
)
T UΣV
T = VΣ
T U
T UΣV
T T2= VΣ
2V
T
(10)

where T1 follows from substituting the SVD of Y , and T2
holds because left singular vectors form an orthogonal basis–
i.e., U
T U = I–so Σ
T U
T UΣ = Σ2
.

This approach enables Swift-SVD to solve the problem
efficiently: it requires only a small amount of extra memory
to store the n × n covariance matrix C and a single
eigenvalue decomposition, bypassing the need for Cholesky
factorization or multiple SVDs. As a result, Swift-SVD is
not only fast (see Section 4.2 for experimental results), but
also exhibits strong numerical stability (see Section 4.3 for
experimental results).

3.2.1. COMPRESSIBILITY ANALYSIS

Swift-SVD’s design of dynamic compression is motivated
by the observation that local compressibility (i.e., how well a
layer can be approximated at low rank) and end-to-end compressibility
(i.e., how much compressing a layer degrades
overall model performance) are negatively correlated.

Algorithm 1 presents the pseudo-code of our method. Given
X containing l input activation vectors, y
T
t yt is computed

To make this intuition concrete, we employ the effective
rank (Roy & Vetterli, 2007) as a quantitative measure of
local compressibility. Specifically, given the singular values
Σ = {σi}
r
i=1 of a layer’s output Y = XW–already
computed by Algorithm 1–the effective rank is,

and the covariance matrix C is updated. After l iterations,
we obtain the full covariance matrix of Y . Finally we perform
eigen-decomposition on C to obtain the singular values
Σ and right singular vectors V of Y .

−
Xr
i=1
pi
ln pi

. (11)

erank(Σ) = exp

3.2. Dynamic Compression

where pi = σi/
Pr
j=1 σj is the normalized spectral distribution.
A lower effective rank indicates a stronger intrinsic

Uniform rank allocation is often suboptimal because different
layers exhibit heterogeneous redundancy. While recent

4
Algorithm 2 Dynamic Rank Allocation Strategy
Input: Frobenius loss ϵ, Layer importance β, Compression
ratio ρ, Scaling factor α, Retained ratio δ, Size of
original weights m, n
Output: Rank allocation k

low-rank structure and thus higher local compressibility.

For end-to-end compressibility, we adopt the standard layer
importance1 metric from prior work (Guo et al., 2025; Shi
et al., 2025; Song et al., 2025; Men et al., 2024), which
estimates the contribution of the i-th layer to the overall
LLM performance. Consequently, lower layer importance
implies higher end-to-end compressibility.

1: β ← (β − min(β))/(max(β) − min(β)) + 1
2: ¯k =
m×n
m+n × ρ ▷ Uniform rank
3: for all layer i ∈ {1 . . . L} do
4: ki ← ¯k · δ ▷ Guaranteed minimal rank
5: si ← (βi
)
α · (log(e + ϵk,i ¯ ))1−α
6: end for
7: b ← ¯k × L −
PL
i=1 ki ▷ Flexible rank pool
8: for all layer i ∈ {1 . . . L} do
9: ki ← ki + [b · (si/
PL
j=1 sj )]
10: end for
11: return k

Figure 3 visualizes the normalized effective rank and normalized
layer importance across all seven weight matrices
in a representative LLM. Strikingly, the two exhibit a clear
negative correlation: layers with high importance tend to
have lower effective rank. This validates our core motivation
and highlights the need to balance these divergent
signals in rank allocation. (Additional results are provided
in Appendix B.4.)

3.2.2. DYNAMIC COMPRESSION STRATEGY

Following above motivation, Swift-SVD first generates candidate
rank allocations based on layer importance and local
compression loss, then selects the best one via lightweight
validation: each candidate is used to compress the model using
the optimal solution in (3), and the resulting models are
evaluated on a validation set to identify the one that yields
the best end-to-end performance. Thanks to the closed-form
spectral solution, this grid search requires no retraining and
is easily parallelized.

Grid search over candidate rank allocations. Swift-SVD
uses a fixed retention ratio δ = 0.5 and 11 scaling factors
α = [0, 0.1, 0.2, ..., 1] to generate 11 candidate rank allocations.
For each candidate corresponding to αi
, the optimal
low-rank approximation of every layer is computed using
the closed-form solution in (3). The resulting compressed
models are then evaluated on a validation set, and the candidate
that yields the best end-to-end performance is selected.
(Elaborate experimental analyses of scaling factors α are
provided in Appendix A.3.)

For clarity, in this subsection we use boldface notation xi
to denote the i-th element of the vector x, and let L denote
the number of layers in an LLM.

4. Experiments and Analysis

Baselines. We evaluate the performance of Swift-SVD
against five state-of-the-art SVD-based LLM compression
methods: FWSVD (Hsu et al., 2022), ASVD (Yuan
et al., 2024), SVD-LLM (Wang et al., 2025b), SVD-LLM
v2 (Wang et al., 2025a), and Dobi-SVD (Qinsi et al., 2025).
Models and Datasets. To evaluate robustness across architectures
and scales, we experiment on a diverse suite of
LLMs: LLaMA-7B, LLaMA2-7B (Touvron et al., 2023),
OPT-6.7B (Zhang et al., 2022), Mistral-7B (Jiang et al.,
2023), and Qwen3 (4B, 8B) (Team, 2025). We assess
performance using nine standard benchmarks: WikiText2
(Merity et al., 2016), C4 (Raffel et al., 2023), and Alpaca
(Taori et al., 2023) for language modeling (perplexity),
and OpenBookQA (Mihaylov et al., 2018), WinoGrande
(Sakaguchi et al., 2019), HellaSwag (Zellers et al.,
2019), ARC-Easy (Clark et al., 2018), PIQA (Bisk et al.,
2020), and MathQA (Amini et al., 2019) for zero-shot common
sense reasoning.

Generating candidate rank allocations. The pseudocode
for dynamic rank allocation is shown in Algorithm 2.
Specifically, given a target compression ratio ρ ∈ (0, 1),
we first compute the uniform rank ¯k that would achieve ρ
under equal allocation. Then, based on a retention ratio
δ ∈ (0, 1], we assign a guaranteed minimal rank to each
layer as ki = ¯k · δ. Next, we compute a compressibility
score for each layer,

si = (βi
)
α

∗
k,i ¯ ))1−α

· (log(e + ϵ

(12)

where βi
is the layer importance of the i-th layer, min-max
normalized to [0, 1] and shifted by 1 (i.e. mapped to [1,2]),
ϵ
∗
k,i ¯ is the minimal reconstruction loss, calculated via (4),
for a rank-¯k approximation of i-th layer, e is the base of the
natural logarithm, α ∈ [0, 1] is a hyperparameter balancing
the influence of global importance and local compressibility.
We then compute the remaining rank budget b = ¯
P
k × L −
L
i=1 ki
, which forms a flexible rank pool. This pool is
distributed proportionally to the score si as ki ← ki + [b ·
(si/
PL
j=1 sj )].

Evaluation metrics. We evaluate model performance using
perplexity (PPL) (Bengio et al., 2003) and zero-shot
accuracy, computational efficiency via compression time
(in seconds), inference efficiency through memory usage

1The referenced code is available at https://github.com/
sramshetty/ShortGPT?tab=readme-ov-file.

5
Table 1. Performance comparison of LLaMA-7B on language modeling and zero-shot tasks. Baseline results are reported from their
original papers. We mark the best and second-best results. Regarding specific variants: Swift-SVD employs a uniform rank allocation
strategy, while Swift-SVD* utilizes our proposed dynamic compression. SVD-LLM(W) denotes the uniform compression of SVD-LLM
with truncation-aware whitening only, and Dobi-SVD(w/o) / (w) represent the method without / with dynamic rank allocation.

Ratio(MEM.) Method PPL (↓) Accuracy (↑)

WikiText-2 C4 ARC_e PIQA Openb. WinoG. HellaS. MathQA Avg.
1.0(12.6GB) Baseline 5.68 7.34 0.76 0.79 0.34 0.70 0.57 0.27 0.57

FWSVD 1727 1511 0.11 0.10 0.09 0.05 0.08 0.05 0.08
ASVD 11.14 15.93 0.53 0.68 0.29 0.64 0.41 0.17 0.45

0.8
(10.1 GB)

SVD-LLM(W) 7.94 15.84 0.62 0.71 0.31 0.61 0.45 0.21 0.49
Dobi-SVD(w/o) 8.87 10.91 - - - - - - -

Dobi-SVD(w) 8.54 10.01 0.63 0.72 0.30 0.62 0.46 0.20 0.49
Swift-SVD 7.91 11.42 0.64 0.73 0.26 0.68 0.47 0.23 0.50
Swift-SVD* 7.84 11.15 0.65 0.73 0.27 0.68 0.48 0.23 0.51

FWSVD 18156 12847 0.05 0.05 0.06 0.02 0.00 0.03 0.04
ASVD 1407 1109 0.11 0.13 0.08 0.09 0.08 0.08 0.10

0.6
(7.7 GB)

SVD-LLM(W) 13.73 75.42 0.33 0.63 0.25 0.55 0.40 0.12 0.38
Dobi-SVD(w/o) 14.96 24.60 - - - - - - -

Dobi-SVD(w) 13.54 23.54 0.45 0.64 0.22 0.58 0.36 0.18 0.41
Swift-SVD 13.42 23.32 0.49 0.66 0.21 0.62 0.37 0.22 0.43
Swift-SVD* 13.29 21.92 0.51 0.67 0.23 0.62 0.38 0.22 0.44

FWSVD 32194 29292 0.02 0.02 0.06 0.01 0.01 0.03 0.03
ASVD 57057 43036 0.04 0.08 0.05 0.06 0.09 0.05 0.06

0.4
(5.3 GB)

SVD-LLM(W) 66.62 471.83 0.05 0.21 0.10 0.17 0.10 0.04 0.11
Dobi-SVD(w/o) 58.02 145.41 - - - - - - -

Dobi-SVD(w) 46.18 190.62 0.25 0.51 0.14 0.48 0.24 0.15 0.30
Swift-SVD 64.16 143.74 0.29 0.56 0.16 0.52 0.27 0.21 0.34
Swift-SVD* 62.32 137.77 0.30 0.57 0.16 0.53 0.28 0.21 0.34

Table 2. Cross-model compression performance under 0.8 compression
ratio. Results show PPL on WikiText-2/C4 and average
accuracy on six common sense reasoning benchmarks.

Our dynamic allocation strategy ensures consistent gains
and avoids the instability observed in Dobi-SVD. For example,
on the C4 dataset at a compression ratio of 0.4,
Dobi-SVD(w) achieves a PPL of 190.62, which is significantly
worse than the uniform counterpart Dobi-SVD(w/o)
that achieves 145.41.

Model OPT-6.7B LLAMA 2-7B MISTRAL-7B

Perplexity↓ Acc↑ Perplexity↓ Acc↑ Perplexity↓ Acc↑
Wiki2 C4 Avg. Wiki2 C4 Avg. Wiki2 C4 Avg.

Method

Performance across Different LLMs. To evaluate the
generalization capability of Swift-SVD across diverse architectures
under uniform compression, we benchmark its
performance against SVD-LLM(W) and ASVD on OPT6.7B,
LLaMA2-7B, and Mistral-7B. As presented in Table 2,
both Swift-SVD and Swift-SVD* outperform the baselines
across all three models, demonstrating superior stability and
universality across different LLM architectures.

Original 10.86 12.52 0.52 5.47 9.30 0.57 5.25 9.28 0.61
ASVD 82.04 102 0.32 10.10 24.02 0.36 13.72 23.34 0.32

SVD-LLM (W) 16.04 21.27 0.41 8.50 12.69 0.53 10.21 13.17 0.42

Swift-SVD 12.12 17.93 0.50 8.41 12.54 0.56 7.40 12.80 0.54
Swift-SVD* 11.65 13.68 0.51 8.27 12.35 0.56 6.63 11.08 0.55

(in GB) and throughput (in tokens/second), and numerical
stability by reconstruction loss, computed from (2).

Cross-dataset Generalization. To validate the effectiveness
of the activation-aware mechanism in Swift-SVD, we
conducted a cross-domain evaluation. We utilized the C4
dataset for calibration and evaluated the compressed model
across three distinct domains.

4.1. Performance Analysis

Compression with Different Methods. First, we benchmark
Swift-SVD against state-of-the-art SVD-based compression
methods across varying compression ratios. To
ensure a fair comparison, all evaluations are conducted under
identical experimental protocols. We evaluate performance
using PPL and zero-shot accuracy on a diverse set
of benchmarks. Evidently in Table 1, Swift-SVD achieves
a comprehensive improvement over baselines, securing the
highest average accuracy across all compression levels while
yielding the lowest PPL in the majority of cases. This robustness
is particularly pronounced under aggressive compression,
where our method maintains high performance.

As illustrated in Table 3, significant PPL degradation is
observed when the calibrated model with C4 is applied to
WikiText-2 or Alpaca, demonstrating that Swift-SVD is
acutely activation-aware.

Impact of Aware Samples Size. We investigate the sensitivity
of Swift-SVD to the calibration sample size N in
Figure 4. The results demonstrate a clear trend: performance
improves rapidly in the low-sample regime but shows diminishing
returns as N increases further. While larger samples
continue to yield marginal gains, we adopt the standard set6
Table 3. Cross-domain PPL of LLaMA-7B. Original uses the
evaluation set for calibration; PPL uses C4-only.

Table 4. End-to-end compression latency (in seconds) evaluated
on the C4 dataset. We report the total time required to compress
the complete model.

nsamples = 256 nsamples = 2048
Ratio Datasets Original PPL Ratio Datasets Original PPL

Samples Method ρ = 0.8 ρ = 0.6 ρ = 0.4 Total (s) Speedup

Dobi-SVD(w/o) 960 650 373 1,983 1.0×
SVD-LLM (W) 542 489 503 1,534 1.3×
Swift-SVD 342 146 133 621 3.2×

0.8
C4 11.34 11.34
WikiText 2 7.86 11.33 WikiText 2 7.81 11.31
Alpaca 8.49 10.51 Alpaca 7.84 10.38

C4 11.42 11.42

16

0.8

Dobi-SVD(w/o) 3,823 2,551 1,508 7,882 1.0×
SVD-LLM (W) 555 530 565 1,650 4.8×
Swift-SVD 346 158 150 654 12.1×

0.6
C4 22.37 22.37
WikiText 2 13.42 37.00 WikiText 2 13.37 35.02
Alpaca 12.31 16.40 Alpaca 9.90 16.82

C4 23.17 23.17

64

0.6

Dobi-SVD(w/o) 15,269 10,468 5,966 31,703 1.0×
SVD-LLM (W) 757 734 722 2,213 14.3×
Swift-SVD 453 152 148 753 42.1×

0.4
C4 136.21 136.21
WikiText 2 64.16 285.87 WikiText 2 63.01 267.98
Alpaca 33.97 78.27 Alpaca 24.18 77.48

C4 137.01 137.01

256

0.4

Dobi-SVD(w/o) 30,862 20,984 11,795 63,641 1.0×
SVD-LLM (W) 1,080 1,069 1,063 3,212 19.8×
Swift-SVD 540 157 130 827 76.9×

512

Ratio 0.8 Ratio 0.6 Ratio 0.4

16
32
64
128
256
512
1k2k4k160k 320k
0.25
0.30
0.35
0.40
0.45
0.50
0.55
Avg. Accuracy 0.45
0.50 0.51
0.36
0.43 0.43
0.31
0.34 0.35
150
200193.5
143.7 136.2
16
32
64
128
256
512
1k2k4k160k 320k
0
20
40
60
14.2 11.4 11.2
38.1
23.3 21.1
PPL

Peak Memory (GB)
Weight Memory (GB)

Throughput (PL=32)
Throughput (PL=64)

Throughput (PL=128)

125
150
175
200
225
250
Throughput (tokens/sec)
154.0
180.0
189.0
196.8
243.0
147.4
171.1
180.8187.7
232.7
135.5
156.1
165.7
171.7
214.5

1.0
0.8
0.6
0.4
0.2
1.0
0.8
0.6
0.4
0.2
1.0
0.8
0.6
0.4
0.2
Compression Ratio
0
2
4
6
8
10
12
14
16
18
20
Memory Usage (GB)
17.33
14.09
10.85
7.62
4.38
17.47
14.21
10.95
7.69
4.43
17.76
14.45
11.14
7.83
4.52

Figure 4. Impact of calibration sample size N on model performance.
We report average accuracy on zero-shot tasks (left) and
PPL on C4 (right) across three compression ratios.

ting of N = 256 to ensure a fair comparison with baselines.

Prompt Length: 32 Prompt Length: 64 Prompt Length: 128

4.2. Computational Efficiency

Figure 5. Throughput improvement and memory efficiency under
batch size of 16. The generated sequence length is 1024.

End-to-end Compression Time. We compare the endto-end
compression time of Swift-SVD against baseline
methods under a uniform compression strategy across three
compression ratios: ρ = 0.8, ρ = 0.6 and ρ = 0.4. The
results, shown in Table 4, demonstrate that Swift-SVD
achieves substantial speedups over all baselines–up to 3.8×
compared to SVD-LLM(W), and up to 76.9× compared to
Dobi-SVD(w/o). This efficiency stems from two key advantages:
First, Algorithm 1 is highly efficient—it only requires
incremental aggregation of the activation covariance matrix
followed by a single eigen-decomposition. In contrast, baseline
methods (notably Dobi-SVD) perform a full SVD for
each input sample, which is computationally prohibitive;
Second, Swift-SVD computes the entire optimal solution
spectrum, as shown in (3) in one pass. Consequently, for
any subsequent compression ratio (e.g., ρ = 0.6 or ρ = 0.4),
no re-compression is needed. As a result, the end-to-end
compression time collapses to the time required to load the
compressed weights into memory.

more, we analyze the scaling behavior of the total model
size, comprising both compressed weights and reduced KV
cache overhead, across various compression ratios. Our
experimental results shown in Figure 5 demonstrate that as
the compression ratio decreases, our method significantly
enhances inference efficiency by boosting throughput and
alleviating HBM pressure.

4.3. Numerical Stability

To evaluate the numerical stability of Swift-SVD and baseline
methods, we generate random matrices of varying sizes
to simulate input activations X and weights W across a
range of dimensions. With a fixed compression ratio of 0.6,
we compare the reconstruction loss of each method against
the theoretical minimum loss. Although both SVD-LLM
and Dobi-SVD are designed to be theoretically optimal,
they all suffer from numerical instabilities, resulting in reconstruction
losses consistently higher than the theoretical
minimum. In contrast, Swift-SVD achieves near-perfect
alignment with the theoretical optimum across all scales.
These results confirm that Swift-SVD provides a more numerically
robust solution.

Inference Speedup and Memory Reduction. To quantify
the acceleration and memory reduction achieved by
Swift-SVD, we evaluate the inference throughput (tokens
per second) and peak memory footprint of LLaMA-7B on a
single NVIDIA 5090 GPU. We report performance under
various batch sizes and sequence lengths (see B.3). Further7
Table 5. Reconstruction loss and absolute error for randomly
generated matrices of varying shapes under ratio of 0.6 (FP32).

2024), or to maximize rank utilization for enhanced modeling
capacity (Bhojanapalli et al., 2020; Boix-Adsera et al.,
2023). With the growing use of LLMs, research has turned
to their inherent low-rank properties. LoRA (Hu et al.,
2022) leverages this structure during fine-tuning, showing
that many weight updates lie in low-dimensional subspaces.
Loki (Singhania et al., 2024) examined the key representations
in attention layers and found that they often reside in
lower-dimensional subspaces across models and datasets,
which can be used for efficient sparse attention. These directions
also motivated growing efforts on compression (Shi
et al., 2024) to address the deployment bottleneck in reading
and storing the model and KV cache (Yu et al., 2022).

Input × Weight [128 × 128]2
[1024 × 1024]2
Minimum 126.2506 841.1812

SVD-LLM 126.7102 (+0.4596) 854.4129 (+13.2317)
Dobi-SVD 128.5441 (+2.2935) 874.7711 (+33.5899)
Swift-SVD 126.2506 (+0.0000) 841.1812 (+0.0000)
Input × Weight [2048 × 2048]2
[4096 × 4096]2
Minimum 1657.4801 3308.6428

SVD-LLM 1686.1804 (+28.7003) 3365.6499 (+57.0071)
Dobi-SVD 1724.2129 (+66.7328) 3442.5242 (+133.8814)
Swift-SVD 1657.4801 (+0.0000) 3308.6428 (+0.0000)

Table 6. PPL (↓) of compressed LLMs of different allocation
strategies across various models under the ratio of 0.8 on C4.

Low-Rank Model Compression. Recent studies explore
low-rank joint KV cache compression to facilitate scalable
inference. MHA2MLA (Ji et al., 2025) and PALU (Chang
et al., 2024) employ SVD to reformulate Multi-Head Attention
(MHA) into Multi-head Latent Attention (MLA). While
effective, these weight-only approaches overlook the intrinsic
low-rank properties of activations; as evidenced by (Yu
& Wu, 2023), transformer weights typically exhibit higher
rank than their corresponding activations, suggesting that
activation-aware compression is more effective. In this direction,
DRONE (Chen et al., 2021b) establishes a closed-form
solution for intermediate representations, yet its heavy reliance
on caching full activations poses significant memory
constraints for LLMs. FWSVD (Hsu et al., 2022) incorporates
Fisher information for importance weighting, albeit at
the cost of expensive gradient computations. ASVD (Yuan
et al., 2024) attempts to normalize activation impact via
diagonal scaling but fails to reach the theoretical minimum
truncation loss. More recently, KV-CoRE (Chen et al.,
2026; Chen et al.), Dobi-SVD (Qinsi et al., 2025), and
SVD-LLM (Wang et al., 2025a) have achieved theoretical
optimum. However, KV-CoRE focuses on compressibility
analysis without practical compressibility scheme.
Dobi-SVD relies on Incremental PCA and gradient-based
training, a combination that leads to numerical instability
and renders the implementation both time-consuming and
memory-intensive. SVD-LLM utilizes Cholesky decomposition,
which necessitates that the matrices remain positivedefinite,
a condition that is challenging in diverse activation
distributions. These limitations necessitate a unified framework
that attains the theoretical optimum of truncation error
without compromising computational efficiency or numerical
stability.

Strategy LLaMA-7B LLaMA-2-7B OPT-6.7B Mistral-7B
Swift-SVD 11.42 12.54 17.93 12.80
Swift-SVD(C) 16.04 22.16 20.69 22.87
Swift-SVD(I) 14.88 17.30 18.94 22.50
Swift-SVD†
(C) 11.78 13.74 15.66 11.67
Swift-SVD†
(I) 11.73 13.27 14.74 11.48
Swift-SVD* 11.15 12.35 13.68 11.08

4.4. Ablation Study

To evaluate the contribution of each component, we compare
our proposed Swift-SVD* against several variants: (1)
Swift-SVD, a baseline that assigns a uniform rank; (2) SwiftSVD(C),
which mirrors the dynamic rank allocation strategy
of SVD-LLM v2 (Wang et al., 2025a) by allocating ranks
solely based on Frobenius loss, and Swift-SVD(I), which
allocate ranks based on layer importance without any preserved
ratio; and (3) Swift-SVD†
(C) and Swift-SVD†
(I),
which incorporate a fixed preserved ratio of δ = 0.5. As
shown in Table 6, unrestricted dynamic allocation proves
detrimental. This confirms that relying exclusively on Frobenius
loss or layer importance risks undermining the module’s
essential representation capacity by over-compressing
specific modules. In contrast, enforcing a fixed preserved ratio
(e.g., δ = 0.5) acts as a stabilizer, effectively mitigating
error propagation and reversing this degradation. Ultimately,
Swift-SVD* achieves the best performance by synergizing
this basic structural preservation with fine-grained redundancy
exploitation.

5. Related Work

Rank Analysis in Language Models. Early work has investigated
the relationship between the rank of transformer
weights or representations and model performance, seeking
either to leverage low-rank structure for efficiency (Chen
et al., 2021a; Hsu et al., 2022; Hajimolahoseini et al., 2022;
Li et al., 2023), to prevent rank collapse that limits expressivity
(Dong et al., 2021; Noci et al., 2022; Yaras et al.,

6. Conclusion

We propose Swift-SVD, a training-free activation-aware
compression framework that reconciles theoretical optimum
with practical efficiency. By formulating compression as a
closed-form eigenvalue decomposition problem, Swift-SVD

8
value ranks in llms. In NeurIPS 2025 Workshop on Evaluating
the Evolving LLM Lifecycle: Benchmarks, Emergent
Abilities, and Scaling.

eliminates the numerical instability and computational bottlenecks.
Swift-SVD further exploits layer-wise compressibility
and importance for dynamic rank allocation strategy.
Extensive experiments demonstrate that Swift-SVD delivers
3–70× end-to-end compression speedups while maintaining
state-of-the-art performance across diverse architectures.

Chen, J., Wang, Z., Qin, J., Li, M., Wang, M., Chen, C.,
Chen, Y., Weng, Q., and Liu, Y. KV-CoRE: Benchmarking
data-dependent low-rank compressibility of kv-caches
in llms. arXiv preprint arXiv:2602.05929, 2026.

References

Chen, P., Yu, H.-F., Dhillon, I., and Hsieh, C.-J. Drone:
Data-aware low-rank compression for large nlp models.
Advances in neural information processing systems, 34:
29321–29334, 2021b.

Amini, A., Gabriel, S., Lin, S., Koncel-Kedziorski, R., Choi,
Y., and Hajishirzi, H. MathQA: Towards interpretable
math word problem solving with operation-based formalisms.
In Proceedings of the 2019 Conference of the
North American Chapter of the Association for Computational
Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers), pp. 2357–2367, Minneapolis,
Minnesota, June 2019. Association for Computational
Linguistics. doi: 10.18653/v1/N19-1245. URL
https://aclanthology.org/N19-1245.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try arc, the ai2 reasoning challenge.
arXiv:1803.05457v1, 2018.

Dong, Y., Cordonnier, J.-B., and Loukas, A. Attention
is not all you need: Pure attention loses rank doubly
exponentially with depth. In International conference on
machine learning, pp. 2793–2803. PMLR, 2021.

Ashkboos, S., Croci, M. L., do Nascimento, M. G., Hoefler,
T., and Hensman, J. Slicegpt: Compress large language
models by deleting rows and columns, 2024. URL https:
//arxiv.org/abs/2401.15024.

Eckart, C. and Young, G. The approximation of one matrix
by another of lower rank. Psychometrika, 1(3):211–218,
1936.

Bengio, Y., Ducharme, R., Vincent, P., and Jauvin, C. A
neural probabilistic language model. Journal of machine
learning research, 3(Feb):1137–1155, 2003.

Guo, J., Chen, X., Tang, Y., and Wang, Y. Slimllm: Accurate
structured pruning for large language models. arXiv
preprint arXiv:2505.22689, 2025.

Bhojanapalli, S., Yun, C., Rawat, A. S., Reddi, S., and
Kumar, S. Low-rank bottleneck in multi-head attention
models. In International conference on machine learning,
pp. 864–873. PMLR, 2020.

Hajimolahoseini, H., Ahmed, W., Rezagholizadeh, M., Partovinia,
V., and Liu, Y. Strategies for applying low rank
decomposition to transformer-based models. In 36th
Conference on Neural Information Processing Systems
(NeurIPS2022), volume 6, 2022.

Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y.
Piqa: Reasoning about physical commonsense in natural
language. In Thirty-Fourth AAAI Conference on Artificial
Intelligence, 2020.

Hsu, Y.-C., Hua, T., Chang, S., Lou, Q., Shen, Y., and Jin,
H. Language model compression with weighted lowrank
factorization, 2022. URL https://arxiv.org/
abs/2207.00112.

Boix-Adsera, E., Littwin, E., Abbe, E., Bengio, S., and
Susskind, J. Transformers learn through gradual rank
increase. Advances in Neural Information Processing
Systems, 36:24519–24551, 2023.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang,
S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation
of large language models. ICLR, 1(2):3, 2022.

Chang, C.-C., Lin, W.-C., Lin, C.-Y., Chen, C.-Y., Hu, Y.-F.,
Wang, P.-S., Huang, N.-C., Ceze, L., Abdelfattah, M. S.,
and Wu, K.-C. Palu: Compressing kv-cache with lowrank
projection. arXiv preprint arXiv:2407.21118, 2024.

Ji, T., Guo, B., Wu, Y., Guo, Q., Shen, L., Chen, Z., Qiu, X.,
Zhang, Q., and Gui, T. Towards economical inference:
Enabling deepseek’s multi-head latent attention in any
transformer-based llms. arXiv preprint arXiv:2502.14837,
2025.

Chen, B., Dao, T., Winsor, E., Song, Z., Rudra, A., and Ré,
C. Scatterbrain: Unifying sparse and low-rank attention.
Advances in Neural Information Processing Systems, 34:
17413–17426, 2021a.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel,
G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-
A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix,
T., and Sayed, W. E. Mistral 7b, 2023. URL https:
//arxiv.org/abs/2310.06825.

Chen, J., Wang, Z., Qin, J., Li, M., Wang, M., Chen, C.,
Chen, Y., Weng, Q., and Liu, Y. Towards dynamic kvcache
compression: Fine-grained evaluation of key and

9
Li, Y., Yu, Y., Zhang, Q., Liang, C., He, P., Chen, W., and
Zhao, T. Losparse: Structured compression of large language
models based on low-rank and sparse approximation.
In International Conference on Machine Learning,
pp. 20336–20350. PMLR, 2023.

Shi, G., Lu, Z., Dong, X., Zhang, W., Zhang, X., Feng, Y.,
and Wu, X.-M. Understanding layer significance in llm
alignment, 2025. URL https://arxiv.org/abs/2410.
17875.

Shi, L., Zhang, H., Yao, Y., Li, Z., and Zhao, H. Keep
the cost down: A review on methods to optimize llm’s
kv-cache consumption. arXiv preprint arXiv:2407.18003,
2024.

Men, X., Xu, M., Zhang, Q., Wang, B., Lin, H., Lu, Y., Han,
X., and Chen, W. Shortgpt: Layers in large language
models are more redundant than you expect, 2024. URL
https://arxiv.org/abs/2403.03853.

Singhania, P., Singh, S., He, S., Feizi, S., and Bhatele,
A. Loki: Low-rank keys for efficient sparse attention.
Advances in Neural Information Processing Systems, 37:
16692–16723, 2024.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer
sentinel mixture models, 2016. URL https://arxiv.
org/abs/1609.07843.

Meyer, C. D. Matrix analysis and applied linear algebra.
SIAM, 2023.

Song, X., Wang, K., Li, P., Yin, L., and Liu, S. Demystifying
the roles of llm layers in retrieval, knowledge, and
reasoning, 2025. URL https://arxiv.org/abs/2510.
02091.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a
suit of armor conduct electricity? a new dataset for open
book question answering. In EMNLP, 2018.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X.,
Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford
alpaca: An instruction-following llama model. https:
//github.com/tatsu-lab/stanford_alpaca, 2023.

Noci, L., Anagnostidis, S., Biggio, L., Orvieto, A., Singh,
S. P., and Lucchi, A. Signal propagation in transformers:
Theoretical perspectives and the role of rank collapse.
Advances in Neural Information Processing Systems, 35:
27198–27211, 2022.

Team, Q. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.

Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng,
N., Grangier, D., and Auli, M. fairseq: A fast, extensible
toolkit for sequence modeling. arXiv preprint
arXiv:1904.01038, 2019.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro,
E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and
Lample, G. Llama: Open and efficient foundation language
models, 2023. URL https://arxiv.org/abs/
2302.13971.

Qinsi, W., Ke, J., Tomizuka, M., Keutzer, K., and Xu,
C. Dobi-SVD: Differentiable SVD for LLM compression
and some new perspectives. In The Thirteenth
International Conference on Learning Representations,
2025. URL https://openreview.net/forum?
id=kws76i5XB8.

Wang, X., Alam, S., Wan, Z., Shen, H., and Zhang, M. SVDLLM
v2: Optimizing singular value truncation for large
language model compression. In Chiruzzo, L., Ritter, A.,
and Wang, L. (eds.), Proceedings of the 2025 Conference
of the Nations of the Americas Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pp. 4287–4296,
Albuquerque, New Mexico, April 2025a. Association
for Computational Linguistics. ISBN 979-8-89176-189-
6. doi: 10.18653/v1/2025.naacl-long.217. URL https:
//aclanthology.org/2025.naacl-long.217/.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,
Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring
the limits of transfer learning with a unified text-totext
transformer, 2023. URL https://arxiv.org/abs/
1910.10683.

Rhee, M., Sim, J., Ahn, T., Lee, S., Yoon, D., Kim, E.,
Park, K., Joo, Y., and Kim, H. Hpu: High-bandwidth
processing unit for scalable, cost-effective llm inference
via gpu co-processing. arXiv preprint arXiv:2504.16112,
2025.

Wang, X., Zheng, Y., Wan, Z., and Zhang, M. SVD-LLM:
Truncation-aware singular value decomposition for large
language model compression. In International Conference
on Learning Representations (ICLR), 2025b. URL
https://openreview.net/forum?id=LNYIUouhdt.

Roy, O. and Vetterli, M. The effective rank: A measure of
effective dimensionality. In 2007 15th European signal
processing conference, pp. 606–610. IEEE, 2007.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.
Winogrande: An adversarial winograd schema challenge
at scale. arXiv preprint arXiv:1907.10641, 2019.

Yaras, C., Wang, P., Balzano, L., and Qu, Q. Compressible
dynamics in deep overparameterized low-rank learning &
adaptation. arXiv preprint arXiv:2406.04112, 2024.

10
Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.-
G. Orca: A distributed serving system for {TransformerBased}
generative models. In 16th USENIX Symposium
on Operating Systems Design and Implementation (OSDI
22), pp. 521–538, 2022.

Yu, H. and Wu, J. Compressing transformers: features are
low-rank, but weights are not! In Proceedings of the
AAAI Conference on Artificial Intelligence, volume 37,
pp. 11007–11015, 2023.

Yuan, Z., Shang, Y., Song, Y., Wu, Q., Yan, Y., and Sun, G.
Asvd: Activation-aware singular value decomposition for
compressing large language models, 2024. URL https:
//arxiv.org/abs/2312.05821.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,
Y. Hellaswag: Can a machine really finish your sentence?
In Proceedings of the 57th Annual Meeting of the
Association for Computational Linguistics, 2019.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M.,
Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., Mihaylov,
T., Ott, M., Shleifer, S., Shuster, K., Simig, D.,
Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer,
L. Opt: Open pre-trained transformer language models,
2022. URL https://arxiv.org/abs/2205.01068.

Zhou, Z., Ning, X., Hong, K., Fu, T., Xu, J., Li, S., Lou,
Y., Wang, L., Yuan, Z., Li, X., et al. A survey on efficient
inference for large language models. arXiv preprint
arXiv:2404.14294, 2024.

11
A. Experimental Setting Details and Analysis

In this section, we provide detailed descriptions of the experimental settings. Additionally, we conduct an analysis of the
hyperparameters used in our dynamic compression experiments.

A.1. Hardware and Software Setup

All experiments are conducted on machines with 2× NVIDIA 5090 GPUs (32GB each), though all evaluations are executed
on a single GPU without distributed computation. We use PyTorch 2.8.0 and Hugging Face Transformers 4.57.3 for model
loading, compression, and inference. All evaluations are conducted in inference mode without gradient computation.

A.2. Calibration Data

During the incremental statistic aggregation phase, we randomly select N = 256 calibration samples. For text datasets such
as WikiText-2 and C4, each sample is processed with a fixed sequence length of 2048 tokens. In contrast, for conversational
datasets like Alpaca and zero-shot common sense reasoning benchmarks (e.g., PIQA), samples are constructed by formatting
individual raw data entries according to their official prompt templates and concatenating them as discrete, individually
separated instances. This ensures that each data entry remains distinct, although it precludes a guaranteed uniform sequence
length across all calibration instances. Under such variable-length conditions, the SVD-LLM method is highly susceptible
to producing non-positive-definite matrices XT X, frequently leading to numerical instability or complete decomposition
failure. Swift-SVD effectively circumvents these limitations by utilizing a direct closed-form eigenvalue decomposition,
ensuring robust performance regardless of sequence irregularity or the presence of padding/formatting boundaries.

A.3. Hyperparameter Analysis in Dynamic Compression

In our dynamic rank allocation strategy, there are two primary hyperparameters: the scaling factor α ∈ [0, 1] and the
preserved ratio δ ∈ [0, 1]. 1) α serves as a critical hyperparameter that modulates the trade-off between reconstruction loss ϵ
and layer importance β. Specifically, α = 0 reduces the strategy to a purely loss-driven allocation, whereas α = 1 prioritizes
layer importance exclusively. 2) δ acts as a performance stabilizer. Without a preserved ratio δ = 0, unrestricted dynamic
allocation can inadvertently destroy the representational capacity of critical layers, leading to significant degradation.
Introducing a fixed baseline (e.g., δ = 0.5) effectively mitigates error propagation and reverses this trend. However,
excessively large δ values constrain the flexible budget available for reallocation, slightly reducing the optimization gains
from dynamic redundancy exploitation. Notably, when δ = 1.0, the rank allocation k matches the uniform target rank ¯k for
all modules, rendering the strategy equivalent to uniform compression.

Swift-SVD eliminates the overhead of redundant SVD operations by performing once eigenvalue decomposition of activation
statistics and caching the resulting spectral components (Σ and V) for subsequent reuse. This allows for direct truncation and
model reconstruction based on dynamic rank assignments, facilitating the rapid acquisition of the full compressed model.
Such computational efficiency provides the necessary foundation for expanding our search grid over the hyperparameter
space. By defining a comprehensive and reasonable search grid, we can effectively identify the optimal dynamic allocation
configuration that maximizes performance of compressed model within the compressed candidates.

(a) Swift-SVD* ( = 0.5)

(b) = 0

0.0 0.2 0.4 0.6 0.8 1.0
Scaling Factor
11.0
11.2
11.4
11.6
11.8
12.0
Perplexity (PPL)
Uniform ( = 1.0)
Optimal: 11.15

16

15

14

13

0.0 0.2 0.4 0.6 0.8 1.0
Scaling Factor

Figure A.1. Impact of hyperparameters α and δ on model performance for dynamic compression. Both configurations exhibit a U-shaped
PPL curve.

12
B. Additional Experiment Results

B.1. Accuracy on Specific Task

To evaluate the impact of calibration data on downstream task performance, we compare three calibration strategies across
seven zero-shot common sense reasoning. As shown in Table B.1, we define three settings: C4 (calibrated on general C4
data), Each (calibrated specifically on the validation set of each respective task), and All (calibrated on a unified mixture of
all validation datasets). We observe a clear hierarchy: Each ≳ All > C4.

Domain Specificity (Each): Achieves the highest accuracy, confirming that task-aligned calibration is optimal for preserving
task-critical features.

Robust Aggregation (All): Matches the oracle Each performance closely, proving that Swift-SVD effectively fuses distinct
feature subspaces into a single projection matrix to some extent.
Generalization Limit (C4): The notable accuracy drop illustrates the inadequacy of general-purpose data in capturing
fine-grained patterns required for specific tasks.

In summary, while domain-specific calibration is optimal, Swift-SVD demonstrates strong capacity for unified calibration,
allowing a single compressed model to serve multiple downstream tasks effectively when provided with mixed calibrations.

Table B.1. Benchmark results categorized by compression ratios and aware methods. The Original row is highlighted with gray text.

Model Ratio Aware ARC_e ARC_c PIQA Openb. WinoG. mathQA HellaS. Avg

Original Object 0.76 0.42 0.79 0.34 0.70 0.27 0.67 0.56

0.8
C4 0.64 0.33 0.73 0.26 0.67 0.23 0.47 0.48
All 0.70 0.38 0.76 0.29 0.68 0.26 0.49 0.51
Each 0.71 0.38 0.76 0.30 0.68 0.26 0.49 0.51

0.6
C4 0.49 0.25 0.66 0.21 0.62 0.22 0.37 0.40
All 0.61 0.31 0.72 0.25 0.62 0.25 0.40 0.45
Each 0.63 0.32 0.72 0.26 0.62 0.26 0.41 0.46

Llama-7B

0.4
C4 0.29 0.20 0.56 0.16 0.52 0.21 0.27 0.32
All 0.46 0.23 0.63 0.18 0.53 0.22 0.30 0.36
Each 0.53 0.25 0.64 0.20 0.56 0.25 0.31 0.39

Original 0.81 0.50 0.75 0.30 0.66 0.46 0.52 0.57

0.8
C4 0.69 0.37 0.70 0.26 0.61 0.29 0.42 0.48
All 0.71 0.41 0.71 0.27 0.61 0.39 0.43 0.50
Each 0.73 0.42 0.74 0.27 0.62 0.44 0.44 0.52

0.6
C4 0.54 0.26 0.65 0.23 0.59 0.23 0.35 0.41
All 0.58 0.32 0.69 0.24 0.60 0.32 0.36 0.45
Each 0.62 0.33 0.71 0.25 0.60 0.36 0.38 0.46

Qwen3-4B

0.4
C4 0.28 0.19 0.56 0.14 0.49 0.20 0.27 0.30
All 0.44 0.23 0.62 0.17 0.52 0.27 0.28 0.36
Each 0.49 0.23 0.64 0.19 0.54 0.31 0.30 0.39

Original 0.83 0.55 0.76 0.31 0.68 0.50 0.57 0.60

0.8
C4 0.75 0.44 0.73 0.27 0.64 0.32 0.47 0.52
All 0.76 0.45 0.74 0.29 0.65 0.45 0.47 0.54
Each 0.77 0.45 0.75 0.30 0.65 0.48 0.48 0.55

0.6
C4 0.57 0.32 0.68 0.23 0.60 0.23 0.38 0.43
All 0.68 0.37 0.71 0.26 0.61 0.39 0.39 0.49
Each 0.70 0.38 0.71 0.26 0.62 0.43 0.41 0.50

Qwen3-8B

0.4
C4 0.36 0.19 0.59 0.14 0.54 0.22 0.30 0.34
All 0.53 0.26 0.65 0.21 0.54 0.29 0.32 0.40
Each 0.56 0.28 0.66 0.22 0.55 0.36 0.33 0.42

B.2. Singular Value Distribution

As visualized in Figure B.1, the singular value distribution of both Key and Value modules exhibits a pronounced spectral
disparity. Even on a logarithmic scale, we observe a massive magnitude gap: the dominant singular values reach up to 105
,
while the median values hover significantly lower. Crucially, this extreme unevenness renders rank allocation methods based
solely on Frobenius norm minimization ineffective. Since the Frobenius loss is disproportionately sensitive to large singular
values, such methods suffer from severe numerical bias, often assigning insufficient ranks to layers with smaller spectral
norms. Furthermore, our empirical analysis reveals a high negative correlation between layer-wise compressibility and
importance. These observations motivate our proposed dynamic allocation strategy.

13
10
5

10
4

Singular Value (Log Scale)

10
4

10
3

10
3

10
2

10
2

10
1

10
1

10
0

L0
L1
L2
L3
L4
L5
L6
L7
L8
L9
L10
L11
L12
L13
L14
L15
L16
L17
L18
L19
L20
L21
L22
L23
L24
L25
L26
L27
L28
L29
L30
L31
Layer Index
10
0

L0
L1
L2
L3
L4
L5
L6
L7
L8
L9
L10
L11
L12
L13
L14
L15
L16
L17
L18
L19
L20
L21
L22
L23
L24
L25
L26
L27
L28
L29
L30
L31
Layer Index

Figure B.1. Singular value distribution of Key (left) and Value (right) modules in Llama-7B. The y-axis is presented on a logarithmic scale
to visualize the magnitude differences across layers.

B.3. Throughput

We further evaluate the system performance across varying batch sizes as shown in Figure B.2. The experimental results

demonstrate that as the compression ratio increases, our method enhances inference efficiency by simultaneously boosting

throughput and alleviating HBM pressure.

Peak Memory (GB)
Weight Memory (GB)

Throughput (PL=32)
Throughput (PL=64)

Throughput (PL=128)

Peak Memory (GB)
Weight Memory (GB)

Throughput (PL=32)
Throughput (PL=64)

Throughput (PL=128)

225
250
Throughput (tokens/sec)

225
250
Throughput (tokens/sec)

1.0
0.8
0.6
0.4
0.2
1.0
0.8
0.6
0.4
0.2
1.0
0.8
0.6
0.4
0.2
Compression Ratio
0
2
4
6
8
10
12
14
16
18
20
Memory Usage (GB)
14.94
12.12
9.29
6.47
3.64
15.02
12.18
9.34
6.51
3.67
15.16
12.30
9.43
6.57
3.71

1.0
0.8
0.6
0.4
0.2
1.0
0.8
0.6
0.4
0.2
1.0
0.8
0.6
0.4
0.2
Compression Ratio
0
2
4
6
8
10
12
14
16
18
20
Memory Usage (GB)
17.33
14.09
10.85
7.62
4.38
17.47
14.21
10.95
7.69
4.43
17.76
14.45
11.14
7.83
4.52

243.0

232.7

214.5

203.7

200

200

198.3

180.0
189.0
196.8

171.1
180.8187.7

185.9

175

175

135.5
156.1
165.7
171.7

156.8162.9169.9

151.5157.7163.6

154.0

140.6146.8
152.5

150

150

147.4

131.5

126.8

125

125

118.8

Prompt Length: 32 Prompt Length: 64 Prompt Length: 128

Prompt Length: 32 Prompt Length: 64 Prompt Length: 128

(a) Batch Size 8

(b) Batch Size 16

Figure B.2. Throughput improvement and memory efficiency under different batch sizes. The generated sequence length is 1024.

B.4. Layer-wise NER and Importance Results

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output gate up down Layer Importance

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output gate up down Layer Importance

(a) LlaMA-7B on C4

(b) LlaMA-7B on WikiText-2

Figure B.3. Layer-wise NER and Importance comparison (Part I).

B.5. Contents Generated From Compressed LLMs

14
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output fc1 fc2 Layer Importance
(c) LlaMA2-7B on C4

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output fc1 fc2 Layer Importance
(d) LlaMA2-7B on WikiText-2

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output gate up down Layer Importance
(f) Mistral-7B on WikiText-2

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output gate up down Layer Importance
(e) Mistral-7B on C4

1.0

0.5

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
0

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Layer Index
query key value output gate up down Layer Importance
(h) OPT-6.7B on WikiText-2

query key value output fc1 fc2 Layer Importance
(g) OPT-6.7B on C4

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
Layer Index
query key value output gate up down Layer Importance
(i) Qwen3-4B on C4

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
Layer Index
query key value output gate up down Layer Importance
(j) Qwen3-4B on WikiText-2

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
Layer Index
query key value output gate up down Layer Importance
(k) Qwen3-8B on C4
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
Layer Index
query key value output gate up down Layer Importance
(l) Qwen3-8B on WikiText-2
Figure B.3. Layer-wise NER and Importance comparison (Part II).
15
RATIO WikiText-2 C4

Original What color is the sky on a clear day? On a clear day, the sky appears blue due to the way sunlight scatters through the
atmosphere, a phenomenon known as Rayleigh scattering.

0.8 What color is the sky on a clear day? Blue, green or white.
The answer to this question should be blue. Because of the
atmosphere and sunlight, it’s blue. However some people
might think that it’s green because they are looking at trees
in front of the sky. Others may say that it’s white due to
clouds. But actually it’s blue. The same way as with other
colors like red or yellow, the color of the sky can change
depending on weather conditions.

What color is the sky on a clear day? The sky is blue
on a clear day. The sky appears blue because sunlight is
scattered by the atmosphere.

0.6 What color is the sky on a clear day? Blue or white. () in
what colors are to be used for the sky. The sun and moon
have their own light, but it’s not that they can change into
red or green, which appears as dark clouds when we observe
them from high altitude. In this case of being able to see
all skies, these colors would be so different with such great
brightness. For example , you could imagine that some
people might find something similar to the sky’s color. I
think perhaps the sky is blue."

What color is the sky on a clear day? The answer is
blue. But this question might be confusing because it’s not
possible to see what colors are in the atmosphere when you
look at it from Earth. So I need to find out what happens if
there’s no clouds, and then how much light would reflect
into the atmosphere? If we have some cloud cover over the
earth and only sunlight comes through, that could make an
effect of reflection by something like reflected sunlight or
light being absorbed.

0.4 What color is the sky on a clear day? It has blue colors.
The sky’s white coloring in black, with yellow hues of
purple that are gray. It s colored as red, and its colors have
green tones which are dark. This landscape area is brown
to some areas where they are not covered by the natural
vegetation but they contain many small plants such trees
and other species like plant wood. With these trees having
a forest structure from their tree and being surrounded by
various species including woodland.

What color is the sky on a clear day? Options: - A) white,
b ) green , c ) blue , d ) yellow. The answer will be blue.

Original What is the responsibility of the AI assistant? The responsibility of an AI assistant primarily lies in providing accurate,
helpful, and ethical assistance to users. This includes adhering to guidelines, maintaining privacy, avoiding harmful or
biased content, and ensuring that the information provided is reliable and up-to-date. Additionally, AI assistants should
respect user autonomy, offer transparency about their capabilities and limitations, and avoid any form of deception or
manipulation.

0.8 What is the responsibility of the AI assistant? First,
to ensure that its responses are accurate and reliable, in
accordance with human standards. The second task of AI
assistants is to be a helpful companion for humans. This
means they can help people by providing information about

What is the responsibility of the AI assistant? The AI
assistant is responsible for providing accurate information
and helping users with their questions.

their own lives, or helping them solve problems. In addition,
AI assistance can serve as an aid to enhance intelligence

through learning from data collected by other AI systems.
And finally, AI assistants may also have responsibilities
related to ethical issues such as privacy protection.

0.6 What is the responsibility of the AI assistant? "In a way
that would be useful for human beings. The AI assistant
can help humans to find out how they are not making their
own decisions, which may lead them to confusion and make
errors. That is also helpful when people try to create new
ideas or thoughts. It’s possible that these actions could have
been used by other people who want to take advantage with
this assistance from AI assistants."

What is the responsibility of the AI assistant? The ai
assistant must do all tasks required by people. In order to
provide a good solution, they need to be responsible for
what happens and can have an accurate response when it’s
needed. Answer: The Ai assistant should perform tasks that
require human attention. It has responsibility for doing tasks
that require humans’ attention so as they will get proper
answers with accuracy.

0.4 What is the responsibility of the AI assistant? The AI
assistant serves as one agent who may serve both or roles
within organizational activities, but not being affiliated to
any organization.

What is the responsibility of the AI assistant? The user
has a total number of tasks, and they can be assigned with
some tools.

Table B.2. Comparison of AI assistant responses across different datasets and compression ratios.

16