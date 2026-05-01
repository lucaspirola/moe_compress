EoRA: Fine-tuning-free Compensation for

Compressed LLM with Eigenspace Low-Rank

Approximation

Shih-Yang Liu1, Maksim Khadkevich, Nai Chit Fung1, Charbel Sakr, Chao-Han Huck Yang,
Chien-Yi Wang, Saurav Muralidharan, Hongxu Yin, Kwang-Ting Cheng1, Jan Kautz,
Yu-Chiang Frank Wang, Pavlo Molchanov, Min-Hung Chen

Abstract: While post-training compression techniques effectively reduce the memory footprint, latency, and
power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation
and remain limited by hardware and kernel constraints that restrict supported compression formats—ultimately
reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA—a novel,
fine-tuning-free method that augments compressed LLMs with low-rank matrices, allowing users to rapidly
enhance task-specific performance and freely balance the trade-off between accuracy and computational
overhead beyond the constraints of compression formats. EoRA consistently outperforms prior fine-tuning-free
low-rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements
(e.g., 10.84% on ARC-Challenge, 6.74% on MathQA, and 11.45% on GSM8K for LLaMA3-8B compressed
to 3-bit). We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4× and reducing
memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracy
of compressed models under varying user requirements, enabling more efficient and flexible deployment of
LLMs. Code is available at https://github.com/NVlabs/EoRA.

arXiv:2410.21271v6 [cs.CL] 17 Mar 2026

Links: NVLabs Code, GPTQModel Support | NV Tech Blog

Figure 1 | An overview of our proposed EoRA, which enables swift task-specific accuracy enhancement for
compressed LLMs without fine-tuning, using only a small amount of downstream calibration data. At
inference time, a single compressed backbone is loaded, while lightweight, task-specific low-rank modules can
be dynamically toggled on and off on demand, enabling efficient and flexible deployment. EoRA with rank
128 boosts the accuracy of the LLaMA3-8B model pruned to 2:4 structured sparsity by 4.53%, 3.48%, and
11.83% on ARC-C, MathQA, and GSM8K, respectively—all achieved within minutes using just 64 calibration
samples per task.

1 affiliated with HKUST. Work done during Shih-Yang’s internship at NVIDIA Research.
© 2026 NVIDIA. All rights reserved.
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

1. Introduction

Although Large Language Models (LLMs) excel in various tasks, their deployment remains challenging due to
high inference costs. Post-training compression methods, like quantization [1, 2, 3, 4] and pruning [5, 6, 7],
aim to reduce computational demands but typically cause accuracy loss or face hardware/kernel constraints,
limiting deployment flexibility. For instance, strict hardware-supported formats, such as 2:4 sparsity on
NVIDIA GPUs or integer-only quantization kernels, prevent intermediate approaches (e.g., 2.X:4 sparsity or
arbitrary-bit quantization) that could offer a more adaptable trade-off between accuracy and latency based on
user needs.

To relax these format constraints and improve the accuracy of the compressed models on specified tasks,
we formulate a new problem, termed customized compensation: Given a compressed LLM, we attach residual
low-rank paths to it to compensate for compression errors and enhance task-specific accuracy, enabling more
flexible control over the trade-off between accuracy and compression ratio to accommodate varying user
requirements. For example, a user may wish to boost the accuracy of a 2:4 sparsity-pruned model on math
reasoning tasks, accepting a modest increase in memory usage and inference latency in return. Importantly, in
our problem setting, the weights of the compressed model are not modified during compensation. This enables
deployment of a single, general compressed backbone alongside lightweight, task-specific low-rank modules
that can be dynamically loaded as needed—allowing for efficient integration with existing multi-adapter
inference frameworks (e.g., vLLM [8]) as illustrated in Figure 1. A naive solution is to apply SVD [9, 10] for
compensation; however, this neglects calibration data and thus fails to enhance task-specific performance.
Alternatively, LoRA-based methods, such as [9, 11], require fine-tuning, limiting their applicability for
rapid task adaptation. These limitations prompt an important question: “How can we swiftly improve the
task-specific accuracy for compressed LLMs without fine-tuning?”

To tackle this research challenge, we introduce fine-tuning-free Eigenspace Low-Rank Approximation (EoRA),
a method designed to efficiently enhance the task-specific accuracy of compressed LLMs while offering users
greater flexibility in managing the trade-off between accuracy and computational overhead. EoRA operates
by projecting the compression error into the task-specific eigenspace of each layer’s input activations, followed
by applying SVD to approximate the projected error. This approach ensures that the SVD approximation
error directly aligns with the task-specific compression loss. As a fine-tuning-free method, EoRA avoids
backpropagation and completes in just a few minutes using minimal calibration data.

We validate the effectiveness of EoRA in boosting the accuracy of compressed LLMs (LLaMA2-7B/13B
and LLaMA3-8B) on language generation, commonsense reasoning, and math tasks. Our method consistently
outperforms other fine-tuning-free baselines, especially for aggressively compressed (including pruned, quantized,
and both) models (e.g., 2.65%, 3.42%, and 10.99% improvement over ZeroQuant-V2 on ARC-Challenge,
MathQA, and GSM8K when compensating 2:4 pruned LLaMA3-8B). To reduce redundant memory transfer
overhead from running low-rank compensation, we design a fused kernel that integrates low-rank and
quantization operations, achieving up to 1.4× speedup.
The summary of our contributions is as follows:

• Flexible and Task-specific Model Compensation: We propose, fine-tuning-free Eigenspace LowRank
Approximation (EoRA), a fine-tuning-free approach that improves the task-specific accuracy of
compressed LLMs in minutes using minimal calibration data, while supporting more flexible compression
ratios unconstrained by hardware or kernel-imposed format limitations.

• Eigenspace Projection: EoRA leverages calibration data to project the compression error into the

task-specific eigenspace and utilizes the corresponding eigenvalues as importance indicators, effectively
aligning the approximation error with task-specific compression loss.

• Efficient Inference: We develop a custom kernel that fuses part of the low-rank matrix multiplication

with a quantization kernel, accelerating EoRA inference by up to 1.4x. EoRA is also robust to quantization,
further minimizing the size-overhead from low-rank compensation matrices.

2
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

2. Preliminaries: Post-training Compression

Post-training compression aims to compress a well-trained model by a targeted compression ratio, utilizing
only a limited set of calibration data. The compression process is often framed as a layer-wise optimization
problem, aiming to minimize the layer-wise output difference between the original weight  ∈ R
× and the
compressed weight ^
 ∈ R
×
for each layer . Then the layer-wise model compression loss can be formed as:

|| − ^ || (1)

arg min
^

where  ∈ R
× is the input activation of layer  and  denotes the Frobenius error between the layer-wise
output. Once the compression is complete, the  for each layer will be substituted with ^

, resulting in
a smaller model size, faster inference, or both. However, their flexibility is often limited by a discrete set
of compression formats (e.g., 2:4 sparsity, 3/4-bit quantization), making it challenging to meet the diverse
accuracy/overhead requirements of different users.

To bypass the limitations of fixed compression formats and enhance the accuracy of compressed models
on user-specified tasks, we introduce a new problem, termed customized compensation: Given an already
compressed model, the objective is to add residual low-rank paths that compensate for compression errors
and enhance task-specific accuracy according to user-defined accuracy/overhead requirements. Crucially, the
compressed model’s weights remain unchanged during compensation, enabling the deployment of a single,
general compressed backbone with lightweight, task-specific low-rank modules that can be dynamically loaded
as needed, facilitating efficient integration with existing inference frameworks, as illustrated in Figure 1.

A simple approach to obtain low-rank residual paths that compensate for compression errors is to directly
apply Singular Value Decomposition (SVD) [9, 10, 12]. More specifically, this method relies on a closedform
solution by using SVD to approximate the compression error Δ =  − ^
 as Δ ≈ Σ


,
where Σ ∈ R
×
is a diagonal matrix containing the top- largest singular value sorted in descending order,
and  ∈ R
×
,  ∈ R
× are orthonormal matrices, with each column representing the singular vectors
corresponding to the singular values in Σ
. The product of  and Σ can then be treated as  = Σ with



being treated as
. Overall, the error approximation loss can be formulated as:

||Δ − || (2)

arg min

,

and SVD is applied on Δ to minimize the above equation. However, naively applying SVD to optimize error
approximation loss (Eq.2) does not ensure minimization of the layer-wise compression loss (Eq.1) and ignores
calibration data, making it ineffective for task-specific accuracy recovery. While LoRA-based methods [9, 11]
address this issue, they require fine-tuning and are less suitable for rapid adaptation. This raises a key
question: “How can we swiftly improve the task-specific accuracy for compressed LLMs without fine-tuning?”.
For simplicity, we omit the subscript , which corresponds to layer  in the following sections.

3. Method: EoRA

To tackle the challenge of improving task-specific accuracy of compressed LLMs without fine-tuning, we
introduce fine-tuning-free Eigenspace Low-Rank Approximation (EoRA)—a method that preserves the
efficiency of existing fine-tuning-free solutions while substantially improving their effectiveness in task-specific
accuracy recovery.

First, we propose projecting the compression error into the eigenspace [13] of the corresponding layer’s input
activations, ensuring a direct alignment between the error approximation loss (Eq. 2) and the overall layer-wise
model compression loss (Eq. 1). Inspired by the classical Principal Component Analysis (PCA) algorithm,
we leverage the eigenvalues of each activation channel as importance scores to indicate the importance of
each column after the eigenprojection. This allows us to allocate more low-rank representation capacity
to approximate the more critical error elements. Following PCA, we perform the eigendecomposition on
˜˜  where ˜ ∈ R
× is the average of the input activations over the task-specific calibration set. The
eigendecomposition ˜˜  = Λ
is then used to derive the eigenspace projection matrix  ∈ R
×
, whose
columns are the eigenvectors, and Λ ∈ R
×
, which is a diagonal matrix with each diagonal element being the

corresponding eigenvalues of the eigenvectors in . We then propose to project the compression error Δ into
the eigenspace with the projection matrix ′ =
√
Λ to obtain the projected error Δ′ ∈ R
× = Δ ′
.

3
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

The proposed new error approximation loss, EoRA loss, can be formulated as:

arg min
′
,′
||Δ′ −
′
′

|| (3)

where SVD is applied to approximate Δ′ as SVD(Δ′
) ≈
′Σ
′
′
, and Σ
′ ∈ R
×
contains the top-
 singular values.
′ ∈ R
× and
′ ∈ R
× are orthonormal matrices with columns representing the
corresponding singular vectors. Then the low-rank matrices ′ and ′ are then assigned as ′ =
′Σ
′ and
′ =
′
. This loss function ensures that error columns associated with larger eigenvalues are approximated
more accurately than those with smaller eigenvalues. We then multiply the low-rank approximation in the
eigenspace Δ′ with ′−1 =
√
Λ
−1

to project back to the original space, obtaining the final task-specific

. ′
is invertible because ′−1 =
√
Λ
−1

,

compression error approximation as Δ = Δ′′−1 ≈ ′′′−1

and ′′−1 =
√
Λ
√
Λ
−1

. Here, the middle term √
Λ
√
Λ
−1
simplifies to the identity matrix, and since
is an orthogonal matrix,  also yields the identity matrix. The product of ′ and ′−1
can be consolidated
into a single matrix with the same dimensions as the original ′
, ensuring no additional inference latency as
 = ′′−1
. Then, the forward pass of one linear layer of the compressed model compensated with EoRA for
the input activation  can be formulated as:

 ^ +

′ (4)

EoRA compensation is applied to each compressed linear layer, and the overall fine-tuning-free optimization
of Eq. 3 across all linear layers can be completed in just a few minutes, enabling users to rapidly enhance
the accuracy of compressed LLMs on their chosen downstream tasks using only a small amount of task-specific
calibration data—without any need for backpropagation. EoRA can also provide better initialization for
further LoRA fine-tuning, offering users the option to further improve accuracy if additional computational
resources are available. Moreover, the low-rank matrices of EoRA are robust to quantization, which can
further reduce the additional memory/inference cost. Please refer to Sec. 4.5 for more details.

Algorithm 1 Eigenspace low-rank approximation (EoRA)

Input: ˜: Average of the input activations of the current layer over the calibration set, : Full-precision Weight, ^ :
Compressed Weight, : Compensation rank
Output: ′
, : Two low-rank matrices for compensation.
1. Δ =  − ^
2. Run Eigendecompostion on ˜˜  = Λ
3. Reformulate Λ = (
√
Λ)(√
Λ ) = ′′
4. Project the compression error to eigenspace Δ′ = Δ ′
5. Run -rank SVD approximation on Δ′
, ′′ = ′Σ′
′ = SVD(Δ′
)
6. Project the approximation back to the original space  = ′′−1
7. The final forward pass of current layer becomes  ^ + ′

Mapping EoRA loss (Eq. 3) to task-specific compression loss (Eq. 1): When Eq. 1 is conditioned on
different task-specific calibration data, it also implies the compressed model’s accuracy on each corresponding
task. Therefore, the objective of task-specific low-rank compensation is to approximate Δ that minimizes
Eq. 1, using input activations  derived from the calibration data of different tasks. To achieve this, we
reformulate the compression objective for each layer as:

||  − (^ + )|| = arg min
,

arg min
,

||Δ  − || (5)

Since the Frobenius norm of a matrix is equal to the square root of its Gram matrix [14, 15], the minimization
problem can be rewritten as:

arg min
,
||Δ  − || = arg min
,
[trace((Δ − )
(Δ − )

)] 1
2

(6)

Directly applying SVD on Δ initially does not guarantee the minimization of the above equation Eq. 6.
To address this issue, EoRA projects Δ into the eigenspace before performing SVD. In the following, we
demonstrate that minimizing Eq. 3 with SVD is the same as minimizing Eq. 6.

Theorem 1. For an activation matrix , whose matrix product  has an eigendecomposition given by
 = Λ
. By projecting the compression error Δ into the eigenspace with
√
Λ as Δ′ = Δ √
Λ,
minimizing Eq. 3 via SVD becomes equivalent to minimizing Eq. 6.

4
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

Proof. First, note that  = Λ

, and by substituting this into Eq. 6, we get

[trace((Δ − )Λ

(Δ − )

)] 1
2

(7)


)] 1
2

= [trace((Δ  − )Λ(Δ  − )

Since Λ = √
Λ
√
Λ and √
Λ = √
Λ

, the above Eq. 7 can further be rewritten as:

[trace((Δ √
Λ − √
Λ)(Δ √
Λ − √
Λ)
)] 1
2 (8)

Let ′ =
√
Λ, then Eq. 8 becomes:

)(Δ ′ − ′
)

)] 1
2

[trace((Δ ′ − ′

)(Δ′ − ′
)

)] 1
2

(9)

= [trace((Δ′ − ′

= ||Δ′ − ′
||

where the square root of the Gram matrix can be transformed back to the corresponding Frobenius norm

according to [14]. By setting ′ = ′′

|| becomes ||Δ′−′′
|| . By the Eckart–Young
theorem [16], the minimization of this Frobenius norm is achieved by running SVD on Δ′
, therefore, we prove

, ||Δ′−′

that minimizing ||Δ′ − ′′
|| via SVD is equivalent to minimizing Eq. 6, where low-rank approximation
of Δ′
is SVD(Δ′
) = ′′
. Note that the above minimization is constrained to the rank of ′ and ′
.

4. Experiments

4.1. Experiments Details

We implement EoRA in PyTorch [17], utilizing the Hugging Face Transformers and Datasets framework [18].

All experiments are conducted on a single NVIDIA H100 GPU. We primarily focus on evaluating EoRA for

compensating LLaMA2-7B/13B and LLaMA3-8B models, compressed using SparseGPT [6], a widely adopted

pruning method, and GPTQ [1] for quantization. Channel-wise asymmetric quantization is applied across all

experiments, and we follow the settings from [19] to construct the calibration dataset for both SparseGPT

and GPTQ.

We compare EoRA with ZeroQuant-V2 [10] which proposes using simple SVD for optimizing Eq. 2.

Although Activation-aware Singular Value Decomposition (ASVD) [20] is designed to replace the entire

model with its low-rank decomposition rather than approximating the compression errors, its strategy of

incorporating activation distribution variance can also be adapted for error compensation using low-rank

matrices. Specifically, we scale the compression error Δ using a diagonal scaling matrix , where each
diagonal entry  is computed based on the average absolute value of the activations ˜ in the -th channel
as  =
(︁
1

∑︀
=1 |˜
 |
)︁ 1
2
. Here,  denotes the number of activation entries in the -th channel. We then

apply SVD to the scaled error Δ” = Δ  to obtain its low-rank approximation. Since  is invertible,

we can project the approximation back to the original space as Δ = Δ”

−1 ≈ ””

−1—same as

how EoRA project its compensation back to the original space. We refer to this method as Act-S in the

remainder of this paper. We also compare EoRA with a training-based method, ApiQ [21], which optimizes

low-rank matrices ( and ) using gradient-based training to minimize Eq. 6. In our comparison, we limit

the evaluation to the layer-wise variant of ApiQ, as other variants require substantially more memory or

training time. These more resource-intensive versions align more closely with PEFT methods rather than

fine-tuning-free low-rank approximation approaches. For instance, when applied at the model level, ApiQ

effectively becomes equivalent to training LoRA on top of a compressed model—shifting its focus toward

fine-tuning rather than fine-tuning-free compensation, and thus falling outside the scope of this study. Note

that the optimization time for both EoRA and Act-S is comparable, with each completing within minutes,

whereas ApiQ requires over hours to optimize.

We evaluate EoRA and the baselines on improving the task-specific accuracy of the compressed LLMs on

language generation, commonsense reasoning, and math reasoning tasks using the LM-Evaluation-Harness

framework [22]. We pick WikiText2 for the language generation task and perplexity as the evaluation metric.

For commonsense reasoning, we select ARC-Challenge (ARC-C) [23], and for math reasoning ability, we

5
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

choose MathQA [24] and GSM8K [25]. We sample 128 concatenated sentences of length 2048 from the
WikiText2 training set as the calibration set for EoRA, Act-S, and ApiQ for the language generation task. For
commonsense reasoning tasks, we sample 32 concatenated sentences of length 2048 from the ARC training set
and combine them with 32 concatenated sentences of the same length from C4 [26] to construct the calibration
set for EoRA, Act-S, and ApiQ. Similarly, for the math reasoning tasks, we sample 32 concatenated sentences
of length 2048 from the MathQA/GSM8K training set and combine them with 32 concatenated sentences
from C4 to form the calibration set for the three methods.

4.2. Main Results

4.2.1. Sparsity Error Compensation

Table 1 | Perplexity and commonsense/math reasoning results for LLaMA3-8B pruned with 2:4 sparsity using
SparseGPT, with all compensation methods evaluated at rank 128.

Model Sparsity Compensation Method Wikitext2 ↓ ARC-C ↑ MathQA ↑ GSM8K ↑

- - 6.13 50.42 40.10 36.23

- 12.32 30.11 26.43 2.12
ZeroQuant-V2 11.31 31.99 26.49 2.956
Act-S 11.32 31.74 26.73 3.26

LLaMA3-8B

2:4

ApiQ 11.08 34.21 28.77 14.55
EoRA (Ours) 11.07 34.64 29.91 13.95

To assess the effectiveness of EoRA in compensating for sparsity error, we compare EoRA with all the
baselines on LLaMA2-7B/13B and LLaMA3-8B models pruned with SparseGPT to 2:4 sparsity—the only
sparsity format that yields actual inference speedups on GPUs. Rank of all the compensation methods
is set to 128, and the results of LLaMA3-8B are presented in Table 1, while the full results, including
LLaMA2-7B/13B, are provided in Table 4 in the appendix. EoRA consistently outperforms all fine-tuning-free

baselines, achieving gains of 2.9%, 2.1%, and 10.7% over Act-S on ARC-C, MathQA, and GSM8K, respectively.
Furthermore, it surpasses ApiQ by 0.4% on ARC-C and 1.1% on MathQA, while delivering comparable
results on GSM8K—all with significantly faster optimization time (15 minutes vs. 2.5 hours). Furthermore,
EoRA proves robustness across different model sizes, continuing to outperform ZeroQuant-V2, Act-S, and
ApiQ in boosting the accuracy of 2:4 pruned LLaMA2-7B/13B across ARC-C and MathQA as shown in
Table 4. We further assess the generalizability and compatibility of EoRA with pruning methods beyond
SparseGPT. Specifically, we evaluate EoRA on LLaMA3-8B pruned to 2:4 sparsity using Wanda [7], where
EoRA continues to outperform all the fine-tuning-free baseline methods. For additional details, please refer to
Section A.4.

4.2.2. Quantization Error Compensation

We evaluate EoRA on LLaMA2-7B/13B and LLaMA3-8B models quantized with GPTQ to 4-bit and 3-bit
to assess the effectiveness of EoRA in compensating for quantization error. The ranks for all the methods
are set to 128. From Table 2, 3-bit quantization causes significant accuracy degradation, with losses up to
29.5%/17.7%/35.8% on ARC-C, MathQA, and GSM8K, respectively. By applying EoRA, we demonstrate
that the accuracy loss can be reduced to 18.7%/10.9%/24.3% on ARC-C, MathQA, and GSM8K—providing
10.8%/6.7%/11.5% improvement, outperforming all the baseline methods for compensating the quantization
error. On the other hand, although 4-bit quantization does not result in as much accuracy loss as 3-bit
quantization, applying EoRA can still generally enhance the performance of the 4-bit model, offering up to
a 2.2% and 3.14% accuracy boost on ARC-C and MathQA, respectively. Comprehensive results, including
those for LLaMA2-7B/13B, are presented in Table 5 in the appendix, where a similar trend of improvement
with EoRA is observed. We also explore the feasibility of using EoRA to improve ultra-compressed models
that combine both pruning and quantization. In this setting, EoRA continues to outperform all baselines on
ARC-C and MathQA. Further details can be found in Section A.3 of the Appendix.

6
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

Table 2 | Perplexity and commonsense/math reasoning results for LLaMA3-8B quantized to 3/4-bits using
GPTQ, with all compensation methods evaluated at rank 128.

Model W bits Compensation Method Wikitext2 ↓ ARC-C ↑ MathQA ↑ GSM8K ↑

- - 6.13 50.42 40.10 36.23

- 7.00 45.90 34.07 27.74
ZeroQuant-V2 6.80 45.24 36.51 31.23
Act-S 6.82 47.86 35.84 29.34
ApiQ 6.87 46.58 36.18 30.09
EoRA (Ours) 6.80 47.44 37.21 30.70

W4

LLaMA3-8B

- 15.64 20.90 22.37 0.45
ZeroQuant-V2 10.24 30.02 26.43 3.79
Act-S 10.19 31.28 25.42 4.09
ApiQ 10.41 30.46 26.86 10.79
EoRA (Ours) 10.06 31.74 29.11 11.90

W3

4.3. Ablation Study: Ranks and Calibration Sizes

Figure 2 | Results of applying EoRAand other baselines with rank set to {64,128,256,512} to improve
LLaMA3-8B models pruned to 2:4 sparsity by SparseGPT on (a) ARC-C/(b) MathQA/(c) GSM8K.

Since one of the advantages of using EoRA is the greater flexibility in adjusting overall model accuracy without
being constrained by specific compression formats, in this section, we investigate the influence of different
ranks on adopting EoRA. We vary the rank of EoRA in {64,128,256,512} on compensating LLaMA3-8B
pruned to 2:4 sparsity. As shown in Figure 2, EoRA consistently outperforms the two fine-tuning-free baselines
(ZeroQuant-V2 and Act-S) across all tested ranks, with the performance gap becoming more prominent at
higher ranks. For instance, on GSM8K, EoRA achieves improvements of 7.43%, 10.69%, 11.9%, and 14.62%
at ranks 64, 128, 256, and 512, respectively. In contrast, the gains on ARC-C remain relatively stable across
ranks, ranging between 2% and 4%. Additionally, EoRA begins to outperform ApiQ on GSM8K at higher
ranks, with improvements of 1.21% and 2.51% observed at ranks 256 and 512, respectively. These experiments
prove that EoRA is robust across different rank settings, offering users a more flexible option upon existing
compression configurations to effectively balance the trade-off between inference overhead and model accuracy.
A similar trend is observed in the results for LLaMA2-7B/13B shown in Table 10 in the appendix.

We also compare the influence of different calibration sizes on EoRA . We vary the calibration size in
{16,32,64}, and compare them on recovering the accuracy of LLaMA3-8B quantized to 3/4-bit and pruned to
2:4 sparsity. Overall, we find that EoRA demonstrates strong robustness and maintains competitive accuracy
even with limited calibration data, as shown in Table 8. Notably, using as few as 32 calibration samples to

7
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

compensate for a 2:4 pruned model can even yield better accuracy improvements than using 64 samples.

4.4. EoRA as LoRA initialization for Fine-tuning Compressed Models

Table 3 | Finetune the 4-bit compressed LLaMA3-8B models with different initialization of the low-rank
matrices for Commonsense/Math reasoning tasks.

Model Compression Method Compression Setting LoRA initialization ARC-C ↑ MathQA ↑

Full-precision - w/o fine-tuning 50.42 40.10

Standard 56.39 53.56

w/o fine-tuning 45.90 34.07
QLoRA 54.09 51.42
LoftQ 54.52 53.96
EoRA (Ours) 55.46 56.04

LLaMA3-8B

GPTQ W4

We show that, with additional computational resources, users can leverage the low-rank matrices from EoRA as
initialization for LoRA fine-tuning, enabling further accuracy improvements for compressed models. We follow
the conventional LoRA fine-tuning framework, which keeps the compressed model frozen and only tunes the
low-rank residual components during fine-tuning. We conduct experiments on compressed LLaMA3-8B models
with {2:4 sparsity, 4-bit, 3-bit} compression. The rank of LoRA is set to 128 and is applied to every linear
layer, initialized using EoRA, SVD following LoftQ [9], and standard initialization following QLoRA [11].
Fine-tuning is performed on the ARC training set for evaluating ARC-C, and on the MathQA training set for
the math reasoning task. We fine-tune the models for 3 epochs with a batch size of 64, a learning rate of 1e-5,
and a cosine learning rate scheduler. As shown in Table 3, initializing with EoRA substantially enhances
the accuracy of compressed models, surpassing both QLoRA and LoftQ when fine-tuning 4-bit quantized
LLaMA3-8B, and achieving accuracy on par with standard full-precision fine-tuning. We also observed that
the improvements over QLoRA and LoftQ are more pronounced on 3-bit quantized and 2:4 pruned models,
aligning with our earlier finding that EoRA is more effective when the compression error is more substantial,
as shown in Table 11.

4.5. Kernel Optimization, Inference Speed Evaluation and Memory Overhead of EoRA

(a) (b)

Figure 3 | (a) We propose fusing the multiplication of  with the weight quantization kernel to minimize data
movement overhead and substantially improve the inference latency. (b) The model size and ARC-C accuracy
of EoRA with rank 128/512, quantized to 4-bit for compensating LLaMA3-8B quantized to 4/3-bit or pruned
to 2:4 sparsity.

8
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

While theoretically, compensating a compressed model with low-rank residual paths introduces minimal
computational overhead, in practice, it leads to a noticeable increase in latency. This is primarily because
input and output must transfer between L2 cache and DRAM twice as often compared to that without a
low-rank residual path, shifting the inference process from being computation-bound to memory-bound. This
phenomenon is also discussed in [12]. To address this, we propose fusing the low-bit weight quantization
kernel with the matrix multiplication of , which shares the same output. By doing so, the shared output
no longer needs to be offloaded and reloaded to the L2 cache, effectively reducing data transfer overhead as
illustrated in Figure 3 (a). Implementation details of our kernel can be found in Section A.6. As shown in
Table 9, our custom EoRA kernel substantially accelerates inference compared to using native PyTorch for the
low-rank residual path on top of the low-bit quantized kernel, achieving a speedup of up to 1.4x over FP16
with EoRA of rank 128 at 3-bit quantization. In contrast, without the EoRA kernel, the initial 1.7x speedup
provided by the 3-bit quantized kernel drops to 1.1x. Similarly, under 4-bit quantization, the EoRA kernel
delivers an extra 0.3x speedup compared to setups without the EoRA kernel.

Finally, EoRA can also be quantized to further reduce the additional cost of residual low-rank compensation
paths. In this section, we quantize EoRA of rank {128, 512} to 4/3-bit on compensating three types of
compressed LLaMA3-8B models (2:4 pruned, 4-bit quantized, and 3-bit quantized). The complete results
are provided in Table 13 in appendix, while the results for LLaMA3-8B are illustrated in Figure 3 (b). As
shown in the figure, EoRA is robust to quantization, which means that when EoRA is quantized, the accuracy

drop from full-precision EoRA is insignificant while the model size is significantly reduced. For example,
when a 512-rank EoRA is quantized from 16-bits to 4-bit on 2:4 pruned LLaMA3-8B, the accuracy drops are
only 0.43% on ARC-C while the total model size reduces by 16.49%. Additionally, compared to the original

uncompensated 2:4 pruned model, quantizing EoRA of rank 128/512 improves accuracy by 4.4%/11.4%
with a total model size increase of just 2%/7%. For 3-bit quantized LLaMA3-8B compensated with a 4-bit
quantized EoRA of rank 128/512 achieves 10.6%/19.1% accuracy improvements, with a corresponding model
size increase of only 3%/14%. Interestingly, we also observe that quantizing EoRA does not always result in
accuracy loss; in some cases, it even slightly improves accuracy, potentially due to quantization acting as a
form of regularization, as discussed in OFQ [27]. Generally, we recommend users quantize EoRA to 4-bit, as
this significantly reduces inference latency and model size with kernel support, without causing any noticeable
drop in accuracy.

5. Related Works

Post-training LLM Compression: As LLMs scale, reducing their size is essential for efficient deployment.
Traditional compression-aware training methods are impractical due to the need for full datasets and heavy
retraining. Post-training compression methods like quantization and pruning have gained popularity as they

require only minimal calibration data and no retraining. PTQ reduces model size by lowering bitwidths [1, 4],
while PTP removes less important weights to reduce computation [6, 7]. Our method, EoRA, is compatible
with all such compression techniques as it operates independently of the base method used.

Low-Rank Decomposition: Low-rank decomposition methods [20, 28, 15, 29, 30, 31, 32, 33, 34] compress
models by replacing weights with low-rank matrices, reducing both latency and size without special kernel
support. However, they are less widely adopted due to weaker accuracy-compression trade-offs. While
FWSVD [29] is designed primarily as a compression method based on low-rank decomposition of model
weights, EoRA instead leverages low-rank modules specifically to compensate for compression errors. Unlike
FWSVD, EoRA provides a theoretical guarantee of minimizing layer-wise compression loss, as demonstrated
in our derivation in Section 3. Using gradient-based information, as required by FWSVD, can be prohibitively
expensive for LLMs, as noted in ASVD [20]. SVD-LLM [15] is conceptually close to our work, which also tries
to align the SVD compression error with the layer-wise compression loss. It relies on the matrix product of
the activation being positive-definite, a condition often unmet in practice. Enforcing this condition typically
requires additional modifications, which introduce noise into the approximation. In contrast, EoRA employs
eigendecomposition, which only requires the matrix product of the activation to be symmetric—a property
that naturally holds—avoiding such issues. Furthermore, although EoRA also utilizes SVD-based low-rank
decomposition, its core objective is fundamentally different. Whereas prior methods aim to replace pre-trained
weight matrices with low-rank approximations to reduce model size and inference cost, EoRA focuses on

9
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

approximating the compression error itself. This allows for improved accuracy recovery in compressed
LLMs and provides greater flexibility in balancing accuracy and computational overhead by overcoming
the constraints of fixed compression formats. Please refer to Section A.13 in Appendix for more detailed
comparison.

6. Conclusion

In this work, we present EoRA, a novel fine-tuning-free approach that rapidly boosts the task-specific accuracy
of compressed LLMs using minimal calibration data, while offering greater flexibility by relaxing compression
format constraints. By projecting compression errors into the task-specific eigenspace of activations, EoRA uses
eigenvalues to guide SVD, aligning approximation error with layer-wise compression loss—without any gradientbased
training. EoRA achieves strong results across language, commonsense, and math reasoning tasks,
outperforming prior low-rank methods [10, 20, 21]. Its fine-tuning-free design allows quick adaptation to
various accuracy-latency trade-offs, and it remains robust under quantization, reducing memory overhead.
Additionally, it can serve as a strong initialization for LoRA fine-tuning. Overall, EoRA is a scalable, efficient
solution for improving compressed LLMs across diverse deployment settings, with potential extensions to new
architectures and modalities.

References

[1] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for
generative pre-trained transformers. In International Conference on Learning Representations, 2023.

[2] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu
Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression
and acceleration. In Machine Learning and Systems, 2024.

[3] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas
Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: Llm quantization with learned rotations. In
International Conference on Learning Representations, 2025.

[4] Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better
llm quantization with hadamard incoherence and lattice codebooks. In International Conference on Machine
Learning, 2024.

[5] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models.
In Neural Information Processing Systems, 2023.

[6] Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In
International Conference on Machine Learning, 2023.

[7] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large
language models. In International Conference on Learning Representations, 2024.

[8] MultiLoRA Inference. https://docs.vllm.ai/en/latest/examples/offline_inference/
multilora_inference/, 2025.

[9] Yixiao Li, Yifan Yu, Chen Liang, Nikos Karampatziakis, Pengcheng He, Weizhu Chen, and Tuo Zhao. Loftq: Lorafine-tuning-aware
quantization for large language models. In International Conference on Learning Representations,
2024.

[10] Zhewei Yao, Xiaoxia Wu, Cheng Li, Stephen Youn, and Yuxiong He. Exploring post-training quantization in llms
from comprehensive study to low rank compensation. In AAAI Conference on Artificial Intelligence, 2024.

[11] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized
llms. In Neural Information Processing Systems, 2023.

[12] Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu,
and Song Han. Svdqunat: Absorbing outliers by low-rank components for 4-bit diffusion models. In International
Conference on Learning Representations, 2025.

[13] Gilbert W Stewart. Matrix Algorithms: Volume II: Eigensystems. SIAM, 2001.

10
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

[14] Ji-Guang Sun. Perturbation bounds for the cholesky and qr factorizations. BIT Numerical Mathematics, 1991.

[15] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. Svd-llm: Truncation-aware singular value decomposition for
large language model compression. In International Conference on Learning Representations, 2025.

[16] Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika,
1(3):211–218, 1936.

[17] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin,
Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. In Neural Information
Processing Systems Workshop, 2017.

[18] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac,
Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language
processing. arXiv preprint arXiv:1910.03771, 2019.

[19] Wei Huang, Xudong Ma, Haotong Qin, Xingyu Zheng, Chengtao Lv, Hong Chen, Jie Luo, Xiaojuan Qi, Xianglong
Liu, and Michele Magno. How good are low-bit quantized llama3 models? an empirical study. arXiv preprint
arXiv:2404.14047, 2024.

[20] Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, and Guangyu Sun. Asvd: Activation-aware
singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821, 2023.

[21] Baohao Liao, Christian Herold, Shahram Khadivi, and Christof Monz. Apiq: Finetuning of 2-bit quantized large
language model. In Empirical Methods in Natural Language Processing, 2024.

[22] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason
Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang,
Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 2024.

[23] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord.
Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457,
2018.

[24] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa:
Towards interpretable math word problem solving with operation-based formalisms. In North American Chapter
of the Association for Computational Linguistics, 2019.

[25] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert,
Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv
preprint arXiv:2110.14168, 2021.

[26] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li,
and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of
machine learning research, 2020.

[27] Shih-Yang Liu, Zechun Liu, and Kwang-Ting Cheng. Oscillation-free quantization for low-bit vision transformers.
In International Conference on Machine Learning, 2023.

[28] Charbel Sakr and Brucek Khailany. Espace: Dimensionality reduction of activations for model compression. In
Neural Information Processing Systems, 2024.

[29] Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. Language model compression
with weighted low-rank factorization. In International Conference on Learning Representations, 2022.

[30] Mohammad Mozaffari, Amir Yazdanbakhsh, and Maryam Mehri Dehnavi. Slim: One-shot quantization and
sparsity with low-rank approximation for llm weight compression. In International Conference on Machine
Learning, 2025.

[31] Stephen Zhang and Vardan Papyan. Oats: Outlier-aware pruning through sparse and low rank decomposition. In
International Conference on Learning Representations, 2025.

[32] Cheng Zhang, Jianyi Cheng, George A Constantinides, and Yiren Zhao. Lqer: Low-rank quantization error
reconstruction for llms. In International Conference on Machine Learning, 2024.

11
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

[33] Cheng Zhang, Jeffrey TH Wong, Can Xiao, George A Constantinides, and Yiren Zhao. Qera: an analytical
framework for quantization error reconstruction. In International Conference on Learning Representations, 2025.

[34] Rajarshi Saha, Naomi Sagan, Varun Srivastava, Andrea Goldsmith, and Mert Pilanci. Compressing large language
models using low rank and low precision decomposition. In Neural Information Processing Systems, 2024.

[35] Meyer Scetbon and James Hensman. Low-rank correction for quantized llms. arXiv preprint arXiv:2412.07902,
2024.

[36] Yelysei Bondarenko, Riccardo Del Chiaro, and Markus Nagel. Low rank quantization-aware training for llms. In
International Conference on Machine Learning Workshop, 2024.

[37] Mohammad Mozaffari, Amir Yazdanbakhsh, Zhao Zhang, and Maryam Mehri Dehnavi. Slope: Double-pruned
sparse plus lazy low-rank adapter pretraining of llms. In International Conference on Learning Representations,
2025.

[38] Geonho Lee, Janghwan Lee, Sukjin Hong, Minsoo Kim, Euijai Ahn, Du-Seong Chang, and Jungwook Choi. Rilq:
Rank-insensitive lora-based quantization error compensation for boosting 2-bit large language model accuracy. In
AAAI Conference on Artificial Intelligence, 2025.

12
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation
A. Appendix
A.1. Sparsity Error Compensation
Table 4 | Perplexity and Commonsense/Math reasoning results of LLaMA2/3 pruned by SparseGPT to 2:4
sparsity, with low-rank compensation of rank 128.
Model Sparsity Compensation Method Wikitext2 ↓ ARC-C ↑ MathQA ↑ GSM8K ↑

- - 6.13 50.42 40.10 36.23

- 12.32 30.11 26.43 2.12
ZeroQuant-V2 11.31 31.99 26.49 2.96
Act-S 11.32 31.74 26.73 3.26
ApiQ 11.08 34.21 28.77 14.55
EoRA (Ours) 11.07 34.64 29.91 13.95

LLaMA3-8B

2:4

- - 5.47 39.84 27.67 14.85

- 8.77 30.11 24.65 1.66
ZeroQuant-V2 8.15 30.54 24.89 1.97
Act-S 8.22 30.20 25.09 2.73
ApiQ 8.03 32.67 26.36 7.58
EoRA (Ours) 7.97 32.67 25.59 6.22

LLaMA2-7B

2:4

- - 4.88 45.56 29.91 21.37

2:4
- 7.10 34.30 25.92 2.65
ZeroQuant-V2 6.82 33.61 25.12 3.56
Act-S 6.92 34.12 25.69 4.09
ApiQ 6.80 36.68 27.16 12.13
EoRA (Ours) 6.75 37.54 27.53 10.91
Table 4 reports more detailed comparison for sparsity error compensation.
A.2. Quantization Error Compensation
Table 5 reports more detailed comparison for quantization error compensation. It is worth noting that
EoRA-enhanced models outperform smaller models quantized at higher precision, while high-precision models
usually outperform low-precision models. For example on GSM8K (Table 5), 3-bit LLaMA2-13B (4.62)
performs worse than 4-bit LLaMA2-7B (9.93), but EoRA-enhanced 3-bit LLaMA2-13B (15.08) outperforms
4-bit LLaMA2-7B. This highlights EoRA’s advantage in terms of greater flexibility: rather than shrinking the
model architecture, it enables more effective compression of larger models while preserving higher accuracy.

LLaMA2-13B

13
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

Table 5 | Perplexity and Commonsense/Math reasoning results of LLaMA2/3 quantized by GPTQ with
different bit-width, with low-rank compensation of rank 128.

Model W bits Compensation Method Wikitext2 ↓ ARC-C ↑ MathQA ↑ GSM8K ↑

- - 6.13 50.42 40.10 36.23

- 7.00 45.90 34.07 27.74
ZeroQuant-V2 6.80 45.24 36.51 31.23
Act-S 6.82 47.86 35.84 29.34
ApiQ 6.87 46.58 36.18 30.09
EoRA (Ours) 6.80 47.44 37.21 30.70

W4

LLaMA3-8B

- 15.64 20.90 22.37 0.45
ZeroQuant-V2 10.24 30.02 26.43 3.79
Act-S 10.19 31.28 25.42 4.09
ApiQ 10.41 30.46 26.86 10.79
EoRA (Ours) 10.06 31.74 29.11 11.90

W3

- - 5.47 39.84 27.67 14.85

- 5.75 38.13 26.73 9.93
ZeroQuant-V2 5.68 37.62 27.06 10.15
Act-S 5.68 39.84 27.50 9.86
ApiQ 5.68 39.59 27.00 11.22
EoRA (Ours) 5.68 38.05 27.13 11.45

W4

LLaMA2-7B

- 7.76 31.65 23.50 0.38
ZeroQuant-V2 6.84 34.47 23.90 2.04
Act-S 6.86 32.67 25.02 2.57
ApiQ 6.86 33.70 26.06 7.13
EoRA (Ours) 6.84 35.83 25.79 7.50

W3

- - 4.88 45.56 29.91 21.37

- 5.06 44.28 29.10 21.00
ZeroQuant-V2 5.03 44.19 28.97 19.48
Act-S 5.04 43.60 29.48 18.49
ApiQ 5.04 42.83 29.64 21.45
EoRA (Ours) 5.03 44.53 28.90 22.36

W4

LLaMA2-13B

- 5.99 37.28 26.26 4.62
ZeroQuant-V2 5.76 37.54 26.83 9.93
Act-S 5.81 38.90 26.26 9.17
ApiQ 5.81 39.67 27.47 14.32
EoRA (Ours) 5.75 39.50 27.20 15.08

W3

14
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.3. Sparsity & Quantization Error Compensation

Table 6 | Perplexity and Commonsense/Math reasoning results of LLaMA2/3 models pruned to 2:4 using
SparseGPT and quantized to 4-bit with GPTQ, with compensation rank set to 128.

Model Sparsity W bits Compensation Method Wikitext2 ↓ ARC-C ↑ MathQA ↑ GSM8K ↑

- - - 6.13 50.42 40.10 36.23

- 86.15 18.34 19.89 0.00
ZeroQuant-V2 12.84 29.35 26.86 1.59
Act-S 12.99 27.90 25.59 1.90
ApiQ 12.77 30.71 28.74 11.06
EoRA (Ours) 12.60 31.22 29.58 10.16

LLaMA3-8B

2:4 W4

- - - 5.47 39.84 27.67 14.85

- 9.37 29.43 23.88 0.99
ZeroQuant-V2 8.42 29.94 24.42 1.67
Act-S 8.24 28.92 24.05 1.97
ApiQ 8.03 30.63 24.12 7.05
EoRA (Ours) 8.24 31.14 25.39 4.93

LLaMA2-7B

2:4 W4

- - - 4.88 45.56 29.91 21.37

- 7.27 33.10 24.75 2.20
ZeroQuant-V2 6.98 33.27 25.29 2.65
Act-S 6.92 34.64 26.09 2.81
ApiQ 6.80 36.17 26.96 12.59
EoRA (Ours) 6.89 35.06 27.06 9.86

LLaMA2-13B

2:4 W4

Here, we examine the feasibility of applying EoRA to compensate for ultra-compressed models that
undergo both pruning and quantization, as shown in Table 6. Specifically, we prune LLaMA2-7B/13B and
LLaMA3-8B to 2:4 sparsity and quantize them to 4-bit. We set the ranks of both EoRA and SVD to 128 to
compensate for the pruning and quantization errors. Similarly to our previous findings, LLaMA3-8B is the
least resilient to compression, experiencing a significant drop in both perplexity for language generation and
accuracy on commonsense and math reasoning tasks. Notably, the accuracy on ARC-C plummets to 18.33%
and MathQA to 19.89%, which is worse than random guessing. However, compensating for the sparsity and
quantization errors with EoRA significantly improves the accuracy of these compressed models, reducing
perplexity by up to 73.55 and boosting accuracy by 12.88%/9.60%/10.16% on ARC-C/MathQA/GSM8K
tasks. Additionally, EoRA consistently outperforms both ZeroQuant-V2 and Act-S across LLaMA2 and
LLaMA3. For instance, EoRA exceeds ZeroQuant-V2 in compensating the compressed LLaMA2-13B on
ARC-C by 1.79% and on MathQA by 1.77%, narrowing the accuracy gap with the uncompressed model to just
2.85% on MathQA. Overall, we find that EoRA tends to offer greater accuracy recovery when addressing more
aggressive compression settings, ensuring the plausibility of adopting EoRA for mitigating severe compression
error.

15
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.4. Compatibility With Various Compression Methods

Table 7 | Comparison between compensation methods of rank set to 128 on compensating LLaMA3-8B models
pruned to 2:4 sparsity with Wanda on Perplexity and Commonsense/Math reasoning tasks.

Compression Method Compression Setting Compensation Method Wikitext2 ↓ ARC-C ↑ MathQA ↑ GSM8K ↑
Full-precision - - 6.13 50.42 40.10 36.23

- 21.42 27.04 25.09 0.76
ZeroQuant-V2 17.16 30.46 26.16 1.28
Act-S 17.37 29.77 26.73 1.51
ApiQ 14.30 31.91 29.61 12.81
EoRA (Ours) 14.04 34.81 30.05 11.52

Wanda 2:4

In this section, we study the generalizability and compatibility of EoRA with different pruning methods
beyond SparseGPT. We adopt Wanda [7], a method that prunes weights with the smallest magnitudes scaled
by their corresponding input activations. For these compression methods, we adhere to the calibration set
construction detailed in 4.1, and maintain the same settings when utilizing EoRA to address compression
errors. We evaluate EoRAon LLaMA3-8B pruned with Wanda to 2:4 structured sparsity. The ranks of all
low-rank compensation methods are set to 128. Table 7 demonstrates that EoRAconsistently outperforms
every fine-tuning-free method, both ZeroQuant-V2 and Act-S, in improving accuracy across all the tasks.
For example, EoRA achieves accuracy gains of 7.77%/4.96%/10.76% on ARC-C/MathQA/GSM8K which
is 5.04%/3.32%/10.01% over the improvement brought by Act-s. Furthermore, EoRA outperforms ApiQ
on both ARC-C and MathQA by 2.9% and 0.44%. Overall, these findings underscore the effectiveness and
generalizability of EoRA across different compression techniques.

A.5. Influence of Different Calibration sizes on EoRA

Table 8 | Ablation studies of calibrating EoRA with different calibration sizes on compensating compressed
LLaMA3-8B.

Model Quantization Format #Calib. Calib. Time (mins) MathQA ↑

FP16 - - 40.10

16 6.40 36.62
32 7.04 36.93
64 8.03 37.21
128 10.40 37.46
256 14.43 37.60
512 21.11 37.30

W4

LLaMA3-8B

16 6.33 26.33
32 7.20 27.57
64 8.16 29.11
128 11.33 30.34
256 14.17 30.21
512 20.89 30.40

W3

We conducted ablation studies to assess how different calibration set sizes—{16, 32, 64, 128, 256, 512}—affect
EoRA’s performance in compensating for 4-bit and 3-bit quantized LLaMA3-8B models on MathQA. As
shown in Table 8, increasing the number of calibration samples from 16 to 256 yields a moderate improvement
in accuracy (from 36.62 to 37.60) for the W4 model. However, further increasing the calibration size to 512
leads to a slight decline, indicating that the accuracy gain saturates beyond a certain threshold. A similar
trend is observed for the W3 model, where performance improves steadily up to 128 samples, after which the

16
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

benefit plateaus. These findings suggest that while EoRA can leverage additional calibration data to improve
accuracy, its performance remains stable and robust even with limited calibration.

A.6. Inference Speed Evaluation

Table 9 | Comparison of the average per-token latency (batch size 1) for 128-token generation on LLaMA3-70B
between full-precision and GPTQ + EoRA with and without our custom EoRA kernel.

Format EoRA Rank EoRA Kernel Latency ↓ Speedup ↑
FP-16 - - 60ms 1x

- - 35ms 1.7x
64 No 52ms 1.2x
64 Yes 44ms 1.4x
128 No 54ms 1.1x
128 Yes 43ms 1.4x
256 No 58ms 1x
256 Yes 48ms 1.3x

3-bit

- - 38ms 1.6x
64 No 60ms 1x
64 Yes 49ms 1.2x
128 No 61ms 1x
128 Yes 51ms 1.2x
256 No 63ms 1x
256 Yes 53ms 1.1x

4-bit

In language generation, the model produces tokens sequentially, making matrix-vector multiplications the
primary factor impacting the inference latency. Consequently, we build our custom EoRA kernel on top of
GPTQ’s low-bit quantized matrix vector product kernel, pre-allocating the shared output prior to matrix
vector multiplication and integrating the full-precision matrix vector multiplication of  into the quantized
kernel reducing redundant memory access. We show the inference speedup of our proposed EoRA kernel in
Table 9. The first row shows the FP16 latency (60ms), followed by the 3-bit quantized-only model (35ms).
The remaining rows present EoRA latencies at different ranks, both with and without our custom CUDA
kernel, alongside 3-bit and 4-bit quantization. As shown in the table, our fused EoRA kernel significantly
improves inference speed for both 3-bit and 4-bit quantized models. While there remains some overhead
compared to the quantized-only baseline, even with our kernel, using no kernel results in significantly higher
latency—sometimes exceeding FP16—due to activation movement overhead, as discussed in Section 4.5. These
results underscore the practical deployability of EoRA when paired with our optimized kernel.

17
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.7. Compensation With Different Ranks

Table 10 | Results of EoRA of different rank on compensating LLaMA2/3 models pruned to 2:4 sparsity by
SparseGPT on Commonsense and Math reasoning tasks.
Model Sparsity r Compensation Method ARC-C ↑ MathQA ↑ GSM8K ↑

- - - 50.42 40.10 36.23

- - 30.11 26.43 2.12

64
ZeroQuant-V2 30.97 26.39 2.27
Act-S 30.46 26.67 3.34
ApiQ 33.10 27.87 11.52
EoRA (Ours) 33.10 28.57 10.77

128
ZeroQuant-V2 31.99 26.49 2.96
Act-S 31.74 26.73 3.26
ApiQ 34.21 28.77 14.55
EoRA (Ours) 34.64 29.91 13.95

LLaMA3-8B

2:4

256
ZeroQuant-V2 34.55 28.74 4.09
Act-S 32.76 27.94 5.16
ApiQ 35.41 30.45 15.85
EoRA (Ours) 37.96 31.59 17.06

512
ZeroQuant-V2 38.73 30.38 6.75
Act-S 36.18 29.65 8.64
ApiQ 36.69 32.63 20.77
EoRA (Ours) 41.89 34.17 23.28

- - - 39.84 27.67 14.85

- - 30.11 24.65 1.66

64
ZeroQuant-V2 30.20 24.48 1.97
Act-S 30.12 25.03 1.74
ApiQ 31.83 25.62 5.91
EoRA (Ours) 32.16 25.62 5.08

128
ZeroQuant-V2 30.54 24.89 1.97
Act-S 30.20 25.09 2.73
ApiQ 32.67 26.36 7.58
EoRA (Ours) 32.67 25.59 6.22

LLaMA2-7B

2:4

256
ZeroQuant-V2 31.99 25.19 2.88
Act-S 32.59 25.39 3.26
ApiQ 34.30 25.99 8.79
EoRA (Ours) 34.47 26.06 7.88

512
ZeroQuant-V2 34.72 24.38 3.34
Act-S 34.73 25.76 3.56
ApiQ 34.98 26.16 9.70
EoRA (Ours) 36.77 25.96 8.79

- - - 45.56 29.91 21.37

- - 34.30 25.92 2.65

64
ZeroQuant-V2 33.95 25.56 2.81
Act-S 32.76 25.93 2.96
ApiQ 35.84 27.17 8.64
EoRA (Ours) 36.00 26.80 8.19

128
ZeroQuant-V2 33.61 25.12 3.56
Act-S 34.12 25.69 4.09
ApiQ 36.68 27.16 12.13
EoRA (Ours) 37.54 27.53 10.91

LLaMA2-13B

2:4

256
ZeroQuant-V2 35.06 26.06 4.93
Act-S 34.56 26.23 4.62
ApiQ 36.69 27.40 14.56
EoRA (Ours) 38.73 27.77 13.04

512
ZeroQuant-V2 36.51 26.39 7.28
Act-S 36.86 26.77 6.14
ApiQ 38.57 27.71 17.21
EoRA (Ours) 40.61 29.17 17.51

Table 10 reports more detailed comparison for error compensation with different ranks.

18
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.8. Fine-tuning Compressed Models with EoRA

Table 11 | Fine-tune the compressed LLaMA3-8B models with various compression settings and different
initialization of the low-rank matrices for Commonsense/Math reasoning tasks.

Model Compression Method Compression Setting LoRA initialization ARC-C ↑ MathQA ↑

Full-precision - w/o fine-tuning 50.42 40.10

Standard 56.39 53.56

w/o fine-tuning 30.11 26.43
QLoRA 41.30 45.42
LoftQ 43.68 48.77
EoRA (Ours) 48.54 54.67

SparseGPT 2:4

w/o fine-tuning 45.90 34.07
QLoRA 54.09 51.42
LoftQ 54.52 53.96
EoRA (Ours) 55.46 56.04

LLaMA3-8B

GPTQ W4

w/o fine-tuning 20.90 22.37
QLoRA 30.29 34.10
LoftQ 44.70 48.17
EoRA (Ours) 47.44 53.90

GPTQ W3

Table 11 reports more detailed comparison for fine-tuning compressed models.

A.8.1. Ablation: Fine-tuning with different numbers of training data

Table 12 | Ablation study on the effect of using different proportions of the dataset for fine-tuning 2:4 pruned
LLaMA3-8B models with varying low-rank matrix initializations on Commonsense/Math reasoning tasks.

Model Dataset Ratio LoRA initialization ARC-C ↑ MathQA ↑

- - 50.42 40.10

QLoRA 41.30 45.42
LoftQ 43.68 48.77
EoRA (Ours) 48.54 54.67

100%

QLoRA 38.56 40.23
LoftQ 41.46 42.51
EoRA (Ours) 46.41 48.91

LLaMA3-8B

50%

QLoRA 36.77 36.71
LoftQ 39.76 40.60
EoRA (Ours) 43.85 44.79

30%

In this section, we show that fine-tuning with the EoRA-compensated model is robust to various ratios
of training data. We follow the setting in Sec. 4.4 on compressed LLaMA3-8B models with 2:4 sparsity
compression. As shown in Table 12, using EoRA for initialization consistently outperforms both standard and
SVD initialization across various dataset ratios, with accuracy improvements (ARC-E/ARC-C/MathQA) of
3.24%/4.95%/6.4% and 1.85%/4.09%/4.19% over SVD when fine-tuning using 50% and 30% training data,
respectively.

19
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.9. Quantizing EoRA To Further Reduce Memory Overhead
Table 13 | Accuracy and the Model Size of quantizing EoRA of rank {128,512} to 4/3-bit on compensating
LLaMA3-8B of {2:4 sparisity, 4/3-bit}.

Compression method Config r W-bit of EoRA Model Size (GB) ARC-C ↑ MathQA ↑
- - - - 15.08 50.42 40.10

- - 9.12 30.11 26.43

16 9.77 34.64 29.91
4 9.28 34.47 29.91
3 9.24 34.72 29.71

128

SparseGPT 2:4

16 11.70 41.89 34.17
4 9.77 41.46 33.63
3 9.64 40.35 32.66

512

- - 5.35 45.90 34.07

16 6.01 47.44 37.21
4 5.50 47.35 36.78
3 5.46 47.18 36.52

128

W4

16 7.85 48.29 38.72
4 6.01 48.80 38.92
3 5.90 46.92 36.88

512

- - 4.63 20.90 22.37

GPTQ

16 5.28 31.74 29.11
4 4.78 31.48 28.64
3 4.74 29.18 26.7

128

W3

16 7.16 38.82 31.89
4 5.28 40.01 31.69
3 5.18 35.4 30.45

512

Table 13 reports more detailed comparison for quantizing EoRA. For example, when a 512-rank EoRA is
quantized from 16-bits to 4-bit on 2:4 pruned LLaMA3-8B, the accuracy drops are only 0.43% on ARC-C
while the total model size reduces by 16.49% (11.70 GB → 9.77 GB). Additionally, compared to the original
uncompensated 2:4 pruned model, quantizing EoRA of rank 128/512 improves accuracy by 4.4%/11.4% with
a total model size increase of just 2% (9.12 GB → 9.28 GB) / 7% (9.12 GB → 9.77 GB). For 3-bit quantized
LLaMA3-8B compensated with a 4-bit quantized EoRA of rank 128/512 achieves 10.6%/19.1% accuracy
improvements, with a corresponding model size increase of only 3% (4.63 GB → 4.78 GB) / 14% (4.63 GB →
5.28 GB). Please see Section 4.5 for more analyses.

20
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.10. EoRA on More Tasks
Table 14 | Comparison of 4-bit and 3-bit quantized LLaMA3-8B on the LLM summarization task.

Model Quantization Format Compensation Method CNN/DailyMail (ROUGE-Lsum) ↑

- 0.1672
ZeroQuant-V2 0.1798
Act-S 0.1786
ApiQ 0.1804
EoRA (Ours) 0.1812

4-bit

LLaMA3-8B

- 0.0650
ZeroQuant-V2 0.0970
Act-S 0.1286
ApiQ 0.1357
EoRA (Ours) 0.1463

3-bit

LLM Summarization. We evaluate EoRA and baseline methods on restoring the summarization capability
of quantized models using the testset of CNN/DailyMail — a widely used English-language corpus containing
over 300k news articles authored by CNN and Daily Mail journalists. This dataset supports both extractive
and abstractive summarization, and we adopt ROUGE-Lsum as the evaluation metric, where higher values
indicate better summary quality. We set the rank to 128 for all the methods. For Act-S, ApiQ, and EoRA, we
use 128 calibration sentences from WikiText2. The results are shown in Table 14. Notably, EoRA consistently
outperforms all baselines across both 4-bit and 3-bit quantized LLaMA3-8B models. Specifically, it improves
the ROUGE-Lsum score from 0.1672 to 0.1812 in the 4-bit setting, and from 0.0650 to 0.1463 in the 3-bit
setting—demonstrating substantial recovery of summarization performance. These results highlight that
EoRA remains effective and competitive in practical, real-world scenarios such as summarization.
Table 15 | Comparison of 4-bit and 3-bit quantized LLaMA3.2-3B on multi-task language understanding.
Model Quantization Format EoRA Rank MMLU ↑

FP16 - 54.19

- 24.16
32 52.53
64 52.49
128 52.93

4-bit

LLaMA3.2-3B

- 22.89
32 39.08
64 38.83
128 39.68

3-bit

Language Understanding. We tested EoRA on 4-bit quantized LLaMA-3.2-3B, varying the rank from
8 to 128. Using 128 WikiText2 calibration samples, we observe substantial recovery—even rank 8 lifts
accuracy from 24.16 to 52.20, as shown in Table 15. This result highlights both the effectiveness of EoRA in
compensating for quantization errors in smaller-scale models and its robustness across a range of low-rank
settings.

21
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.11. GPTQ: Channel-wise vs. Group-wise Quantization

Table 16 | Comparison of 4-bit and 3-bit group-wise quantized LLaMA3-8B. The group size is set as 128.
EoRA rank is set as 128.

Model Quantization Format Compensation Method MathQA ↑

FP16 - 40.10

- 38.34
ZeroQuant-V2 38.92
Act-S 38.49
ApiQ 38.90
EoRA (Ours) 39.16

W4-groupsize 128

LLaMA3-8B

- 32.52
ZeroQuant-V2 32.39
Act-S 33.33
ApiQ 34.80
EoRA (Ours) 35.10

W3-groupsize 128

We initially adopt channel-wise quantization for GPTQ to remain consistent with the original GPTQ
setup. More importantly, our goal is to showcase the robustness of EoRA in recovering from even severe
quantization errors. That said, we conduct additional experiments using group-wise quantization (group
size = 128) to evaluate EoRA’s effectiveness in this more commonly used setting. Table 16 reports results
on MathQA for 4-bit and 3-bit group-wise quantized LLaMA3-8B models, referred to as W4-groupsize 128
and W3-groupsize 128, respectively. We compare EoRA against prior compensation methods and set the
rank to 128. These results highlight EoRA’s consistent advantage over existing methods in compensating
for group-wise quantization. In particular, for the W4-group size 128 configuration, EoRA is able to recover
nearly all of the lost accuracy, achieving performance comparable to the original full-precision model.

A.12. Layer-wise Discrepancy Analysis

Table 17 | Comparison of layer-wise discrepancy (the less the better) on  projector of LLaMA2-7B.

Method Layer 0 Layer 5 Layer 10 Layer 15 Layer 20 Layer 25 Layer 30
GPTQ (4-bit) 2.3 11.3 13.4 12.6 10.7 10.9 11.5
ZeroQuant-V2 1.7 10.2 10.8 11.3 9.7 9.2 9.9
Act-S 2.0 9.7 9.4 10.3 8.4 7.7 8.9
ApiQ 1.5 7.8 8.2 7.4 6.2 5.9 5.1
EoRA (Ours) 1.4 8.5 8.1 7.2 5.8 6.0 5.4

We follow the layer-wise discrepancy analysis setting of the ApiQ [21] and run the analysis on the  projector
of LLaMA2-7B. We compare the discrepancy of layers [0, 5, 10, 15, 20, 25, 30] of different compensation
methods of rank 128 and show that EoRA effectively reduces layer-wise discrepancy compared to existing
baselines, as shown in Table 17. Notably, EoRA maintains consistently low output activation errors across all
layers, especially in the later ones (e.g., 5.8 at layer 20 and 5.4 at layer 30), showing significant improvements
over ZeroQuant-V2 and Act-S. EoRA also matches or outperforms ApiQ, demonstrating its effectiveness.
Importantly, EoRA accomplishes this with dramatically less compute overhead—typically completing in just
a few minutes—whereas ApiQ often requires several hours of optimization. This highlights EoRA’s practical
advantage as a highly efficient solution for layer-wise error minimization for recovering the error of low-bit
quantization.

22
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

A.13. More Comparisons with Recent Low-Rank Approaches

Table 18 | Comparison of 4-bit and 3-bit quantized LLaMA3-8B for more recent low-rank approaches.

Model Quantization Format Compensation Method MathQA ↑

- - 40.10

- 34.07
FWSVD 35.64
ZeroQuant-V2 36.51

Act-S 35.84
ApiQ 36.18
LQER 35.46
LRC 36.40
CALDERA 36.70
QERA 35.90
SLiM 35.90
OATS 36.01
EoRA (Ours) 37.21

4-bit

LLaMA3-8B

- 22.37
FWSVD 26.30
ZeroQuant-V2 26.43

Act-S 25.42
ApiQ 26.86
LQER 25.60
LRC 28.64
CALDERA 28.10
QERA 25.32
SLiM 25.91
OATS 25.30
EoRA (Ours) 29.11

3-bit

Recent Low-Rank approaches can be broadly categorized into three groups: 1) activation-statisticsbased
scaling methods, including SLiM [30], OATS [31], LQER [32], and QERA [33], 2) iterative low-rank
compensation approaches, which are LRC [35] and CALDERA [34], and 3) training-based methods, including
LR-QAT [36] and SLoPe [37].

The first group—SLiM, OATS, LQER, and QERA—scales the compression error based on activation
statistics prior to applying SVD for low-rank approximation. The underlying intuition is that input channels
with higher magnitudes (i.e., outliers) are considered more important, and as a result, their corresponding
weight entries should be prioritized over those linked to lower-magnitude activations. All four methods
differ slightly in how the scaling diagonal matrix is constructed, but they are fundamentally similar to the
ASVD-based scaling baseline already discussed in our paper (see Section 4.1). For example, LQER [32]
mitigates quantization error by scaling the residual using activation statistics—specifically, the maximum
average magnitude per input channel—before applying SVD. Another example, SLiM [30], introduces a lowrank
compensation method that also leverages activation statistics (specifically, average absolute activation
values) to scale residuals. Although SLIM is proposed alongside a custom quantization scheme, we isolate and
evaluate its saliency-based low-rank adapter strategy. Although these methods vary in how they scale the
compression error, none of them are guaranteed to minimize the layer-wise compression error directly—an

23
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

issue we discussed in Section 2 of our paper. In contrast, EoRA is explicitly formulated to minimize this
objective (as detailed in Section 3).

The second group, including iterative low-rank compensation approaches, mainly lack the flexibility
as EoRA. For example, LRC [35] requires iterative updates to weights and low-rank modules, leading to
task-specific quantized models. In contrast, EoRA only adapts the low-rank modules, allowing a shared
compressed backbone and easier integration with multi-adapter frameworks [8]. Although LRC offers a
closed-form solution, it assumes is full-rank—an assumption that often fails and requires extra modification
steps that may introduce noise and instability. EoRA avoids this by only requiring to be symmetric, improving
numerical stability and robustness. Another example, CALDERA [34], employs an iterative optimization
strategy that updates both the quantized weights and the low-rank matrices, using a closed-form solution.
However, because it requires modifying the quantized weights during this process, CALDERA is less efficient
for multi-task scenarios, where separate quantized models would need to be maintained for each task. EoRA,
on the other hand, avoids this limitation by making only the low-rank components task-specific, while keeping
the quantized backbone fixed and shared across tasks. This decoupled design allows for easy integration
with existing multi-adapter inference frameworks [8] and significantly improves the practicality of EoRA for
real-world applications.

The third group requires gradient-based training. For example, LR-QAT [36] is a quantization-aware
training (QAT) method. While it also uses low-rank modules, it operates in the fine-tuning regime, combining
cross-entropy loss with final-layer output alignment (akin to knowledge distillation). SLoPe [37] primarily
targets improving pre-training efficiency and enhancing the accuracy of compressed models through training.
As such, it aligns more closely with quantization-aware training (QAT) as well. In contrast, EoRA is a
post-training quantization (PTQ) method that requires no gradient updates to the quantized model. In the
LLM compression community [1, 2], it is standard practice to distinguish between PTQ and QAT approaches,
as they serve different purposes and are not typically benchmarked against each other.

We summarize the empirical comparison on 4-bit and 3-bit quantized LLaMA3-8B models evaluated on
MathQA in Table 18. All methods use rank of 128 and identical calibration data (see Section 4.1). To fairly
compare CALDERA with EoRA, we adapt the CALDERA method by fixing the quantized weights and only
updating the low-rank matrices. This degenerates the iterative process into a two-step approximation: first
approximate the down-projection matrix, followed by the up-projection. Once the quantized weights are fixed,
additional iterations do not change the approximation.

As the results show, EoRA consistently outperforms other low-rank compensation methods, particularly
those based on activation statistics like LQER, SLiM, etc. This highlights the advantage of EoRA’s mathematical
property, which directly minimizes the layer-wise compression error rather than relying on heuristics
derived from activation statistics. While LRC offers better accuracy than heuristic scaling approaches, it
still lags behind EoRA due to its reliance on less stable approximations. These findings highlight EoRA’s
practicality and effectiveness for post-training quantized LLMs. From the results, we observe that CALDERA
performs well—outperforming both Act-S and ApiQ—but EoRA still consistently achieves higher accuracy.
We attribute this to EoRA’s single-step optimization that jointly solves for both projection matrices, whereas
CALDERA’s sequential two-step process may accumulate slightly more approximation error.

Overall, EoRA consistently surpasses all the recent low-rank methods, including both activation-statisticsbased
approaches and iterative low-rank compensation approaches. Its effectiveness stems from its ability to
directly minimize layer-wise compression error, eliminate the need for heuristic magnitude-based scaling, and
utilize a single-step optimization process. These strengths underscore both the theoretical robustness and
practical efficiency of EoRA compared to existing methods.

A.14. EoRA on 2-bit Quantization

We also evaluate a more challenging setting, 2-bit quantization, where RILQ [38] is one of the state-of-the-art
error compensation methods. RILQ adopts standard backpropagation (i.e., fine-tuning), utilizing a combination
of cross-entropy loss and final-layer output alignment (similar to knowledge distillation). As outlined in the
RILQ paper, RILQ uses gradient descent to collectively tune all adapters, minimizing the discrepancy between
full-precision and quantized activation outputs of the final layer. In addition, as quoted from the RILQ
paper, RILQ also incorporates a causal language modeling objective with Ground Truth in the optimization

24
EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation

Table 19 | Comparison of 2-bit GPTQ-quantized LLaMA3-8B on MathQA.

Model Quantization Format Fine-tuning Strategy MathQA ↑

- 18.22
LoftQ 35.80
EoRA 36.89
RILQ 37.60

LLaMA3-8B 2-bit (GPTQ)

RILQ + EoRA 38.90

of low-rank adapters. Therefore, it is not appropriate to directly compare EoRA to RILQ as methods in
the same category. However, since EoRA can act as an effective initialization for subsequent fine-tuning
(as detailed in Section 4.4), it is complementary to RILQ rather than competing with it. To examine this
synergy, we performed experiments where EoRA was first used for initialization—following the setup outlined
in Section 4.4—and then applied RILQ’s fine-tuning objective to a 2-bit GPTQ-quantized LLaMA3-8B model
on the MathQA dataset. The results are shown in Table 19. As expected, RILQ outperforms LoftQ in
compensating for quantization error in 2-bit models. Moreover, when initialized with EoRA, further fine-tuning
with RILQ’s objective yields an additional 1.3% improvement over RILQ alone.

25