Is Retraining-Free Enough? The Necessity of Router Calibration

for Efficient MoE Compression

Sieun Hyeon 1 Jaeyoung Do 1 2

Abstract

nated by ever-larger models. However, the rapid growth in
parameter counts has exposed fundamental challenges in
efficiency, memory footprint, and deployability.

Mixture-of-Experts (MoE) models scale capacity
efficiently, but their massive parameter footprint
creates a deployment-time memory bottleneck.
We organize retraining-free MoE compression
into three paradigms—Expert Pruning, Expert
Editing, and Expert Merging—and show that
persistent post-compression degradation largely
stems from a neglected factor: router–expert mismatch
when experts are changed but the router is
left untouched. We argue that effective retrainingfree
compression should avoid updating expert
parameters while allowing lightweight router calibration.
To this end, we propose Router Knowledge
Distillation (Router KD), which updates
only a tiny fraction of parameters (the router)
by distilling the original model’s next-token distribution
on unlabeled calibration data. Experiments
across representative methods in all three
paradigms demonstrate consistent performance
recovery, with substantially larger gains in finegrained
MoEs (many small experts) than in coarsegrained
MoEs due to their more complex routing
decision boundaries.

arXiv:2603.02217v1 [cs.LG] 10 Feb 2026

The Mixture-of-Experts (MoE) (Shazeer et al., 2017; Lepikhin
et al., 2020) architecture has emerged as a key response
to this tension. By decoupling total model capacity from
per-token computation, MoE enables models to scale to massive
parameter counts while activating only a sparse subset
of experts at inference time. This property has made MoE
a cornerstone of modern foundation models, offering an
appealing balance between performance and computational
efficiency (Jiang et al., 2024; Team, 2025). Despite these
advantages, MoE models introduce a severe memory bottleneck:
the full parameter set must still be resident in memory,
even though only a fraction is used per token (Fedus et al.,
2022). As a result, MoE LLMs remain prohibitively expensive
to deploy in resource-constrained environments,
limiting their accessibility to most practitioners and users.

To mitigate this issue, a growing body of work has focused
on retraining-free compression of MoE architectures, methods
that reduce memory consumption without full-scale retraining
(Lasby et al., 2025; Chen et al., 2025a; Anonymous,
2025c). Since experts account for the overwhelming majority
of parameters in MoE LLMs, existing approaches have
primarily targeted expert-side compression. These methods
are often presented as mutually competing solutions that
optimize compression ratio while minimizing performance
degradation, and the field has rapidly become crowded with
claims of near-optimal efficiency.

1. Introduction

Large Language Models (LLMs) have driven a paradigm
shift in artificial intelligence, exhibiting remarkable capabilities
across a broad spectrum of tasks from creative generation
and code synthesis to complex mathematical reasoning.
(OpenAI, 2025) Beyond natural language processing, their
impact now extends to domains such as robotics, medicine,
and scientific discovery. Motivated by empirical scaling
laws (Kaplan et al., 2020) that link increased model capacity
to improved performance, recent progress has been domiHowever,
despite this surge of innovation, a fundamental
question remains unresolved: why does performance degradation
persist even when expert compression is carefully
designed? We argue that the core limitation is not the absence
of a “perfect” expert compression method, but rather
a systematic mismatch between compressed experts and
an unmodified router. Expert compression, whether by removal,
modification, or merging, inevitably perturbs the
functional landscape on which routing decisions were originally
learned. Yet, in almost all retraining-free approaches,
the router is left unchanged. This mismatch leads to suboptimal
expert selection and amplifies performance loss.

1Department of Electrical and Computer Engineering, Seoul National
University, Seoul, South Korea 2
Interdisciplinary Program in

Artificial Intelligence, Seoul National University, Seoul, South Korea.
Correspondence to: Jaeyoung Do <jaeyoung.do@snu.ac.kr>.

Preprint. March 4, 2026.

1
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Figure 1. Illustration of the three MoE compression paradigms. The diagram depicts (A) Expert Pruning (selection), (B) Expert Editing
(decomposition or modification), and (C) Expert Merging (aggregation) to reduce model size.

In this work, we formalize this observation by systematizing
retraining-free MoE compression methods into three
categories: Expert Pruning, Expert Editing, and Expert
Merging. Using this taxonomy, we investigate a largely overlooked
dimension of MoE compression: router calibration.
Through theoretical analysis, we show that routing discrepancies
arise even in best-case compression scenarios, and
that these discrepancies compound across layers. Our empirical
analysis further demonstrates that router miscalibration
is a dominant contributor to post-compression performance
degradation across all three compression paradigms.

• We propose a taxonomy of retraining-free MoE

compression—Expert Pruning, Editing, and Merging—
and identify router miscalibration as a primary
source of post-compression performance degradation.
Through theoretical and empirical analysis, we show
that expert compression must be coupled with router
calibration for effective performance preservation.

• We introduce Router Knowledge Distillation as a

lightweight and general recovery strategy that updates
only router parameters. Extensive experiments demonstrate
that Router KD consistently restores performance
across all compression paradigms and is particularly
effective for fine-grained MoE architectures.

These findings lead to a critical re-examination of the notion
of “retraining-free” compression. We show that fully
retraining-free compression—defined as leaving both experts
and router untouched—is often impractical for achieving
strong performance. Instead, we advocate a more precise
interpretation: avoiding updates to expert parameters, while
allowing lightweight router recalibration.

2. Related Works

We categorize retraining-free MoE compression into Expert
Pruning, Editing, and Merging. Crucially, we focus
exclusively on parameter-level compression—reducing the
number of parameters—and explicitly exclude bit-level compression
techniques such as quantization. Expert Pruning
removes redundant experts (k). Strategies include minimizing
reconstruction loss (Lu et al., 2024), differentiable
selection (Bai et al., 2025), and leveraging router magnitudes
(Lasby et al., 2025). Other approaches utilize output
discrepancy bounds (Anonymous, 2025a), token variation
(Dong et al., 2025), trajectory-based importance (Yang
et al., 2025), or coarser layer-level pruning (Jaiswal et al.,
2025; He et al., 2025a). Expert Editing compresses expert
internals via decomposition while retaining the expert
count. Methods employ Singular Value Decomposition
(SVD) (Li et al., 2025), factorization into shared and specific
components (Liu et al., 2025), rank decomposition
with shared bases (Chen et al., 2025b), or tensor decomposition
(Anonymous, 2025c). Expert Merging is grounded in
model merging hypotheses (Wortsman et al., 2022; Matena
& Raffel, 2022). This approach synthesizes experts via output
similarity clustering (Chen et al., 2025a), selective dualmasks
(Zhao et al., 2025), compression matrices (Miao et al.,
2025), or importance-guided coefficient merging (Zhang
et al., 2025a). See Appendix E for extended related works.

To this end, we introduce Router Knowledge Distillation
(Router KD) as a simple yet effective recovery mechanism.
Router KD updates only the router parameters of
the compressed model, distilling knowledge from the original
model’s output distribution. Importantly, this process
incurs minimal computational overhead, as the router constitutes
only a tiny fraction of the total model parameters.
We apply Router KD to representative methods from each
compression category and quantify how much performance
can be recovered by calibrating the router alone.

Our experimental results show that Router KD consistently
and substantially mitigates performance degradation across
Expert Pruning, Editing, and Merging. Moreover, we reveal
that the effectiveness of router calibration depends strongly
on MoE architecture. In particular, fine-grained MoE models
with many small experts (e.g., Qwen3-30B-A3B-Instruct
(Team, 2025)) benefit significantly more from Router KD
than coarse-grained models with fewer, larger experts (e.g.,
Mixtral-8×7B-Instruct (Jiang et al., 2024)), due to the increased
complexity and flexibility of their routing decision
boundaries. Our contributions are summarized as follows:

2
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

3. Causes of Performance Degradation

Let

Despite extensive research on MoE compression, a critical
unresolved issue is the performance degradation observed
in compressed models compared to their original counterparts.
While this might be perceived as an inevitable tradeoff
due to the reduction in parameters, minimizing such
performance loss remains the primary challenge in MoE
compression. In this section, we isolate a key contributor:
router–expert mismatch. We derive how compression
perturbs routing behavior and show that router-induced error
can arise even under favorable compression scenarios,
compounding across layers.

• S be the set of expert indices selected by the original
MoE model,

• P ⊆ {0, . . . , n − 1} be the set of experts that remain
after pruning, and |S| ≤ |P|.

• S
′ be the set of experts effectively used by the pruned
model.

′
.

We analyze the relationship between S, P and S

Gate scores before and after pruning. We denote by
g
orig
i
(x) the expert activation scores produced by the original
gate network for an input x, and by g
pruned
i
(x) the scores

3.1. Original MoE LLMs

produced when running the pruned model end-to-end. For
any index set A, we define the corresponding renormalized
weights as

In the original MoE model before compression, an input (x)
first passes through the gate network, producing n expert
activation scores. Depending on the implementation, the
gate network outputs either the raw logits or the probabilities
after a softmax operation. Let these n expert activation
scores be denoted as (g0, . . . , gn). Likewise, let the n experts
in the same layer as the gate network be represented
as (E0, . . . , En). Assume that this MoE model activates the
top-k experts. In this case, the computation for the input x
is performed as follows.

(x) = g
orig
i
(x)
P
j∈A g
orig
j
(x)

g˜
orig,(A)
i

(x) = g
pruned
i
(x)
P
j∈A g
pruned
j

g˜
pruned,(A)
i

, i ∈ A.

(x)

3.2.1. BEST SCENARIO

S ⊂ {0, 1, . . . , n − 1}, |S| = k, k < n

In the best scenario, all originally selected experts remain
after pruning:
S ⊆ P =⇒ S′ = S.

We define the renormalized expert activation weights as

g˜i = P
gi
j∈S gj

, i ∈ S

The original and pruned MoE outputs for input x can then
be written as

Using these normalized weights, the output of the MoE
layer for input (x) is computed as:

yorig(x) = X
i∈S
g˜
orig,(S)
i

(x) Ei(x)

y =
X
i∈S
g˜i
· Ei(x).

y
best
pruned(x) = X
i∈S
g˜
pruned,(S)
i

(x) Ei(x).

3.2. Expert Pruning (N → N − α)

When the original MoE model undergoes pruning, the inference
can fall into one of the following three scenarios:

The difference between the original and pruned MoE outputs
in the best scenario is then given by:

1. Best scenario: All experts selected by the original model
remain available after pruning, and the same experts are
used without any change.



yorig(x) − y
best
pruned(x)


 =





X
i∈S
g˜
orig,(S)
i
(x) Ei(x) −
X
i∈S
g˜
pruned,(S)
i

(x) Ei(x)





=

2. Most common scenario: Pruning is imperfect, and while
some originally selected experts remain available and are
used as before, others are removed. For the dropped experts,
the model must instead rely on alternative experts as
substitutes.






X
i∈S

g˜
orig,(S)
i

(x)

Ei(x)

(x) − g˜
pruned,(S)
i

3. Worst scenario: All experts that were originally selected
are pruned out, forcing the model to replace every originally
chosen expert with different ones.

In this scenario, the set of active experts is identical to
that of the original model. However, note that even if the

3
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

by g
orig
i
(x) the expert activation scores produced by the
original gate network for an input x. We denote by g
edit
i
(x)
the activation scores produced when running the edited MoE
model end-to-end on the same input. For any index set A,
we define the renormalized weights for the edited model as

selected experts match exactly pre- and post-pruning, the
router’s output values are unlikely to be identical. Because
MoE LLMs are multi-layered, for the router outputs to be
perfectly identical, this best-case scenario must be satisfied
in every layer, which is statistically improbable. Therefore,
it can be trivially stated that g˜
orig,(A)
i
(x) ̸= ˜g
pruned,(A)
i
(x)

(x) = g
edit
i
(x)
P
j∈A g
edit
j
(x)

g˜
edit,(A)
i

, i ∈ A.

3.2.2. MOST COMMON & WORST SCENARIO

For the mathematical formulation of the Most Common and
Worst scenarios, please refer to Appendix B. A key insight
from this analysis is that discrepancies induced by the router
arise even in the rare best-case scenario. Although the router
weight g˜i
is merely a scalar value between 0 and 1, the Expert
itself is a matrix consisting of over 1 million scalar
values; thus, the resulting deviation is by no means negligible.
Furthermore, this difference will inevitably amplify in
the Most Common and Worst scenarios, where the selected
experts differ from those in the original model. This suggests
that the divergence between the pruned and original
models is, in part, attributable to the router.

3.3.1. BEST SCENARIO

In the best scenario, the router’s top-k selection remains
unchanged even after expert editing, i.e.,

S
edit = S.

The original MoE output for input x is

yorig(x) = X
i∈S
g˜
orig,(S)
i

(x) Ei(x),

while the edited MoE output in the best scenario becomes

y
best
edit (x) = X
i∈S
g˜
edit,(S)
i

(x) Xi(x).

′
)

3.3. Expert Editing (N → N, Parameters P → P

Unlike pruning, Expert Editing preserves the total number
of experts. However, as the parameters in preceding layers
are modified, the router’s computation results may shift.
Consequently, for the same input, the edited model may
select different experts compared to the original model. The
inference process can fall into one of the following three
scenarios:

Even though the index set of selected experts is identical,
both the gate scores and the expert functions may differ
between yorig(x) and y
best
edit (x).
The difference between the original and edited MoE outputs
in the best scenario is then given by:

1. Best scenario: The router’s selection remains completely
unchanged after editing, meaning the exact same experts
chosen by the original model are utilized.



yorig(x) − y
best
edit (x)


 =





X
i∈S
g˜
orig,(S)
i
(x) Ei(x) −

X
i∈S
g˜
edit,(S)
i
(x) Xi(x)










X
i∈S

g˜
orig,(S)
i
(x) Ei(x) − g˜
edit,(S)
i
(x) Xi(x)

2. Most common scenario: Due to the router’s altered expert
activation scores following the editing process, the selection
is partially changed. While some originally selected
experts are retained and used as before, others are replaced
by different experts.

Even if sophisticated mathematical approximation techniques
render Ei and Xi nearly identical, they are not
strictly identical; thus, g
orig
i
and g
edit
i
cannot be equal.
Given that each expert is a matrix containing over 1 million
parameters, even slight deviations in router outputs—despite
the similarity between Ei and Xi—inevitably lead to nonnegligible
differences in the final output.

3. Worst scenario: The router’s output shifts drastically such
that none of the originally selected experts are chosen, and
a completely different set of experts is activated.

Let

• S be the set of expert indices selected by the original
MoE model,

• S
edit ⊆ {0, . . . , n − 1} be the set of expert indices
selected by the edited MoE model, and |S| = |Sedit|.

3.3.2. MOST COMMON & WORST SCENARIO

For the mathematical formulation of the Most Common and
Worst scenarios, please refer to Appendix C. These equations
imply that the greater the imperfection in the Expert
Editing method, the more the router’s output diverges from
the original, inevitably altering the set of selected experts.

• Ei denote the original experts, and Xi denote the corresponding
edited experts obtained by modifying Ei
.

Gate scores before and after editing. As before, we denote

4
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Furthermore, it can be inferred that the error stemming from
selecting an incorrect expert—specifically the difference between
the original expert Ei and a mismatched edited expert
Xj—exceeds the error introduced by the editing process
itself (i.e., the difference between Ei and its corresponding
Xi). Given the inherent nature of Sparse MoE, where
experts within the same layer exhibit significant distinctiveness,
the divergence between completely different experts
(i.e., Ei and Xj ) is highly likely to surpass the deviation
between an original expert and its approximated edited counterpart
(i.e., Ei and Xi). Consequently, if router calibration
can mitigate the risk of selecting Xj and ensure the selection
of Xi
, the deviation from the original output (relative
to Ei) can be minimized.

(a) Mean L1 Distance of Routing Probabilities

3.4. Expert Merging (N → M, where M < N)

Unlike Pruning and Editing (three scenarios each), Expert
Merging entails nine distinct scenarios. Due to space limitations,
we refer the reader to Appendix D for the detailed
mathematical formulations. To summarize, similar to Pruning
and Editing, Expert Merging inevitably suffers from
router-induced errors even in the best-case scenario. Furthermore,
in the most common scenario, the merged experts
may be selected differently than originally intended by the
router, potentially resulting in even greater discrepancies.

(b) Top-8 Expert Overlap Ratio
Figure 2. Layer-wise Analysis of Router Behavior on Qwen3-
30B-A3B-Instruct-2507(128 Experts). Results are based on 100
samples from the ELI5 dataset. (a) shows the L1 distance of routing
probabilities, where Pruning methods exhibit larger divergence
compared to Editing and Merging. (b) illustrates the Top-8 expert
overlap ratio, indicating that router consistency degrades in deeper
layers across all compression methods.

3.5. Empirical Analysis of Router Behavior

We empirically examine how routing changes after compression.
Using Qwen3-30B-A3B-Instruct-2507 as the backbone,
we sample 100 questions from ELI5 (Fan et al., 2019)
and compare the router outputs between the original and
compressed models.

As shown in Figure 2a, the routing probabilities assigned to
experts deviate from the original model across most layers
for all three methods: Pruning, Editing, and Merging. Notably,
Pruning models exhibit a relatively larger divergence
in assigned probabilities compared to Editing and Merging.
This is attributed to the fact that Pruning drops experts and
masks the corresponding indices, whereas Editing and Merging
retain the total number of experts, thereby preserving
the dimensionality of the router.

4. Router Knowledge Distillation (Router KD)

Motivated by our analysis in Section 3, we seek to recalibrate
the routing behavior of a compressed MoE model
so that it better reproduces the original model’s next-token
predictions under fixed compressed experts. To this end,
we propose Router Knowledge Distillation (Router KD),
a lightweight distillation procedure that updates only the
student router parameters while keeping all other student
parameters frozen.

The impact of the router post-compression is even more pronounced
in Figure 2b, which illustrates the overlap ratio of
experts selected by the router before and after compression.
We observe that as the layers deepen, the router increasingly
selects experts different from those intended by the original
model. Together, these results support our theoretical analysis:
preserving compressed MoE performance requires not
only expert-side compression but also router calibration to
mitigate mismatch.

Router-only KD objective. Let Dcal be an unlabeled calibration
corpus, and let θT denote the parameters of the original
(Teacher) MoE model. The compressed (Student) MoE
model has parameters θS = (θR, θE), where θR denotes
router (gating) parameters and θE denotes all remaining parameters
(including experts and shared blocks). We freeze
θE and optimize only θR:

Ex∼Dcal [LRKD(x; θT , θR)] . (1)

⋆
R = arg min
θR

θ

5
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

5. Experiments

Distillation loss. For an input token sequence x =
(x1, . . . , xL), let z
(t)
T ∈ R
|V| and z
(t)
S ∈ R
|V| denote the
teacher and student vocabulary logits for predicting the next
token at position t (i.e., conditioned on the prefix x≤t). With
temperature τ > 0, we define the softened next-token distribution
for a model M ∈ {T, S} as:

Our experiments are designed to answer the following
questions: (i) Can Router KD consistently recover performance
lost due to MoE compression? (ii) Does its
effectiveness generalize across different compression
paradigms? (iii) How does MoE architecture influence
the benefits of router calibration? To this end, we select
two representative methods from each of the Expert
Pruning, Editing, and Merging categories. We evaluate
Router KD on two widely used MoE backbones with contrasting
architectures: Qwen3-30B-A3B-Instruct-2507 and
Mixtral-8×7B-Instruct-v0.1. Hybrid compression methods
are intentionally excluded to isolate the independent effect
of router calibration.

(t)
M (·) = Softmax

(t)
M
τ

!

z

p

. (2)

We then minimize the token-level KL divergence from
teacher to student, masking padding tokens. Let mt+1 ∈
{0, 1} be the loss mask for the target position (t+1), and
define the normalizer Nx =
PL−1
t=1 mt+1 + ϵ, where ϵ is a
small constant. The router KD loss for a sequence is:

5.1. Baselines

For Expert Pruning baselines, we adopt REAP (Lasby
et al., 2025), which defines expert importance based on
a criterion that considers both router gate-values and the
magnitude of expert outputs. We also compare against
CFES (Anonymous, 2025a), which utilizes a coarse-to-fine
expert selection strategy to efficiently identify critical experts
by minimizing layer-wise output discrepancy. For
Expert Editing, we utilize MoBE (Chen et al., 2025b) as
a baseline, which employs rank decomposition to separate
experts into unique components and shared basis matrices
to minimize reconstruction error. Although MoBE involves
backpropagation and thus may not be strictly classified as
retraining-free, we adopt it as a suitable baseline because
it is significantly more computationally efficient than standard
fine-tuning and does not require large-scale training
datasets for compression. Additionally, we include TDMoE
(Anonymous, 2025c), which treats experts as correlated
tensors and applies Tucker Decomposition after aligning
the data distribution via whitening. For Expert Merging,
we select HC-SMoE (Chen et al., 2025a) as a baseline,
which utilizes hierarchical clustering based on the similarity
of expert outputs on calibration data. Furthermore, we
employ M-SMoE (Li et al., 2024), which integrates experts
through activation frequency-based weighted averaging or
permutation alignment to preserve collective knowledge.

L
X−1

LRKD(x; θT , θR) = τ
2
Nx

mt+1 DKL



(t)
T

∥ p
(t)
S

p

.

t=1

(3)
Importantly, although the distillation loss is defined on the
output token distribution, gradients are backpropagated and
applied exclusively to the student router parameters θR,
while all expert and backbone parameters remain frozen.
This design directly calibrates the routing behavior of the
compressed model so that it better matches the teacher’s
next-token predictions under fixed compressed experts. In
practice, we follow standard MoE routing implementations
in which gradients flow through the (soft) gating weights of
the selected experts during the forward pass, but parameter
updates are restricted to the router.

By distilling output logits rather than matching router gate
values explicitly, Router KD avoids requiring the teacher
and student to share identical expert sets or gate dimensionalities,
which may not hold after pruning, editing, or merging.
Instead, the student router learns to route tokens to experts
that best reproduce the teacher’s next-token distribution,
thereby partially compensating for routing mismatch and
functional discrepancies introduced by expert compression.
A key advantage of Router KD is its short wall-clock training
time. Since only the router is updated, the number of
trainable parameters is negligible compared to the full MoE
model. Concretely, the router accounts for approximately
0.04% of parameters in Qwen3-30B-A3B-Instruct-2507 and
0.002% in Mixtral-8×7B-Instruct-v0.1. Accordingly, under
the hyperparameter settings used in our experiments, Router
KD required approximately 2 hours for Qwen3 and about 40
minutes for Mixtral, while yielding substantial performance
recovery.

5.2. Benchmark Datasets

We evaluated performance across various types of benchmark
datasets to assess performance changes under diverse
conditions. To assess general reasoning capabilities and QA,
we used BBH (Suzgun et al., 2022) (evaluated separately
for both few-shot and zero-shot settings) and CoQA (Reddy
et al., 2019). For mathematical problem-solving and reasoning,
we utilized GSM8k (Cobbe et al., 2021), GSM8k
Platinum (Vendrow et al., 2025), MATH (Hendrycks et al.,
2021b; Lewkowycz et al., 2022; Kydlicek et al., 2025),

6
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Figure 3. Performance Impact of Router KD on Qwen3 vs. Mixtral. The chart compares performance recovery across Pruning, Editing,
and Merging, where green bars indicate improved benchmarks. Router KD proves significantly more effective for the fine-grained Qwen3
(Left) compared to the coarse-grained Mixtral (Right), which shows limited gains due to its simpler routing decision boundaries.

AIME 1983-2024 (averaged across all problems for those
years) (Veeraboina, 2024), and AIME 2025 (math ai, 2025).
Furthermore, we employed MBPP (Austin et al., 2021) and
HumanEval-Instruct (Chen et al., 2021) to assess coding
proficiency. In addition to these, we evaluated Chain-ofThought
(CoT) (Wei et al., 2023) reasoning capabilities
on the BBH, GSM8k, and GSM8k Platinum datasets (also
evaluated separately for few-shot and zero-shot settings).
Finally, nine benchmark datasets (Clark et al., 2018; Zellers
et al., 2019; Pal et al., 2022; Jin et al., 2020; Mihaylov et al.,
2018; Bisk et al., 2020; Sakaguchi et al., 2019; Hendrycks
et al., 2021a) were utilized to assess Multiple Choice Question
Answering (MCQA) performance.

provides a visual summary of the experimental results, while
detailed numerical data can be found in Appendix G. When
applying Router KD to the Qwen3 backbone, we observed
improvements across the majority of benchmarks. This
trend was consistent across Expert Pruning, Editing, and
Merging, demonstrating that using the teacher’s next-token
distribution as supervision to optimize the student router is
a sufficiently effective recovery strategy. However, this efficacy
was not universal, and we identify specific scenarios
where Router KD proved less effective.

Coarse-grained Experts The most significant disparity
in the effectiveness of Router KD was observed when varying
the backbone architecture. For Qwen3, Router KD
improved benchmark scores across all six compression
methodologies; conversely, the performance gains were
relatively marginal for Mixtral. This distinction can be
attributed to structural differences: Qwen3 (30.5B parameters)
utilizes a fine-grained expert structure with 128 experts
per layer, whereas Mixtral (46.7B parameters) employs a
coarse-grained structure with only 8 experts per layer. In
other words, while Qwen3 consists of many small experts,
Mixtral comprises fewer, larger experts. Consequently, our
results indicate that the beneficial impact of Router KD is
diminished in coarse-grained MoE models like Mixtral.

5.3. Experimental Setup

All experiments were evaluated using the lm-evaluationharness
(Gao et al., 2024) and vLLM (Kwon et al., 2023).
Additionally, Qwen3 was tested in an environment equipped
with NVIDIA A100 40GB GPUs, while Mixtral was tested
with A100 80GB GPUs. The random seed was fixed at
42 for all experiments, and all benchmarks were measured
using greedy decoding with a temperature of 0. Hyperparameters
for Router KD—including epochs, batch size,
max length, number of calibration samples, and learning
rate—were kept consistent across all experiments. A uniform
expert retention rate of 62.5% was applied; in the
context of pruning, this entails a reduction in experts from
128 to 80 for Qwen3 and from 8 to 5 for Mixtral. For all
baselines, Router KD used the same C4 (Dodge et al., 2021)
dataset, and the hyperparameters were set identically. Please
refer to Appendix F for more detailed information.

Router KD can only refine the gating decision—selecting
a top-k subset of experts and reweighting them to match
the teacher. Accordingly, its attainable gain is inherently
limited when (i) the router has few alternative routing paths
to switch to, and (ii) the teacher routing targets provide little
additional information beyond an almost-hard decision.

(1) Small combinatorial routing space. Under top-k routing
with E experts, the number of distinct expert subsets is
|Ω| =
E
k

. For Mixtral-style routing (E=8, k=2), |Ω| =

8
2

= 28, whereas for fine-grained MoEs (e.g., E=128,
k=8), |Ω| =

128
8

≈ 1.43 × 1012. Hence, even if KD

5.4. Results

We confirmed that Router KD effectively mitigates the performance
degradation during model compression. Figure 3

7
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Catastrophic Collapse after Compression Additionally,
we observed that Router KD yields no performance improvements
when the model suffers from catastrophic collapse—that
is, when benchmark scores drop to near-zero
immediately after compression. This suggests that in catastrophic
collapse, the failure likely stems from damage beyond
misrouting (e.g., degraded expert representations),
which router-only calibration cannot recover.

Figure 4. Robustness of Router KD under Milder Compression
(75% Retention). Average benchmark scores for Qwen3 when
retaining 75% of Expert parameters. Comparing standard compression
against Router KD (suffix -R) relative to the original model’s
mean (red line) confirms that Router KD consistently recovers
performance across different compression ratios.

Performance Gains after Compression Furthermore,
upon investigating cases where Router KD led to performance
regression, we found that these often coincided with
instances where compression paradoxically improved model
performance. While compression generally degrades performance,
there are rare instances where it boosts scores in
specific benchmarks or domains. In such scenarios, Router
KD becomes counterproductive because the Teacher (the
original model) effectively underperforms compared to the
Student (the compressed model). However, given that the
original model typically outperforms the compressed version,
such cases are infrequent.

improves the router, the number of qualitatively different
routing paths is fundamentally bounded in small-E MoEs,
since the discrete choice of the top-k subset is limited even
though reweighting within a subset is continuous.

(2) Gradient viewpoint: limited degrees of freedom and
weaker KD signal. For intuition, consider the temperaturescaled
routing distributions over experts gT (x), gS(x), defined
as gM(x) = Softmax(zM(x)/τ ), zM(x) ∈ R
E,
M ∈ {T, S} where zM(x) are router logits. For the KL
objective L = KL(gT ∥gS), the gradient w.r.t. each student
logit is approximately

5.5. Additional Experiments

We additionally examined whether the calibration effect of
Router KD is preserved under different compression ratios.
Specifically, to verify that the observed gains are not limited
to a particular compression setting (e.g., 62.5%), we applied
a milder compression configuration to Qwen3, retaining
75% of the total parameters, and then evaluated the effect
of Router KD. The experimental results are summarized in
Figure 4, with detailed numerical values reported in Table 6.
Even under this alternative compression ratio, Router KD
exhibited pronounced improvements across all benchmark
categories and mitigated the performance drop from the
original model compared to the non-KD baseline. These
results confirm that Router KD provides robust and consistent
recovery effects across compression ratios, rather than
being effective only at a specific pruning level.

∂
∂zS,e

1
τ



gS,e − gT ,e

KL(gT ∥ gS) ≈

(4)

so the learning signal decomposes across the expert dimension
e ∈ {1, . . . , E}. If the router is linear, zS(x) = WSx,
then
∇WS L ∝
gS,e − gT ,e
x
⊤ (5)

highlighting that the router receives adjustment opportunities
along E expert coordinates.

In coarse-grained MoEs, experts typically have broad coverage,
and the teacher routing distribution is often highly
concentrated (low entropy), which means H(g
coarse
T
) ≪
H(g
fine
T
). When gT is already near-hard, (gS − gT ) quickly
becomes small once the student matches the dominant experts,
which deprives the gradients of informative ‘dark
knowledge’ regarding non-selected experts. Moreover,
when E is small, the router has fewer coordinates to reshape
and fewer alternative top-k subsets to switch to, further limiting
the observable benefit of router-only KD. By contrast,
in fine-grained MoEs with large E and E ≫k, the routing
space is vastly larger and the teacher targets are often less
peaky, so KD provides richer ‘dark knowledge’ over many
non-selected experts, acting as a navigational cue across a
substantially more complex decision boundary.

6. Conclusion

In this work, we systematized MoE compression into Expert
Pruning, Editing, and Merging, identifying router-expert
mismatch as a primary cause of performance degradation.
We introduced Router Knowledge Distillation (Router KD),
a lightweight strategy that calibrates only the router parameters
to mitigate this misalignment. Empirical results confirm
that Router KD consistently recovers performance across all
paradigms, proving particularly effective for fine-grained architectures
with complex routing spaces. We conclude that
efficient MoE compression requires accompanying expert
modification with minimal router calibration.

8
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Impact Statement

entiable expert pruning. Advances in neural information
processing systems, 2025.

This paper presents work whose goal is to advance the field
of Machine Learning, specifically addressing the deployment
challenges of Large Language Models (LLMs). By enabling
effective compression of Mixture-of-Experts (MoE)
architectures without the need for resource-intensive full
retraining, our proposed method, Router Knowledge Distillation,
significantly reduces the memory footprint and
computational costs required for inference. This has two
primary societal implications: first, it contributes to environmental
sustainability by lowering the energy consumption
and carbon footprint associated with running large-scale
models. Second, it facilitates the democratization of AI
by making powerful foundation models accessible to researchers
and practitioners with limited hardware resources
(e.g., consumer-grade GPUs), thereby reducing the barrier
to entry in the field. While we acknowledge the general
ethical risks associated with LLMs, such as potential bias
or misuse, our specific algorithmic contribution primarily
focuses on efficiency and does not introduce new negative
societal consequences.

Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y.
Piqa: Reasoning about physical commonsense in natural
language. In Thirty-Fourth AAAI Conference on Artificial
Intelligence, 2020.

Chen, I.-C., Liu, H.-S., Sun, W.-F., Chao, C.-H., Hsu, Y.-C.,
and Lee, C.-Y. Retraining-free merging of sparse moe via
hierarchical clustering, 2025a.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto,
H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N.,
Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov,
M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray,
S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian,
M., Winter, C., Tillet, P., Such, F. P., Cummings, D.,
Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A.,
Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang,
J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W.,
Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra,
V., Morikawa, E., Radford, A., Knight, M., Brundage,
M., Murati, M., Mayer, K., Welinder, P., McGrew, B.,
Amodei, D., McCandlish, S., Sutskever, I., and Zaremba,
W. Evaluating large language models trained on code.
2021.

References

Anonymous. Compressing large moe models via efficient
pruning and data-aware calibration. In Submitted to The
Fourteenth International Conference on Learning Representations,
2025a. URL https://openreview.
net/forum?id=dJy6z9peC7. under review.

Chen, X., Ha, M., Lan, Z., Zhang, J., and Li, J. Mobe:
Mixture-of-basis-experts for compressing moe-based
llms, 2025b. URL https://arxiv.org/abs/
2508.05257.

Anonymous. Drop or merge? hybrid moe LLMs compressors
via metric-driven adaptive allocation. In Submitted
to The Fourteenth International Conference on
Learning Representations, 2025b. URL https://
openreview.net/forum?id=9ps2joMieU. under
review.

Chen, Y., Shao, Y., Wang, P., and Cheng, J. EACMoE:
Expert-selection aware compressor for mixture-ofexperts
large language models. In Che, W., Nabende,
J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
12942–12963, Vienna, Austria, July 2025c. Association
for Computational Linguistics. ISBN 979-8-89176-251-
0. doi: 10.18653/v1/2025.acl-long.633. URL https:
//aclanthology.org/2025.acl-long.633/.

Anonymous. TD-moe: Tensor decomposition for

moe models. In Submitted to The Fourteenth International
Conference on Learning Representations,
2025c. URL https://openreview.net/forum?
id=D9cnZNZfxX. under review.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try arc, the ai2 reasoning challenge.
ArXiv, abs/1803.05457, 2018.

Anonymous. Towards global expert-level mixedprecision
quantization for mixture-of-experts LLMs,
2026. URL https://openreview.net/forum?
id=wAc718O8UM.

Cobbe, K., Kosaraju, V., Bavarian, M., Hilton, J., Nakano,
R., Hesse, C., and Schulman, J. Training verifiers to solve
math word problems, 2021.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski,

H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al.
Program synthesis with large language models. arXiv
preprint arXiv:2108.07732, 2021.

Dodge, J., Sap, M., Marasovic, A., Agnew, W., Ilharco, ´

G., Groeneveld, D., Mitchell, M., and Gardner, M.
Documenting large webtext corpora: A case study on
the colossal clean crawled corpus. In Moens, M.-F.,

Bai, S., Li, H., Zhang, J., Hong, Z., and Guo, S. Diep:

Adaptive mixture-of-experts compression through differ9
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Huang, X., Specia, L., and Yih, S. W.-t. (eds.), Proceedings
of the 2021 Conference on Empirical Methods
in Natural Language Processing, pp. 1286–1305, Online
and Punta Cana, Dominican Republic, November
2021. Association for Computational Linguistics. doi:
10.18653/v1/2021.emnlp-main.98. URL https://
aclanthology.org/2021.emnlp-main.98/.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,
Song, D., and Steinhardt, J. Measuring massive multitask
language understanding. Proceedings of the International
Conference on Learning Representations (ICLR), 2021a.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring
mathematical problem solving with the math dataset. Advances
in neural information processing systems, 2021b.

Dong, Z., Peng, H., Liu, P., Zhao, W. X., Wu, D., Xiao, F.,

and Wang, Z. Domain-specific pruning of large mixtureof-experts
models with few-shot demonstrations, 2025.
URL https://arxiv.org/abs/2504.06792.

Jaiswal, A., Wang, J., Li, Y., Li, P., Chen, T., Wang, Z.,
Wang, C., Pang, R., and Du, X. Finding fantastic experts
in moes: A unified study for expert dropping strategies
and observations, 2025. URL https://arxiv.org/
abs/2504.05586.

Fan, A., Jernite, Y., Perez, E., Grangier, D., Weston, J.,

and Auli, M. ELI5: Long form question answering.
In Korhonen, A., Traum, D., and Marquez, L. (eds.), `
Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pp. 3558–3567,
Florence, Italy, July 2019. Association for Computational
Linguistics. doi: 10.18653/v1/P19-1346. URL
https://aclanthology.org/P19-1346/.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary,
B., Bamford, C., Chaplot, D. S., de las Casas, D., Hanna,
E. B., Bressand, F., Lengyel, G., Bour, G., Lample, G.,
Lavaud, L. R., Saulnier, L., Lachaux, M.-A., Stock, P.,
Subramanian, S., Yang, S., Antoniak, S., Scao, T. L.,
Gervet, T., Lavril, T., Wang, T., Lacroix, T., and Sayed,
W. E. Mixtral of experts, 2024. URL https://arxiv.
org/abs/2401.04088.

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers:

Scaling to trillion parameter models with simple and efficient
sparsity, 2022. URL https://arxiv.org/
abs/2101.03961.

Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., and
Szolovits, P. What disease does this patient have? a
large-scale open domain question answering dataset from
medical exams. 2020. URL https://arxiv.org/
abs/2009.13081.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi,
A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li,
H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,
J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.
The language model evaluation harness, 07 2024. URL
https://zenodo.org/records/12608602.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J.,
and Amodei, D. Scaling laws for neural language models,
2020. URL https://arxiv.org/abs/2001.
08361.

Gu, H., Li, W., Li, L., Qiyuan, Z., Lee, M. G., Sun, S.,

Xue, W., and Guo, Y. Delta decompression for moebased
LLMs compression. In Forty-second International
Conference on Machine Learning, 2025. URL https:
//openreview.net/forum?id=ziezViPoN1.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu,

C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient
memory management for large language model
serving with pagedattention, 2023. URL https://
arxiv.org/abs/2309.06180.

He, S., Dong, D., Ding, L., and Li, A. Towards efficient

mixture of experts: A holistic study of compression techniques,
2025a. URL https://arxiv.org/abs/
2406.02500.

Kydlicek, H., Lozovskaya, A., Habib, N., and Fourrier, C.
Fixing open llm leaderboard with math-verify, 2025.

He, S., Ge, T., Sun, G., Tian, B., Wang, X., and Yu, D.
Router-tuning: A simple and effective approach for dynamic
depth. In Christodoulopoulos, C., Chakraborty, T.,
Rose, C., and Peng, V. (eds.), Proceedings of the 2025
Conference on Empirical Methods in Natural Language
Processing, pp. 1925–1938, Suzhou, China, November
2025b. Association for Computational Linguistics. ISBN
979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.
99. URL https://aclanthology.org/2025.
emnlp-main.99/.

Lasby, M., Lazarevich, I., Sinnadurai, N., Lie, S., Ioannou,

Y., and Thangarasa, V. Reap the experts: Why pruning
prevails for one-shot moe compression, 2025. URL
https://arxiv.org/abs/2510.13999.

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y.,
Krikun, M., Shazeer, N., and Chen, Z. Gshard: Scaling
giant models with conditional computation and automatic
sharding, 2020. URL https://arxiv.org/abs/
2006.16668.

10
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E.,
Michalewski, H., Ramasesh, V., Slone, A., Anil, C.,
Schlag, I., Gutman-Solo, T., et al. Solving quantitative
reasoning problems with language models. Advances in
neural information processing systems, 35:3843–3857,
2022.

of moe models via expert output merging, 2025. URL
https://arxiv.org/abs/2510.14436.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can
a suit of armor conduct electricity? a new dataset for
open book question answering, 2018. URL https://
arxiv.org/abs/1809.02789.

Li, P., Zhang, Z., Yadav, P., Sung, Y.-L., Cheng, Y., Bansal,
M., and Chen, T. Merge, then compress: Demystify efficient
SMoe with hints from its routing policy. In The
Twelfth International Conference on Learning Representations,
2024. URL https://openreview.net/
forum?id=eFWG9Cy3WK.

OpenAI. Introducing gpt-5. https://openai.com/
index/introducing-gpt-5/, 2025. OpenAI
Blog.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. Medmcqa:
A large-scale multi-subject multi-choice dataset
for medical domain question answering. In Flores, G.,
Chen, G. H., Pollard, T., Ho, J. C., and Naumann, T.
(eds.), Proceedings of the Conference on Health, Inference,
and Learning, volume 174 of Proceedings of
Machine Learning Research, pp. 248–260. PMLR, 07–
08 Apr 2022. URL https://proceedings.mlr.
press/v174/pal22a.html.

Li, W., Li, L., Gu, H., Huang, Y.-L., Lee, M. G., Sun, S.,
Xue, W., and Guo, Y. MoE-SVD: Structured mixture-ofexperts
LLMs compression via singular value decomposition.
In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S.,
Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J.
(eds.), Proceedings of the 42nd International Conference
on Machine Learning, volume 267 of Proceedings of
Machine Learning Research, pp. 35209–35230. PMLR,
13–19 Jul 2025. URL https://proceedings.mlr.
press/v267/li25az.html.

Reddy, S., Chen, D., and Manning, C. D. CoQA: A
conversational question answering challenge. Transactions
of the Association for Computational Linguistics,
7:249–266, 2019. doi: 10.1162/tacl a 00266. URL
https://aclanthology.org/Q19-1016/.

Liu, E., Zhu, J., Lin, Z., Ning, X., Blaschko, M. B., Yan, S.,
Dai, G., Yang, H., and Wang, Y. Efficient expert pruning
for sparse mixture-of-experts language models: Enhancing
performance and reducing inference costs, 2024. URL
https://arxiv.org/abs/2407.00945.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.
Winogrande: An adversarial winograd schema challenge
at scale, 2019. URL https://arxiv.org/abs/
1907.10641.

Liu, Z., Wu, H., She, R., Fu, X., Han, X., Zhong,
T., and Yuan, M. Molae: Mixture of latent experts
for parameter-efficient language models, 2025. URL
https://arxiv.org/abs/2503.23100.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q.,
Hinton, G., and Dean, J. Outrageously large neural networks:
The sparsely-gated mixture-of-experts layer, 2017.
URL https://arxiv.org/abs/1701.06538.

Lu, X., Liu, Q., Xu, Y., Zhou, A., Huang, S., Zhang, B.,
Yan, J., and Li, H. Not all experts are equal: Efficient
expert pruning and skipping for mixture-of-experts large
language models. In Ku, L.-W., Martins, A., and Srikumar,
V. (eds.), Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume
1: Long Papers), pp. 6159–6172, Bangkok, Thailand,
August 2024. Association for Computational Linguistics.
doi: 10.18653/v1/2024.acl-long.334. URL https:
//aclanthology.org/2024.acl-long.334/.

Suzgun, M., Scales, N., Scharli, N., Gehrmann, S., Tay, ¨
Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H.,
Zhou, D., , and Wei, J. Challenging big-bench tasks and
whether chain-of-thought can solve them. arXiv preprint
arXiv:2210.09261, 2022.

Team, Q. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.

Matena, M. and Raffel, C. Merging models with fisherweighted
averaging, 2022. URL https://arxiv.
org/abs/2111.09832.

Veeraboina, H. Aime problem set 1983-
2024, 2024. URL https://www.kaggle.
com/datasets/hemishveeraboina/
aime-problem-set-1983-2024.

math ai. Aime problem set 2025, 2025. URL
https://huggingface.co/datasets/
math-ai/aime25.

Vendrow, J., Vendrow, E., Beery, S., and Madry, A. Do large
language model benchmarks test reliability?, 2025. URL
https://arxiv.org/abs/2502.03461.

Miao, R., Yao, Y., Wang, Z., Wang, Z., Yi, B., Liu, L.,
Zhao, Y., and Yang, T. Mergemoe: Efficient compression

11
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter,
B., Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-ofthought
prompting elicits reasoning in large language
models, 2023. URL https://arxiv.org/abs/
2201.11903.

2025, pp. 15169–15186, Suzhou, China, November 2025.
Association for Computational Linguistics. ISBN 979-
8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.
820. URL https://aclanthology.org/2025.
findings-emnlp.820/.

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R.,
Gontijo-Lopes, R., Morcos, A. S., Namkoong, H.,
Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt,
L. Model soups: averaging weights of multiple finetuned
models improves accuracy without increasing inference
time, 2022. URL https://arxiv.org/abs/
2203.05482.

Yang, C., Sui, Y., Xiao, J., Huang, L., Gong, Y., Duan,
Y., Jia, W., Yin, M., Cheng, Y., and Yuan, B. MoE-i2
:

Compressing mixture of experts models through interexpert
pruning and intra-expert low-rank decomposition.
In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.),
Findings of the Association for Computational Linguistics:
EMNLP 2024, pp. 10456–10466, Miami, Florida,
USA, November 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.findings-emnlp.
612. URL https://aclanthology.org/2024.
findings-emnlp.612/.

Yang, X., Tian, Y., and Song, Y. Moe pathfinder: Trajectorydriven
expert pruning, 2025. URL https://arxiv.
org/abs/2512.18425.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,
Y. Hellaswag: Can a machine really finish your sentence?
In Proceedings of the 57th Annual Meeting of the
Association for Computational Linguistics, 2019.

Zhang, D., Ma, X., Ni, Z., Wu, Z., Shu, H., Jiang, X.,
and Chen, X. Expert merging: Model merging with
unsupervised expert alignment and importance-guided
layer chunking, 2025a. URL https://arxiv.org/
abs/2509.25712.

Zhang, G., Han, Y., Lou, Y., Zhao, W., Zhang, Y., and You,
Y. Mone: Replacing redundant experts with lightweight
novices for structured pruning of moe, 2025b. URL
https://arxiv.org/abs/2507.00390.

Zhao, Y., Wang, Z., and Zhang, M. Puzzlemoe: Efficient
compression of large mixture-of-experts models
via sparse expert merging and bit-packed inference, 2025.
URL https://arxiv.org/abs/2511.04805.

Zhou, Y., Zhao, Z., Cheng, D., Wu, Z., Gui, J., Yang, Y.,
Wu, F., Cheng, Y., and Fan, H. Dropping experts, recombining
neurons: Retraining-free pruning for sparse
mixture-of-experts LLMs. In Christodoulopoulos, C.,
Chakraborty, T., Rose, C., and Peng, V. (eds.), Findings of
the Association for Computational Linguistics: EMNLP

12
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

A. Original MoE (before Compression)

In the original MoE model before compression, an input (x) first passes through the gate network, producing n expert
activation scores. Depending on the implementation, the gate network outputs either the raw logits or the probabilities after
a softmax operation. Let these n expert activation scores be denoted as (g0, . . . , gn). Likewise, let the n experts in the same
layer as the gate network be represented as (E0, . . . , En).

Assume that this MoE model activates the top-k experts. In this case, the computation for the input x is performed as
follows.

S ⊂ {0, 1, . . . , n − 1}, |S| = k, k < n

We define the renormalized expert activation weights as

g˜i = P
gi
j∈S gj
, i ∈ S

Using these normalized weights, the output of the MoE layer for input (x) is computed as:

y =
X
i∈S
g˜i
· Ei(x).

B. Expert Pruning (N → N − α)

When the original MoE model undergoes pruning, the inference process can fall into one of the following three scenarios:

1. Best scenario: All experts selected by the original model remain available after pruning, and the same experts are used
without any change.

2. Most common scenario: Pruning is imperfect, and while some originally selected experts remain available and are used
as before, others are removed. For the dropped experts, the model must instead rely on alternative experts as substitutes.

3. Worst scenario: All experts that were originally selected are pruned out, forcing the model to replace every originally
chosen expert with different ones.

Let

• S be the set of expert indices selected by the original MoE model,

• P ⊆ {0, . . . , n − 1} be the set of experts that remain after pruning, and |S| ≤ |P|.

′ be the set of experts effectively used by the pruned model.

• S

′
.

We analyze the relationship between S, P and S

Gate scores before and after pruning. We denote by g
orig
i
(x) the expert activation scores produced by the original gate
network for an input x, and by g
pruned
i
(x) the scores produced when running the pruned model end-to-end. For any index
set A, we define the corresponding renormalized weights as

(x) = g
orig
i
(x)
P
j∈A g
orig
j
(x)

g˜
orig,(A)
i

(x) = g
pruned
i
(x)
P
j∈A g
pruned
j
(x)
, i ∈ A.

g˜
pruned,(A)
i

13
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

B.1. Best Scenario

In the best scenario, all originally selected experts remain after pruning:

S ⊆ P =⇒ S′ = S.

The original and pruned MoE outputs for input x can then be written as

yorig(x) = X

g˜
orig,(S)
i

(x) Ei(x)

i∈S

g˜
pruned,(S)
i

y
best
pruned(x) = X

(x) Ei(x).

i∈S

The difference between the original and pruned MoE outputs in the best scenario is then given by:



yorig(x) − y

best
pruned(x)


 =



(x) Ei(x)

g˜
orig,(S)
i

g˜
pruned,(S)
i

X

X

(x) Ei(x) −

=

i∈S

i∈S



(x)

Ei(x)


g˜
orig,(S)
i

X

(x) − g˜
pruned,(S)
i

i∈S

In this scenario, the set of active experts is identical to that of the original model. However, note that even if the selected

experts match exactly pre- and post-pruning, the router’s output values are unlikely to be identical. Because MoE LLMs are
multi-layered, for the router outputs to be perfectly identical, this best-case scenario must be satisfied in every layer, which

is statistically improbable. Therefore, it can be trivially stated that g˜
orig,(A)
i

(x) ̸= ˜g
pruned,(A)
i

(x)

B.2. Most Common Scenario (Partial Overlap)

Some selected experts survive, but some are removed:

∅ ̸= S ∩ P ̸= S.

Let the surviving experts be:

T = S ∩ P.

Let the dropped experts be:

D = S \ P.

Then the pruned model must select replacement experts for D, forming:

S

′ = T ∪ R where R ⊆ P \ S.

In this case, the original MoE output is

yorig(x) = X

g˜
orig,(S)
i

(x)Ei(x)

i∈S

while the pruned model output decomposes as

g˜
pruned,(S
′
)
i

g˜
pruned,(S
′
)
i

y
common
pruned (x) = X

(x)Ei(x) + X

(x)Ei(x)

i∈T

i∈R

where the first term corresponds to the experts shared with the original model and the second term to the substituted experts.

14
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

B.3. Worst Scenario

All originally selected experts are pruned out:

S ∩ P = ∅.

Thus, all experts must be substituted:

′ ⊆ P, |S′

′ ∩ S = ∅.

S

| = k, S

In this case, the original MoE output is

yorig(x) = X

g˜
orig,(S)
i

(x)Ei(x)

i∈S

while the pruned model output becomes

g˜
pruned,(S
′
)
i

y
worst
pruned(x) = X

(x)Ei(x)

i∈S′

B.4. Difference between original and pruned MoE outputs

B.4.1. BEST SCENARIO

′ = S), the output discrepancy is not zero. This

In this scenario, although the set of selected experts remains identical (S

error is solely attributable to the deviation in normalized router scores (g˜
orig vs. g˜
pruned). It highlights that even in the ideal

case of expert retention, router calibration is necessary to align the weighting distribution.



yorig(x) − y

best
pruned(x)


 =





X

g˜
orig,(S)
i

X

g˜
pruned,(S)
i

(x) Ei(x) −

(x) Ei(x)

=

i∈S

i∈S






g˜
orig,(S)
i

(x)

Ei(x)

X

(x) − g˜
pruned,(S)
i

i∈S

B.4.2. MOST COMMON SCENARIO (PARTIAL OVERLAP)

Here, the total output difference decomposes into three distinct components:

• Weight Shift (T ): The discrepancy caused by altered router weights on the shared experts.

• Information Loss (D): The contribution of the original experts that were dropped, representing missing knowledge.

• Substitution Noise (R): The impact of newly activated experts that were not selected by the original model, potentially

introducing distributional shifts.

This formulation clearly shows that the error is driven not just by which experts are lost, but also by how the router

re-distributes probability mass among the remaining and new experts.



yorig(x) − y

common
pruned (x)


 =



 X

!

g˜
pruned,(S
′
)
i

g˜
pruned,(S
′
)
i

X

g˜
orig,(S)
i

(x) Ei(x) + X

(x) Ei(x) −

(x) Ei(x)

=

i∈S

i∈T

i∈R






g˜
orig,(S)
i

(x) − g˜
pruned,(S
′
)
i

(x)

g˜
pruned,(S
′
)
i

X

Ei(x) +X

g˜
orig,(S)
i

X

(x) Ei(x) −

(x) Ei(x)

i∈T

i∈D

i∈R

15
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

B.4.3. WORST SCENARIO

In the worst-case scenario, the sets of active experts are disjoint (S ∩ S′ = ∅). Consequently, the model completely loses the

original computational path and relies entirely on a different set of experts. This results in the maximum divergence, as the
model fails to utilize any of the originally intended parameter knowledge.



yorig(x) − y

worst
pruned(x)


 =



(x) Ei(x)

g˜
pruned,(S
′
)
i

g˜
orig,(S)
i

X

X

(x) Ei(x) −

i∈S

i∈S′

′
)

C. Expert Editing (N → N, Parameters P → P

Unlike pruning, Expert Editing preserves the total number of experts. However, as the parameters in preceding layers are

modified, the router’s computation results may shift. Consequently, for the same input, the edited model may select different
experts compared to the original model. The inference process can fall into one of the following three scenarios:

1. Best scenario: The router’s selection remains completely unchanged after editing, meaning the exact same experts chosen
by the original model are utilized.

2. Most common scenario: Due to the router’s altered expert activation scores following the editing process, the selection is
partially changed. While some originally selected experts are retained and used as before, others are replaced by different

experts.

3. Worst scenario: The router’s output shifts drastically such that none of the originally selected experts are chosen, and a

completely different set of experts is activated.

Let

• S be the set of expert indices selected by the original MoE model,

edit ⊆ {0, . . . , n − 1} be the set of expert indices selected by the edited MoE model, and |S| = |Sedit|.

• S

• Ei denote the original experts, and Xi denote the corresponding edited experts obtained by modifying Ei

.

Gate scores before and after editing. As before, we denote by g
orig
i
(x) the expert activation scores produced by the
original gate network for an input x. We denote by g
edit
i
(x) the activation scores produced when running the edited MoE
model end-to-end on the same input. For any index set A, we define the renormalized weights for the edited model as

(x) = g
edit
i
(x)
P
j∈A g
edit
j
(x)
, i ∈ A.

g˜
edit,(A)
i

C.1. Best Scenario

In the best scenario, the router’s top-k selection remains unchanged even after expert editing, i.e.,

S
edit = S.

The original MoE output for input x is

yorig(x) = X

g˜
orig,(S)
i

(x) Ei(x),

i∈S

while the edited MoE output in the best scenario becomes

y
best
edit (x) = X

g˜
edit,(S)
i

(x) Xi(x).

i∈S

Even though the index set of selected experts is identical, both the gate scores and the expert functions may differ between

best
edit (x).

yorig(x) and y

16
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

The difference between the original and edited MoE outputs in the best scenario is then given by:

best
edit (x)


 =



yorig(x) − y





g˜
orig,(S)
i

g˜
edit,(S)
i

X

X

(x) Ei(x) −

(x) Xi(x)

i∈S

i∈S



(x) Xi(x)


g˜
orig,(S)
i

(x) Ei(x) − g˜
edit,(S)
i

X

i∈S

Even if sophisticated mathematical approximation techniques render Ei and Xi nearly identical, they are not strictly

identical; thus, g
orig
i
and g
edit
i

cannot be equal. Given that each expert is a matrix containing over 1 million parameters, even

slight deviations in router outputs—despite the similarity between Ei and Xi—inevitably lead to non-negligible differences

in the final output.

C.2. Most Common Scenario (Partial Overlap)

In this scenario, the router’s selection is partially preserved but also altered due to the editing process. We define the sets of

shared (T ), dropped (D), and newly introduced (R) experts as:

T = S ∩ Sedit

, D = S \ Sedit

edit \ S

, R = S

edit = T ∪ R. In this case, the original MoE output is

The effective set of experts for the edited model is formed by S

g˜
orig,(S)
i

yorig(x) = X

(x)Ei(x)

i∈S

while the edited model output decomposes as

g˜
edit,(S
edit)
i

g˜
edit,(S
edit)
i

y
common
edit (x) = X

(x)Xi(x) + X

(x)Xi(x)

i∈T

i∈R

where the first term represents the edited versions of the originally selected experts, and the second term represents the

newly activated experts.

C.3. Worst Scenario

In the worst-case scenario, the router’s behavior shifts drastically such that there is no overlap between the original and

edited selections (S ∩ Sedit = ∅). Consequently, the model relies entirely on a disjoint set of experts. The original output is

yorig(x) = X

g˜
orig,(S)
i

(x)Ei(x)

i∈S

whereas the edited model output becomes

g˜
edit,(S
edit)
i

y
worst
edit (x) = X

(x)Xi(x)

i∈Sedit

Here, both the selected expert indices and the underlying expert parameters differ entirely from the original model.

17
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

C.4. Difference between original and edited MoE outputs

C.4.1. BEST SCENARIO

edit = S), the discrepancy arises

Even in the best-case scenario where the router selects the exact same expert indices (S

from two sources: (1) the approximation error of the experts themselves (Ei → Xi) and (2) the shift in router weights
(g˜
orig → g˜
edit).

best
edit (x)



yorig(x) − y





g˜
orig,(S)
i

g˜
edit,(S)
i

X

X

=

(x) Ei(x) −

(x) Xi(x)

i∈S

i∈S






g˜
orig,(S)
i

(x) Ei(x) − g˜
edit,(S)
i

X

=

(x) Xi(x)

i∈S

C.4.2. MOST COMMON SCENARIO (PARTIAL OVERLAP)

The total output difference decomposes into three components, reflecting both structural changes in the router and parameter

approximation in the experts:

• Approximation & Weight Shift (T ): The compound error arising from the parameter compression of shared experts

(Ei → Xi) and the deviation in their router weights.

• Information Loss (D): The loss of knowledge from the original experts (Ei) that were dropped.

• Substitution Noise (R): The impact of newly introduced, compressed experts (Xi) that were not originally selected.

common
edit (x)∥

∥yorig(x) − y



 X

!

g˜
edit,(S
edit)
i

g˜
edit,(S
edit)
i

X

g˜
orig,(S)
i

(x) Xi(x) + X

=

(x) Ei(x) −

(x) Xi(x)

i∈S

i∈T

i∈R






g˜
orig,(S)
i

(x) Ei(x) − g˜
edit,(S
edit)
i


+
X

g˜
edit,(S
edit)
i

X

g˜
orig,(S)
i

X

=

(x) Xi(x)

(x) Ei(x) −

(x) Xi(x)

i∈T

i∈D

i∈R

C.4.3. WORST SCENARIO

With disjoint active sets (S ∩ Sedit = ∅), the model relies entirely on a different set of compressed experts (Xi). This

maximizes divergence as the original parameters are completely abandoned.





g˜
edit,(S
edit)
i

g˜
orig,(S)
i

worst
edit (x)

X

X



yorig(x) − y



 =

(x)Ei(x) −

(x)Xi(x)

i∈S

i∈Sedit

18
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression
D. Expert Merging (N → M, where M < N)
Expert Merging is primarily implemented by replacing the original experts with a merged representation to reduce the
memory footprint. Structurally, multiple original experts are clustered into a single representative expert. For instance, if
Expert 0 and Expert 1 are merged, they share the same parameters in the compressed model, effectively mapping multiple
original indices to a single merged index.
To formalize this, let N be the number of original experts and M be the number of merged experts (M < N). We define a
surjective mapping function ϕ : {1, . . . , N} → {1, . . . , M} that assigns each original expert index i to a merged expert
cluster index c. Consequently, the output of the merged model is computed using the merged parameters Mc:
Mc ≈ Ei
, ∀i such that ϕ(i) = c

D.1. Set Definitions and Output Formulation
Since the original and merged models operate in different index spaces, direct comparison of selected indices is not possible.
We analyze the discrepancy by projecting the original selection into the merged cluster space.
Let S be the set of indices selected by the original router, where |S| = k. We define the Projected Original Set Cproj as the
set of unique clusters required by the original selection:
Cproj = {ϕ(i) | i ∈ S}
Let S
merge be the set of indices selected by the merged model’s router, where |Smerge| = kmerge. We define the intersection
(T ), dropped (D), and newly introduced (R) sets within the cluster space:
• Shared Clusters (T = Cproj ∩ Smerge): Clusters intended by the original model and selected by the merged model.
• Dropped Clusters (D = Cproj \ Smerge): Clusters required by the original input but missed by the merged router.
• Substituted Clusters (R = S
merge \ Cproj): Clusters selected by the merged router that were not implied by the
original selection.

Based on these definitions, we analyze the scenarios below based on the relationship between the number of required clusters
(|Cproj|) and the merged router’s capacity (kmerge).

19
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

D.2. Case 1: All Original Experts Co-located in One Cluster

∗ be that unique cluster

This case occurs when all selected original experts map to a single merged cluster (|Cproj| = 1). Let c

index.

∗
.

Scenario 1.1: Best Case. The merged router selects exactly the cluster c

S
merge = {c
∗
}

y
(1.1)
merge(x) = ˜g
merge
c
∗ (x)Mc
∗ (x)

∗ but also activates irrelevant clusters

Scenario 1.2: Most Common (Partial Noise). The router selects the correct cluster c

(R).

c
∗ ∈ Smerge

merge \ {c
∗
} ̸= ∅

, R = S

y
(1.2)
merge(x) = ˜g
merge
c
∗ (x)Mc

∗ (x) + X

g˜
merge
c
(x)Mc(x)

c∈R

∗
.

Scenario 1.3: Worst Case (Miss). The router completely misses the required cluster c

c
∗ ∈ S /
merge

y
(1.3)
merge(x) = X

g˜
merge
c
(x)Mc(x)

c∈Smerge

D.3. Case 2: Distributed Experts (Within Capacity)

Here, the original experts map to multiple distinct clusters, but the number of required clusters is within the merged router’s

selection capacity (1 < |Cproj| ≤ kmerge).

Scenario 2.1: Best Case (Perfect Match). The router selects exactly the set of projected clusters.

S
merge = Cproj

y
(2.1)
merge(x) = X

g˜
merge
c
(x)Mc(x)

c∈Cproj

Scenario 2.2: Most Common (Partial Overlap). Some correct clusters are selected (T ), some are missed (D), and noise

is added (R).

T = Cproj ∩ Smerge ̸= ∅, D = Cproj \ T ̸= ∅, R = S

merge \ T ̸= ∅

y
(2.2)
merge(x) = X

g˜
merge
c

(x)Mc(x) + X

g˜
merge
c
(x)Mc(x)

c∈T

c∈R

Scenario 2.3: Worst Case (Disjoint). No correct clusters are selected.

Cproj ∩ Smerge = ∅

y
(2.3)
merge(x) = X

g˜
merge
c
(x)Mc(x)

c∈Smerge

D.4. Case 3: Over-Distributed (Capacity Exceeded)

This critical scenario occurs when the original experts are scattered across more clusters than the merged router is allowed to

select (|Cproj| > kmerge). This implies inevitable structural information loss.

20
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Scenario 3.1: Best Possible (Capacity Saturation). Even in the best case, the router can only select a subset of the

required clusters. Let Tmax ⊂ Cproj be the largest possible subset (|Tmax| = kmerge).

merge = Tmax, Dinevitable = Cproj \ Tmax ̸= ∅

S

y
(3.1)
merge(x) = X

g˜
merge
c

(x)Mc(x)

c∈Tmax

Scenario 3.2: Most Common (Sub-optimal Selection). The router selects fewer correct clusters than its capacity allows,

or picks wrong ones.

T = Cproj ∩ Smerge

merge \ T ̸= ∅

, |T | < kmerge, R = S

y
(3.2)
merge(x) = X

(x)Mc(x) + X

g˜
merge
c

g˜
merge
c

(x)Mc(x)

c∈T

c∈R

Scenario 3.3: Worst Case. The router selects clusters completely disjoint from Cproj.

Cproj ∩ Smerge = ∅

y
(3.3)
merge(x) = X

g˜
merge
c

(x)Mc(x)

c∈Smerge

D.5. Difference between original and merged MoE outputs

We analyze the output discrepancy ∥yorig(x) − ymerge(x)∥ across the scenarios defined in previous sections.

CASE 1: ALL ORIGINALLY SELECTED EXPERTS BELONG TO THE SAME CLUSTER

Scenario 1.1: Best Case

⋆

The router correctly identifies the unique cluster c

. The error arises solely from the Merging Approximation.




yorig(x) − y



(1.1)
merge(x)





X

g˜
orig,(S)
i

(x) Ei(x) − g˜
merge
c
∗ (x) Mc

=

∗ (x)

.

i∈S

Scenario 1.2: Most Common (Partial Noise)

The router activates irrelevant clusters (R). The error decomposes into Merging Approximation and Substitution Noise.




yorig(x) − y



(1.2)
merge(x)



!



g˜
orig,(S)
i

X

g˜
merge
c
∗ (x) Mc

∗ (x) + X

g˜
merge
c

(x) Ei(x) −

=

(x) Mc(x)

.

i∈S

c∈R

Scenario 1.3: Worst Case (Miss)

Complete Information Loss of the original knowledge, replaced entirely by Substitution Noise.




yorig(x) − y



(1.3)
merge(x)





X

g˜
orig,(S)
i

X

g˜
merge
c

=

(x) Ei(x) −

(x) Mc(x)

.

i∈S

c∈Smerge

21
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

CASE 2: ORIGINALLY SELECTED EXPERTS SHARE SOME CLUSTERS (MIXED)

Scenario 2.1: Best Case (Perfect Match)

The discrepancy is purely due to Merging Approximation across the active clusters Cproj.




yorig(x) − y



(2.1)
merge(x)





g˜
orig,(S)
i

X

X

g˜
merge
c

=

(x) Ei(x) −

(x) Mc(x)

.

i∈S

c∈Cproj

Scenario 2.2: Most Common (Partial Overlap)

This involves: Merging Approximation (T ), Information Loss (D), and Substitution Noise (R).




yorig(x) − y



(2.2)
merge(x)



!

 X

g˜
orig,(S)
i

X

g˜
merge
c

(x) Mc(x) + X

g˜
merge
c

=

(x) Ei(x) −

(x) Mc(x)

i∈S

c∈T

c∈R









X

g˜
orig
i

X

g˜
merge
c

X

g˜
orig
i

X

g˜
merge
c

=

(x)Ei(x) −

(x)Mc(x)

 +

(x) Ei(x) −

(x) Mc(x)

.



i∈S,ϕ(i)∈T

c∈T

i∈S,ϕ(i)∈D

c∈R

Scenario 2.3: Worst Case (Disjoint)

Maximum divergence due to complete mismatch (T = ∅).




yorig(x) − y



(2.3)
merge(x)





g˜
orig,(S)
i

X

X

g˜
merge
c

=

(x) Ei(x) −

(x) Mc(x)

.

c∈Smerge

i∈S

CASE 3: ALL ORIGINALLY SELECTED EXPERTS BELONG TO DISTINCT CLUSTERS

Scenario 3.1: Best Possible (Capacity Saturation)

Even with optimal selection (Tmax), the model suffers from structural loss (Dinevitable).




yorig(x) − y



(3.1)
merge(x)





X

g˜
orig,(S)
i

X

g˜
merge
c

=

(x) Ei(x) −

(x) Mc(x)

.

i∈S

c∈Tmax

Scenario 3.2: Most Common (Sub-optimal Selection)

Structurally prone to Information Loss (D) due to limited capacity kmerge and sub-optimal selection.




yorig(x) − y



(3.2)
merge(x)



 X

!

X

g˜
orig,(S)
i

(x) Mc(x) + X

g˜
merge
c

g˜
merge
c

=

(x) Ei(x) −

(x) Mc(x)

.

i∈S

c∈T

c∈R

Scenario 3.3: Worst Case

Complete mismatch.




yorig(x) − y



(3.3)
merge(x)





g˜
orig,(S)
i

X

X

g˜
merge
c

(x) Ei(x) −

=

(x) Mc(x)

.

i∈S

c∈Smerge

22
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

E. Related Works

Due to the massive VRAM consumption of Large Language Models (LLMs) based on the Mixture-of-Experts (MoE)
architecture, deploying these models remains challenging, particularly in resource-constrained environments. Consequently,
there is a growing need for MoE compression techniques that minimize the degradation of existing high-performance
models while being retraining-free and computationally efficient. To address this, extensive research has been conducted. In
this paper, we propose a taxonomy for MoE compression, categorizing existing methodologies into three classes: Expert
Pruning, Expert Editing, and Expert Merging. In this work, we focus exclusively on parameter-level compression. We
do not consider bit-level compression of individual parameters (i.e., quantization). In other words, our scope is limited to
methods that reduce the number of parameters efficiently, rather than techniques that optimize the bit representation of each
parameter, which lies outside the boundaries of this study.

E.1. Expert Pruning (N → N − α)

Expert Pruning reduces the total number of experts by permanently removing α out of the total N experts. Since this method
simply drops α experts, the remaining N −α experts are preserved without modification. Consequently, after the parameters
are removed, routing is performed exclusively among the remaining N − α experts. The core challenge of this approach lies
in identifying the importance of each expert; it is based on the hypothesis that certain experts are less important, redundant,
or make negligible contributions to the model’s performance, and thus can be removed without significant degradation.

A wide array of methodologies has been proposed in the field of Expert Pruning. NAEE (Lu et al., 2024) proposes a method
that retains only the combination of experts that minimizes layer-wise reconstruction loss after inferring the MoE LLM
with specific calibration data. DiEP (Bai et al., 2025) approaches the problem by transforming the discrete expert selection
task into a differentiable continuous optimization. REAP (Lasby et al., 2025) demonstrates that expert merging can lead
to functional subspace collapse, resulting in lower performance on text generation tasks compared to pruning; instead, it
introduces a pruning criterion that considers both router gate-values and the magnitude of expert outputs. Another study
(Anonymous, 2025a) mathematically proved that the output discrepancy of the entire model is bounded by the cumulative

sum of layer-wise output discrepancies, thereby proposing a layer-wise search instead of a global search. Furthermore,
EASY-EP (Dong et al., 2025), observing that experts in large-scale MoE models are highly specialized and identifiable with

limited data, proposes an effective domain-specific pruning method based on expert output magnitude and token variation.
MoE Pathfinder (Yang et al., 2025) evaluates global importance based on the ‘expert activation trajectory’ across all layers to
perform pruning. Beyond these specific algorithms, a comprehensive set of evaluation criteria, MC-Suite, was suggested to
determine the optimal experts for removal (Jaiswal et al., 2025), and another study has explored more aggressive approaches,
such as removing entire MoE layers or even transformer blocks rather than individual experts (He et al., 2025a).

′

E.2. Expert Editing (N → N, Parameters P → P

)

Expert Editing maintains the total number of experts N but reduces the total number of parameters from P to P
′ by
compressing the internal structure of each expert and adjusting the computation order. This approach is grounded in the
hypothesis that the weight matrices (W) within experts are over-parameterized and can be approximated by matrices with

fewer parameters. The core mechanism involves mathematically decomposing or re-parameterizing the expert matrices.
Techniques such as Singular Value Decomposition (SVD), Rank Decomposition, and Tucker Decomposition are employed,
and the operation order of the decomposed matrices may be rearranged to maximize efficiency or performance. Furthermore,
some approaches attempt to replace expert matrices with smaller, lighter structures, such as vectors.

Expert Editing has recently gained significant attention, with various methodologies being actively proposed. MoE-SVD (Li
et al., 2025) applies SVD to each expert’s weight matrix, demonstrating that it is possible to effectively compress experts and
reduce parameters while preserving performance. MoLAE (Liu et al., 2025) proposes factorizing an expert into two matrices
(A and B) using SVD; it designates B as a latent mapping matrix shared across all experts (or expert groups), and A as
an expert-specific transformation matrix that operates within the low-dimensional space. Building on this, MoBE (Chen
et al., 2025b) utilizes rank decomposition to decompose experts into a unique matrix A and a shared linear combination of
basis matrices. It optimizes these components to minimize the reconstruction error with the original weight matrix, thereby
achieving effective compression and performance preservation. Additionally, TD-MoE (Anonymous, 2025c) treats MoE
experts not as independent matrices but as correlated tensors. It aligns the data distribution via whitening and then applies
Tucker Decomposition, aiming to capture the correlations among experts through Joint Tensor Decomposition.

23
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

E.3. Expert Merging (N → M, where M < N)

Expert merging reduces the total number of experts from N to a smaller number M by combining functionally similar

experts into synthesized experts. It is motivated by model merging hypotheses (Wortsman et al., 2022; Matena & Raffel,
2022), which posit that when experts exhibit redundancy, their knowledge can be fused to preserve most of the collective
capability of the original set. This process often involves (i) identifying redundancy via clustering or similarity matching
(e.g., hierarchical clustering or k-means) and (ii) fusing experts via parameter- or output-level mechanisms (e.g., weighted
averaging or learned linear combinations).

Various methodologies have been proposed for MoE expert merging. HC-SMoE (Chen et al., 2025a) reduces parameters by
performing hierarchical clustering based on each expert’s output similarity on calibration data, thereby merging similar
experts in a retraining-free manner. PuzzleMoE (Zhao et al., 2025) is a training-free approach that targets expert pairs by
constructing a dual-mask consisting of an entry-wise similarity mask and an activation-weighted saliency mask, enabling
selective merging of redundant parameters while preserving expert-specific knowledge. MergeMoE (Miao et al., 2025)
reinterprets MoE expert merging from the perspective of output merging rather than parameter averaging, and computes a
compression (dimensionality reduction) matrix via least squares based on sample inputs. Complementary to MoE expert
merging, the Expert Merging method (Zhang et al., 2025a) studies merging multiple domain experts (SFT models) by
learning layer-wise (and importance-guided chunk-wise) coefficients from unlabeled calibration data to align hidden states
and logits across experts.

E.4. Hybrid Approaches

Methodologies that combine the three aforementioned approaches are also being actively proposed. These strategies include
merging after pruning, editing after pruning, and editing after merging.

DM-MoE (Anonymous, 2025b) proposed a ‘drop-then-merge’ hybrid MoE compression method that first prunes (drops)
redundant experts and subsequently merges the remaining experts using a graph-based approach. DERN (Zhou et al., 2025)
compresses MoE models without retraining by pruning redundant experts based on router statistics, decomposing and

reallocating the pruned experts into neuron-level segments, and finally merging these segments via clustering. EEP (Liu et al.,
2024) introduced a method that reduces inference costs while maintaining or improving performance without retraining; it
utilizes a gradient-free evolutionary strategy to prune MoE experts and preserves knowledge through expert merging.

MoNE (Zhang et al., 2025b) proposes a pruning-and-editing approach that prunes redundant experts and replaces them
by editing them into lightweight, input-agnostic ‘Novice’ vectors. MoE-I2 (Yang et al., 2024) suggests a two-stage
(Pruning+Decomposition) framework to compress MoE models. It performs non-uniform Inter-Expert Pruning based on
layer/expert importance analysis, further compresses the remaining experts via non-uniform Low-Rank Decomposition, and
recovers performance using LoRA fine-tuning. D2
-MoE (Gu et al., 2025) proposes a delta-based compression method that
constructs a shared base weight via Fisher-weighted merging and stores the difference of each expert as a low-rank edit
using truncation-aware SVD. MC-SMoE (Li et al., 2024) employs M-SMoE to capture expert redundancy using routing
policy statistics, align neurons via permutation alignment, and integrate experts through activation frequency-based weighted
averaging. Subsequently, it further compresses the integrated experts via low-rank decomposition to maximize memory and
parameter efficiency.

Since each individual method possesses distinct advantages and limitations, hybridizing two or more of these approaches
represents a promising direction for future research.

E.5. Importance Of Router

MoE compression is often framed as modifying expert parameters (e.g., pruning, merging, editing, or quantization), but
recent studies consistently highlight that the router is a disproportionately high-leverage component: even small perturbations
to expert weights or numerics can shift expert outputs, which in turn changes token-to-expert assignments and cascades into
larger performance drops. This phenomenon is particularly evident in low-bit quantization. EAC-MoE attributes a major
failure mode of quantized MoEs to expert-shift—a distortion of expert selection after quantization—and proposes router
calibration to mitigate the cumulative accumulation of expert selection shift across layers, using a TopK-MSE objective that
focuses alignment on the experts most likely to be selected (Chen et al., 2025c). Similarly, GEMQ observes that quantization
substantially distorts router behavior and shows that merely reusing full-precision router signals is insufficient; instead, a
lightweight global router fine-tuning step (updating only router parameters) provides substantial gains in perplexity and

24
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

downstream accuracy under low-bit regimes (Anonymous, 2026) . Beyond quantization, Router-Tuning further demonstrates
that training only a lightweight routing module can effectively steer computation by deciding when to skip modules, and the
approach can be deployed on MoE backbones as well, reinforcing that routing adaptation alone can be impactful (He et al.,
2025b).

While these works mainly focus on bit-level compression (mixed-precision quantization) or compute-efficient routing, our
setting is complementary: we study parameter-level MoE compression (e.g., expert pruning, merging, and editing), where
experts are structurally modified or removed. In this regime, the pre-trained router is often no longer well-matched to
the post-compression expert set, making router–expert mismatch a key driver of performance degradation. Our findings
therefore support a broader conclusion: effective MoE compression requires not only modifying experts, but also calibrating
the router to remain consistent with the altered expert landscape.

F. Experiment Settings

F.1. Common protocol in Qwen3.

For all experiments on the Qwen3 backbone, we followed the default configurations provided by each baseline’s official code
release as closely as possible. We only introduced minimal, unavoidable changes to accommodate our server constraints
(e.g., the number of GPUs and distributed/runtime configuration), while keeping all algorithmic hyperparameters identical to
the defaults. Unless otherwise stated, all compression baselines were executed in a one-shot manner (i.e., without additional
post-hoc fine-tuning), and we used a uniform expert retention rate of 62.5% for Qwen3 (reducing experts from 128 to 80 per
MoE layer) to ensure fair comparison.

REAP (Expert Pruning). We used the authors’ official implementation of REAP and kept all pruning-related hyperparameters
at their default values. For calibration, we adopted the same calibration data recipe used in the REAP paper: a
50/50 mixture of allenai/c4 and theblackcat102/evol-codealpaca-v1. All pruning runs for REAP were
performed on two NVIDIA A100 40GB GPUs.

CFES (Expert Pruning). We used the authors’ official implementation of CFES, and followed the default settings except
for unavoidable runtime/distributed configurations. For calibration, we adopted the same calibration dataset composition
used in the CFES paper: rstar coder, openr1 math220, and c4 (using the official sampling/preprocessing routine).
(These correspond to code, mathematical reasoning, and knowledge domains, respectively.) All pruning runs were conducted
on two NVIDIA A100 40GB GPUs. The main hyperparameters are as follows:

• batch size=32

• num routed expert=80

• feature form=segment mean

• metric name=l2

• max length=4096

• prune method=c2f

MoBE (Expert Editing). We used the authors’ official implementation of MoBE and adopted the same optimization
setup as the default configuration, except for the following explicitly specified hyperparameters. We set the number of basis
matrices to m=8 (i.e., Basis(Num B)=8) and compressed the model using a single NVIDIA A100 40GB GPU. The
MoBE factorization was trained with:

• --num epochs 10000

• --batch size 32

• --num batches 4

• --learning rate 0.07

25
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression
• --activation "silu"
• --truncation 768

All remaining options (including preprocessing and layer-wise handling) followed the official defaults.
HC-SMoE (Expert Merging). We used the authors’ official implementation of HC-SMoE and kept all hyperparameters
at their default values, except for the following settings explicitly chosen to match our experimental protocol. All merging
runs were performed on eight NVIDIA A100 40GB GPUs. We did not use dominant experts (DOMINANT="no"),
computed expert similarity based on expert outputs (SIM BASE="expert-output"), and used zipit as the default
merge method (MERGE METHOD="zipit"). We ran HC-SMoE in normal mode (MODE="normal") with the following
configuration:

• Number SENTENCES=4, max block size=2048, TRAIN Batch Size=2
• START LAYER=0, GROUP LIMIT=4
• CLUSTER METHOD="hierarchical",
• LINKAGE METHOD="average"
• STOP METRIC="silhouette"
• INGREDIENT="act"

All other implementation details (including calibration-data sampling and merging pipeline internals) followed the official
defaults.

TD-MoE (Expert Editing; Tensor Decomposition). We used the official TD-MoE implementation and followed its
default configuration unless explicitly stated. For Qwen3-30B-A3B-Instruct-2507, we used c4 as the calibration dataset and
applied a global clustering strategy (CLUSTER TYPE="global"). We set the target compression ratio to RATIO=0.4
and compressed the following MoE layers:

• LAYERS TO COMPRESS = {4,5,7,8,9,10,11,13,14,15,18,33,34,35,36,37,38,39}
The TD-MoE decomposition was run with:
• --whitening nsamples 256
• --cluster type "global"
• --model seq len 2048
• --whiten type "output"
• --layers to compress (as above)
• --ratio 0.4
• --decomposition method "svd"

All remaining settings (including data sampling and runtime/distributed configurations) followed the official defaults, with
only minimal adjustments required by our server environment.
26
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression
M-SMoE (Expert Merging). We used the official M-SMoE implementation and kept the default settings unless explicitly
stated. We used c4 as the calibration dataset and performed merging on eight NVIDIA A100 40GB GPUs. We computed
expert similarity based on router logits and used a small calibration subset:
• --similarity base="router-logits"
• --subset ratio=0.01
The remaining key configuration follows:
• block size = 512
• batch size = 1
• num fewshot = 5

All other hyperparameters and the end-to-end merging pipeline followed the official defaults, except for unavoidable
runtime/distributed adjustments to match our hardware constraints.
F.2. Common protocol in Mixtral.

For all experiments on the Mixtral backbone, we followed the default configurations provided by each baseline’s official code
release as closely as possible. We only introduced minimal, unavoidable changes to accommodate our server constraints
(e.g., number of GPUs and runtime/distributed settings), while keeping the algorithmic hyperparameters unchanged unless
explicitly specified below. Unless otherwise stated, all compression baselines were executed in a one-shot manner (i.e.,
without additional post-hoc fine-tuning). For pruning/merging-style baselines on Mixtral, we used a uniform expert retention
rate of 62.5% (reducing experts from 8 to 5 per MoE layer) for fair comparison.

REAP (Expert Pruning). We used the authors’ official implementation of REAP and kept all pruning-related hyperparameters
at their default values. For calibration, we adopted the same calibration data recipe used in the REAP paper: a
50/50 mixture of allenai/c4 and theblackcat102/evol-codealpaca-v1. All pruning runs for REAP were
conducted on two NVIDIA A100 80GB GPUs.

CFES (Expert Pruning). We used the authors’ official implementation of CFES and followed the default settings except
for unavoidable runtime/distributed configurations. For calibration, we adopted the same calibration dataset composition
used in the CFES paper: rstar coder, openr1 math220, and c4 (using the official sampling/preprocessing routine).
All pruning runs for CFES were conducted on two NVIDIA A100 80GB GPUs. The main hyperparameters are:
• batch size=32
• num routed expert=5
• feature form=segment mean
• metric name=l2
• max length=4096
• prune method=c2f

MoBE (Expert Editing). For MoBE, we followed the official implementation and kept the default settings except for the
basis count and the listed training hyperparameters. We set the number of bases to Num B = 2 and compressed the model
on one NVIDIA A100 80GB GPU. The MoBE factorization was trained with:
• --num epochs 10000
• --batch size 32

27
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression
• --num batches 4
• --learning rate 0.07
• --activation "silu"
• --truncation 1672

All remaining options (including preprocessing and layer-wise handling) followed the official defaults.
HC-SMoE (Expert Merging). We used the authors’ official implementation of HC-SMoE and kept all hyperparameters
at their default values, except for the following settings explicitly chosen to match our experimental protocol. All merging
runs were performed on eight NVIDIA A100 80GB GPUs.
• DOMINANT="no" (no dominant expert)
• SIM BASE="expert-output" (expert-output-based similarity)
• MERGE METHOD="zipit" (default merge method)
• MODE="normal"
• Number SENTENCES=32,
• max block size=2048,
• TRAIN Batch Size=2
• START LAYER=0,
• GROUP LIMIT=4
• CLUSTER METHOD="hierarchical",
• LINKAGE METHOD="average"
• STOP METRIC="silhouette"
• INGREDIENT="act"

All other implementation details (including calibration-data sampling and merging pipeline internals) followed the official
defaults.

TD-MoE (Tensor Decomposition). We used the authors’ official TD-MoE implementation with c4 as the calibration
dataset and global clustering (CLUSTER TYPE="global"). We set the target compression ratio to RATIO=0.4 and
compressed the following Mixtral MoE layers:

• LAYERS TO COMPRESS = (3, 5, 6, 7, 9, 10, 12, 22, 23, 24, 25, 26).
The decomposition was run with the following key arguments:
• MODEL PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"
• DATASET="c4",
• --cluster type "global",
• --ratio 0.4
• --whitening nsamples 256

28
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression
• --model seq len 2048
• --whiten type "output"
• --layers to compress (as above)
• --decomposition method "svd"

All remaining options followed the official defaults, with only minimal runtime/distributed adjustments required by our
server environment.

M-SMoE (Expert Merging). We used the authors’ official M-SMoE implementation and preserved the default settings
except for unavoidable runtime/distributed configurations. Merging was performed on eight NVIDIA A100 80GB GPUs.
We computed expert similarity based on router logits and used a small calibration subset:
• --similarity base="router-logits"
• --subset ratio=0.01
The remaining key configuration is:
• block size=512
• batch size=1
• subset ratio=0.01
• num fewshot=5

All other hyperparameters and merging pipeline details followed the authors’ defaults, except for unavoidable runtime/distributed
adjustments to match our hardware constraints.
F.3. Router Knowledge Distillation Hyperparameter Settings

Note. The Router KD hyperparameters and the calibration dataset (c4) (Dodge et al., 2021) are identical across all
experimental cases. (See Table 1)

Table 1. Router KD hyperparameters (shared across all cases). We used exactly the same Router KD hyperparameters for every
experiment case and model variant, and consistently used c4 as the calibration dataset; only the teacher–student pair (i.e., the backbone
and the compressed baseline) was changed.

Parameter Value
Calibration dataset c4
Epochs 1
Batch size 2
Gradient accumulation steps 4
Learning rate 5 × 10−5
Max sequence length 512
KD temperature (T) 1.0
Max calibration samples 3000

29
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

G. Experiment Result Tables

Table 2. Performance comparison of compression methods with Router KD on Qwen3-30B-A3B-Instruct-2507. This table compares
the performance of representative methods from each category—Expert Pruning (REAP), Expert Editing (MoBE), and Expert Merging
(HC-SMoE)—against their Router KD calibrated versions (denoted with -R). The results demonstrate that Router KD consistently
recovers performance across diverse benchmarks on the fine-grained Qwen3-30B-A3B-Instruct-2507 architecture.

Original REAP REAP-R MoBE MoBE-R HC-SMoE HC-SMoE-R
Method Original Prune Prune Edit Edit Merge Merge

# Total Params 30.53B 19.66B 19.66B 19.66B 19.66B 19.66B 19.66B

BBH-Fewshot 0.3039 0.4866 0.4810 0.0023 0.4715 0.3107 0.3173
BBH-Zeroshot 0.4222 0.4446 0.4468 0.4016 0.4128 0.3665 0.3726
CoQA 0.4283 0.4107 0.4015 0.2553 0.2820 0.3688 0.3642
GSM8k 0.8628 0.8802 0.8832 0.8006 0.8575 0.7339 0.7498

General

GSM8k Platinum 0.8875 0.8999 0.9090 0.8354 0.8776 0.7585 0.7750
MATH 0.2712 0.2736 0.2786 0.2438 0.4608 0.2518 0.2578
AIME 1983-2024 0.3912 0.3483 0.3451 0.2069 0.2347 0.1854 0.1919
AIME 2025 0.2333 0.2667 0.2333 0.1333 0.1333 0.0667 0.1667

Math

Coding MBPP 0.6780 0.6220 0.6180 0.5860 0.5640 0.6340 0.6240
HumanEval-instruct 0.9390 0.9146 0.9207 0.8049 0.8171 0.8841 0.8963

BBH 0.0488 0.0644 0.0660 0.0103 0.4181 0.0364 0.0367
GSM8k 0.8241 0.8757 0.8749 0.7847 0.8355 0.7506 0.7650
GSM8k Platinum 0.8594 0.9065 0.9074 0.8147 0.8718 0.7783 0.7883

CoT-Fewshot

BBH 0.3863 0.3850 0.3853 0.4065 0.4170 0.3130 0.3288
GSM8k 0.5671 0.6149 0.6080 0.6277 0.6217 0.5868 0.5951
GSM8k Platinum 0.6005 0.6460 0.6394 0.6634 0.6543 0.6179 0.6352
ARC-challenge 0.4787 0.4215 0.4181 0.4462 0.4369 0.3788 0.3899
ARC-easy 0.7256 0.6936 0.6957 0.6991 0.7138 0.6549 0.6599
HellaSwag 0.4153 0.4198 0.4245 0.4429 0.4522 0.3684 0.3778

CoT-Zeroshot

MedMCQA 0.5420 0.4100 0.4109 0.4109 0.4282 0.3371 0.3359
MedQA 0.4556 0.4218 0.4266 0.3174 0.3244 0.2883 0.2914
OpenbookQA 0.3320 0.2960 0.3020 0.2740 0.2860 0.2260 0.2360
PIQA 0.7258 0.7203 0.7247 0.7356 0.7383 0.6534 0.6496
WinoGrande 0.5683 0.5683 0.5841 0.5706 0.5848 0.5391 0.5493
MMLU 0.7112 0.6483 0.6507 0.6369 0.6347 0.4313 0.4517

Multi-Choice

30
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Table 3. Performance comparison of alternative compression baselines with Router KD on Qwen3-30B-A3B-Instruct-2507. We evaluate
additional baselines for each category: CFES (Pruning), TD-MoE (Editing), and M-SMoE (Merging). Consistent with Table 2, applying
Router KD (denoted with -R) yields performance improvements across most tasks, reinforcing the generalizability of our proposed
calibration strategy.

Original CFES CFES-R TD-MoE TD-MoE-R M-SMoE M-SMoE-R
Method Original Prune Prune Edit Edit Merge Merge

# Total Params 30.53B 19.66B 19.66B 19.66B 19.66B 19.66B 19.66B

BBH-Fewshot 0.3039 0.3416 0.3285 0.3446 0.3442 0.3758 0.3791
BBH-Zeroshot 0.4222 0.3843 0.3992 0.4105 0.4147 0.4124 0.4136
CoQA 0.4283 0.2458 0.2443 0.3095 0.2990 0.3097 0.3252
GSM8k 0.8628 0.6209 0.6649 0.8052 0.8127 0.5330 0.5527

General

GSM8k Platinum 0.8875 0.6443 0.7047 0.8478 0.8536 0.5633 0.5806
MATH 0.2712 0.2786 0.2950 0.2748 0.2794 0.0330 0.0336
AIME 1983-2024 0.3912 0.0986 0.1565 0.1800 0.1844 0.0000 0.0000
AIME 2025 0.2333 0.0667 0.1000 0.1333 0.1333 0.0000 0.0000

Math

Coding MBPP 0.6780 0.1040 0.1860 0.5620 0.5740 0.0000 0.0000
HumanEval-instruct 0.9390 0.0183 0.0854 0.8171 0.8110 0.0000 0.0000

BBH 0.0488 0.3281 0.3231 0.0264 0.0256 0.1921 0.1599
GSM8k 0.8241 0.6133 0.6641 0.8256 0.8317 0.5625 0.5732
GSM8k Platinum 0.8594 0.6336 0.6940 0.8553 0.8619 0.5864 0.5972

CoT-Fewshot

BBH 0.3863 0.3571 0.3734 0.3634 0.3655 0.3070 0.3090
GSM8k 0.5671 0.3927 0.4466 0.5709 0.5679 0.1736 0.1979
GSM8k Platinum 0.6005 0.4152 0.4682 0.5955 0.5997 0.1844 0.2126
ARC-challenge 0.4787 0.3183 0.3166 0.4352 0.4394 0.4147 0.4232
ARC-easy 0.7256 0.5109 0.5253 0.6848 0.6890 0.6827 0.6928
HellaSwag 0.4153 0.4534 0.4753 0.4655 0.4639 0.4456 0.4532

CoT-Zeroshot

MedMCQA 0.5420 0.3015 0.3330 0.3832 0.3870 0.4898 0.4994
MedQA 0.4556 0.3048 0.3229 0.2891 0.2899 0.4643 0.4941
OpenbookQA 0.3320 0.2060 0.2300 0.2820 0.2780 0.3260 0.3200
PIQA 0.7258 0.7220 0.7383 0.7497 0.7519 0.7252 0.7307
WinoGrande 0.5683 0.5872 0.6212 0.5801 0.5691 0.5896 0.5872
MMLU 0.7112 0.5486 0.5614 0.4803 0.4920 0.6696 0.6716

Multi-Choice

31
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Table 4. Performance comparison of compression methods with Router KD on Mixtral-8x7B-Instruct-v0.1. This table presents the
evaluation results using the coarse-grained Mixtral architecture as the backbone. We compare REAP, MoBE, and HC-SMoE with
their Router KD counterparts. As discussed in Section ??, the performance gains from Router KD are relatively marginal compared to
Qwen3-30B-A3B-Instruct-2507, due to the simpler routing decision boundaries of the coarse-grained architecture.

Original REAP REAP-R MoBE MoBE-R HC-SMoE HC-SMoE-R
Method Original Prune Prune Edit Edit Merge Merge

# Total Params 46.70B 29.79B 29.79B 29.79B 29.79B 29.79B 29.79B

BBH-Fewshot 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
BBH-Zeroshot 0.4661 0.2224 0.2133 0.2992 0.2953 0.4038 0.4030
CoQA 0.1698 0.3145 0.3400 0.3083 0.2805 0.1342 0.1468
GSM8k 0.6247 0.0311 0.0334 0.0243 0.0318 0.3859 0.3920

General

GSM8k Platinum 0.6592 0.0306 0.0339 0.0331 0.0314 0.3921 0.4078
MATH 0.2798 0.0346 0.0308 0.0428 0.0432 0.1026 0.1170
AIME 1983-2024 0.0021 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
AIME 2025 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000

Math

Coding MBPP 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
HumanEval-instruct 0.5427 0.0000 0.0122 0.0122 0.0000 0.2683 0.2927

BBH 0.6481 0.2657 0.2780 0.1605 0.1754 0.5228 0.5356
GSM8k 0.6892 0.0364 0.0364 0.0826 0.0781 0.3700 0.3882
GSM8k Platinum 0.7146 0.0372 0.0521 0.0902 0.0794 0.3772 0.3978

CoT-Fewshot

BBH 0.4909 0.1193 0.1539 0.2500 0.2520 0.4360 0.4319
GSM8k 0.6050 0.0227 0.0288 0.1130 0.0910 0.3692 0.3783
GSM8k Platinum 0.6154 0.0265 0.0265 0.1117 0.0935 0.3879 0.3879
ARC-challenge 0.5026 0.2816 0.3020 0.4155 0.4249 0.4983 0.5068
ARC-easy 0.7189 0.4566 0.4575 0.6797 0.6734 0.7677 0.7668
HellaSwag 0.6364 0.3981 0.4008 0.5043 0.4870 0.6065 0.6092

CoT-Zeroshot

MedMCQA 0.5377 0.2678 0.2728 0.3753 0.3610 0.4052 0.4028
MedQA 0.5727 0.2584 0.2773 0.3943 0.3661 0.4273 0.4297
OpenbookQA 0.3440 0.1940 0.2240 0.3240 0.3300 0.3480 0.3560
PIQA 0.7448 0.5827 0.5892 0.7388 0.7296 0.7807 0.7791
WinoGrande 0.6156 0.5627 0.4996 0.6093 0.6290 0.6875 0.6811
MMLU 0.6765 0.2829 0.2874 0.4515 0.4623 0.5753 0.5758

Multi-Choice

32
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Table 5. Performance comparison of alternative compression baselines with Router KD on Mixtral-8x7B-Instruct-v0.1. Evaluation of
CFES, TD-MoE, and M-SMoE on the Mixtral backbone. Similar to Table 4, the results show the impact of Router KD on a coarse-grained
MoE model. While some recovery is observed, it highlights the structural limitations of router calibration in models with fewer experts.

Original CFES CFES-R TD-MoE TD-MoE-R M-SMoE M-SMoE-R
Method Original Prune Prune Edit Edit Merge Merge

# Total Params 46.70B 29.79B 29.79B 29.79B 29.79B 29.79B 29.79B

BBH-Fewshot 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
BBH-Zeroshot 0.4643 0.2049 0.1955 0.3986 0.3961 0.2264 0.3110
CoQA 0.1668 0.2888 0.2938 0.1878 0.1895 0.1832 0.0130
GSM8k 0.6361 0.0197 0.0182 0.3753 0.3844 0.0546 0.1380

General

GSM8k Platinum 0.6592 0.0149 0.0199 0.4028 0.4185 0.0571 0.1266
MATH 0.2834 0.0236 0.0260 0.1442 0.1500 0.0306 0.0400
AIME 1983-2024 0.0021 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
AIME 2025 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000

Math

Coding MBPP 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
HumanEval-instruct 0.5427 0.0000 0.0000 0.3476 0.3476 0.0976 0.1402

BBH 0.6481 0.2646 0.2542 0.4970 0.5004 0.3130 0.3706
GSM8k 0.6892 0.0273 0.0303 0.3601 0.3472 0.0614 0.1274
GSM8k Platinum 0.7146 0.0265 0.0207 0.3672 0.3631 0.0620 0.1373

CoT-Fewshot

BBH 0.4909 0.1146 0.1181 0.4072 0.4006 0.2321 0.2786
GSM8k 0.6050 0.0174 0.0182 0.3859 0.3859 0.0743 0.1175
GSM8k Platinum 0.6154 0.0141 0.0149 0.4003 0.4111 0.0835 0.1141
ARC-challenge 0.5026 0.2543 0.2611 0.4915 0.4753 0.3942 0.4394
ARC-easy 0.7189 0.4019 0.3851 0.7218 0.7210 0.6650 0.6801
HellaSwag 0.6364 0.3795 0.3782 0.6137 0.6138 0.4954 0.5368

CoT-Zeroshot

MedMCQA 0.5377 0.2768 0.2704 0.4260 0.4236 0.3325 0.3352
MedQA 0.5727 0.2482 0.2608 0.4690 0.4635 0.3056 0.3307
OpenbookQA 0.3440 0.1840 0.1780 0.3220 0.3380 0.2580 0.3360
PIQA 0.7448 0.5696 0.5664 0.7622 0.7541 0.7247 0.7563
WinoGrande 0.6156 0.5359 0.5241 0.6393 0.6361 0.6488 0.6598
MMLU 0.6765 0.2481 0.2512 0.5649 0.5670 0.3825 0.4284

Multi-Choice

33
Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression

Table 6. Robustness analysis of Router KD under milder compression (75% Parameter Retention) on Qwen3-30B-A3B-Instruct-2507. To
verify that the efficacy of Router KD is not limited to a specific compression ratio, we evaluated performance while retaining 75% of the
expert parameters (≈ 23.28B total parameters). The results confirm that Router KD consistently mitigates performance degradation even
under this alternative compression setting.

Original CFES CFES-R MoBE MoBE-R HC-SMoE HC-SMoE-R
Method Original Prune Prune Edit Edit Merge Merge

# Total Params 30.53B 23.28B 23.28B 23.28B 23.28B 23.28B 23.28B

BBH-Fewshot 0.3039 0.2268 0.2273 0.0026 0.3575 0.3000 0.3018
BBH-Zeroshot 0.4222 0.3764 0.4162 0.4165 0.4170 0.3755 0.3812
CoQA 0.4283 0.1462 0.1810 0.3565 0.3412 0.3955 0.3953
GSM8k 0.8628 0.4117 0.6475 0.8658 0.8582 0.8393 0.8415

General

GSM8k Platinum 0.8875 0.4342 0.6782 0.8892 0.8900 0.8635 0.8668
MATH 0.2712 0.2274 0.2824 0.2396 0.2940 0.2696 0.2726
AIME 1983-2024 0.3912 0.0086 0.0911 0.3891 0.3805 0.3708 0.3708
AIME 2025 0.2333 0.0000 0.0333 0.2333 0.2333 0.2667 0.2667

Math

Coding MBPP 0.6780 0.1140 0.4720 0.4440 0.6480 0.6500 0.6440
HumanEval-instruct 0.9390 0.1341 0.7012 0.9085 0.8963 0.9146 0.9146

BBH 0.0488 0.3098 0.3797 0.0361 0.0359 0.0412 0.0429
GSM8k 0.8241 0.4579 0.6300 0.8461 0.8378 0.8340 0.8408
GSM8k Platinum 0.8594 0.4806 0.6609 0.8768 0.8685 0.8644 0.8726

CoT-Fewshot

BBH 0.3863 0.3213 0.3422 0.3738 0.3774 0.3672 0.3671
GSM8k 0.5671 0.2631 0.4776 0.5588 0.5519 0.6353 0.6482
GSM8k Platinum 0.6005 0.2796 0.5136 0.5947 0.5889 0.6741 0.6791
ARC-challenge 0.4787 0.3609 0.3831 0.4753 0.4821 0.4480 0.4505
ARC-easy 0.7256 0.5467 0.6237 0.7273 0.7269 0.7235 0.7294
HellaSwag 0.4153 0.4655 0.4982 0.4588 0.4584 0.4212 0.4245

CoT-Zeroshot

MedMCQA 0.5420 0.3622 0.3918 0.5078 0.5104 0.3662 0.3739
MedQA 0.4556 0.2969 0.3771 0.3841 0.3881 0.3480 0.3535
OpenbookQA 0.3320 0.2380 0.2780 0.3040 0.3000 0.2840 0.3020
PIQA 0.7258 0.7307 0.7508 0.7486 0.7476 0.7057 0.7018
WinoGrande 0.5683 0.5991 0.6030 0.5841 0.5896 0.5746 0.5833
MMLU 0.7112 0.5472 0.6288 0.6958 0.6954 0.5691 0.5766

Multi-Choice

34