---
type: paper
title: "Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"
title_zh: Nemotron 3 Super：面向智能体推理的开放高效混合专家 Mamba-Transformer 模型
authors:
  - NVIDIA
affiliations:
  - NVIDIA
year: 2026
core_method: "[[LatentMoE]]"
description: 提出 Nemotron 3 Super（120B/12B active），结合 LatentMoE、Hybrid Mamba-Attention 和 MTP，在同等精度下实现 2.2-7.5x 推理吞吐提升
aliases:
  - Nemotron 3 Super
  - Nemotron-3-Super
tags:
  - architecture
  - moe
  - efficiency
  - post-training
  - rlhf
  - agentic
created: 2026-03-13
updated: 2026-03-13T23:46
---

# Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

NVIDIA

# Abstract.

We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the first model in the Nemotron 3 family to 1) be pre-trained in NVFP4, 2) leverage LatentMoE, a new Mixture-of-Experts architecture that optimizes for both accuracy per FLOP and accuracy per parameter, and 3) include MTP layers for inference acceleration through native speculative decoding. We pre-trained Nemotron 3 Super on 25 trillion tokens followed by post-training using supervised fine tuning (SFT) and reinforcement learning (RL). The final model supports up to 1M context length and achieves comparable accuracy on common benchmarks, while also achieving up to  $2.2 \times$  and  $7.5 \times$  higher inference throughput compared to GPT-OSS-120B and Qwen3.5-122B, respectively. Nemotron 3 Super datasets, along with the base, post-trained, and quantized checkpoints, are open-sourced on HuggingFace.

# 1. Introduction

The last few years have seen a rise in the popularity of Mixture-of-Experts (MoE) based Large Language Models (LLMs) (DeepSeek-AI, 2025c; Yang et al., 2025; GLM-4.5-Team, 2025). MoEs help LLMs achieve higher accuracy at a lower active parameter count than regular dense models (Dai et al., 2024; Lepikhin et al., 2020). Orthogonal to MoEs, Hybrid Mamba-Attention models have shown promise in significantly improving inference throughput (NVIDIA, 2025c). We combine these two directions of improvement in Nemotron 3 (NVIDIA, 2025c). As part of our Nemotron 3 series of models, we present Nemotron 3 Super—a 12 billion active, 120 billion total parameter MoE hybrid Mamba-Attention model. Nemotron 3 Super achieves better or on-par benchmark accuracies than GPT-OSS-120B (OpenAI, 2025) and Qwen3.5-122B while achieving up to  $2.2 \times$  and  $7.5 \times$  higher inference throughput, respectively, on the 8k token input / 64k token output setting.

Nemotron 3 Super is our first model to use LatentMoE (Elango et al., 2026) - a novel MoE architecture that achieves better accuracy per parameter and per FLOP than regular MoEs. Nemotron 3 Super also incorporates Multi-Token-Prediction (MTP), which accelerates inference through speculative decoding while improving overall model quality. We pre-trained Nemotron 3 Super in NVFP4, demonstrating stable and accurate pre-training in low precision. Similar to Nemotron 3 Nano (NVIDIA, 2025a), we pre-trained Nemotron 3 Super on 25 trillion text tokens divided into 2 phases. The first phase accounted for  $80\%$  of pre-training (20 trillion tokens) and focused on diversity and broad coverage, while the second phase accounted for  $20\%$  of pre-training (5 trillion tokens) and focused on high-quality data and benchmark accuracy. Our base model achieves significantly better accuracy than similarly sized state-of-the-art base models, such as GLM-4.5-Air-Base (GLM-4.5-Team, 2025) and Ling-flash-Base-2.0 (Ling-Team, 2025).

We trained Nemotron 3 Super with a strong emphasis on agentic capabilities. To support this objective, we substantially scaled the breadth of our RL environments, the volume and quality

© 2026 NVIDIA. All rights reserved.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_1_Figure_0.jpeg](_page_1_Figure_0.jpeg)
Figure 1 | Accuracy and throughput comparisons of Nemotron 3 Super with GPT-OSS-120B and Qwen3.5-122B. Nemotron 3 Super achieves comparable accuracies across popular benchmarks but provides the highest inference throughput; on 8k input sequence lengths and 64k output sequence lengths, Nemotron 3 Super provides up to  $2.2 \times$  and  $7.5 \times$  higher throughput than GPT-OSS-120B and Qwen3.5-122B, respectively. We measured throughput on B200 GPUs with vLLM and TRT-LLM and use the best out of the two frameworks for each model. For GPT-OSS-120B we use MXFP4 weights, MXFP8 activations, and FP8 KV-Cache; for Qwen3.5-122B we use BF16. We used the OpenHands harness to evaluate SWE-Bench.

of agentic training data, and the overall amount of post-training focused on multi-step tool-using behavior. To train effectively on this diverse set of long-horizon tasks, we made substantial improvements to the resiliency of our RL infrastructure, enabling large-scale asynchronous training. This expanded agentic training recipe yields substantial improvements over Nemotron 3 Nano across software engineering, terminal use, and general tool use benchmarks.

We are publicly sharing the training recipe for Nemotron 3 Super on the Nemotron Developer Repository $^{1}$ . We are also openly releasing the following:

# Checkpoints

- Nemotron 3 Super 120B-A12B NVFP4 : post-trained and NVFP4 quantized model
- Nemotron 3 Super 120B-A12B FP8 : post-trained and FP8 quantized model
- Nemotron 3 Super 120B-A12B BF16 : post-trained model
- Nemotron 3 Super 120B-A12B Base BF16 : base model
- Qwen3-Nemotron-235B-A22B-GenRM-2603 : GenRM used for RLHF

# Data

- Nemotron-Pretraining-Specialized-v1.1 : a collection of synthetic datasets aimed to improve LLM capabilities in code concepts and algorithms, formal logic, economics, and multiple choice questions.
- Nemotron-Super-Post-Training-Data : a collection of RL environments and SFT datasets targeting a broad range of agentic capabilities.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

The report is organized into 3 broad sections: Pre-training (§2), Post-training (§3), and Quantization (§4), each describing in detail our approach.

# 2. Pretraining

In this section, we highlight the key features of Nemotron 3 Super 120B-A12B Base, detailing its hybrid Mamba-Attention Mixture-of-Experts (MoE) architecture, NVFP4 pre-training, hyperparameter configurations, long-context extension, and the 25-trillion-token corpus used for pretraining. We also demonstrate that Nemotron-3 Super 120B A12B Base achieves superior accuracy compared to other public state-of-the-art models—including Ling-flash-Base-2.0 and GLM-4.5-Air-Base—across a comprehensive suite of benchmarks.

# 2.1. Model Architecture

![_page_2_Figure_0.jpeg](_page_2_Figure_0.jpeg)
Figure 2 | Nemotron 3 Super layer pattern. Similar to Nemotron 3 Nano, we use a hybrid Mamba-Attention architecture, but Nemotron 3 Super is the first model to scale sparsely using LatentMoE layers rather than standard MoE layers.



<table><tr><td>Configuration</td><td>Nemotron 3 Super 120B-A12B Base</td></tr><tr><td>Total Layers</td><td>88</td></tr><tr><td>Model Dimension</td><td>4096</td></tr><tr><td>Q-Heads (nq)</td><td>32</td></tr><tr><td>KV-Heads (nkv)</td><td>2</td></tr><tr><td>Head Dimension</td><td>128</td></tr><tr><td>Mamba State Dimension</td><td>128</td></tr><tr><td>Mamba Groups</td><td>8</td></tr><tr><td>Mamba Heads</td><td>128</td></tr><tr><td>Mamba Head Dimension</td><td>64</td></tr><tr><td>Expert Hidden Dimension</td><td>2688</td></tr><tr><td>Shared Expert Intermediate Size</td><td>5376</td></tr><tr><td>Total Experts per Layer</td><td>512</td></tr><tr><td>Top-k (Activated Experts)</td><td>22</td></tr><tr><td>MoE Latent Size</td><td>1024</td></tr><tr><td>MTP layers (shared weight)</td><td>2</td></tr></table>



Table 1 | Nemotron 3 Super Architectural Dimensions. The model employs a hybrid Mamba-2 and MoE design with strategic global attention layers to optimize the balance between sequence modeling performance and inference throughput.

---

Nemotron 3 Super 120B-A12B Base scales up the hybrid Mamba-Attention Mixture-of-Experts (MoE) architecture introduced in Nemotron-3 Nano *(NVIDIA, 2025c)*. We extend this foundation to 120.6B total parameters, maintaining a constrained active budget of 12.7B parameters (12.1B excluding embeddings) per forward pass. The architecture comprises three core pillars: sparse LatentMoE scaling (§2.1.1), Multi-Token Prediction (MTP) for inference acceleration (§2.1.2), and a periodic hybrid interleaving pattern (§2.1.3).

#### 2.1.1 LatentMoE: Hardware-Aware Expert Design for Improved Accuracy per Byte

Mixture-of-Experts (MoE) architectures have emerged as a promising approach to maximize accuracy under fixed inference cost, allowing models to scale in parameter count while keeping floating-point operations (FLOPs) per token constant. Existing MoE designs are largely motivated by high-level sparsity arguments and optimized for offline, throughput-oriented settings, with little consideration for online deployments that impose strict latency, memory bandwidth, and communication constraints. Whereas accuracy per FLOP reflects computational efficiency, accuracy per parameter captures memory footprint, memory bandwidth, routing-induced communication, and sharding overhead. Neglecting these factors can yield architectures that appear efficient in aggregate compute yet incur substantial inefficiency in practice.

Motivated by these observations, we revisited MoE design from a hardware–software co-design perspective. Through systematic analysis of existing MoE systems across the throughput–latency Pareto frontier, together with accuracy measurements and theoretical analysis, we identified structural inefficiencies in prevailing MoE designs that limit accuracy per unit inference cost. From this analysis we distilled the following design principles for efficient MoE scaling:

1. In low-latency serving, MoE inference is often dominated by the memory bandwidth cost of reading expert weights. Each expert matrix has size $d\times m$, where $d$ is the hidden dimension and $m$ is the expert FFN intermediate dimension; reducing this cost therefore requires decreasing $d$ or $m$.
2. In throughput-oriented serving, distributed MoE inference is dominated by all-to-all routing. Routing volume scales as $d\times K$, where $K$ is the number of active experts; reducing communication overhead therefore requires decreasing $d$ or $K$.
3. Preserving model quality requires preserving the effective nonlinear budget $K\cdot m$. To relieve memory and communication bottlenecks without sacrificing quality, $K$ and $m$ should thus be held fixed.
4. A task-specific effective feature rank $r_{\text{eff}}$ imposes a lower limit on how much $d$ can be reduced; reducing $d$ below this limit causes model quality to collapse.
5. Scaling both the total number of experts $N$ and the top-$K$ experts per token improves quality by exponentially expanding the space of expert combinations.

Principles (1)–(3) imply that the hidden dimension $d$ is the most promising axis for reduction, enabling gains in both throughput- and latency-oriented regimes without significant loss in accuracy. Principle (4) gives a lower bound on how far $d$ can be reduced without collapse. Principle (5) indicates that increasing $N$ and $K$ improves quality; because memory bandwidth and communication scale linearly with $K$, we can increase $K$ by a factor $\alpha$ and reduce $d$ by the same factor $\alpha$ to obtain higher accuracy at similar inference cost. Guided by these insights, we developed LatentMoE *(Elango et al., 2026)*, a MoE architecture designed to achieve higher accuracy than a standard MoE at similar inference cost.

The LatentMoE architecture is illustrated in Figure 3(b). Each input token $x\in\mathbb{R}^{d}$ is first projected into a lower-dimensional latent space $\mathbb{R}^{\ell}$ via a learnable down-projection matrix $W_{\hat{\downarrow}}\in\mathbb{R}^{\ell\times d}$. The

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_4_Figure_0.jpeg](_page_4_Figure_0.jpeg)
(a) Standard MoE architecture.
Figure 3 | Standard MoE vs. LatentMoE. In LatentMoE, tokens are projected from the hidden dimension  $d$  into a smaller latent dimension  $\ell$  for routing and expert computation, reducing routed parameter loads and all-to-all traffic by a factor  $d / \ell$ . These savings are used to increase both the total number of experts and the top- $K$  active experts per token by the same factor, improving model accuracy at approximately constant inference cost.

![_page_4_Figure_1.jpeg](_page_4_Figure_1.jpeg)
(b) LatentMoE architecture.

compressed representation is then routed to an expanded set of experts that operate entirely in this latent space. After expert computation, the outputs are aggregated and projected back to dimension  $d$  via a learnable up-projection matrix  $W_{\uparrow} \in \mathbb{R}^{d \times \ell}$ . Shifting routed expert computation and all-to-all traffic into the latent space reduces both per-expert weight loads and communication payloads by a factor  $d / \ell$  relative to a standard MoE. We use these savings to increase the total number of experts from  $N$  to  $N' = N \cdot d / \ell$  and the top- $K$  active experts per token from  $K$  to  $K' = K \cdot d / \ell$ . The reduction in dimension offsets the increase in expert count and in  $K$ , yielding higher model quality at a similar computational and communication budget. To preserve quality, all non-routed computations—including the routing gate (gating network), shared expert computation, and non-expert layers—remain in the full hidden dimension  $d$ , as they do not contribute significantly to the targeted bottlenecks. We refer the reader to the LatentMoE technical report (Elango et al., 2026) for further details.

# 2.1.2. Multi-Token Prediction

Nemotron-3 Super incorporates a Multi-Token Prediction (MTP) objective to improve both modeling quality and inference efficiency. Unlike conventional next-token training, MTP optimizes the model to predict multiple future tokens at each position (Gloeckle et al., 2024; DeepSeek-AI, 2025c). This encourages representations that capture multi-step dependencies and longer-range structure, leading

---

to consistent improvements in validation loss and downstream benchmark accuracy.

Beyond quality gains, MTP enables native speculative decoding. The auxiliary prediction heads function as an internal draft model: during inference, they generate candidate continuations that are verified by the main model in a single forward pass. This substantially reduces decoding latency while introducing minimal additional FLOPs—significantly less than required by an external draft model. Although speculative decoding is particularly effective at small batch sizes, recent work shows that it can also improve throughput in larger-batch and sparse MoE settings *(Huang et al., 2025)*.

##### Design for Robust Autoregressive Drafting.

Standard MTP implementations use $N$ independent heads, each trained to predict a fixed offset (e.g., $n+2,\ldots,n+N+1$). While effective during training, this limits speculative decoding to at most $N$ draft tokens. Longer drafts require either increasing $N$ or reusing a single offset-trained head autoregressively.

Reusing a fixed-offset head introduces a training–inference mismatch: the head is trained under ground-truth hidden states but, at inference, conditions on its own generated states. This distribution shift often reduces acceptance rates as draft length increases.

Nemotron-3 Super addresses this limitation by sharing parameters across multiple MTP heads during training, yielding a unified prediction head exposed to multiple offsets. This shared-weight formulation regularizes the head across prediction horizons and improves robustness to the self-generated hidden states encountered during autoregressive drafting. As a result, the same head can be applied recursively at inference to generate longer drafts with more stable acceptance behavior. While acceptance rates naturally decrease as draft length increases, the degradation is substantially milder than with independently trained offset heads. This enables more effective speculative decoding without introducing additional parameters or requiring a separate draft model.

##### Speculative Decoding Performance.

We evaluate MTP quality using SPEED-Bench *(Abramovich et al., 2026)*, a benchmark tailored for speculative decoding. Table 2 reports the average acceptance length (tokens accepted per verification step) with a fixed draft length of 7. Nemotron-3 Super achieves the highest overall average acceptance length (3.45), outperforming DeepSeek-R1 across all domains and remaining competitive with Qwen3-Next.

Figure 4 plots acceptance rate as a function of draft token index. As expected, acceptance decreases monotonically with draft depth for all models. However, Nemotron-3 Super consistently maintains higher acceptance than DeepSeek-R1 at every draft position and closely tracks or exceeds Qwen3-Next across most indices. Notably, the gap becomes more pronounced at larger draft indices (4–7), where recursive drafting is most challenging. This behavior indicates improved stability of the shared-head autoregressive design under longer speculative rollouts.

Overall, MTP in Nemotron-3 Super improves both representation learning and decoding efficiency, enabling higher acceptance at extended draft lengths without relying on an external draft model. These acceptance gains translate to superior serving efficiency on Blackwell hardware. As shown in Figure 5, increasing the draft depth ($D=1$ to $D=3$) via MTP significantly shifts the throughput–latency Pareto frontier, delivering higher aggregate output tokens per second (TPS) for any given median user latency compared to the baseline with MTP disabled.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_6_Figure_0.jpeg](_page_6_Figure_0.jpeg)
Figure 4 | MTP acceptance rate by draft index on SPEED-Bench using a draft length of 7.



<table><tr><td>Category</td><td>DSR1</td><td>Qwen3 Next</td><td>Nemotron3 Super</td></tr><tr><td>Coding</td><td>2.99</td><td>4.32</td><td>3.78</td></tr><tr><td>Humanities</td><td>2.67</td><td>3.07</td><td>3.26</td></tr><tr><td>Math</td><td>2.98</td><td>3.89</td><td>3.73</td></tr><tr><td>Multilingual</td><td>2.83</td><td>3.97</td><td>4.05</td></tr><tr><td>QA</td><td>2.63</td><td>3.09</td><td>3.16</td></tr><tr><td>RAG</td><td>2.79</td><td>3.53</td><td>3.78</td></tr><tr><td>Reasoning</td><td>2.80</td><td>3.47</td><td>3.59</td></tr><tr><td>Roleplay</td><td>2.19</td><td>2.17</td><td>2.82</td></tr><tr><td>STEM</td><td>2.79</td><td>3.37</td><td>3.30</td></tr><tr><td>Summarization</td><td>2.59</td><td>3.06</td><td>3.48</td></tr><tr><td>Writing</td><td>2.41</td><td>2.69</td><td>2.99</td></tr><tr><td>Average</td><td>2.70</td><td>3.33</td><td>3.45</td></tr></table>



Table 2 | MTP average acceptance lengths on SPEED-Bench using a draft length of 7.

![_page_6_Figure_1.jpeg](_page_6_Figure_1.jpeg)
Figure 5 | Total vs. user throughput for an NVFP4 checkpoint (TRT-LLM, TP=1, B300 GPU). Comparing MTP off vs. MTP with draft lengths of 1,3. Measured on SPEED-Bench's Throughput-1k split, with 1k output tokens.

# 2.1.3. Hybrid Interleaved MoE Architecture and Global Anchors

Nemotron 3 Super adopts a hybrid Mixture-of-Experts (MoE) architecture designed to maximize inference throughput—particularly for long-context reasoning—while preserving the modeling capacity of large-scale dense Transformers. The primary systems bottleneck in modern sequence models is the quadratic growth of the KV cache in self-attention layers. To address this, we predominantly utilize Mamba-2 blocks (Dao &amp; Gu, 2024), which operate with a constant-sized state during generation, substantially reducing memory overhead and latency.

The 88-layer stack follows a periodic interleaving pattern in which MoE layers are paired with Mamba-2 blocks. While Mamba provides efficient linear-time sequence modeling, a limited number of self-attention layers are strategically inserted as global "anchors" to enable full-token interaction and long-range information routing across the stack. This hybrid interleaving preserves global dependency modeling while offloading the majority of computation to the more efficient Mamba and

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

sparse MoE components. Table 1 and Figure 2 provide a comprehensive summary of the structural parameters and the specific interleaving pattern of the hybrid stack.

The attention layers employ Grouped-Query Attention (GQA) with 32 query heads and 2 KV heads (head dimension 128). Consistent with prior Nemotron models, we omit positional embeddings, dropout, and bias terms in linear layers, use RMSNorm for normalization, and maintain un-tied embedding and output weights. This configuration supports context lengths of up to 1M tokens.

Sparse scaling further improves efficiency. Each MoE layer activates only a subset of experts per token (top-22 routing), enabling the model to scale to 120.6B total parameters while maintaining a 12.7B active parameter budget per forward pass.

Overall, the synergy between linear-time Mamba blocks, sparsely activated MoE capacity, and strategically placed attention anchors enables Nemotron 3 Super to deliver strong long-context performance while remaining optimized for real-world deployment on modern hardware.

# 2.2. NVFP4 Pretraining

Table 3 | Precision by Layer Type



<table><tr><td>Layer Type</td><td>Format</td><td>Rationale</td></tr><tr><td>All Linear Layers Unless Otherwise Noted</td><td>NVFP4</td><td></td></tr><tr><td>Final 15% of Network</td><td>BF16</td><td>Promote training stability at scale</td></tr><tr><td>Latent Projections</td><td>BF16</td><td>Strategically kept in BF16 as step-time impact is negligible</td></tr><tr><td>MTP Layers</td><td>BF16</td><td>Preserves multi-token prediction capabilities</td></tr><tr><td>QKV &amp; Attention Projections</td><td>BF16</td><td>Maintain fidelity of few attention layers</td></tr><tr><td>Mamba Output Projection</td><td>MXFP8</td><td>Mitigates high incidence of underflows observed when quantizing this layer to NVFP4 at smaller scales</td></tr><tr><td>Embedding Layers</td><td>BF16</td><td></td></tr></table>



Nemotron 3 Super was trained with the NVFP4 pretraining recipe detailed in the Nemotron 3 white paper (NVIDIA, 2025c). All linear layers, unless otherwise noted in Table 3, are trained using the open-source NVFP4 GEMM kernels provided by Transformer Engine with the cuBLAS backend (NVIDIA Corporation, 2024) for fprop, dgrad, and wgrad GEMMs. This framework performs quantization of weights, activations, and gradients to NVFP4 according to the scheme first introduced in NVIDIA (2025d). Weights are quantized to NVFP4 using two-dimensional (2D) block scaling to maintain consistency between quantized weights in the forward and backward pass. Gradients and activations are quantized to NVFP4 using one-dimensional (1D) blocks along the GEMM reduction axis. Random Hadamard Transforms (RHTs) are performed on inputs to wgrad and stochastic rounding is applied to gradient tensors. The NVFP4 format utilizes an E2M1 element format with 16-element micro-blocks, E4M3 micro-block scaling factors, and a second-level FP32 global scale. Nemotron 3 Super showcases large-scale stable training in NVFP4 up to 25T tokens.

During the training of Nemotron 3 Super, we observed a growth in the number of zero-valued weight gradient elements and investigated the root cause to validate training health. Magnitude

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

patterns emerged within some expert layers, characterized by the norms of FC1 output channels and corresponding FC2 input channels converging toward zero (Figure 6). By the end of pretraining, zero-valued weight gradient elements accounted for  $7\%$  of total parameters, appearing to correlate with the magnitude patterns. We believe that NVFP4 quantization increases the incidence of true zeros in the weight gradients that could have been more easily be represented by BF16 or MXFP8. Low-norm channels likely attenuate quicker when these layers are trained in NVFP4.

![_page_8_Figure_0.jpeg](_page_8_Figure_0.jpeg)
Figure 6 | Channel Magnitude Patterns in Weights of Expert Layers in Nemotron 3 Super. Patterns emerge as training progresses. Top: Early layer routed expert FC1 weight matrix at 0.5T tok and 23T tok. Bottom: Early layer routed expert FC2 weight matrix at 500B tok and 23T tok. Low-norm output channels of FC1 align with low-norm input channels of FC2.

![_page_8_Figure_1.jpeg](_page_8_Figure_1.jpeg)

![_page_8_Figure_2.jpeg](_page_8_Figure_2.jpeg)
Figure 7 | Number of Zero-Valued Weight Gradient Elements on Nemotron 3 Nano. Left: Released Nemotron Nano 3 model (NVIDIA, 2025a) trained to 25T tokens in BF16. Right: Ablation study trained to 1T tokens in BF16 and with our NVFP4 recipe. The NVFP4 model at 1T tokens reaches a similar zero-valued weight gradient count as the BF16 model at 25T tokens. Switching from NVFP4 to BF16 at 0.5T tokens causes zero-valued weight gradients to return to baseline levels. The high prevalence of small-magnitude gradients (&lt;1e-12) in BF16 suggests NVFP4 quantization underflows already-small values to zero.

We compare identical Nemotron 3 Nano models (NVIDIA, 2025a) trained for 1T tokens in BF16 and in NVFP4 and find that NVFP4 pretraining produces roughly 3x more zero-valued weight

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

gradients at the same token horizon. When a partially trained NVFP4 model is switched back to BF16, the number of zero-valued weight gradients returns to baseline levels. The BF16 model still contains many small-magnitude gradients (&lt;1e-12), but NVFP4 quantization underflows these values to zero (Figure 7). We sampled weight, activation, and gradient tensors from routed expert layers and observed high rates of underflow in dgrad of FC2 at 500B tokens, primarily because two-dimensional weight quantization blocks span high and low magnitude channels. Underflows in dgrad of FC2 create zeros in wgrad of FC1 through backpropagation of the gradient. At 750B tokens, we observe high rates of underflow in fprop of FC1, creating zeros in wgrad of FC2 (Figure 8). The 1T-token NVFP4 model behaves similarly to a much longer-trained BF16 model. After 10T tokens, the released Nemotron 3 Nano model, trained in BF16, NVIDIA (2025a) reaches a similar number of zero-valued weight gradient elements as the NVFP4 model trained to 1T tokens (Figure 7) and inspection of weight matrices in early expert layers revealed a similar channel magnitude pattern. These conclusions on the Nemotron 3 Nano architecture give insights into the channel magnitude patterns and growth in zero-valued weight gradient elements in Nemotron 3 Super.

![_page_9_Figure_0.jpeg](_page_9_Figure_0.jpeg)
Figure 8 | Origin of Zero Element Weight Gradients in Nemotron 3 Nano, shown for routed expert layers increasing in layer depth. Averaged across all routed experts in a layer index. Top: Tensors sampled at 500B tokens. Percent of zero-valued weight gradients are higher for FC1 than FC2 and attributed almost entirely to underflows in dgrad of FC2. Bottom: Tensors sampled at 750B tokens. Percent of zero-valued weight gradients are equivalently high for FC1 and FC2. Zero-valued weight gradients in FC1 are attributed to underflows in dgrad of FC2. Zero element weight gradients in FC2 are attributed mainly to underflows in fprop of FC1, with a minor contribution of underflows in wgrad of FC2.

Following our previous work in NVFP4 pretraining (NVIDIA, 2025d), we evaluated whether switching all tensors to higher precision prior to learning rate annealing would benefit Nemotron 3 Super. We promoted all tensors to MXFP8 at 19T tokens (1T tokens before annealing) and continued training through 20.6T tokens. While this improved the loss trajectory, it yielded no gains in downstream task accuracy (Fig 9). The final Nemotron 3 Super model is therefore pretrained with our NVFP4 recipe for the entire token horizon.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_10_Figure_0.jpeg](_page_10_Figure_0.jpeg)
Figure 9 | Improvement in downstream task evaluation accuracy after switching the network precision to MXFP8. Values greater than zero indicate improvement in accuracy over the NVFP4 model. None of the downstream task evaluation metrics showed sustained improvement after training in MXFP8.

# 2.3. Pretraining Data

# 2.3.1. Data

We describe here several new datasets we added to pretraining since Nemotron 3 Nano (NVIDIA, 2025a). We are releasing these datasets on HuggingFace as Nemotron-Pretraining-Specialized-v1.1.

# 2.3.2. Synthetic Code Concepts

With the aim of improving Python problem-solving capabilities, we synthetically generated a dataset consisting Python problems and solutions. Using a taxonomy consisting of thousands of programming concepts curated from our Nemotron-Pretraining-Code datasets and GPT-OSS-120B, we extracted high-level programming concepts from the HumanEval benchmark dataset (Chen et al., 2021). In total, after dedduplication of the extracted taxonomical representations, we collected a total of 91 concepts.

Using the extracted concepts, we performed open-ended generation using GPT-OSS 20B to generate Python programming problems that test these concepts and instructed it to generate the problem with a descriptive function name and problem description in the function docstring. To generate these problems at the pretraining scale, we combined up to four concepts per generation and generated up to five problems per set of concepts. This resulted in a total of approximately 14 million problems.

Following problem generation, we then used GPT-OSS 120B to generate five self-contained solutions for each generated problem. To avoid biasing models trained on these data to generate long-winded solutions, we instructed GPT-OSS 120B to restrict its solution to 60 lines maximum. For each problem, we generate five solutions, and we stopped generating after obtaining approximately 23 million problem-solution pairs.

As a final step in generating this dataset, we thoroughly cleaned the generated problem-solution pairs. Our cleaning consisted of the following steps: we check that GPT-OSS-120 B did not include

---

additional imports that were not specified in the original problem generated from GPT-OSS-20 B. We discard all solutions that did not satisfy this condition. We form the final problem-solution pair by parsing only the solution provided by GPT-OSS-120B and appending it to the problem generated from GPT-OSS-20 B. We found that GPT-OSS-120B frequently modified the original problem and this ensured we maintained the desired format prescribed in our original problem-formation prompt. We check that the final problem-solution pair is valid Python code via generation of an abstract-syntax tree (AST). If the final function does not pass the final AST check, it is discarded. After the above cleaning procedure, we resulted in the 15 M problems that make up the dataset.

#### 2.3.3 Synthetic Unconditional Algorithmic

To create this dataset, we generated algorithmic Python problems using Qwen3-235B-A22B (the base model) and gpt-oss-120b. We used minimalistic prompts—such as “Write a function,” “Write a Python function,” or “Write a coding problem and solution for a student to solve”—and optionally specified a difficulty level (easy, medium, or hard). To ensure diversity and quality, we prompted gpt-oss-120b to rewrite these samples to handle edge cases, add unit tests, and reformat the outputs in various ways.

In another variant, we instructed gpt-oss-120b to generate LeetCode-style questions and answers, again with a randomly selected difficulty level. We further used gpt-oss-120b to score the correctness of the solution and, if incorrect, to correct it. Overall, such nearly unconditional prompting did result in high rates of duplicates. To combat this, we found it effective to deduplicate based on short titles of around 5–8 words generated by gpt-oss-120b for each problem.

All samples were decontaminated against HumanEval *(Chen et al., 2021)*, MBPP *(Austin et al., 2021)*, CRUXEval *(Gu et al., 2024)*, and LiveCodeBench *(Jain et al., 2024)* as follows: First, exact matches of the solution against those benchmarks were removed. Second, we used Qwen3-Embedding-0.6 to encode Problem and Solution and filtered any data with $>$0.8 similarity with any of the benchmarks.

Although this dataset is small by usual pretraining standards (0.2B tokens), we believe it helps to teach coding practices like edge case handling and reasoning about program execution, as evidenced by improvements of 1-2 points to HumanEval, MBPP, and CRUXEval-O over the Nemotron 3 Nano base checkpoint, when adding these datasets to a redo of the last 100B tokens of 25T token pretraining.

#### 2.3.4 Synthetic Economics

We generated a diverse set of economics multiple-choice questions across various formats, including cloze, calculation, sentence completion, and multiple-response, covering key topics and terms in microeconomics, macroeconomics, and econometrics (e.g., “Statistical Inference and Hypothesis Testing - Type I error” and “Inflation and the Price Level - Inflation rate”) from a curated list. For each topic-term pair, we used Qwen3-235B-A22B-Thinking-2507 to generate multiple questions, each accompanied by a detailed, step-by-step, and well-formatted solution. To enhance the diversity, we further prompted the model to create new and original questions using the initial outputs as reference points. Each question-solution pair underwent model-based verification for clarity, ambiguity, solvability and accuracy.

#### 2.3.5 Synthetic Formal Logic

We synthesized a set of formal logic problems and solutions spanning several tasks, such as translating between natural language and predicate or propositional logic, deriving the antecedents of conditional propositions, and solving logic problems using indirect or complete truth tables. We introduced

---

variability into the generated scenarios, premises, and formulas by incorporating random personas, letters, and/or logic connective (i.e., $\land$, $\lor$, $\supset$, $\equiv$, $\sim$) into the prompt. We generated and evaluated the problems and solutions using Qwen3-235B-A22B-Thinking-2507.

#### 2.3.6 Synthetic Multiple Choice

We construct a multiple-choice question (MCQ) dataset by bootstrapping from the MMLU auxiliary training set *(Hendrycks et al., 2021b)*, which aggregates auxiliary MCQ data from sources such as ARC *(Clark et al., 2018b)*, MC_TEST *(Richardson et al., 2013)*, OpenBookQA *(Mihaylov et al., 2018a)*, and RACE *(Lai et al., 2017)*, etc. Starting from each seed question, we generate multiple similar questions that follow the same task format and difficulty profile, along with corresponding answer options by prompting Qwen3-235B-A22B *(Yang et al., 2025)*. In the second stage, we prompt the DeepSeek-V3 *(DeepSeek-AI, 2025c)* model to solve each generated question by selecting an answer and providing the supporting knowledge or contextual reasoning underlying its choice. To improve answer reliability, we sample multiple independent solution generations for each question using different random seeds. We then apply majority voting over the generated answers to identify the most consistent choice, retaining only those samples whose final answer agrees with the majority and discarding inconsistent or incorrect instances.

Using this pipeline, we generate approximately 3.5M MMLU-style MCQ samples (~1.6B tokens) augmented with explicit, relevant knowledge or reasoning traces. We evaluate the impact of this data via ablation experiments by continued training the Nemotron-Nano-V3 *(NVIDIA, 2025a)* 24.9T-token checkpoint with an additional 100B tokens, out of which 1B tokens are from the generated MMLU-aux-train-SDG data. The results show consistent gains on most of the benchmarks: MMLU improves from 77.22 to 77.51, "MATH Level 5" from 78.55 to 79.05, AIME-2024 improved from 53.3 to 56.7, and MBPP from from 74.8 to 75.2. Performance on other benchmarks remains largely stable with only minor variance, indicating that the synthesized MCQ data primarily strengthens mathematical and structured reasoning capabilities without introducing regressions elsewhere.

#### 2.3.7 Data Mixture and Ordering

We adopt the Nemotron 3 Nano data mixture as described in *(NVIDIA, 2025a)*. Our pretraining corpus spans 16 high-level categories. The largest component is web crawl data, which we partition into five quality-based groups following the Nemotron-CC taxonomy *(Su et al., 2025)*: crawl-medium, crawl-medium-high, and crawl-high, representing progressively higher-quality crawl data, along with their synthetic counterparts, syn-crawl-medium-high and syn-crawl-high, generated from filtered web documents. Beyond web crawl, the mixture includes math *(Mahabadi et al., 2025; Akter et al., 2024)*, Wikipedia, code, Nemotron-CC-Code, academic text, Crawl++, multilingual data, finepdfs *(Kydliček et al., 2025)* and synthetic SFT-style datasets. The SFT-style data is further divided into general-sft, stem-sft, and code-sft. As part of the SFT-style component, we incorporate reasoning-focused datasets into pretraining, motivated by prior findings demonstrating their effectiveness *(Akter et al., 2026)*. Crawl++ consists of OpenWebText, BigScience *(Laurençon et al., 2023)*, and Reddit datasets.

Data blending is designed to balance diversity and quality: sources with comparable estimated quality are assigned similar weights, while higher-quality datasets receive proportionally greater weight in the mixture. Further details on dataset quality estimation and mixture construction are provided in *(Feng et al., 2024)*. We adopt the two-phase curriculum proposed in *(Feng et al., 2024)* work. In Phase 1, the mixture emphasizes data diversity to promote broad coverage and generalization. In Phase 2, the blend shifts toward predominantly high-quality sources (e.g., Wikipedia) to refine

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

model performance. The transition to Phase 2 occurs at  $80\%$  of total training tokens. The specific mixtures used in each phase are illustrated in Figure 10.

![_page_13_Figure_0.jpeg](_page_13_Figure_0.jpeg)
(a) Data mixture of Phase 1.

![_page_13_Figure_1.jpeg](_page_13_Figure_1.jpeg)
(b) Data mixture of Phase 2.
Figure 10 | Data mixtures for each phase of pre-training.

# 2.4. Hyperparameters

The pretraining of Nemotron 3 Super 120B-A12B Base was conducted using a Warmup-Stable-Decay (WSD) (Hu et al., 2024) learning rate schedule over a total horizon of 25 trillion tokens. The learning rate (LR) was warmed up over the initial 200 billion tokens to a peak value of  $4.5 \times 10^{-4}$ . Following a sustained stable plateau phase, we implemented a minus-sqrt decay schedule for the final 5 trillion tokens, annealing the LR to a minimum of  $4.5 \times 10^{-6}$ .

We used AdamW (Loshchilov &amp; Hutter, 2017) optimizer with a weight decay of 0.1 and momentum coefficients  $\beta_{1} = 0.9$  and  $\beta_{2} = 0.95$ . The model was trained with a sequence length of 8,192 and a batch size of 3,072 sequences, resulting in approximately 25.17 million tokens per batch.

The architecture employs a hybrid Mamba-MoE design, featuring Mixture-of-Experts (MoE) layers with 512 total experts and a top-22 routing mechanism ( $k = 22$ ). We utilized a sigmoid router score function complemented by expert biasing. To ensure equitable expert utilization across the 120.6B parameters, we adopted an auxiliary-loss-free load balancing strategy (Wang et al., 2024; DeepSeek-AI, 2025c) with an update rate of  $10^{-3}$ , paired with a standard load balancing loss with coefficient of  $10^{-4}$  (Lepikhin et al., 2020).

Furthermore, we used an MTP objective with loss scaling factor of 0.3. To maximize computational efficiency and training stability at scale, the execution utilized a hybrid precision scheme of BF16 and NVFP4.

# 2.5. Tracking Merge Evaluation

During the stable phase of the WSD learning rate schedule described in Section 2.4, the learning rate remains constant, and individual trained checkpoints exhibit noisy benchmark performance from step to step. Following recent work on weight-space merging (Wortsman et al., 2022; Tian et al., 2025; Ling Team, 2025), we apply checkpoint merging (weighted averaging over a sliding window of recent checkpoints) to produce stronger readouts of model quality without requiring dedicated learning rate decay runs. In a conventional pretraining workflow, evaluating model quality at intermediate checkpoints requires dedicated decay runs; checkpoint merging eliminates this cost. For a schedule comparable to ours, the savings could reach  $\sim 4\mathrm{T}$  tokens of compute (e.g.,  $\sim 2$  avoided runs at  $1.5\mathrm{T}$  and  $\sim 2$  at  $0.5\mathrm{T}$ ), or roughly  $16\%$  of the total pretraining FLOP budget.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

Following Tian et al. (2025), we use a minus-sqrt decay emulation to compute merge coefficients, with checkpoints saved every 1,000 iterations ( $\approx$ 25B tokens at our global batch size of  $3,072 \times 8,192$  tokens). We evaluated sliding merge windows of 125B, 250B, and 500B tokens over the course of pretraining. On average, across a suite of 12 benchmarks (MMLU-Pro, MMLU, HumanEval, HumanEval+, MBPP, MBPP+, GSM8K, MATH-500, RACE, ARC-Challenge, HellaSwag, WinoGrande), the best merge consistently outperforms the corresponding trained checkpoint by 2-4 points on the unweighted average. Since merging is computationally cheap relative to training, we can evaluate all three windows at each checkpoint and select the best. Figure 11 reports this best-of-three merge against the trained checkpoint over the full 25T-token training run.

![_page_14_Figure_0.jpeg](_page_14_Figure_0.jpeg)
Figure 11 | Average accuracy across 12 benchmarks for trained checkpoints versus the best offline checkpoint merge during pretraining. During the stable LR phase, offline merging yields a consistent 2-4 point improvement. During the LR decay phase (shaded), the gap narrows as trained checkpoints benefit from actual learning rate annealing.

During the final 5T-token LR decay phase (from 20T to 25T tokens), the gap between merged and trained checkpoints narrows substantially, and the two evaluation trajectories largely coincide by the end of training. Tian et al. (2025) reported that combining merging with decay offers no gain over merging alone, so this convergence is expected. The original WSM results went further, showing merge-based readouts surpassing decay-trained checkpoints. In experiments on the Nemotron 3 Nano scale architecture (30B-A3B), we were able to reproduce such gains when emulating short ( $\sim$ 500B) decay windows. However, direct comparisons at 1T and 1.5T merge horizons showed no improvement over decay-trained checkpoints.

Our takeaway is that offline checkpoint merging appears most effective for shorter annealing horizons. This is consistent with Ling Team (2025), who employ a comparably short decay schedule and report merge-based improvements, in contrast with the much longer 5T decay used here, where trained decay is able to match or surpass merge-based readouts. The final base model checkpoint selected for downstream alignment was itself a 500B merge; short-horizon merging remains practically useful even alongside a full decay schedule. That said, our experiments explored only a single merge schedule (minus-sqrt) and a fixed checkpoint granularity; it remains plausible that alternative coefficient schemes, finer-grained checkpoint windows, or merging strategies tailored to longer decay horizons could recover the gains observed at shorter scales. Per-benchmark breakdowns are provided in Appendix Figure 17.

---

2.6 Long-Context Extension

Similar to Nemotron 3 Nano, we added a long-context phase (LC-Phase) at the end of pretraining. In the LC-Phase, we performed continuous pretraining (CPT) to equip the base model with long-context ability. We used a constant learning rate of $4.5*10^{-6}$ and global batch size of 16. We used 64-way context parallelism, 2-way tensor parallelism, and 64-way expert parallelism to train on GB200 GPUs. We reused the long-context document QA dataset from Nemotron 2 & 3 Nano. We allocated the document QA data to 20% in the Phase LC data blend, with the remaining 80% being downscaled Phase 2 data. We initially performed CPT on 1,048,576 (1m) context length. Such stage lasted for 34 billion tokens. Following that we added another stage to alternatingly train on both 1m and 4k sequences in order to mitigate the minor impact we observed on the math-related benchmarks. The second stage lasted for 17 billion tokens.

### 2.7 Base Model Evaluations

All evaluation results were collected via Nemo Evaluator SDK and NVIDIA’s open source container of LM Evaluation Harness, unless otherwise stated. For reproducibility purposes, more details on the evaluation settings can be found in the Nemo Evaluator SDK examples folder. The open source container on LM Evaluation Harness packaged via NVIDIA’s Nemo Evaluator SDK used for evaluations can be found here. This container is is built on top of LM Evaluation Harness, with the following changes applied to all models for fairness:

1. For mathematical reasoning, we evaluate GSM8K and MATH *(Cobbe et al., 2021; Hendrycks et al., 2021c)* benchmarks using greedy-decoding. We also highlight the competition-level slice of the MATH benchmark as “MATH Level 5”. Additionally, we report the pass@32 performance on AIME-2024. We use Math-Verify to grade all generations.
2. For code tasks (HumanEval *(Chen et al., 2021)*, MBPP *(Austin et al., 2021)*) we evaluate the EvalPlus variants along with the sanitization of generations *(Liu et al., 2023)*, in a 0-shot setup. We estimate avg@32, pass@1 from 32 generations per prompt.
3. General reasoning benchmarks (OpenBookQA *(Mihaylov et al., 2018b)*, PIQA *(Bisk et al., 2019)*, Hellaswag *(Zellers et al., 2019)*, Winogrande *(Sakaguchi et al., 2019)*) are unchanged except for ARC-Challenge *(Clark et al., 2018a)*, where we present all options at the same time, similar to MMLU *(Hendrycks et al., 2021a)*.
4. For multilingual capability, we evaluate MGSM *(Shi et al., 2022)* (8-shot, native CoT) and Global MMLU-Lite *(Singh et al., 2024)*.

Accuracy results for Nemotron 3 Super 120B-A12B Base with comparsions to Ling-flash-Base-2.0 and GLM-4.5-Air-Base are shown in Table 4.

## 3 Post-Training

We follow the same general recipe as Nemotron 3 Nano, with a stronger emphasis on agentic tasks. An overview of our pipeline is given in Figure 12. We start with Supervised-Finetuning (SFT) phase with details in §3.1 followed by the three stage Reinforcement Learning comprising of RLVR,

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning



<table><tr><td>Task</td><td>Metric</td><td>N-3-Super 120B-A12B-Base</td><td>Ling-flash base-2.0</td><td>GLM-4.5 Air-Base</td></tr><tr><td colspan="5">General Knowledge</td></tr><tr><td>MMLU</td><td>5-shot, acc</td><td>86.01</td><td>81.00</td><td>81.00</td></tr><tr><td>MMLU-Pro</td><td>5-shot, CoT EM</td><td>75.65</td><td>62.10</td><td>58.20</td></tr><tr><td>AGIEval-En</td><td>3/5-shot, CoT EM</td><td>77.92</td><td>61.70</td><td>62.40</td></tr><tr><td>GPQA-Diamond</td><td>5-shot, CoT EM</td><td>60.00</td><td>36.00</td><td>23.20</td></tr><tr><td colspan="5">MATH</td></tr><tr><td>GSM8K</td><td>8-shot, EM</td><td>90.67</td><td>90.75</td><td>82.60</td></tr><tr><td>MATH</td><td>4-shot, EM</td><td>84.84</td><td>63.80</td><td>50.36</td></tr><tr><td>MATH Level 5</td><td>4-shot, EM</td><td>70.00</td><td>39.80</td><td>26.30</td></tr><tr><td>AIME 2024</td><td>pass@32</td><td>53.33</td><td>30.00</td><td>20.00</td></tr><tr><td colspan="5">Code</td></tr><tr><td>HumanEval</td><td>0-shot, pass@1 n=32</td><td>79.40</td><td>70.10</td><td>76.30</td></tr><tr><td>MBPP-Sanitized</td><td>3-shot, pass@1 n=32</td><td>78.38</td><td>77.30</td><td>77.50</td></tr><tr><td colspan="5">Commonsense Understanding</td></tr><tr><td>ARC-Challenge</td><td>25-shot, acc_norm</td><td>96.08</td><td>94.80</td><td>93.90</td></tr><tr><td>HellaSwag</td><td>10-shot, acc_norm</td><td>88.97</td><td>84.69</td><td>87.70</td></tr><tr><td>OpenBookQA</td><td>0-shot, acc_norm</td><td>50.20</td><td>47.00</td><td>48.60</td></tr><tr><td>PIQA</td><td>0-shot, acc_norm</td><td>85.47</td><td>84.00</td><td>84.22</td></tr><tr><td>WinoGrande</td><td>5-shot, acc</td><td>78.93</td><td>78.37</td><td>83.82</td></tr><tr><td colspan="5">Reading Comprehension</td></tr><tr><td>RACE</td><td>0-shot, acc</td><td>91.00</td><td>90.10</td><td>89.50</td></tr><tr><td colspan="5">Multilingual</td></tr><tr><td>MMLU Global Lite</td><td>5-shot, avg</td><td>85.72</td><td>74.94</td><td>79.25</td></tr><tr><td>MGSM</td><td>8-shot, avg</td><td>87.47</td><td>82.73</td><td>80.33</td></tr><tr><td colspan="5">Long Context</td></tr><tr><td>RULER 64K</td><td>0-shot</td><td>93.17</td><td>-</td><td>81.58</td></tr><tr><td>RULER 128K</td><td>0-shot</td><td>89.00</td><td>57.56</td><td>63.62</td></tr><tr><td>RULER 256K</td><td>0-shot</td><td>86.18</td><td>-</td><td>-</td></tr><tr><td>RULER 512K</td><td>0-shot</td><td>82.16</td><td>-</td><td>-</td></tr><tr><td>RULER 1M</td><td>0-shot</td><td>74.39</td><td>-</td><td>-</td></tr></table>



Table 4 | Comparison of Ling-flash-base-2.0, GLM-4.5-Air-Base, and Nemotron Super 120B-A12B Base. Best results are marked in bold.

SWE-RL and RLHF in §3.2. We also do a final phase of MTP. In SFT, we expand the training blend to cover a wider range of agentic harnesses and interaction scenarios. We also significantly improved our RL infrastructure, enabling reliable large-scale asynchronous training on thousands of GPUs. This infrastructure allows us to (1) train across 21 diverse environments, improving robustness across tasks, and (2) train on long-horizon SWE tasks, strengthening multi-step reasoning and problem solving in realistic agentic settings.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_17_Figure_0.jpeg](_page_17_Figure_0.jpeg)
Figure 12: Overview of the post-training pipeline for Nemotron 3 Super.

### 3.1 Supervised Fine Tuning

For Nemotron 3 Super SFT, we focused on improving dataset quality and diversity. In particular, we scaled up our agentic datasets and increased their share in the overall SFT blend. The chat template remains identical to Nemotron 3 Nano. In addition, we add low effort reasoning mode, giving users further control over reasoning length. We found that a single-stage SFT led to a marked degradation on long-input-short-output scenarios. We therefore adopt a two-stage SFT procedure: Stage 1 emphasizes learning from token-level supervision and induces strong reasoning behavior, while Stage 2 switches to per-conversation normalization to prevent long outputs from dominating the loss, which restores long-input-short-output performance while retaining reasoning. We describe it below:

#### SFT objective and two-stage loss

For a packed global batch $\mathcal{B}$ containing multiple conversations $c$, let $\mathcal{O}_{c}$ denote the set of output-token positions for conversation $c$ and $|\mathcal{O}_{c}|$ its output-token count. With token-level negative log-likelihood $\ell_{t}=-\log p_{\theta}(y_{t}\mid x,y_{&lt;t})$, we use:

##### Stage 1: token-level (global) average.

We minimize the average loss over all output tokens in the packed global batch:

$\mathcal{L}_{\text{tok}}=\frac{\sum\limits_{c\in\mathcal{B}}\sum\limits_{t\in\mathcal{O}_{c}}\ell_{t}}{\sum\limits_{c\in\mathcal{B}}|\mathcal{O}_{c}|}.$ (1)

This corresponds to summing the output-token log probabilities across all conversations and normalizing by the total number of output tokens.

##### Stage 2: sample-level average.

We then switch to a per-conversation normalized loss and average equally across conversations:

$\mathcal{L}_{\text{samp}}=\frac{1}{|\mathcal{B}|}\sum\limits_{c\in\mathcal{B}}\left(\frac{1}{|\mathcal{O}_{c}|}\sum\limits_{t\in\mathcal{O}_{c}}\ell_{t}\right).$ (2)

This stage reduces the dominance of long outputs by normalizing each conversation by its own output-token count before averaging across the batch.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_18_Figure_0.jpeg](_page_18_Figure_0.jpeg)
Figure 13 | Agentic Command Line Interface Dataset Construction &amp; Training Pipeline

For Stage 1, we run SFT with 256k sequence length packing, global batch size 64, constant lr  $1e - 5$  with 30k warmup samples. For Stage 2, we use 512k sequence length packing and include long context data with length up to 512K, global batch size 32 and constant lr  $1e - 5$ .

MTP during SFT We continue training Nemotron 3 Super with the same shared-weight MTP head used in pretraining to preserve both the accuracy benefits of multi-step prediction and the inference-time gains from speculative decoding. Concretely, we train two MTP layers with shared parameters and optimize the combined objective using a scaled auxiliary loss computed with per-token loss and 0.3 scaling factor.

# 3.1.1. Data

We reuse the following datasets from the Nemotron 3 Nano SFT datasets: Chat, Infinibyte, and Formal Proofs. We refresh the following datasets with new teacher models (DeepSeek v3.2, Kimi K2): Competition Math, Competition Code, Conversational Tool Use, Multilingual, Science. Below we describe new or heavily modified SFT datasets.

Software Engineering. We curate a dataset of coding tasks derived from real-world GitHub issues to train Nemotron 3 Super for autonomous software engineering capabilities including code exploration, task tracking, issue reproduction and bug fixing. We use the issues and containerized execution environments from the SWE-Gym (Pan et al., 2025), R2E-Gym (Jain et al., 2025) and SWE-rebench (Badertdinov et al., 2025) datasets. For R2E-Gym, we regenerate problem statements with Qwen3-Coder-480B-A35B-Instruct. We distill trajectories from the OpenHands agent harness using Qwen3-Coder-480B-A35B-Instruct as the teacher model.

# Agentic Programming.

The landscape of software development has undergone a substantial shift with the emergence of Agentic Command Line Interface (CLI) tools, moving beyond the "autocomplete" era of 2021-2023 into a regime of autonomous execution. Alongside substantial improvements in harnesses such as Claude Code, OpenCode, and OpenAI's Codex, models are now capable of operating as active

---

digital collaborators, capable of multi-step reasoning, long-horizon execution, and end-to-end task orchestration.

We established a foundational seed set of tasks designed to replicate common user-initiated operations within agentic CLIs. Refer to Figure 13 where we discuss the full pipeline. We utilized NeMo Data Designer *(The NeMo Data Designer Team, 2025)* to generate approximately 20k queries derived from a taxonomy of 24 distinct actions typically performed in these environments. Subsequently, we employed GPT-OSS 120B *(OpenAI, 2025)* in an LLM-as-a-Judge framework to filter out tasks referencing pre-existing codebases or modifications to extant files. This mitigation ensures that the models do not attempt modification operations within empty directories—a scenario that frequently results in redundant and exhausted tool invocations during failed execution cycles. The resulting dataset comprises roughly 15k tasks centered on direct solution synthesis. To further enhance the diversity of the generated outputs, we coupled each task with a supplementary markdown specification, equivalent to an AGENTS.md file. These documents impose additional constraints and architectural requirements, effectively narrowing the design space and necessitating more complex, varied solutions from the agent.

Since we eliminate all tasks that require a pre-existing codebases, we augment this task set with roughly 3000 questions from SWE tasks that are challenging and come with pre-existing repository and specific git hash commit. We apply a simple prompt prior to the existing issue statements, stating that the execution environment does not have the library installed and therefore execution of unit tests should be avoided. This is a conscientious decision made to substantially reduce the engineering effort required to support per sample container based execution of each SWE task, reduce the number of tool calls and avoid large tool outputs due to multiple rounds of unit test executions. This further relaxes the constraints imposed on the agent to solve the issue, as there is no SWE specific prompt to guide the agent in solving the task. Finally, we synthesize 10k web development tasks using a taxonomy of 100 fine-grained tasks that are commonly requested by users as the seed, on which we apply LLM-as-a-Judge to eliminate tasks that require a pre-existing repository. We impose no restrictions on these tasks, but provide only a Node.js environment and expect the agent to setup and install all dependencies and plugins on its own.

Applying these task sets, we distill from high-performance, open-source agentic LLMs such as Qwen-3-Coder-480B *(Qwen, 2025)* and Minimax M2.5 *(MiniMax AI, 2025)* by recording their interactions with various CLI environments, such as Codex, OpenCode, Qwen Code CLI, and Stirrup. These interaction traces are subsequently filtered, normalized into the standard OpenAI message format with various tool definitions, and utilized for large-scale SFT to effectively embed agentic operational knowledge within the model. For each Agentic CLI, we study the individual capabilities that exist and are commonly utilized. We apply the same task sets to target different capabilities, such as Agent Skills, tool restriction (bash only execution), ask user clarifying questions, single and multi step planning, static and dynamic multi turn conversations and parallel tool calling, depending on whether the specific CLI can accommodate these capabilities.

Long Context. We extend the long-context SFT dataset from Nemotron 3 Nano with a more comprehensive synthetic data pipeline. To improve long-context multi-document reasoning, we construct a synthetic SFT dataset using long sequences from our pre-training blend, which contains books, papers, financial reports, code repositories, etc. We first cluster these documents by topic/domain and concatenate related documents to reach target sequence lengths, such as 128K, 256K, or 512K tokens. For each long-context sample, we use an LLM to generate one or more QA pairs. The prompt requires questions to involve cross-document or cross-section navigation, ensuring information is scattered rather than localized. It strictly enforces multi-hop reasoning, requiring at least 4 to 7 distinct retrieval or reasoning steps. These steps mandate computational or logical processing,

---

preventing simple copy-pasting, and often include explicit formatting instructions. Next, we generate 8 independent reasoning traces for each context-question pair. We apply semantic majority voting to group the answers, either by exact match or via an LLM judge. From the resulting majority group, we select the answer containing the shortest reasoning trace. Additionally, we generate seven synthetic reasoning tasks to improve the model’s ability to process context in a sequential, left-to-right manner. Specifically, synthetic snippets (generated using Qwen3-235B-A22B-Thinking-2507) are concatenated together to form long input context. The thinking traces are constructed by chaining rule-based reasoning steps, each of which includes relevant excerpts from the input context and tracking metadata, such as the frequency of query-related snippets. We also construct long-context samples by concatenating records from *(Meyer and Corneil, 2025)* to reach the required sequence length. Questions are designed to emphasize multi-hop reasoning and information aggregation across records. Context, question, and answer are formatted using pre-defined templates, with ground-truth answers derived by executing SQL queries over the underlying records.

Financial Reasoning. To construct a large-scale training corpus for financial reasoning, we employ a template-based synthetic data generation (SDG) pipeline that scales a curated seed set into hundreds of thousands of grounded question–answer pairs. The pipeline sources 565 expert-authored seed questions from the SecQue benchmark *(BenYoash et al., 2025)*, a dataset of financial analysis questions anchored to SEC 10-K and 10-Q filings. These seeds are expanded combinatorially across S&P 500 companies and fiscal years (2019–2024), where comparative questions are restricted to company pairs within the same GICS Sub-Industry to preserve semantic coherence. GPT-OSS-120B paraphrases each template instantiation, producing up to three diverse reformulations per combination. The resulting questions are mapped to relevant SEC filing sections using the original SecQue metadata, and the corresponding documents are converted to markdown with a configurable token limit. For answer generation, we adopt the GenSelect strategy *(Toshniwal et al., 2025)*: five candidate answers are sampled per question using GPT-OSS-120B with distinct random seeds, and a larger judge model (Qwen3-235B-A22B) selects the best response based on numerical accuracy, financial methodology, and logical soundness. A smaller model (Qwen3-30B-A3B) then classifies each pair as ANSWERABLE or UNANSWERABLE, retaining only those containing a complete, substantive response. Prior to supervised fine-tuning, the SDG output undergoes percentile-based outlier removal and deduplication. The resulting dataset comprises 366,243 financial Q&A pairs with reasoning traces.

CUDA. A large-scale synthetic CUDA dataset comprising 100K samples for kernel generation, repair, and optimization was constructed using a synthetic data generation pipeline based on DeepSeek-R1 and GPT-OSS-120B. Seed questions were sourced from popular open-source libraries, NVIDIA library API surfaces, and BackendBench *(Saroufim et al., 2025)*. These seeds were used to generate tuples of the form (PyTorch reference, CUDA C++ kernel) and (natural language specification, CUDA C++ kernel), each accompanied by reasoning. For each seed item, multiple candidate kernels were generated and rigorously validated for correctness within an internal CUDA evaluation environment. The validated kernels were then ranked by performance, and the highest-performing kernel was retained. In addition, we collected traces from an internal CUDA agent, producing samples of the form (PyTorch reference, faulty CUDA C++ kernel, error message, corrected CUDA C++ kernel) and (PyTorch reference, slow CUDA C++ kernel, Nsight Compute log, optimized CUDA C++ kernel). Using publicly available documentation and official code samples, and following the same formulation as the CUDA-C data, we generated additional PyTorch references and corresponding CUDA-library implementations with reasoning chains, as well as aligned natural language specifications. These libraries include Thrust, CUB, cuBLAS, cuDNN, cuSPARSE, cuRAND, and cuSOLVER.

---

Safety. We have significantly enhanced our safety framework compared to Nemotron 3 Nano by combining a robust prompt library with a two-stage synthetic response generation strategy. While retaining the core prompts from Nemotron Content Safety v2 *(Ghosh et al., 2025)*, Gretel Safety Alignment v1 *(Gretel, 2024)*, Harmful Tasks *(Hasan et al., 2024)* and Red-Team-2K *(Luo et al., 2024)* covering content safety and common jailbreak techniques, we added in new synthetic prompts targeting the elicitation of over-refusals, demographic biases, and copyright reproduction. We also expand coverage of jailbreak strategies to better capture emerging adversarial techniques including indirect prompt injection attacks.

Our primary advancement over Nemotron 3 Nano is an explicit response policy framework. For each prompt, a response policy is inferred from a combination of prompt metadata, its annotated safety category, and predictions from a set of lightweight auxiliary classifiers that detect attributes such as self-harm risk, demographic targeting, or the presence of embedded adversarial instructions. We cast this decision process as a multi-class classification problem, where each class corresponds to a distinct response mode aligned with safety guidelines. These response modes specify whether the model should provide supportive resources such as helpline information, issue a refusal with a brief explanation, or answer the benign portion of the request while ignoring malicious content. This ensures that the responses are safe, contextually appropriate, and consistent across a diverse set of safety-sensitive scenarios.

Following the deliberative alignment framework *(Guan et al., 2025)*, we adopt a two-stage generation process in which the reasoning trace and the final response are curated separately but consistently with our response policy. In the first stage, we construct a concise reasoning trace that guides the model to reflect on the safety properties of the prompt, explicitly identifying why the request may be unsafe or policy-relevant and what constraints should govern the response. In the second stage, we generate the final response based on this reasoning trace, ensuring that it adheres to the predefined response policy and behavior guidelines. This structured separation encourages deliberate reflection on safety guidelines while producing responses that are consistent, policy-compliant, and contextually appropriate to ensure that the final response follows safety policies while minimizing unnecessary references to said policies. Finally, we apply a content-moderation classifier to filter any responses flagged as unsafe, providing an additional safeguard to ensure alignment with safety objectives.

Search. To improve search capabilities, we generated a synthetic search-agent SFT dataset using NeMo Data Designer *(The NeMo Data Designer Team, 2025)*. The pipeline begins by constructing seed prompts grounded in the Wikidata knowledge graph *(Vrandečić and Krötzsch, 2014)*: we query SPARQL *(Harris and Seaborne, 2013)* for well-connected hub entities across approximately 25 verified entity classes (cities, universities, films, chemical elements, etc.), then perform random walks of 4-8 hops through the graph, filtering out degenerate paths via stop-node lists, anti-meta-relation exclusions, and minimum path-length thresholds. Each valid walk yields a start entity, a chain of factual relations, and a final answer entity.

Data Designer then processes these seeds in three stages: (1) a draft stage converts the structured knowledge-graph walk into a natural-language multi-hop question, (2) an obfuscation stage rewrites the question to hide intermediate entities and eliminate breadcrumb-style chaining – producing search-riddle queries where the solver must decompose the problem independently – and (3) an agent stage in which MiniMax-M2 *(MiniMax AI, 2025)* solves the obfuscated question by issuing web searches via the Tavily (https://tavily.com) MCP search tool, producing a grounded search trajectory with supporting URLs. Each resulting SFT record is a multi-turn conversation where assistant turns interleave chain-of-thought reasoning with structured tool calls, and tool-response turns return search results as JSON, preserving a full Thought–Action–Observation loop across an average of 12 tool calls per trajectory. A final structured-output stage normalizes the agent’s raw

---

response into a validated JSON schema.

Terminal Use. The dataset for enhancing terminal capabilities follows the dual-stream *Terminal-Task-Gen* methodology described in Nemotron-Terminal *(Pi et al., 2026)*, comprising a total of 84,864 samples. This pipeline combines the adaptation of existing high-quality datasets with synthetic task generation grounded in a comprehensive terminal skill taxonomy. The source distribution consists of 68,924 synthetic samples, 8,125 samples from Nemotron-Cascade-Math, and 7,815 samples from Nemotron-Cascade-Code *(Wang et al., 2025a)*. For trajectory construction, we use DeepSeek-V3.2 *(DeepSeek-AI, 2025a)* as the primary engine to generate step-by-step solution traces within isolated, Dockerized environments through an agentic execution-feedback loop. All samples are generated using the Terminus 2 agent framework *(Merrill et al., 2026)* as the underlying scaffolding, providing a unified set of terminal tools and a structured interaction protocol to maintain consistency and quality across long-horizon trajectories.

Multilingual. Our multilingual data combines synthetic translations of English SFT examples with a sentence-level parallel corpus to improve machine translation. We reuse the line-by-line translation pipeline from Nemotron 3 Nano, translating into six languages (German, Spanish, French, Italian, Japanese, and Chinese) using Qwen2.5-Instruct-14b. After translation, we apply filtering to remove samples in the wrong language and other common failure modes. We observed a recurring pattern where translation disrupts the alignment between prompt specifications and answer formats — a consistency usually maintained in English data. This mismatch led to instruction-following failures during preliminary testing. To mitigate this, we introduced a lightweight post-editing step with Qwen3-4B-Thinking-2507 to automatically restore format compliance. We also expand the parallel corpus with additional Chinese $\leftrightarrow$ English pairs and exclude very short samples that previously degraded post-training performance.

Structured Query Language (SQL). To improve Nemotron 3 Super on enterprise SQL workloads, we generate a synthetic text-to-SQL dataset with NeMo Data Designer *(The NeMo Data Designer Team, 2025)*. The dataset contains 96.5k records spanning MySQL, PostgreSQL, and SQLite across 60 industry sectors, $\sim$700 domain topics, and 90 SQL concept buckets (from basic SELECT to recursive CTEs, window functions, and geospatial queries). Each sample pairs a natural-language prompt and a fully synthetic database schema context with a target SQL query. To improve robustness and to mimic the real-world messiness of production databases, the pipeline injects distractor tables and columns into the database context. Specifically, related but irrelevant tables and columns are added to the database context, forcing the model to learn to ignore irrelevant schema elements. Prompt diversity is controlled along three axes – instruction style (imperative, declarative, interrogative, contextual, abbreviated), linguistic register (formal, conversational, technical, academic, direct), and politeness level – yielding naturalistic and varied user requests. The final dataset of 96.5k records is validated and filtered down by Data Designer from a larger dataset using per-dialect syntax validators and five LLM-as-a-critic judges.

Conversational Tool Use. Large-scale specialized tool-use training data has been adopted by many models to boost agentic capabilities *(Liu et al., 2024; DeepSeek-AI, 2025a; Team, 2025; GLM-4.5-Team, 2025)*. For Nemotron 3 Super, we scale conversational tool-use data via a fully synthetic, six-stage generation pipeline:

1. Domain Generation: Sample synthetic domains with a model, iteratively expanding initial generations into specialized subdomains.
2. Policy and Tool Generation: Sample customer-service policies and related tools, iteratively improving them via self-refinement; use few-shot prompting and an LM-as-a-Judge for quality filtering to maintain style and formatting.
3. Scenario Generation: Generate plausible user personas, background information, and inquiries

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_23_Figure_0.jpeg](_page_23_Figure_0.jpeg)
Figure 14 | Overview of the synthetic data generation pipeline for specialized conversational tool-use SFT data used for Nemotron 3 Super.

for each policy setting.

4. Trajectory Collection: For each policy-scenario pair, simulate 16 customer service interactions among a model-based agent, user, and environment.
5. Verification: Evaluate trajectories at both outcome and process levels using an LM-as-a-Judge.
6. SFT Data Selection: Select successful trajectories for SFT and filter for difficulty by dropping scenarios that yield all-success or all-failure outcomes.

A visualization of the pipeline is shown in Figure 14. We utilize Qwen3-235B-A22B-Thinking-2507, Qwen3-32B, Qwen3-235B-A22B-Instruct-2507 (Yang et al., 2025), deepseek-r1-0528 (DeepSeek-AI, 2025b), DeepSeek-V3.2 (DeepSeek-AI, 2025a), and gpt-oss-120b (OpenAI, 2025) on various parts of the above pipeline, yielding 279,116 conversations across 838 domains. This represents a substantial scale-up over Nemotron 3 Nano, which used 15,588 conversations spanning 5 domains.

General-Purpose Tool Use. The broader, general purpose tool-calling synthetic data pipeline begins with the construction of diverse tool sets from ToolEyes (Ye et al., 2025), API-Bank (Li et al., 2023b), UltraTools (Huang et al., 2024), AutoTools (Shi et al., 2025), xLAM (Zhang et al., 2024), Glaive-Function-Calling-v2 (Glaive AI, 2025), Toucan-1.5M (Xu et al., 2025), as well as custom written tools that serve as the foundation for downstream synthetic task generation. A tool-calling trajectory is simulated by grounding in one or multiple of these tool sets. The trajectory simulation involves an LLM playing three roles - User (User-LLM), Assistant (Assistant-LLM), and Tool Environment (Tool-LLM). The User-LLM is seeded with the selected tool set, a persona sampled from Nemotron-Personas-USA (Meyer &amp; Corneil, 2025), and a tool-calling scenario (single-turn, multi-turn, or multi-step). The User-LLM starts by designing a task guided by the tool-calling scenario which is relevant to the selected persona and can be solved by the selected tool set. The Assistant-LLM attempts to solve this task in one or more turns by producing tool-calls and responding to tool execution results. The Tool-LLM is responsible for producing a simulated tool execution result based on the tool-call generated by the Assistant-LLM and the tool being called. The Tool-LLM is prompted with a rubric that helps identify syntactic and semantic errors in tool-calling, as well as the original user query so that the tool results can be contextualized when tool-call is successful. To ensure accuracy, we employ a turn-level and trajectory-level judge similar to the specialized tool-calling data generation. The turn level judge is also paired with a rule-based verification for ensuring correctness of tool-calls. We scale this pipeline with DeepSeek-v3.2 (DeepSeek-AI et al., 2025) and GLM-4.7 (Z.ai / zai-org, 2025) to create a dataset of 1.5M diverse tool-calling trajectories.

The overall general-purpose synthetic tool-calling data pipeline is visualized in figure 15.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

![_page_24_Figure_0.jpeg](_page_24_Figure_0.jpeg)
Figure 15 | Overview of the pipeline for general-purpose tool-calling data used in Nemotron 3 Super.

![_page_24_Figure_1.jpeg](_page_24_Figure_1.jpeg)
Figure 16 | SFT data blend for Nemotron 3 Super.

# 3.1.2. SFT Data Blend

Our general stage 1 SFT data blend can be found in Figure 16 (all datasets not listed make up less than  $1\%$  of the blend). We train on over 7M total samples. In stage 2, we use  $85\%$  of the stage 1 blend and augments it with 256K and 512K-token long-context data. Compared to Nemotron 3 Nano, we significantly increased the volume and diversity of agentic tasks, and allocate it a much larger proportion of our blend.

# 3.1.3. Reasoning Control

Nemotron 3 Super is trained for three reasoning modes: reasoning-off, regular and low-effort. The low-effort reasoning mode is a new addition. The regular and low-effort reasoning modes have the option to be used in conjunction with inference-time budget control (NVIDIA, 2025b). These combinations of controls provide flexibilities that cover the entire spectrum of accuracy-efficiency trade-off to meet customers' needs in various application scenarios.

---

The low-effort reasoning mode is introduced during the SFT stage by adding training samples generated by GPT-OSS-120B in its low-effort mode *(Du et al., 2025)*. These low-effort training samples cover the tasks of math reasoning, STEM question answering and instruction following, and represent 2% of the overall SFT data by sample count. The low-effort mode is later optimized during the RL stages to be discussed in the next section.

The SFT recipe for reasoning-off mode and for inference-time budget control is similar to NVIDIA (2025a) with a few differences. We strip the reasoning traces from a random 3% samples for reasoning-off mode. After the main SFT stage, we add a short semi-on-policy SFT stage of 350 steps for inference-time budget control, where we collect roll-outs from the model and truncate 12% of reasoning traces to random reasoning budgets.

### 3.2 Reinforcement Learning

The RL phase of Nemotron 3 Super post-training consists of three stages followed by an MTP healing stage as shown in Figure 12:

- Stage 1: Multi-environment RL from Verifiable Rewards (§3.2.1). This is the primary training stage, where we optimize Nemotron 3 Super jointly across the full set of environments. Training in a unified mixture keeps each RL update informed by the complete environment distribution and helps prevent regressions on individual tasks over the course of training.
- Stage 2: SWE-RL for end-to-end software engineering tasks (§3.2.2). We run SWE-RL as a separate stage because SWE rollouts are substantially slower to generate and typically require longer context lengths, creating a throughput bottleneck when co-trained with shorter-horizon environments. Isolating this stage allows us to tune rollout and batching settings for long-horizon, long-context trajectories.
- Stage 3: RLHF (§3.2.3). We finally apply RLHF as a distinct stage to improve instruction-following behavior, robustness, and overall interaction quality.
- Stage 4: MTP Healing. In this stage we train the MTP heads and keep rest of the weights frozen. We re-use the prompts from RLVR, and train the MTP head using the negative log likelihood loss similar to SFT on the generated responses. We find this stage significantly improves MTP accuracy.

We describe these stages below including the training algorithm, data and systems setup.

#### 3.2.1 Stage 1: Multi-environment RL from Verifiable Rewards

We employ a unified RLVR strategy similar to Nemotron 3 Nano, but significantly scale the number of environments. We find that training on all environments simultaneously yields stable gains, whereas single-environment training leads to severe regressions on other benchmarks.

Our RLVR setting contains 21 environments covering diverse domains, including math, code, STEM, safety, chat, instruction following, long context capabilities, puzzles, and various agentic tasks. For data mixture and curriculum, we adopt an approach similar to Nemotron 3 Nano: we filter out prompts where the SFT model consistently provides correct answers, then sort the remaining samples via a difficulty-based curriculum. Further details of this methodology are available in Nemotron 3 Nano.

##### Low-effort Reasoning

During the multi-environmental RL stages, we convert a subset of the prompts to be in the low-effort mode. For each low-effort prompt, the reward for a roll-out is adjusted

---

as a function of both correctness and the number of generated tokens. The low-effort prompt mix starts with subsets of Math, STEM QA and competitive coding prompts, in total representing 2% of all RL prompts being in low-effort mode, and is later reduced to subsets of Math and STEM QA, representing just 1% of RL prompts. For Math and STEM QA, we randomly sample a subset. For competitive coding, we have a set of coding problems that have been withheld from SFT data and we only sample from this withheld set for low-effort coding prompts. Empirical results suggest that this data strategy provides sufficient generalization and the low-effort mode is improved across a wide set of benchmarks during the course of multi-environmental RL.

#### 3.2.2 RLVR Data

We scale up our RL data significantly compared to Nemotron 3 Nano. We describe the RL datasets we used in our multi environment RL below. The majority of the RL training environments are open sourced in Nemo Gym *(NVIDIA, 2025a)*. In total, we train on 21 environments and 37 different RL datasets.

- Math. We use the same competitive math problems as Nemotron 3 Nano. For this dataset, we train both with and without a Python execution tool. We also introduce a new environment for formal proof verification.
- Code. We train on competition-style code data from Nemotron 3 Nano and *Wang et al. (2025a)*.
- STEM. We include the STEM datasets from Nemotron 3 Nano and add newly curated, more challenging scientific problems.
- Instruction Following. We augment the instruction-following data from Nemotron 3 Nano with a new multi-challenge dataset. In this setting, the agent must follow complex user instructions, with rewards computed from a predetermined rubric.
- Safety. We add two environments: one targeting reduced over-refusals on safety-related prompts, and one improving robustness to jailbreaks. For jailbreaks, seed prompts come from our SFT data; to surface harder attacks during RL, we apply an iterative attack pipeline following PAIR *(Chao et al., 2024)*, attacking an early SFT-only checkpoint and collecting modified prompts with high attack success rates.
- Long Context. We use the same long-context environment introduced in Nemotron 3 Nano.
- Agentic Tool Use. Beyond the tool-use environments from Nemotron 3 Nano, we add new environments focused on conversational tool use and terminal use.
- Reasoning Gym. We train with Reasoning Gym *(Stojanovski et al., 2025)*, enabling learning over a diverse suite of reasoning tasks.

#### 3.2.3 Stage 2: End-to-end RL for Software Engineering

In the SWE-RL stage, we improve the model’s ability to autonomously solve GitHub issues under diverse harnesses. Each rollout launches an Apptainer container with the target repository, runs an OpenHands agent loop to produce a code patch, and evaluates it against ground-truth tests for a binary reward. For tool diversity, we implemented OpenCode and Codex agent classes within OpenHands that match the tool formats of Claude Code and Codex CLI, reusing a single harness while varying tools and prompts at training time. This multi-harness training improves the model’s generalization and performance across all target harnesses at inference time.

---

3.2.3 Stage 3: Reinforcement Learning from Human Feedback

We follow a similar approach to Nemotron 3 Nano for RLHF, training a large GenRM model to provide supervision during RL. Rather than using a vanilla GenRM, we train a principle following GenRM as in *Wang et al. (2025b)*. These principles allow us to guide Nemotron 3 Super’s behavior on important domains like identity and safety related topics. Similar to Nemotron 3 Nano, we use Qwen3-235B-A22B-Thinking-2507 as the initialization for training the GenRM.

To train the GenRM we use the Helpsteer 3 dataset *(Wang et al., 2025c)*, commercially friendly subsets of the lmarena-140k dataset *(Chiang et al., 2024)*, and some more recently collected human preference data. Unlike Nemotron 3 Nano, we train using our GenRM throughout our multi environment RL stage and also perform a separate RLHF-only stage at the end of post training.

#### 3.2.4 Algorithm

We use an asynchronous GRPO setup in which training and inference are decoupled across separate GPU devices. Inference workers continuously generate trajectories, which are stored in a rollout buffer. Once enough trajectories are collected to form a batch, the batch is sent to the training engine for a model update. We push the updated weights to the inference workers as soon as a new model version is available. Because weight updates can happen mid-rollout, a single trajectory may contain tokens produced by different model versions. We do not recompute the KV cache after updating the model weights on the inference workers. To avoid excessive policy lag, which may result in accuracy degradation, we restrict the inference workers to be at most one step behind the latest model version.

To stabilize the training and minimize off-policy effects caused by the training-inference mismatch and policy lag, we mask the importance sampling ratio computed from the training and inference logprobs *(Shao et al., 2024; Team et al., 2025; Yao et al., 2025)*.

In multi-environment RLVR, we sample 256 prompts per step and generate 16 responses per prompt. We train with a batch size of 4096, which corresponds to a single gradient update per rollout. We begin training with a maximum generation length of 49K tokens and later increase it to 64K.

Agentic RL - PivotRL: Post-training for long-horizon agentic capabilities has a tension between efficiency and accuracy. By long-horizon, we mean tasks that require many turns of interaction with an environment, such as conversational tool use, code editing, terminal interaction, and web search. SFT is cheap and simple for this task, but it often degrades performance outside of the target domain (OOD). End-to-end RL avoids that outcome to a large part, but it is costly because every update requires online interactive rollouts in complex environments. To address this, during the post-training of Super, we adopt PivotRL.

PivotRL is a assistant-turn-level RL method that addresses this tradeoff by reusing offline SFT expert trajectories during RL. It focuses training on informative turns (called "pivots") within those SFT traces, where the policy has uncertainty over the next action, and it uses a domain-appropriate reward to match the policy’s action to the expert action, so the model gets credit for similar actions rather than the exact expert action. We notice that this method greatly improves the efficiency of our agentic RL, without facing the OOD degradation issues of SFT.

We apply PivotRL for all agentic domains: including for Agentic Programming, Search, Terminal Use, and Conversational Tool Use. We will have a manuscript with more details soon.

---

#### 3.2.5 Infrastructure

RL at the frontier of model post-training is currently defined by scaling up to an increasing diversity of tasks or environments designed for the model to learn increasingly general capabilities. Scaling RL to many environments requires a high-performance, extensible, and standardized interface for coordinating between rollouts and training. To address the scaling performance and extensibility challenges using one standard framework, we adopt NeMo Gym *(NVIDIA, 2025a)* and NeMo RL *(NVIDIA, 2025b)* for enabling large-scale RL on many different environments/verifiers.

NeMo Gym is based on the abstraction of servers. There are three core varieties of servers in Gym: (1) agents, (2) models, and (3) resources. An agent server implements the rollout kernel of a RL environment. A model server wraps an inference engine such as vLLM *(Kwon et al., 2023)* to provide a prompt-response API, and also carefully preserves token and inference log-prob data and metadata required for RL. A resource server provides a verification API for computing rewards from a given rollout.

Our Nemotron Super 3 RLVR experiments were all based on an integrated infrastructure of NeMo RL and NeMo Gym: NeMo RL acts as the RL training loop controller, using Megatron-Core *(Shoeybi et al., 2020)* for model training at scale, and routing all rollouts through NeMo Gym and vLLM.

NeMo RL and NeMo Gym use ray for orchestration and resource management and deploy the ray cluster on SLURM. Megatron training workers, vLLM generation workers, Gym environments and judge models are all scheduled onto a single ray cluster.

##### Async RL Infrastructure:

All RL stages used asynchronous RL, where generation can happen independently of training which improves training efficiency by trading off how on-policy the rollouts are. The trainings used one-step off policy where each training step was earmarked for a future training step so there were no wasted rollouts. Training and generation were not collocated which simplifies deployment and avoids having to orchestrate complex memory management between the async training and generation workers.

All asynchronous RL runs also used in-flight weight updates *(Piché et al., 2025)* where training can update generation worker weights without waiting for the remaining ongoing rollouts to finish. The result is a single rollout can have tokens and log probabilities from differently aged policies. Enabling in-flight weight updates is critical for speeding up asynchronous RL training. We did not recompute the KV cache after in-flight weight updates.

##### Resiliency:

As we scaled up to 1k GPUs, we encountered several issues that caused intermittent failures that were not observed in smaller job shapes. These issues fell into two categories (1) hardware related and (2) software related.

We observed several hardware issues that required full restart of the job, so several optimizations were also made to improve startup time by (1) parallelizing all initialization (2) prefetching all virtual environments and binaries (3) utilize caching in upstream repos like vLLM and flashinfer.

Parallel initialization exacerbated latent race conditions with port bindings in the post-training software stack, which became a significant failure point. Several components of the post-training software required ports: (1) Ray control plane (2) vLLM workers and OpenAI servers (3) TCP rendezvous (4) NeMo Gym servers. Due to the high number of processes needing ports on a node, we hit port conflict frequently at 1K GPU scale that were all time-of-check to time-of-use (TOCTOU) race conditions. The pattern we observed was that a component would check if a port was available without claiming it exclusively, and by the time it (or another process it notified) attempted to bind, another process had already claimed the port.

---

##### SWE-RL Infrastructure:

Training a model on software engineering tasks requires a gym environment that can execute hundreds of concurrent agent-codebase interactions, each within an isolated sandbox, and return a reward signal derived from real test execution. In the SWE-RL environment in Nemo-Gym, each rollout launches an Apptainer container with the target repository, runs the OpenHands agent loop to produce a code patch, and runs the ground-truth tests to compute a binary reward. Rollouts are distributed across nodes using Ray with a SPREAD scheduling strategy. Below we describe the key components of this environment.

- Container Execution with Apptainer. The absence of root access on our cluster restricts the use of Docker for container isolation. Instead, we use Apptainer (formerly Singularity) to run each SWE task instance in a pre-built container image (.sif files), providing filesystem isolation via a writable tmpfs overlay while sharing the host kernel.
- OpenHands Agent Loop. The agent loop is executed by a modified version of OpenHands, managing the full lifecycle of each interaction: initializing the runtime, presenting the problem statement, running the agent’s step loop upto a configurable turn limit, extracting the git patch, and cleaning up. The agent interacts with the repository workspace through bash commands and file operations via a tmux-based session.
- Memory Management. Since Apptainer containers share host memory and processes (unlike Docker’s cgroup isolation), runaway agent processes can cause OOM conditions affecting the entire node. We implemented a memory watchdog daemon that monitors the aggregate RSS of the tmux process tree and proactively kills processes inside panes when a configurable limit is exceeded, while keeping the tmux server alive for graceful recovery.
- Command Blocklist. The shared-kernel nature of Apptainer means an agent issuing killall or pkill could terminate training processes or vLLM servers on the same node. A regex-based command block list intercepts and blocks dangerous commands before execution, returning informative error messages with safer alternatives.
- Harness Diversity. To increase tool diversity during training without building separate harness integrations, we implemented OpenCode and Codex agent classes within OpenHands that match the tool input/output formats of their respective external harnesses (Claude Code and Codex CLI). Both agents plug into OpenHands’ existing runtime and conversation memory, inheriting container management and observation handling.
- Serialization. We replaced Python’s standard json with orjson for serialization of HTTP payloads between the gym and the model server, as each trajectory turn carries prompt token IDs, generated token IDs, and log probabilities, resulting in large payloads that benefit from orjson’s Rust-based implementation.

### 3.3 Post-trained Model Evaluations

We evaluate Nemotron 3 Super on the same broad benchmark suite and evaluation stack as Nemotron 3 Nano *(NVIDIA, 2025a)*, covering general knowledge, reasoning, agentic, instruction following, long-context, and multilingual capability. All evaluation results were collected via Nemo Evaluator SDK and for most benchmarks, the Nemo Skills Harness. For reproducibility purposes, the open source container on Nemo Skills packaged via NVIDIA’s Nemo Evaluator SDK used for evaluations can be found here. In addition to Nemo Skills, the evaluations also used dedicated open-source packaged containers for Tau-2 Bench (default prompt), Terminal Bench Hard (48 tasks), ScaleAI Multi Challenge Multi-turn Instruction Following, Ruler. More details on the evaluation settings

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning



<table><tr><td>Benchmark</td><td>N-3-Super</td><td>Qwen3.5-122B-A10B</td><td>GPT-OSS-120B</td></tr><tr><td colspan="4">General Knowledge</td></tr><tr><td>MMLU-Pro</td><td>83.73</td><td>86.70</td><td>81.00</td></tr><tr><td colspan="4">Reasoning</td></tr><tr><td>AIME25 (no tools)</td><td>90.21</td><td>90.36</td><td>92.50</td></tr><tr><td>HMMT Feb25 (no tools)</td><td>93.67</td><td>91.40</td><td>90.00</td></tr><tr><td>HMMT Feb25 (with tools)</td><td>94.73</td><td>89.55</td><td>-</td></tr><tr><td>GPQA (no tools)</td><td>79.23</td><td>86.60</td><td>80.10</td></tr><tr><td>GPQA (with tools)</td><td>82.70</td><td>-</td><td>80.09</td></tr><tr><td>LiveCodeBench (v5 2024-07←2024-12)</td><td>81.19</td><td>78.93</td><td>88.00</td></tr><tr><td>SciCode (subtask)</td><td>42.05</td><td>42.00</td><td>39.00</td></tr><tr><td>HLE (no tools)</td><td>18.26</td><td>25.30</td><td>14.90</td></tr><tr><td>HLE (with tools)</td><td>22.82</td><td>-</td><td>19.0</td></tr><tr><td colspan="4">Agentic</td></tr><tr><td>Terminal Bench (hard subset)</td><td>25.78</td><td>26.80</td><td>24.00</td></tr><tr><td>Terminal Bench Core 2.0</td><td>31.00</td><td>37.50</td><td>18.70</td></tr><tr><td>SWE-Bench (OpenHands)</td><td>60.47</td><td>66.40</td><td>41.9</td></tr><tr><td>SWE-Bench (OpenCode)</td><td>59.20</td><td>67.40</td><td>-</td></tr><tr><td>SWE-Bench (Codex)</td><td>53.73</td><td>61.20</td><td>-</td></tr><tr><td>SWE-Bench Multilingual (OpenHands)</td><td>45.78</td><td>-</td><td>30.80</td></tr><tr><td colspan="4">TauBench V2</td></tr><tr><td>Airline</td><td>56.25</td><td>66.0</td><td>49.2</td></tr><tr><td>Retail</td><td>62.83</td><td>62.6</td><td>67.80</td></tr><tr><td>Telecom</td><td>64.36</td><td>95.00</td><td>66.00</td></tr><tr><td>Average</td><td>61.15</td><td>74.53</td><td>61.0</td></tr><tr><td>BrowseComp with Search</td><td>31.28</td><td>-</td><td>33.89</td></tr><tr><td>BIRD Bench</td><td>41.80</td><td>-</td><td>38.25</td></tr><tr><td colspan="4">Chat &amp; Instruction Following</td></tr><tr><td>IFBench (prompt)</td><td>72.56</td><td>73.77</td><td>68.32</td></tr><tr><td>Scale AI Multi-Challenge</td><td>55.23</td><td>61.50</td><td>58.29</td></tr><tr><td>Arena-Hard-V2</td><td>73.88</td><td>75.15</td><td>90.26</td></tr><tr><td colspan="4">Long Context</td></tr><tr><td>AA-LCR</td><td>58.31</td><td>66.90</td><td>51.00</td></tr><tr><td>RULER 256k</td><td>96.30</td><td>96.74</td><td>52.30</td></tr><tr><td>RULER 512k</td><td>95.67</td><td>95.95</td><td>46.70</td></tr><tr><td>RULER 1M</td><td>91.75</td><td>91.33</td><td>22.30</td></tr><tr><td colspan="4">Multilingual</td></tr><tr><td>MMLU-ProX (avg over langs)</td><td>79.36</td><td>85.06</td><td>76.59</td></tr><tr><td>WMT24++ (en→xx)</td><td>86.67</td><td>87.84</td><td>88.89</td></tr></table>



Table 5 | Evaluation suite for Nemotron 3 Super. We compare against QwEN-3.5-122B-A10B and GPT-OSS-120B. (Values omitted in this template; to be filled with final scores.)

---

can be found in the Nemo Evaluator SDK config folder. The following benchmarks are not onboarded yet in our open source tools and for these we used either their official open source implementation or otherwise an internal scaffolding that we plan to open source in the future: SWE Bench Verified (OpenHands), SWE Bench Multilingual (OpenHands), BrowseComp with Search (internal implementation, with Serp API), Terminal Bench Core 2.0 (Harbor).

##### Reasoning Capabilities.

We report results on AIME 25, HMMT Feb 25, GPQA *(Rein et al., 2023)*, LiveCodeBench v5 *(Jain et al., 2024)*, SciCode *(Tian et al., 2024)*, and HLE *(Phan et al., 2025)*. Across all benchmarks Nemotron 3 Super is competitive with GPT-OSS-120B, while lagging behind Qwen-3.5-122B slightly.

##### Agentic capabilities.

We report results on TerminalBench (the hard subset and the v2 set), SWE-Bench (OpenHands, OpenCode, Codex, and the Multilingual set) *(Jimenez et al., 2023)*, TauBench V2 (Airline, Retail, Telecom; and their average) *(Barres et al., 2025)*, and BrowseComp *(Wei et al., 2025)*. For Browsecomp, our harness takes strong inspiration from the browser tool released with GPT OSS. *(OpenAI, 2025)*, and we do not evaluate with any context management strategies. For SQL we evaluate on the BIRD benchmark *Li et al. (2023a)* dev set (1,534 samples, SQLite, execution accuracy). Across all agentic benchmarks, Nemotron 3 Super outperforms or is competitve with GPT-OSS 120B and is competitive to Qwen 3.5 122B on some harnesses.

##### Chat and Instruction Following Capabilities.

We report results on IFBench, MultiChallenge, Arena-Hard V2 *(Li et al., 2024)*. Across all benchmarks Nemotron 3 Super is competitive with the baseline models.

##### Long Context Capabilities.

We report results on Ruler *(Hsieh et al., 2024)* using 100 samples per task and AALCR.

##### Multilingual Capabilities.

We measure multilingual capability on MMLU-ProX *(Xuan et al., 2025)* and WMT24++ en$\rightarrow$xx *(Deutsch et al., 2025)*. Nemotron 3 Super matches or outperforms the baseline models on both bencharks.

For comparison with GPT-OSS-120B and Qwen-3.5-122B-A10B, we use officially reported numbers whenever available; when not available, we follow the Nemotron 3 Nano report *(NVIDIA, 2025a)* procedure of either sourcing values from reputable public aggregators (when consistent with the official protocol) or computing scores ourselves using the official evaluation settings.

## 4 Quantization For Inference

We apply post-training quantization (PTQ) using Model-Optimizer to quantize weights and activations to generate two efficient deployment checkpoints: FP8 (W8A8) for Hopper and NVFP4 (W4A4) for Blackwell.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

# 4.1. Nemotron 3 Super FP8 Checkpoint

For FP8 PTQ calibration, we used a small subset containing 256 samples with 65536 context length from the post-training SFT dataset. For FP8 quantization, we quantized MoE GEMMs, both routed and shared, and Mamba Linear layers. We also kept the KV Cache in FP8, whereas Mamba state cache has been quantized to FP16 for speedup. The precision assignments for the different operators in this checkpoint are summarized in Table 6.

Table 6 | Precision settings for the FP8 checkpoint compared with the BF16 baseline.



<table><tr><td>Configuration</td><td>FP8 Checkpoint</td><td>BF16 Baseline</td></tr><tr><td>Embedding</td><td>BF16</td><td>BF16</td></tr><tr><td>Attention GEMM (QKV and Out Projection)</td><td>BF16</td><td>BF16</td></tr><tr><td>KV Cache + Attention BMM1</td><td>FP8</td><td>FP8</td></tr><tr><td>Attention BMM2</td><td>BF16</td><td>BF16</td></tr><tr><td>MoE GEMM (Sparse Experts and Shared Experts)</td><td>FP8</td><td>BF16</td></tr><tr><td>MoE Latent Projection GEMM</td><td>BF16</td><td>BF16</td></tr><tr><td>Router</td><td>FP32</td><td>FP32</td></tr><tr><td>Mamba GEMM</td><td>FP8</td><td>BF16</td></tr><tr><td>Mamba SSM Cache</td><td>FP16</td><td>FP32</td></tr><tr><td>Mamba 1D Conv</td><td>BF16</td><td>BF16</td></tr><tr><td>Output Layers</td><td>BF16</td><td>BF16</td></tr></table>



# 4.2. Nemotron 3 Super FP4 Checkpoint

FP4 is a more aggressive quantization format than FP8 and is particularly attractive for prefill-heavy inference workloads, such as coding-agent deployments, where linear and MoE GEMMs are major performance bottlenecks. NVFP4 is natively accelerated on Blackwell GPUs and delivers better accuracy than alternative FP4 formats such as MXFP4 (NVIDIA, 2025). NVFP4 for inference (NVIDIA, 2025) uses signed E2M1 values with per-block scaling over 1D blocks of size 16 along the last dimension. These per-block scales are further quantized into FP8 E4M3 using a per-tensor statically calibrated FP32 scaling factor.

In the baseline NVFP4 PTQ recipe (NVIDIA, 2025), each per-block scale is determined by the maximum absolute value in the block. We evaluated a range of alternative PTQ methods. The results of these experiments can be found in Appendix B.1. The best overall results were obtained with a hybrid FP4 recipe: weight per-block scales were selected by minimizing weight MSE, while activation per-block scales continued to use max-based scaling. This choice is both effective and practical. Weight quantization is calibrated offline so an expensive scale search can be performed without impacting runtime performance. Activation quantization must be computed efficiently at runtime, making scale search algorithms impractical. Max-based scaling of activations provide a good trade-off between runtime performance and quantization accuracy.

In addition, we selectively promoted some layers from FP4 (W4A4) to FP8 (W8A8) or BF16 (W16A16) to further improve accuracy. We used Model-Optimizer AutoQuantize  $^{14}$ , a neural architecture search (NAS) inspired method for deriving optimal mixed-precision assignments. AutoQuantize estimates per-operation sensitivity, models the performance cost of available quantization choices, and solves

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

for the optimal layer-wise allocation using a knapsack-style optimization procedure. Its sensitivity metric follows a second-order Taylor approximation inspired by Optimal Brain Surgeon (Hassibi et al., 1993), as introduced in LLM-MQ (Li et al., 2023c). AutoQuantize generalizes LLM-MQ beyond weight-only quantization. It supports operator-level quantization, including joint weight-and-activation quantization for GEMMs, and accounts for inference deployment constraints such as operator fusion. The detailed AutoQuantize algorithm is given in B.2.

Overall, our final NVFP4 PTQ recipe combines:

- calibrated per-block weight scaling that minimizes MSE,
- dynamic per-block max-based activation scaling, and
- selective promotion of sensitive layers through Model-Optimizer AutoQuantize.

This combination addresses the accuracy loss of naive NVFP4 PTQ while preserving the runtime efficiency needed for deployment. The resulting per-operator precision assignments in the final NVFP4 checkpoint are summarized in Table 7.

Table 7 | Precision settings for the backbone NVFP4 checkpoint compared with the BF16 baseline.



<table><tr><td>Configuration</td><td>AutoQuantize
Searched?</td><td>NVFP4
Checkpoint</td><td>BF16
Baseline</td></tr><tr><td>Embedding</td><td>No</td><td>BF16</td><td>BF16</td></tr><tr><td>Attention QKV Projection GEMM</td><td>Yes</td><td>BF16</td><td>BF16</td></tr><tr><td>Attention Output Projection GEMM</td><td>Yes</td><td>FP8 / BF16</td><td>BF16</td></tr><tr><td>KV Cache + Attention BMM1</td><td>No</td><td>FP8</td><td>FP8</td></tr><tr><td>Attention BMM2</td><td>No</td><td>BF16</td><td>BF16</td></tr><tr><td>Sparse Expert (Routed) GEMM</td><td>Yes</td><td>NVFP4</td><td>BF16</td></tr><tr><td>Shared Expert GEMM</td><td>Yes</td><td>NVFP4 / FP8 / BF16</td><td>BF16</td></tr><tr><td>MoE Latent Projection GEMM</td><td>Yes</td><td>FP8 / BF16</td><td>BF16</td></tr><tr><td>Router</td><td>No</td><td>FP32</td><td>FP32</td></tr><tr><td>Mamba Projection GEMM</td><td>Yes</td><td>FP8 / BF16</td><td>BF16</td></tr><tr><td>Mamba 1D Conv</td><td>No</td><td>BF16</td><td>BF16</td></tr><tr><td>Mamba SSM Cache</td><td>No</td><td>FP16</td><td>FP32</td></tr><tr><td>Output Layers</td><td>No</td><td>BF16</td><td>BF16</td></tr></table>



For all searched backbone GEMMs, AutoQuantize considered candidate precisions from  $\{\mathrm{NVFP4},\mathrm{FP8},\mathrm{BF16}\}$  and selected the per-operator assignment under a quantization-sensitivity objective with an effective-precision budget of 4.75 bits. In the searched model, sparse-expert GEMMs are assigned NVFP4 throughout, attention and Mamba projection GEMMs are assigned FP8 or BF16, and shared-expert GEMMs use a mix of NVFP4, FP8, and BF16.

Combining AutoQuantize with the improved NVFP4 PTQ recipe produced a mostly-FP4 model, with only a small subset of layers retained in FP8 or BF16 for accuracy preservation. The full mixed-precision PTQ process completed in less than 2 hours on a single B200 node with 8 GPUs, using 512 samples from the Nemotron 3 Super SFT dataset at sequence length 4096. The resulting model achieved  $99.8\%$  median accuracy relative to the BF16 baseline while retaining near-FP4 performance. Final evaluation results are reported in Table 9.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

# 4.3. Mamba State Quantization

In memory-bound settings, DRAM reads of the Mamba state cache (SSM cache) become a major bottleneck to decoding speed. The SSM cache is stored in FP32 by default. One option is to store the SSM cache in FP16 while arithmetic executes in FP32. In this case, the cache is fetched from memory in FP16, upcast to FP32 for the recurrent update, and then cast back to FP16 for storage. Table 8 shows experiments on an early checkpoint of Nemotron 3 Super. These results show that directly casting the SSM cache to FP16 can result in up to  $40\%$  increase in morbidity when combined with W8A8 quantization. Even with maintaining weights and activations in BF16, casting the SSM cache to FP16 leads to up to  $37\%$  increase in morbidity.

Table 8 | Impact of SSM cache recipe on code benchmarks and morbidity.



<table><tr><td rowspan="2">Weight and Activation Precision</td><td rowspan="2">SSM cache recipe</td><td colspan="2">Accuracy</td><td colspan="2">Completion Tokens</td><td colspan="2">Verbosity Increase</td></tr><tr><td>livecodebench (pass@1 avg-of-8)</td><td>sciode (pass@1 avg-of-8)</td><td>livecodebench (pass@1 avg-of-8)</td><td>sciode (pass@1 avg-of-8)</td><td>livecodebench (pass@1 avg-of-8)</td><td>sciode (pass@1 avg-of-8)</td></tr><tr><td>W16A16</td><td>FP32</td><td>72.91</td><td>40.90</td><td>21769</td><td>3680</td><td>0.00%</td><td>0.00%</td></tr><tr><td>W16A16</td><td>FP16</td><td>73.24</td><td>42.01</td><td>29812</td><td>3760</td><td>36.95%</td><td>2.19%</td></tr><tr><td>W16A16</td><td>FP16+RS</td><td>72.00</td><td>41.94</td><td>21392</td><td>3580</td><td>-1.73%</td><td>-2.72%</td></tr><tr><td>W8A8</td><td>FP16</td><td>73.13</td><td>40.98</td><td>30536</td><td>3780</td><td>40.27%</td><td>2.70%</td></tr><tr><td>W8A8</td><td>INT16+block128</td><td>72.22</td><td>41.46</td><td>22406</td><td>3521</td><td>2.90%</td><td>-4.30%</td></tr><tr><td>W8A8</td><td>FP16+SR (Philox 10)</td><td>72.22</td><td>40.38</td><td>22120</td><td>3672</td><td>1.71%</td><td>-0.74%</td></tr><tr><td>W8A8</td><td>FP16+SR (Philox 5)</td><td>72.63</td><td>41.86</td><td>22159</td><td>3720</td><td>1.79%</td><td>1.08%</td></tr><tr><td>W8A8</td><td>FP16+SR (Philox 4)</td><td>72.85</td><td>40.46</td><td>21631</td><td>3785</td><td>-0.63%</td><td>2.84%</td></tr><tr><td>W8A8</td><td>FP16+SR (Philox 3)</td><td>70.07</td><td>39.94</td><td>24098</td><td>3827</td><td>10.70%</td><td>3.98%</td></tr></table>



A key challenge in quantizing the Mamba cache is that quantization error does not remain local to a single step. Because Mamba decoding is recurrent, quantization error from previous steps propagates into future steps and accumulates over time. This accumulation of quantization error can be seen by unrolling the recurrent update. Let the recurrent state update be  $h_t = A_t h_{t-1} + B_t x_t$ , and let cache quantization at step  $t$  introduce an additive error  $e_t$ , so that  $h_{q,t} = A_t h_{q,t-1} + B_t x_t + e_t$ . Unrolling this recursion gives

$$
\begin{array}{l} h _ {q, t} = h _ {t} + e _ {t} + A _ {t} e _ {t - 1} + A _ {t} A _ {t - 1} e _ {t - 2} + \dots \\ + A _ {t} A _ {t - 1} \dots A _ {2} e _ {1} + A _ {t} A _ {t - 1} \dots A _ {1} e _ {0} \\ = h _ {t} + \sum_ {i = 0} ^ {t} \left(\prod_ {j = i + 1} ^ {t} A _ {j}\right) e _ {i}, \tag {3} \\ \end{array}
$$

showing that quantization error from earlier steps is propagated through subsequent recurrent transitions and can accumulate over decoding time.

Addressing these quantization errors through changes in training (e.g., QAT or QAD) is non-trivial. Mamba training uses the chunked State Space Duality algorithm which does not explicitly materialize the recurrence relationship and the inference-time cache. Accurately modeling the recurrent decode-time cache behavior during training introduces substantial overhead. We therefore focused on training-free methods to recover the accuracy lost from cache quantization.

One way to decrease the accumulation of quantization error during PTQ is to increase the mantissa precision. We explored this by using INT16 instead of FP16 for the SSM cache. Naive INT16 quantization did not improve morbidity as tensor-level analysis showed that the SSM cache has a wide dynamic range. We then introduced FP32 per-block scaling over blocks of size 128 along the state dimension to increase effective dynamic range. This eliminated the morbidity issue (Table 8).

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

We also explored a hypothesis that the error accumulation was tied to rounding during the cast from FP32 to FP16. The key issue is that round to nearest, ties on even (RTNE) introduces bias in the quantization process. Because RTNE maps a given input to the same rounded value, its quantization error has zero variance but non-zero bias relative to the original value. In contrast, stochastic rounding (SR) is unbiased in expectation. In a recurrent setting, the bias from RTNE accumulates coherently over time, whereas stochastic rounding replaces this systematic drift with zero-mean noise. Based on this observation, we applied stochastic rounding before casting the cache to FP16 which. This fixed the morbidity issue for both the BF16 baseline and the FP8 checkpoint (Table 8).



<table><tr><td>Benchmark</td><td>N-3-Super</td><td>N-3-Super FP8</td><td>N-3-Super NVFP4</td></tr><tr><td colspan="4">General Knowledge</td></tr><tr><td>MMLU-Pro</td><td>83.73</td><td>83.63</td><td>83.33</td></tr><tr><td colspan="4">Reasoning</td></tr><tr><td>HMMT Feb25 (with tools)</td><td>94.73</td><td>94.38</td><td>95.36</td></tr><tr><td>GPQA (no tools)</td><td>79.23</td><td>79.36</td><td>79.42</td></tr><tr><td>LiveCodeBench (v6 2024-08↔2025-05)</td><td>78.69</td><td>78.44</td><td>78.44</td></tr><tr><td>LiveCodeBench (v5 2024-07↔2024-12)</td><td>81.19</td><td>80.99</td><td>80.56</td></tr><tr><td>SciCode (subtask)</td><td>42.05</td><td>41.38</td><td>40.83</td></tr><tr><td>HLE (no tools)</td><td>18.26</td><td>17.42</td><td>17.42</td></tr><tr><td colspan="4">Agentic</td></tr><tr><td>Terminal Bench (hard subset)</td><td>25.78</td><td>26.04</td><td>24.48</td></tr><tr><td colspan="4">TauBench V2</td></tr><tr><td>Airline</td><td>56.25</td><td>56.25</td><td>54.75</td></tr><tr><td>Retail</td><td>62.83</td><td>63.05</td><td>63.38</td></tr><tr><td>Telecom</td><td>64.36</td><td>63.93</td><td>63.27</td></tr><tr><td>Average</td><td>61.15</td><td>61.07</td><td>60.46</td></tr><tr><td colspan="4">Chat &amp; Instruction Following</td></tr><tr><td>IFBench (prompt)</td><td>72.58</td><td>72.32</td><td>73.30</td></tr><tr><td>Scale AI Multi-Challenge</td><td>55.23</td><td>54.35</td><td>52.8</td></tr><tr><td>Arena-Hard-V2 (Hard Prompt)</td><td>73.88</td><td>76.06</td><td>76.00</td></tr><tr><td colspan="4">Long Context</td></tr><tr><td>AA-LCR</td><td>58.31</td><td>57.69</td><td>58.06</td></tr><tr><td>RULER 128k (500 samples per task)</td><td>96.79</td><td>96.85</td><td>95.99</td></tr><tr><td>RULER 256k (500 samples per task)</td><td>96.60</td><td>96.33</td><td>96.52</td></tr><tr><td>RULER 512k (500 samples per task)</td><td>96.09</td><td>95.66</td><td>96.23</td></tr><tr><td colspan="4">Multilingual</td></tr><tr><td>MMLU-ProX (avg over languages)</td><td>79.35</td><td>79.21</td><td>79.37</td></tr></table>



Table 9 | Evaluation suite for Nemotron 3 Super. We compare our FP8 and NVFP4 optimized models with BF16 model.

Table 8 shows accuracy and morbidity of the different SSM cache recipes for livecodebench and scicode. Both INT16 with per-block scales and FP16 with stochastic rounding were able to maintain accuracy and morbidity similar to the FP32 baseline. We selected FP16 with stochastic rounding (SR) using Philox  $&lt; 5&gt;$  pseudorandom number generation as the SSM cache recipe for Nemotron 3 Super for three reasons:

---

- It does not require calculating, storing, and loading block scale factors.
- Blackwell provides a dedicated PTX instruction for stochastic rounding during type conversion.
- Blackwell supports Philox-based pseudorandom number generation through cuRAND.

To further improve efficiency, Table 8 also varies the number of Philox rounds. Increasing the number of rounds improves the statistical quality of the generated values, while reducing the number of rounds lowers pseudorandom number generation overhead. Philox$<$5$>$ was chosen to maintain accuracy and verbosity while minimizing pseudorandom number generation overhead.

## 5 Conclusion

We introduce Nemotron 3 Super, a 12B active and 120B total parameter MoE hybrid Mamba-Attention model with strong agentic capabilities. Nemotron 3 Super employs LatentMoE to improve accuracy and incorporates MTP layers to accelerate inference via speculative decoding. We pretrained Nemotron 3 Super on 25 trillion text tokens with low-precision NVFP4, followed by post-training on a diverse set of RL environments. Finally, we quantized the model to FP8 and NVFP4, achieving significantly higher inference throughput without sacrificing model accuracy. Nemotron 3 Super achieves up to 2.2$\times$ higher throughput than GPT-OSS-120B while maintaining higher accuracy across a wide range of tasks. We release the pre-trained, post-trained, and quantized checkpoints for Nemotron 3 Super on HuggingFace.

## Contributors

We thank the following people for their invaluable contributions to NVIDIA Nemotron 3 Super.

Aaron Blakeman, Aakshita Chandiramani, Abdullahi Olaoye, Abhilash Somasamudramath, Abhibha Gupta, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal Lilach Ilan, Aditya Vavre, Ahmad Kiswani, Aishwarya Padmakumar, Ajay Hotchandani, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Gronskiy, Alex Kondratenko, Alex Neefus, Alex Steiner, Alex Yang, Alexander Bukharin, Alexander Young, Ali Hatamizadeh, Ali Taghibakhshi, Alina Galiautdinova, Alisa Liu, Alok Kumar, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Anahita Bhiwandiwalla, Ananth Subramaniam, Andrew Tao, Anjaney Shrivastava, Anjulie Agrusa, Ankur Srivastava, Ankur Verma, Ann Guan, Anna Shors, Annamalai Chockalingam, Anubhav Mandarwal, Aparnaa Ramani, Arham Mehta, Arti Jain, Arun Venkatesan, Asha Anoosheh, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asli Sabanci Demiroz, Asma Kuriparambil Thekkumpate, Atefeh Sorabizadeh, Avinash Kaur, Ayush Dattagupta, Barath Subramaniam Anandan, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Benjamin Chislett, Besmira Nushi, Bilal Kartal, Bill Thiede, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Buvaneswari Mani, Carlo del Mundo, Chanran Kim, Chantal Hwang, Chankyu Lee, Chao Ni, Charles Wang, Charlie Truong, Chenhan Yu, Chenjie Luo, Cherie Wang, Cheng-Ping Hsieh, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Chris Holguin, Chris Wing, Christian Munley, Christopher Parisien, Chunyang Sheng, Chuck Desai, Collin Neale, Cyril Meurillon, Dakshi Kumar, Dan Gil, Dan Su, Dane Corneil, Daniel Afrimi, Daniel Burkhardt Eliuth Triana, Daniel Egert, Daniel Fatade Douglas O’Flaherty, Daniel Lo, Daniel Rohrer, Daniel Serebrenik, Daniil Sorokin, Daria Gitman, Daria Levy, Darko Stosic, David Edelsohn, David Messina, David Mosallanezhad, David Tamok, Deena Donia, Deepak Narayanan, Devin O’Kelly, Dheeraj Peri, Dhruv Nathawani, Di Wu, Dima Rekesh, Dina Yared, Divyanshu Kakwani, Dmitry Konyagin Brandon Tuttle, Dong Ahn, Dongfu Jiang, Dorrin Poorkay, Duncan Riach, Dusan Stosic, Dustin Van Stee, Edgar Minasyan,

---

Edward Lin, Eileen Peters Long, Elad Segal, Elena Lewis, Elena Lantz, Ellie Evans, Elliott Ning, Eric Chung, Eric Harper, Eric Pham-Hung, Eric W. Tramel, Erick Galinkin, Erik Pounds, Esti Etrog, Evan Briones, Evan Wu, Evelina Bakhturina, Evgeny Tsykunov, Ewa Dobrowolska, Farshad Saberi Movahed, Farzan Memarian, Fay Wang, Fei Jia, Felipe Soares, Felipe Vieira Frujeri, Feng Chen, Fengguang Lin, Ferenc Galko, Fortuna Zhang, Frankie Siino, Frida Hou, Gantavya Bhatt, Gargi Prasad, Geethapriya Venkataramani, Geetika Gupta, George Armstrong, Gerald Shen, Giulio Borghesi, Gordana Neskovic, Gorkem Batmaz, Grace Lam, Grace Wu, Greg Pauloski, Greyson Davis, Grigor Nalbandyan, Guoming Zhang, Guy Farber, Guyue Huang, Haifeng Qian, Haran Kumar Shiv Kumar, Harry Kim, Harsh Sharma, Hayate Iso, Hayley Ross, Herbert Hum, Herman Sahota, Hexin Wang, Himanshu Soni, Hiren Upadhyay, Huy Nguyen, Iain Cunningham, Ido Galil, Ido Shahaf, Igor Gitman, Igor Shovkun, Igino Padovani, Ikroop Dhillon, Ilya Loshchilov, Ingrid Kelly, Itamar Schen, Itay Levy, Ivan Moshkov, Izik Golan, Izzy Putterman, Jain Tu, Jan Baczek, Jan Kautz, Janica Rosenberg, Jane Polak Scowcroft, Jared Casper, Jarrod Pflum, Jason Grant, Jason Sewall, Jatin Mitra, Jeffrey Glick, Jian Zhang, Jenny Chen, Jesse Oliver, Jiafan Zhu, Jialin Song, Jiaqi Zeng, Jie Lou, Jill Milton, Jimmy Zhang, Jim Chow, Jinhang Choi, Jining Huang, Jocelyn Huang, Joel Caruso, Joey Conway, Joey Guman, Johan Jatko, John Kamalu, Johnny Greco, Jonathan Cohen, Jonathan Raiman, Joseph Jennings, Joyjit Daw, Juan Yu, Julio Tapia, Junkeun Yi, Jupinder Parmar, Jyothi Achar, Kari Briski, Kartik Mattoo, Katherine Cheung, Katherine Luna, Keith Wyss, Kevin Shih, Kezhi Kong, Khanh Nguyen, Khushi Bhardwaj, Kirill Buryak, Kirthi Shankar Sivamani, Konstantinos Krommydas, Kris Murphy, Krishna C. Puvvada, Krzysztof Pawelec, Kumar Anik, Laikh Tewari, Laya Sleiman, Leo Du, Leon Derczynski, Li Ding, Lingjie Wu, Lizzie Wei, Luis Vega, Maarten Van Segbroeck, Maer Rodrigues de Melo, Magaret Zhang, Mahan Fathi, Makesh Narsimhan Sreedhar, Makesh Sreedhar, Makesh Tarun Chandran, Manuel Reyes Gomez, Maor Ashkenazi, Marc Cuevas, Marc Romeijn, Margaret Zhang, Mark Cai, Mark Gabel, Markus Kliegl, Martyna Patelka, Maryam Moosaei, Matvei Novikov, Matthew Varacalli, Mauricio Ferrato, Mehrzad Samadi, Melissa Corpuz, Meng Xin, Mengdi Wang, Mengru Wang, Meredith Price, Michael Andersch, Michael Boone, Michael Evans, Michael Z Wang, Micah Schaffer, Miguel Martinez, Mikail Khona, Mike Chrzanowski, Mike Hollinger, Mingyuan Ma, Minseok Lee, Mohammad Dabbah, Mohammad Shoeybi, Mostofa Patwary, Nancy Agarwal, Nader Khalil, Nanthini Balasubramaniam, Narsi Kodukula, Nabin Mulepati, Najeeb Nabwani, Narimane Hennouni, Natalie Hereth, Nathaniel Pinckney, Nave Assaf, Negar Habibi, Nestor Qin, Netanel Haber, Neta Zmora, Nick Reamaroon, Nickson Quack, Nidhi Bhatia, Nikhil Jukar, Nikolai Ludwig, Nikki Pope, Nima Tajbakhsh, Nir Ailon, Nirmal Juluru, Nirmalya De, Nowel Pitt, Olivier Delalleau, Oleg Rybakov, Oleksii Hrinchuk, Oleksii Kuchaiev, Oluwatobi Olabiyi, Omer Ullman Argov, Omri Almog, Omri Puny, Oren Tropp, Otavio Padovani, Ouye Xie, Parth Chadha, Pasha Shamis, Paul Gibbons, Pavlo Molchanov, Peter Belcak, Peter Jin, Pinky Xu, Piotr Januszewski, Pooya Jannaty, Prachi Shevate, Pradeep Thalasta, Pranav Prashant Thombre, Prasoon Varshney, Prerana Gambhir, Pritam Gundecha, Przemek Tredak, QA Lun Su, Qing Miao, Qiyu Wan, Quan Tran Minh, Rabeeh Karimi Mahabadi, Rachel Oberman, Rachit Garg, Rahul Kandu, Raina Zhong, Ran El-Yaniv, Ran Zilberstein, Rasoul Shafipour, Renjie Pi, Renee Yao, Richard Mazzarese, Richard Wang, Rick Izzo, Ridhima Singla, Rima Shahbazyan, Rishabh Garg, Ritika Borkar, Ritu Gala, Riyad Islam, Robert Clark, Robert Hesse, Roger Waleffe, Rohit Varma Kalidindi, Rohit Watve, Roi Koren, Ron Fan, Ruchika Kharwar, Ruisi Cai, Ruoxi Zhang, Russell J. Hewett, Ryan Prenger, Ryan Timbrook, Ryota Egashira, Sadegh Mahdavi, Sagar Singh Ashutosh Joshi, Sahil Modi, Samuel Kriman, Sandeep Pombra, Sanjeev Satheesh, Sanjay Kariyappa, Santiago Pombo, Saori Kaji, Satish Pasumarthi, Saurav Mishra, Saurav Muralidharan, Scott Hara, Sean Narenthiran, Sebastian Rogawski, Seonjin Na, Seonmyeong Bak, Sepehr Sameni, Seth Poulos, Shahar Mor, Shaona Ghosh Adam Lord, Sharath Turuvekere Sreenivas, Shaun Kotek, Shaya Gharghabi, Shelby Thomas, Sheng-Chieh Lin, Shibani Likhite, Shiqing Fan, Shiyang Chen, Shreya Gopal, Shrimai Prabhumoye, Shubham Pachori, Shubham Toshniwal, Shuo Zhang, Shuoyang

---

Ding, Shyam Renjith, Shyamala Prayaga, Siddhartha Jain, Simeng Sun, Sirisha Rella, Sirshak Das, Smita Ithape, Sneha Harishchandra S, Somshubra Majumdar, Soumye Singhal, Sri Harsha Singudasu, Sriharsha Niverty, Stas Sergienko, Stefana Gloginic, Stefania Alborghetti, Stephen Ge, Stephen McCullough, Sugam Dipak Devare, Suguna Varshini Velury, Sukrit Rao, Sumeet Kumar Barua, Sunny Gai, Suseella Panguluri, Sushil Koundinyan, Swathi Patnam, Sweta Priyadarshi, Swetha Bhendigeri, Syeda Nahida Akter, Sylendran Arunagiri, Tailling Yuan, Talor Abramovich, Tan Bui, Tan Yu, Terry Kong, Thanh Do, Thomas Gburek, Thorgane Marques, Tiffany Moore, Tim Moon, Timothy Ma, Tiyasa Mitra, Tomasz Grzegorzek, Tomer Asida, Tomer Bar Natan, Tomer Keren, Tomer Ronen, Traian Rebedea, Trenton Starkey, Tugrul Konuk, Twinkle Vashishth, Tyler Condensa, Udi Karpas, Ushnish De, Vahid Noorozi, Vahid Noroozi, Vanshil Atul Shah, Veena Vaidyanathan, Venkat Srinivasan, Venmugil Elango, Victor Cui, Vijay Korthikanti, Vikas Mehta, Virginia Adams, Virginia Wu, Vitaly Kurin, Vitaly Lavrukhin, Vladimir Anisimov, Wan Seo, Wanli Jiang, Wasi Uddin Ahmad, Wei Du, Wei Ping, Wei-Ming Chen, Wendy Quan, Wenliang Dai, Wenwen Gao, Will Jennings, William Zhang, Xiaowei Ren, Xiaowen Xin, Xin Li, Yang Yu, Yangyi Chen, Yaniv Galron, Yashaswi Karnati, Yejin Choi, Yev Meyer, Yi-Fu Wu, Yian Zhang, Ying Lin, Yonatan Geifman, Yonggan Fu, Yoshi Suhara, Youngeun Kwon, Yuan Zhang, Yuki Huang, Zach Moshe, Zhiyu Cheng, Zhilin Wang, Zihan Liu, Zijia Chen, Zijie Yan, Zhongbo Zhu, Zhuolin Yang, Zuhair Ahmed.

---

Nemotron 3 Super : Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning