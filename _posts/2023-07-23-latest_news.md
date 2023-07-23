---
title: CoreDNS and Kube-DNS in Kubernetes
date: 2023-07-23 00:00:00 +0800
categories: [Blogging, Tutorial]
tags: [favicon]
---

Meta-Transformer: A Unified Framework for Multimodal Learning

Meta-Transformer is a framework that performs unified learning across 12 modalities - probably the first of its kind to do so!

It can handle tasks that include fundamental perception (text, image, point cloud, audio, video), practical application (X-Ray, infrared, hyperspectral, and IMU), and data mining (graph, tabular, and time-series).

The code coming soon!

https://arxiv.org/abs/2307.10802

Comparison with other foundational models. 

Multimodal models are obviously the next step and this work looks super ambitious in trying to build a unified framework for even more modalities beyond the typical ones.


![image](https://pbs.twimg.com/media/F1hete3XwAIj5Ea?format=jpg&name=small)

What the architecture looks like.

Data-to-sequence tokenization --> unified feature encoding --> down-stream task learning

![image](https://pbs.twimg.com/media/F1hfOkDWIAAYkoC?format=jpg&name=small)

There is a bunch of results/comparisons in the paper for all sorts of tasks and modalities that range from time-series forecasting to tabular data understanding to graph data understanding. 

While the model is competitive compared to other specialized models on some tasks such as X-ray image recognition and image understanding, it falls behind in other modalities such as graph data understanding and tabular data understanding. Very interesting results overall. 

The limitations, as reported in the paper, are computational complexity that makes it hard to scale and a lack of temporal and structural awareness which is needed for tasks such as social network prediction.


--- 

Brain2Music: Reconstructing Music from Human Brain Activity

paper page: https://huggingface.co/papers/2307.11078


The process of reconstructing experiences from human brain activity offers a unique lens into how the brain interprets and represents the world. In this paper, we introduce a method for reconstructing music from brain activity, captured using functional magnetic resonance imaging (fMRI). Our approach uses either music retrieval or the MusicLM music generation model conditioned on embeddings derived from fMRI data. The generated music resembles the musical stimuli that human subjects experienced, with respect to semantic properties like genre, instrumentation, and mood. We investigate the relationship between different components of MusicLM and brain activity through a voxel-wise encoding modeling analysis. Furthermore, we discuss which brain regions represent information derived from purely textual descriptions of music stimuli.

---

[new blog post! 🔥] FlashAttention 2.0 was published yesterday. It's a huge deal having sped up standard attention 5-9x! I coincidentally spent the whole last week learning about it and I wrote a blog post explaining it in detail:

https://lnkd.in/dV_E79g7

ELI5 style.

I start from the very basics so even if the MLSys world is completely foreign to you - you should still be able to pick it up and learn a thing or two 

You might ask yourself: why didn’t anyone invent FlashAttention before? Given how crucial this piece of computation is to all modern ML workloads. Given how many engineering hours are spent, across different tech organizations, trying to squeeze the last ounce of performance out of these systems. Why did a Stanford student (shout out to Tri Dao for the amazing work!) come up with this and not e.g. NVIDIA engineers? <- One of the questions I ask in the blog :)

Special thanks to Tri Dao, Horace He and Amrit S. for providing feedback on the earlier draft of this blog post!