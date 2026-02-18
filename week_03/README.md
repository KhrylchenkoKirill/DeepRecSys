# Week 3
## Two-tower models

Lecture and seminar by [Kirill Khrylchenko](https://github.com/KhrylchenkoKirill)

Youtube:
* [lecture](https://www.youtube.com/watch?v=XQDGaW4xwWs)
* [seminar](https://www.youtube.com/watch?v=tnrw1quaZF0)

The third week of the course focuses on two-tower architectures for candidate generation.

In the lecture, we provide an in-depth discussion of:
1. Why two-tower models are needed in the early stages of recommender systems and how they are deployed in production.
2. Why typical ranking objectives are not suitable for the retrieval stage (including a discussion of the folding effect).
3. The sampled softmax loss and its motivation.
4. Contrastive learning and InfoNCE: cosine similarity, temperature scaling, and key properties of softmax.
5. Negative sampling strategies and the biases introduced by in-batch negatives.
6. A principled correction for in-batch sampling bias - logQ correction.

During the seminar, we focus on research design for recommender system experiments. The discussion centers on best practices for offline experimentation, including how to formulate research questions, choose objectives, select datasets, design baselines, and evaluate models with appropriate metrics.