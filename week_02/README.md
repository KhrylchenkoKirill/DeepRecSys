# Week 2
## ML Design of Recommender Systems

Lecture by [Kirill Khrylchenko](https://github.com/KhrylchenkoKirill)

Seminar by [Vladimir Baikalov](https://github.com/NonameUntitled)

Youtube:
* [lecture](https://www.youtube.com/watch?v=2CLyLE3oOTQ)
* [seminar](https://www.youtube.com/watch?v=q55mPzvw-AM)

This week we continue our journey through recommender systems from a more classical perspective and focus on the ML design of real-world recommender systems.

The lecture is structured in a way that resembles a typical ML design interview for recommender systems.

Imagine that you are stranded on an uninhabited island and need to build a recommender system from scratch. What would you do?

We go step by step through the key components of such a system:
1. Metrics - defining goals and understanding what we optimize.
2. Data - logs, impressions, metadata, and their implications.
3. Retrieval - candidate generation, multi-stage design, and practical challenges.
4. Ranking - model inputs, objectives, and multi-signal optimization.
5. Bonus - discussion of the lecturer’s R&D experience.

The seminar is complementary to the lecture:
1. We review strong classical baselines that can be tried before deep learning.
2. We analyze the [Yambda](https://arxiv.org/abs/2505.22238) paper, discuss baseline results, and highlight evaluation caveats.
