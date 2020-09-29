---
layout:     post
title:      单智能体强化学习算法
subtitle:   提纲
date:       2020-12-01
author:     MY
header-img: img/post-bg-hacker.jpg
catalog: true
top: true
tags:
    - RL
    - RL advanced algorithms
---
---
以下是基于<a href="https://spinningup.openai.com/en/latest/index.html">OpenAI Spinning Up</a>整理并扩充的单智能体深度强化学习中值得阅读的论文列表以及我的阅读总结与思考。该列表并不全面，但建议对强化学习感兴趣的同学朋友们进行阅读~
  * Model-Free RL
    * Deep Q-Learning
      * <a href="https://mayi1996.top/2020/08/03/Playing-Atari-with-Deep-Reinforcement-Learning/">DQN: Playing Atari with Deep Reinforcement Learning</a>
      * <a href="https://mayi1996.top/2020/08/03/Deep-Recurrent-Q-Learning-for-Partially-Observable-MDPs/">DRQN：Deep Recurrent Q-Learning for Partially Observable MDPs</a>
      * <a href="https://mayi1996.top/2020/08/03/Dueling-Network-Architectures-for-Deep-Reinforcement-Learning/">Dueling DQN：Dueling Network Architectures for Deep Reinforcement Learning</a>
      * <a href="https://mayi1996.top/2020/08/04/Deep-Reinforcement-Learning-with-Double-Q-learning/">Double DQN：Deep Reinforcement Learning with Double Q-learning</a>
      * <a href="">Prioritized Experience Replay (PER)：Prioritized Experience Replay</a>
      * <a href="https://mayi1996.top/2020/08/05/Rainbow-Combining-Improvements-in-Deep-Reinforcement-Learning">Rainbow DQN：Rainbow Combining Improvements in Deep Reinforcement Learning</a>
    * Policy Gradients
      * <a href="">A3C: Asynchronous Methods for Deep Reinforcement Learning</a>
      * <a href="">TRPO: Trust Region Policy Optimization</a>
      * <a href="">GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation</a>
      * <a href="">PPO-Clip, PPO-Penalty: Proximal Policy Optimization Algorithms</a>
      * <a href="">PPO-Penalty: Emergence of Locomotion Behaviours in Rich Environments</a>
      * <a href="">ACKTR: Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation</a>
      * <a href="">ACER: Sample Efficient Actor-Critic with Experience Replay</a>
      * <a href="">SAC: Soft Actor-Critic Off-Policy Maximum Entropy Deep Reinforcement Learning With a Stochastic Actor</a>
      * <a href="">ERO: Experience Replay Optimization</a>
    * Deterministic Policy Gradients
      * <a href="https://mayi1996.top/2020/08/06/Deterministic-Policy-Gradient-Algorithms/">DPG: Deterministic Policy Gradient Algorithms</a>
      * <a href="https://mayi1996.top/2020/08/06/Continuous-Control-With-Deep-Reinforcement-Learning/">DDPG: Continuous Control With Deep Reinforcement Learning</a>
      * <a href="https://mayi1996.top/2020/08/07/Addressing-Function-Approximation-Error-in-Actor-Critic-Methods/">TD3: Addressing Function Approximation Error in Actor-Critic Methods</a>
    * Distributional RL
    * Policy Gradients with Action-Dependent Baselines
    * Path-Consistency Learning
    * Other Directions for Combining Policy-Learning and Q-Learning
    * Evolutionary Algorithms
  * Exploration
    * Count based Exploration
      * <a href="https://mayi1996.top/2020/08/07/Exploration-by-random-network-distillation/">RND: Exploration by random network distillation</a>
    * Curiosity based Exploration
      * <a href="https://mayi1996.top/2020/08/05/Curiosity-Driven-Exploration-by-Self-Supervised-Prediction/">ICM: Curiosity-Driven Exploration by Self-Supervised Prediction</a>
      * <a href="https://mayi1996.top/2020/08/05/Episodic-Curiosity-through-Reachability/">EC: Episodic Curiosity through Reachability</a>
    * Information Gain    
  * Transfer and Multitask RL
    * <a href="https://mayi1996.top/2020/08/10/Dynamic-Weights-in-Multi-Objective-Deep-Reinforcement-Learning/">Dynamic Weights in Multi-Objective Deep Reinforcement Learning</a>
  * Hierarchy RL
  * Memory
  * Model-Based RL
  * Meta-RL
  * Scaling RL
  * Offline RL
    * <a href="https://mayi1996.top/2020/09/01/%E7%A6%BB%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/#21-off-policy-deep-reinforcement-learning-without-exploration">Off-Policy Deep Reinforcement Learning Without Exploration</a>
    * <a href="https://mayi1996.top/2020/09/01/%E7%A6%BB%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/#22-benchmarking-batch-deep-reinforcement-learning-algorithms">Benchmarking Batch Deep Reinforcement Learning Algorithms</a>
    * <a href="https://mayi1996.top/2020/09/01/%E7%A6%BB%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/#23-stabilizing-off-policy-q-learning-via-bootstrapping-error-reduction">Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction</a>
    * <a href="https://mayi1996.top/2020/09/01/%E7%A6%BB%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/#31-an-optimistic-perspective-on-offline-reinforcement-learning">An Optimistic Perspective on Offline Reinforcement Learning</a>
    * <a href="https://mayi1996.top/2020/09/01/%E7%A6%BB%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/#32-iris-implicit-reinforcement-without-interaction-at-scale-for-learning-control-from-offline-robot-manipulation-data">IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline Robot Manipulation Data</a>
    * <a href="https://mayi1996.top/2020/09/15/Behavior-Regularized-Offline-Reinforcement-Learning/">Behavior Regularized Offline Reinforcement Learning</a>
 * Representation Learning in RL
   * Contrastive Learning
     * <a href="https://mayi1996.top/2020/08/20/Unsupervised-State-Representation-Learning-in-Atari/">Unsupervised State Representation Learning in Atari</a>
     * <a href="https://mayi1996.top/2020/08/26/CURL-Contrastive-Unsupervised-Representations-for-Reinforcement-Learning/">CURL: Contrastive Unsupervised Representations for Reinforcement Learning</a>
   * Others
     * <a href="https://mayi1996.top/2020/08/27/Data-Efficient-Reinforcement-Learning-with-Momentum-Predictive-Representations/">Data-Efficient Reinforcement Learning with Momentum Predictive Representations</a>
     * <a href="https://mayi1996.top/2020/09/29/Diversity-is-All-You-Need-Learning-skills-without-a-Reward-Function/">Diversity is All You Need: Learning Skills without a Reward Function</a>
  * Generalized RL
    * <a href="https://mayi1996.top/2020/08/10/Generalization-to-New-Actions-in-Reinforcement-Learning/">Generalization to New Actions in Reinforcement Learning</a>
  * RL in the Real World
    * <a href="">Suphx Mastering Mahjong with Deep Reinforcement Learning</a>
  * Safety
  * Imitation Learning and Inverse Reinforcement Learning
  * Reproducibility, Analysis, and Critique
    * <a href="">The Mirage of Action-Dependent Baselines in Reinforcement Learning</a>
  * Classic Papers in RL Theory or Review


