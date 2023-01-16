# CMA-ES_Optimized_RL

Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES) for hyperparameter optimization on Deep-Q-Network

## What is CMA-ES?

From [The CMA Evolution Strategy](http://cma.gforge.inria.fr/) homepage:

> The CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is an evolutionary algorithm for difficult non-linear non-convex black-box optimization problems in continuous domain. It is considered as state-of-the-art in evolutionary computation and has been adopted as one of the standard tools for continuous optimization in many (probably hundreds of) research labs and industrial environments around the world. 

Useful links:

* [A quick start guide with a few usage examples](https://pypi.python.org/pypi/cma)

* [CMA-ES Tutorial](https://arxiv.org/pdf/1604.00772.pdf)

* [A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)

## Why use CMA-ES?

> RL problems are [notoriously sensitive](https://cloud.google.com/blog/products/ai-machine-learning/deep-reinforcement-learning-on-gcp-using-hyperparameters-and-cloud-ml-engine-to-best-openai-gym-games) to hyperparameters, which means it’s necessary to evaluate many different hyperparameters. The prominant obstacle in this process is laborious effort required by human experts. Using CMA-ES is an attempt to solve this problem, by automating the hyperparameter searching process.

* [DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [stable-baseline](https://stable-baselines.readthedocs.io/en/master/index.html)

## Result
> Agents trained with hyperparameters optimzied by CMA-ES performs significantly better than the baseline DQN from [stable-baseline](https://stable-baselines.readthedocs.io/en/master/index.html). For more explanation refer to the poster `Tokyo_Tech_Poster.pdf`

Go to the link to see the results yourself!
* [Cartpole environment](https://www.youtube.com/watch?v=7nTh392Lwms)
* [LunarLandar-v2 environment](https://www.youtube.com/watch?v=A6YEB_7PSgo)

## Create Conda Environment
Create the environment from the `cmaes.yml` file:
```bash
conda env create -f cmaes.yml
```
