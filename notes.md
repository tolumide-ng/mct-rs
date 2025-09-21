### Upper Confidence Tress (UCT)
1. When we selecte nodes, we select using some [multi-armed bandit algorithm](https://gibberblot.github.io/rl-notes/single-agent/multi-armed-bandits.html#sec-multi-armed-bandits). We can use any multi-armed bandit, but in practise, using a slight variation of the UCB1 algorithm has proved to be successful in MCTS.
2. The Upper Confidence Trees (UCT) algorithm is the combination of MCTS with the UCB1 strategy for selecting the next node to follow:
```
UCT = MCTS + UCB1
```
3. The UCT selection strategy is similar to the UCB1 strategy:
```
```