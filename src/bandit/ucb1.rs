use crate::bandit::selector::{MultiArmedBandit, Selector};

/// Given that this node is fully expanded i.e all the direct children of this node have been explored
/// This method helps us calculate the best child of this node to exploit further
/// Selects an action for the state from a list given a Q-function(???) (https://gibberblot.github.io/rl-notes/single-agent/multi-armed-bandits.html#id5)
/// this can be: Softmax strategy, UCB1 e.t.c
pub struct UCB1;

impl<S, A, Q> MultiArmedBandit<S, A, Q> for UCB1
where
    Q: Selector<S, A>,
{
    fn select(&self, state: S, actions: Vec<A>, q_function: Q) {
        //
    }
}
