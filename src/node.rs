use std::sync::{Arc, Weak};

pub(crate) struct Node<S, A, R> {
    /// Records the unique node id to distinguish duplicated state
    id: u16,
    /// Records the number of times this state has been visited
    visits: u16,
    state: S,
    action: A,
    reward: R,
    parent: Weak<Arc<Node<S, A, R>>>,
}

impl<S, A, R> Node<S, A, R> {
    pub(crate) fn new(
        id: u16,
        state: S,
        action: A,
        reward: R,
        parent: Weak<Arc<Node<S, A, R>>>,
    ) -> Self {
        Self {
            id,
            visits: 0,
            state,
            action,
            reward,
            parent,
        }
    }

    pub(crate) fn select(&self) {}

    pub(crate) fn expand(&self) {}

    /// BackPropagate the reward back to the parent node
    pub(crate) fn back_propagate(&self, reward: f64) {}

    /// Return the value of this node
    pub(crate) fn get_value(&self) {
        // max_q_value = self.qfunction.get_max_q(
        //     self.state, self.mdp.get_actions(self.state)
        // )
        // return max_q_value
    }
}

// Needs to implement MDP
// Needs to implement QFunction
// Needs to implement Bandit
