use std::rc::{Rc, Weak};

use crate::{action::Action, mdp::MDP, rand};

pub(crate) struct Node<S, A, R>
where
    S: MDP<S, A, R>,
    A: Action,
{
    /// Records the unique node id to distinguish duplicated state
    id: u16,
    /// Records the number of times this state has been visited
    visits: u16,
    pub(crate) state: S,
    action: Option<A>,
    reward: Option<R>,
    parent: Weak<Rc<Node<S, A, R>>>,
}

impl<S: MDP<S, A, R>, A: Action, R> Node<S, A, R> {
    pub(crate) fn new(
        id: u16,
        state: S,
        action: Option<A>,
        parent: Weak<Rc<Node<S, A, R>>>,
    ) -> Self {
        Self {
            id,
            visits: 0,
            state,
            action,
            reward: None,
            parent,
        }
    }

    pub(crate) fn select(&self) -> Self {
        todo!()
    }

    pub(crate) fn expand(&self) -> Self {
        todo!()
    }

    pub(crate) fn simulate(&self) -> R {
        todo!()
    }

    /// BackPropagate the reward back to the parent node
    pub(crate) fn back_propagate(&self, reward: R) {}

    /// Return the value of this node
    pub(crate) fn get_value(&self) {
        // max_q_value = self.qfunction.get_max_q(
        //     self.state, self.mdp.get_actions(self.state)
        // )
        // return max_q_value
    }

    fn choose(&self, actions: Vec<A>) -> A {
        if actions.len() == 1 {
            return actions[0];
        }

        let index = rand::genrand(0, actions.len());
        return actions[index];
    }
}

// Needs to implement QFunction
// Needs to implement Bandit
