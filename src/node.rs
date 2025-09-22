use std::rc::{Rc, Weak};

use crate::{action::Action, mdp::MDP, rand};

#[derive(Debug, Clone)]
pub(crate) struct Node<S, A, R> {
    /// Records the unique node id to distinguish duplicated state
    id: u16,
    /// Records the number of times this state has been visited
    visits: u16,
    pub(crate) state: S,
    action: Option<A>,
    reward: Option<R>,
    parent: Weak<Rc<Node<S, A, R>>>,
    children: Vec<Rc<Node<S, A, R>>>,
}

impl<S, A: Action, R> Node<S, A, R> {
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
            children: vec![],
        }
    }

    /// Select a node that is not fully expanded
    pub(crate) fn select<M: MDP<S, A, R>>(&self, mdp: &M) {
        if !self.is_full_expanded(mdp) || mdp.is_terminal(&self.state) {
            // return Rc::clone(&self);
        }

        // Assuming this node is already fully expanded
        // (i.e. all it's children have been explored),
        // we need to make an informed decision about which of it's
        // children to select to become the next node under scope
    }

    pub(crate) fn expand<M: MDP<S, A, R>>(&self, mdp: &M) -> Rc<Self> {
        // let actions = mdp.get_actions(&self.state);
        // if actions.is_empty() {
        //     return Rc::clone(&self);
        // }
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

    /// Returns true if and only if all child actions have been expanded
    fn is_full_expanded<M: MDP<S, A, R>>(&self, mdp: &M) -> bool {
        mdp.get_actions(&self.state).len() == self.children.len()
    }
}

// Needs to implement QFunction
// Needs to implement Bandit
