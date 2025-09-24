use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::{action::Action, bandit::selector::MultiArmedBandit, mdp::MDP, rand};

#[derive(Debug, Clone)]
pub struct Node<S, A, R>
where
    R: Clone,
{
    /// Records the unique node id to distinguish duplicated state
    // id: u16,
    /// Records the number of times this node has been visited
    pub(crate) visits: usize,
    pub(crate) state: S,
    /// The action that resulted in this Node(State)
    pub(crate) action: Option<A>,
    reward: Option<R>,
    parent: Weak<Node<S, A, R>>,
    /// rather than storing stats(time visited for the bandit) in UCB1, we only store children and times visited here
    /// In UCB1 where we need to explore all the actions first before we start exploiting
    /// All we just do is compare total actions on this state with the total children (explored children of this node)
    /// If they're the same, we've explored everything, else we haven't, and we just add the missing action(child)
    /// Since each node has `action` we can easily use this to know which action/child has been checked or not
    pub(crate) children: RefCell<Vec<Rc<Node<S, A, R>>>>,
}

impl<S, A: Action, R> Node<S, A, R>
where
    R: Clone,
{
    pub(crate) fn new(
        state: S,
        action: Option<A>,
        reward: Option<R>,
        parent: Weak<Node<S, A, R>>,
    ) -> Self {
        Self {
            // id,
            visits: 0,
            state,
            action,
            reward,
            parent,
            children: RefCell::new(vec![]),
        }
    }

    /// Simulate the outcome of an action, and return the child node
    pub(crate) fn get_outcome_child<M>(self: &Rc<Self>, mdp: &M, action: &A) -> Rc<Node<S, A, R>>
    where
        M: MDP<S, A, R>,
    {
        // Chose one outcome based on transition probabilities
        let (next_state, reward, _) = mdp.execute(&self.state, action);

        // Find the corresponding state and return if this already exists
        // We do that here by checking if any of the children(node) was a product of the action A
        for child in self.children.borrow().iter() {
            if let Some(child_action) = child.action {
                if child_action == *action {
                    return Rc::clone(self);
                }
            }
        }

        // This outcome has not occured from this state-action pair previously
        let new_child = Rc::new(Node::new(
            next_state,
            Some(*action),
            Some(reward),
            Rc::downgrade(self),
        ));

        self.children.borrow_mut().push(Rc::clone(&new_child));

        return new_child;
    }

    /// Select a node that is not fully expanded
    pub(crate) fn select<M, B>(self: &Rc<Self>, mdp: &M, bandit: &B) -> Rc<Node<S, A, R>>
    //  -> Rc<Node<S, A, R>>
    where
        M: MDP<S, A, R>,
        B: MultiArmedBandit<S, A, R>,
    {
        if !self.is_full_expanded(mdp) || mdp.is_terminal(&self.state) {
            return Rc::clone(&self);
        }

        // Assuming this node is already fully expanded
        // (i.e. all it's children have been explored),
        // we need to make an informed decision about which of it's
        // children to select to become the next node under scope
        let actions = mdp.get_actions(&self.state);
        let action = bandit.select(&self, actions);
        return self.get_outcome_child(mdp, &action).select(mdp, bandit);
    }

    pub(crate) fn expand<M: MDP<S, A, R>>(self: &Rc<Self>, mdp: &M) -> Rc<Self> {
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
        mdp.get_actions(&self.state).len() == self.children.borrow().len()
    }
}

// Needs to implement QFunction
// Needs to implement Bandit
