use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::{
    action::Action,
    mdp::MDP,
    rand::{genrand, get_id},
    ucb1::UCB1,
};

#[derive(Debug, Clone)]
pub struct Node<S, A> {
    /// Records the unique node id to distinguish duplicated state
    // id: u16,
    /// Records the number of times this node has been visited
    pub(crate) visits: RefCell<usize>,
    pub(crate) state: S,
    pub(crate) id: u64,
    /// The action that resulted in this Node(State)
    pub(crate) action: Option<A>,
    reward: Option<f64>,
    parent: Weak<Node<S, A>>,
    /// rather than storing stats(time visited for the bandit) in UCB1, we only store children and times visited here
    /// In UCB1 where we need to explore all the actions first before we start exploiting
    /// All we just do is compare total actions on this state with the total children (explored children of this node)
    /// If they're the same, we've explored everything, else we haven't, and we just add the missing action(child)
    /// Since each node has `action` we can easily use this to know which action/child has been checked or not
    pub(crate) children: RefCell<Vec<Rc<Node<S, A>>>>,
}

impl<S, A: Action> Node<S, A> {
    pub(crate) fn new(
        state: S,
        id: u64,
        action: Option<A>,
        reward: Option<f64>,
        parent: Weak<Node<S, A>>,
    ) -> Self {
        Self {
            id,
            visits: RefCell::new(0),
            state,
            action,
            reward,
            parent,
            children: RefCell::new(vec![]),
        }
    }

    /// Simulate the outcome of an action, and return the child node
    pub(crate) fn get_outcome_child<M>(self: &Rc<Self>, mdp: &M, action: &A) -> Rc<Node<S, A>>
    where
        M: MDP<S, A>,
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
            get_id(),
            Some(*action),
            Some(reward),
            Rc::downgrade(self),
        ));

        self.children.borrow_mut().push(Rc::clone(&new_child));

        return new_child;
    }

    /// Select a node that is not fully expanded
    pub(crate) fn select<M>(self: &Rc<Self>, mdp: &M, bandit: &UCB1) -> Rc<Self>
    //  -> Rc<Node<S, A, R>>
    where
        M: MDP<S, A>,
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

    pub(crate) fn expand<M: MDP<S, A>>(self: &Rc<Self>, mdp: &M) -> Rc<Self> {
        if mdp.is_terminal(&self.state) {
            return Rc::clone(&self);
        }

        // Randomly select an unexpected action to expand
        let actions = mdp.get_actions(&self.state);
        let actions = actions
            .iter()
            .filter(|a| {
                let already_executed = self
                    .children
                    .borrow()
                    .iter()
                    .any(|child| child.action.is_some_and(|ac| ac == **a));

                return !already_executed;
            })
            .collect::<Vec<_>>();

        let index = genrand(0, actions.len());
        let action = actions[index];
        return self.get_outcome_child(mdp, action);
    }

    /// BackPropagate the reward back to the parent node
    pub(crate) fn back_propagate(self: &Rc<Self>, reward: f64, q_function: &mut UCB1) {
        *self.visits.borrow_mut() += 1;

        let qvalue = q_function.get_q_value(&self.id);
        let delta = (1f64 / *self.visits.borrow() as f64) * (reward - qvalue);
        q_function.update(&self, delta);

        if let Some(parent) = self.parent.upgrade() {
            let sreward = self.reward.as_ref().map_or(0f64, |r| *r);
            parent.back_propagate(reward + sreward, q_function);
        }
    }

    /// Returns true if and only if all child actions have been expanded
    fn is_full_expanded<M: MDP<S, A>>(&self, mdp: &M) -> bool {
        mdp.get_actions(&self.state).len() == self.children.borrow().len()
    }
}
