use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::{action::Action, mdp::MDP, policy::RolloutPolicy, ucb1::UCB1};

#[derive(Debug)]
pub struct Node<S, A> {
    pub state: S,
    /// The action that resulted in this Node(State)
    // pub(crate) action: Option<A>,
    pub(crate) action: Option<A>,
    // pub reward: Option<f64>,
    parent: Weak<Node<S, A>>,
    /// rather than storing stats(time visited for the bandit) in UCB1, we only store children and times visited here
    /// In UCB1 where we need to explore all the actions first before we start exploiting
    /// All we just do is compare total actions on this state with the total children (explored children of this node)
    /// If they're the same, we've explored everything, else we haven't, and we just add the missing action(child)
    /// Since each node has `action` we can easily use this to know which action/child has been checked or not
    /// IF ALL CHILDREN NODES OF THIS NODE ARE VISITED, THIS NODE IS CONSIDERED FULLY EXPANDED, otherwise it's not full expanded
    // pub(crate) children: RefCell<Vec<Rc<Node<S, A>>>>,
    pub(crate) children: RefCell<Vec<Rc<Node<S, A>>>>,
    /// Records the number of times this node has been on the backpropagation path
    /// N(v) - A node is considered visited if it has been evaluated at least once.
    pub(crate) visits: RefCell<usize>,
    /// Q(v) - Total simulation reward
    // pub(crate) score: RefCell<f64>,
    pub(crate) score: RefCell<f64>,
}

impl<S, A: Action> Node<S, A>
where
    S: Eq + PartialEq,
{
    pub(crate) fn new(
        state: S,
        action: Option<A>,
        score: Option<f64>,
        parent: Weak<Node<S, A>>,
    ) -> Self {
        Self {
            visits: RefCell::new(0),
            state,
            action,
            score: RefCell::new(score.unwrap_or(0.0)),
            parent,
            children: RefCell::new(vec![]),
            // score: RefCell::new(0f64),
        }
    }

    pub(crate) fn q_value(&self) -> f64 {
        let visits = *(self.visits.borrow());
        if visits == 0 {
            0.0
        } else {
            *self.score.borrow() / (visits as f64)
        }
    }

    // /// Simulate the outcome of an action, and return the child node
    pub(crate) fn get_outcome_child<M>(self: &Rc<Self>, mdp: &M, action: &A) -> Rc<Node<S, A>>
    where
        M: MDP<S, A>,
    {
        // Chose one outcome based on transition probabilities
        let (next_state, reward, _) = mdp.execute(&self.state, action);

        // If a child already exists for this *resulting state* and action, return it.
        // We do that here by checking if any of the children(node) was a product of the action A
        for child in self.children.borrow().iter() {
            if let Some(child_action) = child.action {
                if child_action == *action {
                    return Rc::clone(child);
                }
            }
        }

        // for child in self.children.borrow().iter() {
        //     if next_state == child.state {
        //         return Rc::clone(child);
        //     }
        // }

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

    /// TODO:  This should be considered as a trait, but a default value just incase the user wants to provide something custom here
    pub fn ucb1(self: &Rc<Self>, exploration_constant: f64) -> f64 {
        let parent_visits = if let Some(parent) = self.parent.upgrade() {
            *(parent.visits.borrow()) as f64
        } else {
            1.0
        }
        .max(1f64);

        // self.q_value()
        //     + exploration_constant
        //         * (parent_visits.ln() / (*self.visits.borrow() as f64 + 1e-6)).sqrt()

        let child_visits = (*self.visits.borrow()).max(1) as f64;
        self.q_value() + (exploration_constant * (parent_visits.ln() / child_visits).sqrt())
        // self.q_value() + (exploration_constant * (parent_visits.ln() / child_visits))
        // self.q_value() + f64::sqrt((2f64 * parent_visits.ln()) / child_visits)
    }

    /// Select a node that is not fully expanded
    pub(crate) fn select<M>(self: &Rc<Self>, mdp: &M, bandit: &UCB1) -> Rc<Self>
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

    pub(crate) fn expand<M, P>(self: &Rc<Self>, mdp: &M, policy: &P) -> Rc<Self>
    where
        M: MDP<S, A>,
        P: RolloutPolicy<M, S, A>,
    {
        if mdp.is_terminal(&self.state) {
            return Rc::clone(&self);
        }

        let explored = self
            .children
            .borrow()
            .iter()
            .flat_map(|x| x.action)
            .collect::<Vec<_>>();

        // let children = self.children.borrow();
        // Randomly select an unexpected action to expand
        let actions = mdp.get_actions(&self.state);
        let expandable_actions = actions
            .into_iter()
            .filter(|a| !explored.contains(a))
            .collect::<Vec<_>>();

        // let index = genrand(0, expandable_actions.len());
        let action = policy.pick(&self.state, &expandable_actions);
        // let action = expandable_actions[index];

        return self.get_outcome_child(mdp, &action);
    }

    /// BackPropagate the reward back to the parent node
    pub(crate) fn back_propagate(self: &Rc<Self>, reward: f64, q_function: &mut UCB1) {
        *self.visits.borrow_mut() += 1;
        *self.score.borrow_mut() += reward;

        if let Some(parent) = self.parent.upgrade() {
            parent.back_propagate(reward, q_function);
        }
    }

    /// Returns true if and only if all child actions have been expanded
    fn is_full_expanded<M: MDP<S, A>>(&self, mdp: &M) -> bool {
        let actions = mdp.get_actions(&self.state);
        let explored = self
            .children
            .borrow()
            .iter()
            .flat_map(|c| c.action)
            .collect::<Vec<_>>();

        return actions.len() == explored.len();
    }
}

#[cfg(test)]
mod tests {
    use crate::policy::RandomRollout;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestAction {
        A,
        B,
    }

    impl Action for TestAction {}
    struct DummyMDP;

    impl MDP<u32, TestAction> for DummyMDP {
        fn execute(&self, state: &u32, action: &TestAction) -> (u32, f64, bool) {
            let next_state = match action {
                TestAction::A => *state + 1,
                TestAction::B => *state + 2,
            };
            let reward = *state as f64;
            (next_state, reward, false)
        }

        fn get_actions(&self, _state: &u32) -> Vec<TestAction> {
            vec![TestAction::A, TestAction::B]
        }

        fn is_terminal(&self, state: &u32) -> bool {
            *state >= 10
        }

        fn get_states(&self) -> Vec<u32> {
            todo!()
        }

        fn get_transitions(&self, state: &u32, action: &TestAction) -> Vec<(u32, f64)> {
            todo!()
        }

        fn get_reward(&self, state: &u32, action: &TestAction, next_state: &u32) -> f64 {
            todo!()
        }

        fn get_discount_factor(&self) -> f64 {
            todo!()
        }

        fn get_initial_state(&self) -> u32 {
            todo!()
        }

        fn get_goal_states(&self) -> Vec<u32> {
            todo!()
        }
    }

    struct DummyUCB;

    #[test]
    fn test_node_new() {
        let node: Node<u32, TestAction> = Node::new(0, None, None, Weak::new());
        assert_eq!(*node.visits.borrow(), 0);
        assert_eq!(node.state, 0);
        assert!(node.action.is_none());
        assert!(node.children.borrow().is_empty());
    }

    #[test]
    fn test_get_outcome_child_adds_new_child() {
        let root = Rc::new(Node::new(0, None, None, Weak::new()));
        let mdp = DummyMDP;

        let child = root.get_outcome_child(&mdp, &TestAction::A);

        assert_eq!(root.children.borrow().len(), 1);
        assert_eq!(child.state, 1); // 0 + 1
        assert!(Rc::ptr_eq(&child.parent.upgrade().unwrap(), &root));
    }

    #[test]
    fn test_get_outcome_child_returns_existing_child() {
        let root = Rc::new(Node::new(0, None, None, Weak::new()));
        let mdp = DummyMDP;

        let child1 = root.get_outcome_child(&mdp, &TestAction::A);
        let child2 = root.get_outcome_child(&mdp, &TestAction::A);

        assert!(Rc::ptr_eq(&child1, &child2));
        assert_eq!(root.children.borrow().len(), 1);
    }

    #[test]
    fn test_is_full_expanded() {
        let node = Rc::new(Node::new(0, None, None, Weak::new()));
        let mdp = DummyMDP;

        assert!(!node.is_full_expanded(&mdp));

        // Expand all actions
        node.get_outcome_child(&mdp, &TestAction::A);
        node.get_outcome_child(&mdp, &TestAction::B);

        assert!(node.is_full_expanded(&mdp));
    }

    #[test]
    fn test_expand_adds_one_child() {
        let node = Rc::new(Node::new(0, None, None, Weak::new()));
        let mdp = DummyMDP;
        let policy = RandomRollout::new();

        assert_eq!(node.children.borrow().len(), 0);

        let child = node.expand(&mdp, &policy);

        assert_eq!(node.children.borrow().len(), 1);
        assert_eq!(
            *node.children.borrow()[0].score.borrow(),
            *child.score.borrow()
        );
        assert_eq!(node.children.borrow()[0].state, child.state);
        assert_eq!(node.children.borrow()[0].visits, child.visits);
    }

    #[test]
    fn test_expand_terminal_returns_self() {
        let node = Rc::new(Node::new(10, None, None, Weak::new())); // terminal state
        let mdp = DummyMDP;
        let policy = RandomRollout::new();

        let child = node.expand(&mdp, &policy);

        assert!(Rc::ptr_eq(&node, &child));
    }

    #[test]
    fn test_back_propagate_increments_visits() {
        let root = Rc::new(Node::new(0, None, None, Weak::new()));
        let child = Rc::new(Node::new(
            1,
            Some(TestAction::A),
            Some(1.0),
            Rc::downgrade(&root),
        ));
        root.children.borrow_mut().push(Rc::clone(&child));

        let mut q = UCB1::default();
        child.back_propagate(10.0, &mut q);

        assert_eq!(*child.visits.borrow(), 1);
        assert_eq!(*root.visits.borrow(), 1);
    }

    #[test]
    fn test_select_returns_terminal_node() {
        let root = Rc::new(Node::new(10, None, None, Weak::new())); // terminal state
        let mdp = DummyMDP;
        let bandit = UCB1::default();

        let selected = root.select(&mdp, &bandit);
        assert!(Rc::ptr_eq(&selected, &root));
    }

    #[test]
    fn test_select_traverses_fully_expanded() {
        let root = Rc::new(Node::new(0, None, None, Weak::new()));
        let mdp = DummyMDP;
        let bandit = UCB1::default();

        // Expand both actions
        root.get_outcome_child(&mdp, &TestAction::A);
        root.get_outcome_child(&mdp, &TestAction::B);

        let selected = root.select(&mdp, &bandit);

        // Should return one of the children
        // assert!(root.children.borrow().contains(&selected));

        assert_eq!(root.children.borrow().len(), 2);

        assert_eq!(
            *root.children.borrow()[0].score.borrow(),
            *selected.score.borrow()
        );
        assert_eq!(root.children.borrow()[0].state, selected.state);
        assert_eq!(root.children.borrow()[0].visits, selected.visits);
    }
}
