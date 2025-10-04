use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::{action::Action, mdp::MDP, rand::genrand, ucb1::UCB1};

#[derive(Debug)]
pub struct Node<S, A> {
    /// Records the unique node id to distinguish duplicated state
    // id: u16,
    /// Records the number of times this node has been visited
    pub(crate) visits: RefCell<usize>,
    pub(crate) state: S,
    pub(crate) id: usize,
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

impl<S, A: Action> Node<S, A>
where
    S: Eq + PartialEq,
{
    pub(crate) fn new(
        state: S,
        id: usize,
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

    pub fn most_visited_child(&self) -> Option<Rc<Node<S, A>>> {
        self.children
            .borrow()
            .iter()
            .max_by_key(|c| *c.visits.borrow())
            .map(Rc::clone)
    }

    pub fn visits(&self) -> usize {
        *self.visits.borrow()
    }

    /// Simulate the outcome of an action, and return the child node
    pub(crate) fn get_outcome_child<M>(
        self: &Rc<Self>,
        mdp: &M,
        action: &A,
        next_id: &RefCell<usize>,
    ) -> Rc<Node<S, A>>
    where
        M: MDP<S, A>,
    {
        // Chose one outcome based on transition probabilities
        let (next_state, reward, _) = mdp.execute(&self.state, action);

        // If a child already exists for this *resulting state* and action, return it.
        // We do that here by checking if any of the children(node) was a product of the action A
        for child in self.children.borrow().iter() {
            if next_state == child.state {
                return Rc::clone(child);
            }
        }
        // for child in self.children.borrow().iter() {
        //     if let Some(child_action) = child.action {
        //         if child_action == *action {
        //             return Rc::clone(child);
        //         }
        //     }
        // }

        let id = *next_id.borrow();
        *next_id.borrow_mut() += 1;

        // This outcome has not occured from this state-action pair previously
        let new_child = Rc::new(Node::new(
            next_state,
            id,
            Some(*action),
            Some(reward),
            Rc::downgrade(self),
        ));

        self.children.borrow_mut().push(Rc::clone(&new_child));

        return new_child;
    }

    /// Select a node that is not fully expanded
    pub(crate) fn select<M>(
        self: &Rc<Self>,
        mdp: &M,
        bandit: &UCB1,
        next_id: &RefCell<usize>,
    ) -> Rc<Self>
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
        let child = self.get_outcome_child(mdp, &action, &next_id);

        return child.select(mdp, bandit, &next_id);
    }

    pub(crate) fn expand<M: MDP<S, A>>(
        self: &Rc<Self>,
        mdp: &M,
        next_id: &RefCell<usize>,
    ) -> Rc<Self> {
        if mdp.is_terminal(&self.state) {
            return Rc::clone(&self);
        }

        // let xx = id.borrow();
        // *id.borrow_mut() += 1;

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
        return self.get_outcome_child(mdp, action, next_id);
    }

    /// BackPropagate the reward back to the parent node
    pub(crate) fn back_propagate(self: &Rc<Self>, reward: f64, q_function: &mut UCB1) {
        *self.visits.borrow_mut() += 1;

        let qvalue = q_function.get_q_value(&self.id);
        let delta = (1f64 / *self.visits.borrow() as f64) * (reward - qvalue);
        q_function.update(&self, delta);

        if let Some(parent) = self.parent.upgrade() {
            let self_reward = self.reward.as_ref().unwrap_or(&0.0);
            parent.back_propagate(reward + 0.99 * self_reward, q_function);
        }
    }

    /// Returns true if and only if all child actions have been expanded
    fn is_full_expanded<M: MDP<S, A>>(&self, mdp: &M) -> bool {
        mdp.get_actions(&self.state).iter().all(|a| {
            self.children
                .borrow()
                .iter()
                .find(|c| c.action.is_some_and(|ca| ca == *a))
                .is_some()
        })
    }
}

#[cfg(test)]
mod tests {
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
        let node: Node<u32, TestAction> = Node::new(0, 1, None, None, Weak::new());
        assert_eq!(*node.visits.borrow(), 0);
        assert_eq!(node.state, 0);
        assert!(node.action.is_none());
        assert!(node.children.borrow().is_empty());
    }

    #[test]
    fn test_get_outcome_child_adds_new_child() {
        let root = Rc::new(Node::new(0, 1, None, None, Weak::new()));
        let next_id = RefCell::new(2);
        let mdp = DummyMDP;

        let child = root.get_outcome_child(&mdp, &TestAction::A, &next_id);

        assert_eq!(root.children.borrow().len(), 1);
        assert_eq!(child.state, 1); // 0 + 1
        assert_eq!(next_id.borrow().clone(), 3);
        assert!(Rc::ptr_eq(&child.parent.upgrade().unwrap(), &root));
    }

    #[test]
    fn test_get_outcome_child_returns_existing_child() {
        let root = Rc::new(Node::new(0, 1, None, None, Weak::new()));
        let next_id = RefCell::new(2);
        let mdp = DummyMDP;

        let child1 = root.get_outcome_child(&mdp, &TestAction::A, &next_id);
        let child2 = root.get_outcome_child(&mdp, &TestAction::A, &next_id);

        assert!(Rc::ptr_eq(&child1, &child2));
        assert_eq!(root.children.borrow().len(), 1);
    }

    #[test]
    fn test_is_full_expanded() {
        let node = Rc::new(Node::new(0, 1, None, None, Weak::new()));
        let mdp = DummyMDP;

        assert!(!node.is_full_expanded(&mdp));

        // Expand all actions
        let next_id = RefCell::new(2);
        node.get_outcome_child(&mdp, &TestAction::A, &next_id);
        node.get_outcome_child(&mdp, &TestAction::B, &next_id);

        assert!(node.is_full_expanded(&mdp));
    }

    #[test]
    fn test_expand_adds_one_child() {
        let node = Rc::new(Node::new(0, 1, None, None, Weak::new()));
        let mdp = DummyMDP;
        let next_id = RefCell::new(2);

        assert_eq!(node.children.borrow().len(), 0);

        let child = node.expand(&mdp, &next_id);

        assert_eq!(node.children.borrow().len(), 1);
        assert_eq!(node.children.borrow()[0].id, child.id);
        assert_eq!(node.children.borrow()[0].reward, child.reward);
        assert_eq!(node.children.borrow()[0].state, child.state);
        assert_eq!(node.children.borrow()[0].visits, child.visits);
    }

    #[test]
    fn test_expand_terminal_returns_self() {
        let node = Rc::new(Node::new(10, 1, None, None, Weak::new())); // terminal state
        let mdp = DummyMDP;
        let next_id = RefCell::new(2);

        let child = node.expand(&mdp, &next_id);

        assert!(Rc::ptr_eq(&node, &child));
    }

    #[test]
    fn test_back_propagate_increments_visits() {
        let root = Rc::new(Node::new(0, 1, None, None, Weak::new()));
        let child = Rc::new(Node::new(
            1,
            2,
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
        let root = Rc::new(Node::new(10, 1, None, None, Weak::new())); // terminal state
        let mdp = DummyMDP;
        let bandit = UCB1::default();
        let next_id = RefCell::new(2);

        let selected = root.select(&mdp, &bandit, &next_id);

        assert!(Rc::ptr_eq(&selected, &root));
    }

    #[test]
    fn test_select_traverses_fully_expanded() {
        let root = Rc::new(Node::new(0, 1, None, None, Weak::new()));
        let mdp = DummyMDP;
        let bandit = UCB1::default();
        let next_id = RefCell::new(2);

        // Expand both actions
        root.get_outcome_child(&mdp, &TestAction::A, &next_id);
        root.get_outcome_child(&mdp, &TestAction::B, &next_id);

        let selected = root.select(&mdp, &bandit, &next_id);

        // Should return one of the children
        // assert!(root.children.borrow().contains(&selected));

        assert_eq!(root.children.borrow().len(), 2);

        assert_eq!(root.children.borrow()[0].id, selected.id);
        assert_eq!(root.children.borrow()[0].reward, selected.reward);
        assert_eq!(root.children.borrow()[0].state, selected.state);
        assert_eq!(root.children.borrow()[0].visits, selected.visits);
    }
}
