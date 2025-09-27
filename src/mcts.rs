use std::{
    cell::RefCell,
    rc::{Rc, Weak},
    time::Instant,
};

use crate::{action::Action, mdp::MDP, node::Node, rand::genrand, ucb1::UCB1};

pub struct MCTS<M, S, A>
where
    M: MDP<S, A>,
    A: Action,
    S: Clone,
{
    mdp: M,
    root: Rc<Node<S, A>>,
    next_id: RefCell<usize>,
    bandit: UCB1,
}

impl<M, S, A> MCTS<M, S, A>
where
    M: MDP<S, A>,
    A: Action,
    S: Clone,
{
    pub fn new(mdp: M) -> Self {
        let state = mdp.get_initial_state();
        Self {
            root: Rc::new(Node::new(state, 0, None, None, Weak::new())),
            next_id: RefCell::new(1),
            mdp,
            bandit: UCB1::default(),
        }
    }

    pub fn best_action(&self) -> Option<A> {
        self.root.most_visited_child().and_then(|c| c.action)
    }

    /// Execute the MCTS algorithm from the initial state given, with timeout in seconds
    /// After how many milliseconds, the mcts should timeout
    /// TODO: Move this to be more dynamic, and support max-depth timeout
    pub fn mcts(&mut self, timeout: u128) {
        let start_time = Instant::now();

        while start_time.elapsed().as_millis() < timeout {
            let selected_node = self.root.select(&self.mdp, &self.bandit, &self.next_id);
            if !self.mdp.is_terminal(&selected_node.state) {
                let child = selected_node.expand(&self.mdp, &self.next_id);
                let reward = self.simulate(&child);
                child.back_propagate(reward, &mut self.bandit);
            }
        }
    }

    fn choose(&self, state: &S) -> A {
        let actions = self.mdp.get_actions(state);
        if actions.len() == 1 {
            return actions[0];
        }

        let index = genrand(0, actions.len());
        return actions[index];
    }

    /// Simulate until a terminal state
    pub(crate) fn simulate(&self, node: &Rc<Node<S, A>>) -> f64 {
        let mut state = node.state.clone();
        let mut cumulative_reward = 0.0;
        let mut depth = 0;

        while !self.mdp.is_terminal(&state) {
            // Choose an action to execute
            let action = &self.choose(&state);

            // Execute the action
            let (next_state, reward, ..) = self.mdp.execute(&state, action);

            // Discount the reward
            cumulative_reward += f64::powi(self.mdp.get_discount_factor(), depth) * reward;
            depth += 1;

            state = next_state;
        }

        return cumulative_reward;
    }
}
