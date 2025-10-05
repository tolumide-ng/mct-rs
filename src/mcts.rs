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
    exploration_constant: f64,
}

impl<M, S, A> MCTS<M, S, A>
where
    M: MDP<S, A>,
    A: Action,
    S: Clone + Eq + PartialEq,
{
    pub fn new(mdp: M) -> Self {
        let state = mdp.get_initial_state();
        Self {
            root: Rc::new(Node::new(state, 0, None, None, Weak::new())),
            next_id: RefCell::new(1),
            mdp,
            bandit: UCB1::default(),
            exploration_constant: 1.4142135623730951,
        }
    }

    pub fn best_action(&self) -> Option<A> {
        self.root.most_visited_child().and_then(|c| c.action)
    }

    // fn ucb1(&self, parent_visits: f64, child: &Rc<Node<S, A>>) -> f64 {
    //     child.q_value()
    //         + self.exploration_constant
    //             * (parent_visits.ln() / (*child.visits.borrow() as f64 + 1e-6)).sqrt()
    // }

    /// Execute the MCTS algorithm from the initial state given, with timeout in seconds
    /// After how many milliseconds, the mcts should timeout
    /// TODO: Move this to be more dynamic, and support max-depth timeout
    pub fn mcts(&mut self, timeout: u128) {
        let start_time = Instant::now();

        while start_time.elapsed().as_millis() < timeout {
            // Find a state node to expand
            let selected_node = self.root.select(&self.mdp, &self.bandit, &self.next_id);

            if !self.mdp.is_terminal(&selected_node.state) {
                let child = selected_node.expand(&self.mdp, &self.next_id);
                let reward = self.simulate(&child, start_time, timeout);
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

    /// TODO: This would eventually be moved to a trait that must be implemented on state!, this MCTS or whatever!
    pub(crate) fn heuristic_eval(&self, _state: &S) -> f64 {
        0.0
    }

    /// Simulate until a terminal state
    pub(crate) fn simulate(
        &self,
        node: &Rc<Node<S, A>>,
        start_time: Instant,
        timeout: u128,
    ) -> f64 {
        let mut state = node.state.clone();
        let mut cumulative_reward = 0.0;
        let mut depth = 0;

        while !self.mdp.is_terminal(&state) && start_time.elapsed().as_millis() < timeout {
            // Choose an action to execute
            let action = &self.choose(&state);

            // Execute the action
            let (next_state, reward, ..) = self.mdp.execute(&state, action);

            // Discount the reward
            cumulative_reward += f64::powi(self.mdp.get_discount_factor(), depth) * reward;
            depth += 1;

            state = next_state;
        }

        if !self.mdp.is_terminal(&state) {
            cumulative_reward += self.heuristic_eval(&state);
        }

        return cumulative_reward;
    }
}
