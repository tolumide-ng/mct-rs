use core::f64;
use std::{
    cell::RefCell,
    fmt::Display,
    rc::{Rc, Weak},
    time::Instant,
};

use crate::{
    action::Action, mdp::MDP, node::Node, policy::RolloutPolicy, rand::genrand, strategy::Strategy,
    ucb1::UCB1,
};

pub struct MCTS<M, S, A, P>
where
    M: MDP<S, A>,
    A: Action,
    S: Clone,
    P: RolloutPolicy<M, S, A>,
{
    mdp: M,
    root: Rc<Node<S, A>>,
    next_id: RefCell<usize>,
    bandit: UCB1,
    policy: P,
}

impl<M, S, A, P> MCTS<M, S, A, P>
where
    M: MDP<S, A>,
    A: Action,
    S: Clone + Eq + PartialEq,
    P: RolloutPolicy<M, S, A>,
{
    pub fn new(mdp: M, policy: P) -> Self {
        let state = mdp.get_initial_state();
        Self {
            root: Rc::new(Node::new(state, 0, None, None, Weak::new())),
            next_id: RefCell::new(1),
            mdp,
            bandit: UCB1::default(),
            policy,
        }
    }

    /// Execute the MCTS algorithm from the initial state given, with timeout in seconds
    /// After how many milliseconds, the mcts should timeout
    /// TODO: Move this to be more dynamic, and support max-depth timeout
    pub fn mcts(&mut self, timeout: u128) {
        let start_time = Instant::now();

        while start_time.elapsed().as_millis() < timeout {
            // Find a state node to expand
            let selected_node = self.root.select(&self.mdp, &self.bandit, &self.next_id);
            // let xx = !self.mdp.is_terminal(&selected_node.state);
            if !self.mdp.is_terminal(&selected_node.state) {
                let child = selected_node.expand(&self.mdp, &self.policy, &self.next_id);
                let reward = self.simulate(&child, start_time, timeout);
                child.back_propagate(reward, &mut self.bandit);
            }
        }
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
        // let mut depth = 0;

        while !self.mdp.is_terminal(&state) && start_time.elapsed().as_millis() < timeout {
            let actions = self.mdp.get_actions(&state);

            // Choose an action to execute
            let action = self.policy.pick(&state, &actions);

            // Execute the action
            let (next_state, reward, ..) = self.mdp.execute(&state, &action);

            // Discount the reward
            // cumulative_reward += f64::powi(self.mdp.get_discount_factor(), depth) * reward;
            cumulative_reward += reward;
            // depth += 1;

            state = next_state;
        }

        if !self.mdp.is_terminal(&state) {
            // todo! this needs to be a trait
            cumulative_reward += self.heuristic_eval(&state);
        }

        return cumulative_reward;
    }

    pub fn best_action(&self, strategy: Strategy) -> Option<A> {
        let root = &self.root;
        let children = root.children.borrow();

        if children.is_empty() {
            return None;
        }

        match strategy {
            Strategy::MostVisited => children
                .iter()
                .max_by_key(|c| *c.visits.borrow())
                .and_then(|c| c.action),

            Strategy::HighestQValue => children
                .iter()
                .max_by(|a, b| {
                    a.q_value()
                        .partial_cmp(&b.q_value())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .and_then(|c| c.action),

            Strategy::Probabilistic => {
                // Softmax over Q-values
                let qvalues = children.iter().map(|c| c.q_value()).collect::<Vec<_>>();

                let maxq = qvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                // subtract maxq for numerical stability
                let expq: Vec<f64> = qvalues.iter().map(|q| ((q - maxq).exp())).collect();
                let sum = expq.iter().sum::<f64>().max(f64::MIN_POSITIVE);

                let probs = expq.iter().map(|x| x / sum).collect::<Vec<_>>();

                // sample based on probabilities
                let mut r = genrand(0, 10_000) as f64 / 10_000.0;
                for (i, p) in probs.iter().enumerate() {
                    r -= p;
                    if r <= 0.0 {
                        return children[i].action;
                    }
                }

                // fallback
                children[0].action
            }
            Strategy::HeuristicWin => {
                // prioritize terminal winning moves
                let mut winning_mvs = vec![];
                let mut bestq = f64::NEG_INFINITY;
                let mut best_mvs = vec![];

                for child in children.iter() {
                    let q = child.q_value();

                    // if child is terminal with positive reward (win)
                    // if let Some(reward) = child.score.borrow() {}

                    if *child.score.borrow() > 0.0 {
                        winning_mvs.push(child);
                        continue;
                    }

                    if q > bestq {
                        bestq = q;
                        best_mvs = vec![child];
                    } else if (q - bestq).abs() < 1e-9 {
                        best_mvs.push(child);
                    }
                }

                let chosen = if !winning_mvs.is_empty() {
                    &winning_mvs[genrand(0, winning_mvs.len())]
                } else {
                    &best_mvs[genrand(0, best_mvs.len())]
                };

                chosen.action
            }
        }
    }
}
