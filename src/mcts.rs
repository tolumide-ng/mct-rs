use std::{rc::Weak, time::Instant};

use crate::{mdp::MDP, node::Node};

pub struct MCTS<S: MDP<S, A, R>, A, R> {
    root: Node<S, A, R>,
    next_id: u16,
    /// After how many milliseconds, the mcts should timeout
    /// TODO: Move this to be more dynamic, and support max-depth timeout
    timeout: u128,
}

impl<S: MDP<S, A, R>, A, R> MCTS<S, A, R> {
    pub fn new(state: S, timeout: u128) -> Self {
        let next_id = 0;
        Self {
            root: Node::new(next_id, state, None, Weak::new()),
            next_id: next_id + 1,
            timeout,
        }
    }

    /// Execute the MCTS algorithm from the initial state given, with timeout in seconds
    pub fn mcts(&self) {
        let start_time = Instant::now();

        while start_time.elapsed().as_millis() < self.timeout {
            // Fund a state node to expand
        }
    }
}

// Impl QFunction for MCTS {}
// Impl MDP for MCTS {}
// Impl Bandit for MCTS {}
