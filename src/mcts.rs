use std::{rc::Weak, time::Instant};

use crate::{action::Action, mdp::MDP, node::Node};

pub struct MCTS<M, S, A, R>
where
    M: MDP<S, A, R>,
    A: Action,
{
    mdp: M,
    root: Node<S, A, R>,
    next_id: u16,
    /// After how many milliseconds, the mcts should timeout
    /// TODO: Move this to be more dynamic, and support max-depth timeout
    timeout: u128,
}

impl<M: MDP<S, A, R>, S, A: Action, R> MCTS<M, S, A, R> {
    pub fn new(mdp: M, timeout: u128) -> Self {
        let state = mdp.get_initial_state();
        Self {
            root: Node::new(0, state, None, Weak::new()),
            mdp,
            next_id: 1,
            timeout,
        }
    }

    /// Execute the MCTS algorithm from the initial state given, with timeout in seconds
    pub fn mcts(&self) {
        let start_time = Instant::now();

        while start_time.elapsed().as_millis() < self.timeout {
            let selected_node = self.root.select();
            if !self.mdp.is_terminal(&selected_node.state) {
                let child = selected_node.expand(&self.mdp);
                let reward = child.simulate();
                child.back_propagate(reward);
            }
        }
    }
}

// Impl QFunction for MCTS {}
// Impl MDP for MCTS {}
// Impl Bandit for MCTS {}
