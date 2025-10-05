use core::f64;
use std::collections::HashMap;

use crate::action::Action;
use crate::node::Node;

/// Given that this node is fully expanded i.e all the direct children of this node have been explored
/// This method helps us calculate the best child of this node to exploit further
/// Selects an action for the state from a list given a Q-function(???) (https://gibberblot.github.io/rl-notes/single-agent/multi-armed-bandits.html#id5)
/// this can be: Softmax strategy, UCB1 e.t.c
#[derive(Debug, Default)]
pub struct UCB1 {
    // TODO: Readup/watch videos on Qfunctions
    q_table: HashMap<usize, f64>,
}

impl UCB1 {
    // pub(crate) fn get_q_value(&self, id: &usize) -> f64 {
    //     *self.q_table.get(id).unwrap_or(&0.0)
    // }

    // pub(crate) fn update<S, A>(&mut self, state: &Node<S, A>, delta: f64)
    // where
    //     A: Action,
    // {
    //     // let entry = self.q_table.entry(state.id).or_insert(0.0);
    //     // *entry += delta
    //     self.q_table.insert(state.id, delta);
    // }

    const C: f64 = 1.4142135623730951;

    pub(crate) fn select<S, A>(&self, node: &Node<S, A>, actions: Vec<A>) -> A
    where
        A: Action,
        S: PartialEq + Eq,
    {
        let children = node.children.borrow();

        if let Some(untried) = actions
            .iter()
            .find(|&a| !children.iter().any(|c| c.action == Some(*a)))
        {
            return *untried;
        }

        let max = children
            .iter()
            .max_by(|a, b| (a.ucb1(Self::C)).total_cmp(&b.ucb1(Self::C)))
            .unwrap();

        return max.action.unwrap();
    }
}
