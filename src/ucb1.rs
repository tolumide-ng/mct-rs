use core::f64;
use std::collections::HashMap;

use crate::action::Action;
use crate::node::Node;
use crate::rand::genrand;

/// Given that this node is fully expanded i.e all the direct children of this node have been explored
/// This method helps us calculate the best child of this node to exploit further
/// Selects an action for the state from a list given a Q-function(???) (https://gibberblot.github.io/rl-notes/single-agent/multi-armed-bandits.html#id5)
/// this can be: Softmax strategy, UCB1 e.t.c

#[derive(Debug, Default)]
pub struct UCB1 {
    // TODO: Readup/watch videos on Qfunctions
    q_table: HashMap<u64, f64>,
}

impl UCB1 {
    pub(crate) fn get_q_value(&self, id: &u64) -> f64 {
        *self.q_table.get(id).unwrap_or(&0.0)
    }

    pub(crate) fn update<S, A>(&mut self, state: &Node<S, A>, delta: f64)
    where
        A: Action,
    {
        let entry = self.q_table.entry(state.id).or_insert(0.0);
        *entry += delta
    }

    pub(crate) fn select<S, A>(&self, node: &Node<S, A>, actions: Vec<A>) -> A
    where
        A: Action,
    {
        if let Some(untried) = actions.iter().find(|&action| {
            !node
                .children
                .borrow()
                .iter()
                .any(|c| c.action == Some(*action))
        }) {
            return *untried;
        }

        let mut max_actions: Vec<A> = Vec::with_capacity(actions.len());
        let mut max_value = f64::NEG_INFINITY;
        let total = node
            .children
            .borrow()
            .iter()
            .map(|x| *x.visits.borrow())
            .sum::<usize>() as f64;

        for child in node.children.borrow().iter() {
            let value = self.get_q_value(&node.id)
                + f64::sqrt(2f64 * f64::log10(total)) / (*child.visits.borrow() as f64);

            if value > max_value {
                max_actions = vec![child.action.unwrap()];
                max_value = value;
            } else {
                max_actions.push(child.action.unwrap());
            }
        }

        // If there are multiple actions with the highest value, choose one randomly
        let index = genrand(0, max_actions.len());
        return max_actions[index];
    }
}
