use core::f64;
use std::rc::Rc;

use crate::action::Action;
use crate::bandit::selector::MultiArmedBandit;
use crate::node::Node;
use crate::rand::genrand;

/// Given that this node is fully expanded i.e all the direct children of this node have been explored
/// This method helps us calculate the best child of this node to exploit further
/// Selects an action for the state from a list given a Q-function(???) (https://gibberblot.github.io/rl-notes/single-agent/multi-armed-bandits.html#id5)
/// this can be: Softmax strategy, UCB1 e.t.c

#[derive(Debug, Default)]
pub struct UCB1;

impl<S, A, R> MultiArmedBandit<S, A, R> for UCB1
where
    A: Action,
{
    fn q_function(&self, state: &Node<S, A, R>, action: A) -> f64 {
        todo!()
    }

    fn select(&self, node: Node<S, A, R>, actions: Vec<A>) -> A {
        if let Some(untried) = actions
            .iter()
            .find(|&action| !node.children.iter().any(|c| c.0.action == Some(*action)))
        {
            return *untried;
        }

        let mut max_actions: Vec<A> = Vec::with_capacity(actions.len());
        let mut max_value = f64::NEG_INFINITY;
        let total = node.children.iter().map(|x| x.1).sum::<usize>() as f64;

        let children: Vec<&Rc<(Node<S, A, R>, usize)>> = node.children.iter().collect();

        for child in children {
            let value = self.q_function(&node, child.0.action.unwrap())
                + f64::sqrt(2f64 * f64::log10(total)) / (child.1 as f64);

            if value > max_value {
                max_actions = vec![child.0.action.unwrap()];
                max_value = value;
            } else {
                max_actions.push(child.0.action.unwrap());
            }
        }

        // If there are multiple actions with the highest value, choose one randomly
        let index = genrand(0, max_actions.len());
        return max_actions[index];
    }
}

impl UCB1 {
    pub fn new() -> Self {
        Self
    }
}
