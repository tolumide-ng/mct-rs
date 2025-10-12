use core::f64;

use crate::action::Action;
use crate::node::Node;
use crate::rand::genrand;

/// Given that this node is fully expanded i.e all the direct children of this node have been explored
/// This method helps us calculate the best child of this node to exploit further
/// Selects an action for the state from a list given a Q-function(???) (https://gibberblot.github.io/rl-notes/single-agent/multi-armed-bandits.html#id5)
/// this can be: Softmax strategy, UCB1 e.t.c
#[derive(Debug, Default)]
pub struct UCB1;

impl UCB1 {
    const C: f64 = 1.4142135623730951;

    pub(crate) fn select<S, A>(&self, node: &Node<S, A>, actions: Vec<A>) -> A
    where
        A: Action,
        S: PartialEq + Eq,
    {
        let children = node.children.borrow();
        let child_actions = children.iter().flat_map(|c| c.action).collect::<Vec<_>>();

        for action in actions.iter() {
            if !child_actions.contains(action) {
                return *action;
            }
        }

        let mut max_actions = Vec::new();
        let mut max_value = f64::NEG_INFINITY;

        for child in children.iter() {
            let value = child.ucb1(Self::C);

            if value > max_value {
                max_actions = vec![child.action.unwrap()];
                max_value = value;
            } else if value == max_value {
                max_actions.push(child.action.unwrap());
            }
        }

        //  if there are multiple actions with the highest value choose one randomly
        let index = genrand(0, max_actions.len());
        return max_actions[index];
    }
}
