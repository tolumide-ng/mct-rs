use crate::{action::Action, mdp::MDP, rand::genrand};

pub trait RolloutPolicy<M, S, A> {
    fn pick(&self, state: &S, actions: &Vec<A>) -> A;
}

pub struct RandomRollout;

impl RandomRollout {
    pub fn new() -> Self {
        Self
    }
}

impl<M, S, A> RolloutPolicy<M, S, A> for RandomRollout
where
    M: MDP<S, A>,
    A: Action,
{
    fn pick(&self, _state: &S, actions: &Vec<A>) -> A {
        if actions.len() == 1 {
            return actions[0];
        }

        let index = genrand(0, actions.len());
        return actions[index];
    }
}
