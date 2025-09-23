use crate::node::Node;

pub trait MultiArmedBandit<S, A, R> {
    fn q_function(&self, state: &Node<S, A, R>, action: A) -> f64;

    fn select(&self, state: Node<S, A, R>, actions: Vec<A>) -> A;
}
