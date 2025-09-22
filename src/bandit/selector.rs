pub trait Selector<S, A> {
    fn get(&self, state: S, action: A);
}


pub trait MultiArmedBandit<S, A, Q>
where
    Q: Selector<S, A>,
{
    fn select(&self, state: S, actions: Vec<A>, q_function: Q);
}
