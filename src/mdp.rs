/// currently rethinking MDP to be implemented by State, i.e. making MDP itself state
/// Markov Decision Processes
pub trait MDP<S, A, R> {
    /// Returns all states of this MDP
    fn get_states(&self) -> Vec<S>;

    /// Returns all actions with non-zero probability from this state
    fn get_actions(&self, state: &S) -> Vec<A>;

    /// Returns all non-zero probability transitions for this action from state,
    /// as a list of (state, probability) pairs
    fn get_transitions(&self, state: &S, action: A) -> Vec<(S, f64)>;

    /// Returns the reward for transitioning from state to nextState via action
    fn get_reward(&self, state: &S, action: A, next_state: S) -> R;

    /// Returns true if and only if state is a terminal state of this MDP
    fn is_terminal(&self, state: &S) -> bool;

    /// Returns the discount dactor for this MDP
    fn get_discount_factor(&self) -> f64;

    /// Returns the initial state of this MDP
    fn get_initial_state(&self) -> S;

    /// Returns all goal states of this MDP
    fn get_goal_states(&self) -> Vec<S>;
}
