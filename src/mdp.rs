use crate::rand::genrand;

/// currently rethinking MDP to be implemented by State, i.e. making MDP itself state
/// Markov Decision Processes
pub trait MDP<S, A> {
    /// Returns all states of this MDP
    fn get_states(&self) -> Vec<S>;

    /// Returns all actions with non-zero probability from this state
    fn get_actions(&self, state: &S) -> Vec<A>;

    /// Returns all non-zero probability transitions for this action from state,
    /// as a list of (state, probability) pairs
    fn get_transitions(&self, state: &S, action: &A) -> Vec<(S, f64)>;

    /// Returns the reward for transitioning from state to nextState via action
    fn get_reward(&self, state: &S, action: &A, next_state: &S) -> f64;

    /// Returns true if and only if state is a terminal state of this MDP
    fn is_terminal(&self, state: &S) -> bool;

    /// Returns the discount dactor for this MDP
    fn get_discount_factor(&self) -> f64;

    /// Returns the initial state of this MDP
    fn get_initial_state(&self) -> S;

    /// Returns all goal states of this MDP
    fn get_goal_states(&self) -> Vec<S>;

    /// Returns the new state after the application of the provided action on it, and the reward/outcome of such move(application)
    fn execute(&self, state: &S, action: &A) -> (S, f64, bool) {
        let mut transitions = self.get_transitions(state, &action);
        assert!(!transitions.is_empty(), "No transitions for this action");

        // Sample from probabilities
        let r = (genrand(0, 1000) as f64) / 1000.0; // uniform (0, 1)
        let mut cumulative = 0.0;
        // let mut chosen_state = transitions[0].0;

        let chosen_index = transitions
            .iter()
            .position(|(_, p)| {
                cumulative += p;
                return cumulative >= r;
            })
            .unwrap_or(0);

        let (chosen_state, _) = transitions.swap_remove(chosen_index);

        let reward = self.get_reward(state, &action, &chosen_state);
        let done = self.is_terminal(&chosen_state);

        return (chosen_state, reward, done);
    }
}
