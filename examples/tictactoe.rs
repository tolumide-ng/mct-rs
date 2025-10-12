use mct_rs::{action::Action, mcts::MCTS, mdp::MDP, policy::RandomRollout, strategy::Strategy};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
enum Player {
    #[default]
    O,
    X,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TicTacToeState {
    board: [[Option<Player>; 3]; 3],
    current: Player,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TicTacToeAction {
    Pos(usize, usize),
}

impl Action for TicTacToeAction {}

#[derive(Debug, Default)]
pub(crate) struct TicTacToeMDP {
    player: Player,
    state: TicTacToeState,
}

impl TicTacToeMDP {
    fn get_winner(&self, state: &TicTacToeState) -> Option<Player> {
        let lines = [
            // rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            // columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            // diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ];

        for line in lines.iter() {
            let [a, b, c] = line;
            if let (Some(p1), Some(p2), Some(p3)) = (
                state.board[a.0][a.1],
                state.board[b.0][b.1],
                state.board[c.0][c.1],
            ) {
                if p1 == p2 && p2 == p3 {
                    return Some(p1);
                }
            }
        }
        None
    }
}

impl MDP<TicTacToeState, TicTacToeAction> for TicTacToeMDP {
    fn get_initial_state(&self) -> TicTacToeState {
        TicTacToeState::default()
    }

    fn get_actions(&self, state: &TicTacToeState) -> Vec<TicTacToeAction> {
        let mut actions = vec![];
        for i in 0..3 {
            for j in 0..3 {
                if state.board[i][j].is_none() {
                    actions.push(TicTacToeAction::Pos(i, j));
                }
            }
        }

        actions
    }

    fn execute(
        &self,
        state: &TicTacToeState,
        action: &TicTacToeAction,
    ) -> (TicTacToeState, f64, bool) {
        let mut new_state = state.clone();

        // let TicTacToeAction::Pos(i, j) = action {}
        let (i, j) = match action {
            TicTacToeAction::Pos(i, j) => (*i, *j),
        };

        new_state.board[i][j] = Some(state.current);
        new_state.current = match state.current {
            Player::X => Player::O,
            Player::O => Player::X,
        };

        let terminal = self.is_terminal(&new_state);
        let reward = match terminal {
            true => {
                if self.get_winner(&new_state) == Some(state.current) {
                    1.0
                } else {
                    0.0
                }
            }
            false => 0.0,
        };

        (new_state, reward, terminal)
    }

    fn is_terminal(&self, state: &TicTacToeState) -> bool {
        self.get_winner(state).is_some()
            || state
                .board
                .iter()
                .all(|row| row.iter().all(|c| c.is_some()))
    }

    fn get_discount_factor(&self) -> f64 {
        1.0
    }

    fn get_goal_states(&self) -> Vec<TicTacToeState> {
        vec![]
    }

    fn get_reward(
        &self,
        _state: &TicTacToeState,
        _action: &TicTacToeAction,
        next_state: &TicTacToeState,
    ) -> f64 {
        match self.get_winner(next_state) {
            Some(p) if p == self.player => 1.0, // current player wins
            Some(_) => -1.0,                    // current player loses
            None => 0.0,                        // draw or ongoing
        }
    }

    fn get_states(&self) -> Vec<TicTacToeState> {
        vec![self.get_initial_state()]
    }

    fn get_transitions(
        &self,
        state: &TicTacToeState,
        action: &TicTacToeAction,
    ) -> Vec<(TicTacToeState, f64)> {
        // Tic-Tac-Toe is deterministic: only one outcome per action
        let (next_state, _, _) = self.execute(state, action);
        vec![(next_state, 1.0)]
    }
}

fn main() {
    let mdp = TicTacToeMDP::default();
    let mut mcts = MCTS::new(mdp, RandomRollout::new());

    // The player from whom's perspective we want to run MCTS

    // Run MCTS for 100ms
    mcts.mcts(100);

    // this would return the best possible move from the perspective of mdp.player
    // if you need to check the best move for a different player X, change player to X in mdp, and test from it's perspective.
    // Pick best child from root (most visits)
    let best_child = mcts
        .best_action(Strategy::Probabilistic)
        .expect("No children found");
    println!("Best action: {:?}", best_child);
}
