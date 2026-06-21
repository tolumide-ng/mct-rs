/// Different strategies for selecting the final action after MCTS
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
pub enum Strategy {
    MostVisited,
    HighestQValue,
    Probabilistic,
    HeuristicWin, // terminal/winning move aware
}
