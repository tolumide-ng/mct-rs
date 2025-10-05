/// Different strategies for selecting the final action after MCTS
pub enum Strategy {
    MostVisited,
    HighestQValue,
    Probabilistic,
    HeuristicWin, // terminal/winning move aware
}
