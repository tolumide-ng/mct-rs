use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

pub trait Action: Debug + Eq + PartialEq + Clone + Copy + Display + Hash {
    // type Reward;
}
