use std::fmt::Debug;

pub trait Action: Debug + Eq + PartialEq + Clone + Copy {}
