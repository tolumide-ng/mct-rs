use std::fmt::{Debug, Display};

pub trait Action: Debug + Eq + PartialEq + Clone + Copy + Display {}
