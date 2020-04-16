pub mod position;

use super::util::dynamic::{DynEq, Dynamic};
use position::TextPosition;
use std::any::Any;

pub trait AstNode: std::fmt::Debug + Any + DynEq + Dynamic {
    fn pos(&self) -> &TextPosition;
}

pub mod debug_printer;
pub mod error;
pub mod identifier;
pub mod prelude;
pub mod print;
pub mod program;
pub mod program_parallel_for;
#[macro_use]
pub mod test;
