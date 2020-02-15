pub mod position;

use std::any::Any;
use position::TextPosition;
use super::util::dynamic::{ DynEq, Dynamic };

pub trait AstNode: std::fmt::Debug + Any + DynEq + Dynamic
{
	fn pos(&self) -> &TextPosition;
}

pub mod error;
pub mod identifier;
pub mod print;
pub mod program;
pub mod prelude;
pub mod nazgul_printer;
#[macro_use]
pub mod test;