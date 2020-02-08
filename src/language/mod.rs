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
pub mod program;
pub mod prelude;

// #[cfg(test)]
// pub mod test_rename;

// pub mod obj_type;
// pub mod scope;
// pub mod inline;
