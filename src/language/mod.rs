pub mod position;

use super::util::dynamic::DynEq;
use position::TextPosition;
use std::any::Any;

pub trait AstNodeFuncs: std::fmt::Debug + Any + DynEq {
    fn pos(&self) -> &TextPosition;
}

dynamic_trait!{ AstNode: AstNodeFuncs; AstNodeDynCastable }

pub mod compiler;
pub mod error;
pub mod identifier;
pub mod prelude;
pub mod program;
pub mod program_parallel_for;
pub mod types;
pub mod view;
pub mod gwaihir_writer;

#[cfg(test)]
#[macro_use] 
pub mod ast_test;