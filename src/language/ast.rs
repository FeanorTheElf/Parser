use super::super::util::dynamic::{DynEq, Dynamic};
use super::position::TextPosition;
use super::identifier::Name;
use super::error::CompileError;
use std::any::Any;

pub trait AstNodeFuncs: std::fmt::Debug + Any + DynEq {
    fn pos(&self) -> &TextPosition;
}

dynamic_trait!{ AstNode: AstNodeFuncs; AstNodeDynCastable }

pub enum TraversePreorderCancel {
    RealError(CompileError), DoNotRecurse
}

impl From<CompileError> for TraversePreorderCancel {
    fn from(e: CompileError) -> TraversePreorderCancel {
        TraversePreorderCancel::RealError(e)
    }
}

pub type TraversePreorderResult = Result<(), TraversePreorderCancel>;

pub fn TraverseErr(e: CompileError) -> TraversePreorderResult {
    Err(TraversePreorderCancel::RealError(e))
}

pub const DONT_RECURSE: TraversePreorderResult = Err(TraversePreorderCancel::DoNotRecurse);