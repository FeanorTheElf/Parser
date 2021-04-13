use super::super::util::dynamic::DynEq;
use super::position::TextPosition;
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

pub trait IgnoreTraversePreorderResultCancel {

    fn ignore_cancel(self) -> Result<(), CompileError>;
}

impl IgnoreTraversePreorderResultCancel for TraversePreorderResult {

    fn ignore_cancel(self) -> Result<(), CompileError> {
        match self {
            Ok(()) | Err(TraversePreorderCancel::DoNotRecurse) => Ok(()),
            Err(TraversePreorderCancel::RealError(e)) => Err(e)
        }
    }
}

#[allow(non_snake_case)]
pub fn TraverseErr(e: CompileError) -> TraversePreorderResult {
    Err(TraversePreorderCancel::RealError(e))
}

pub const DONT_RECURSE: TraversePreorderResult = Err(TraversePreorderCancel::DoNotRecurse);
pub const RECURSE: TraversePreorderResult = Ok(());