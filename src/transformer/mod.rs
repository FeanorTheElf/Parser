use super::parser::prelude::*;
use std::cell::RefCell;

pub type Program = Vec<RefCell<FunctionNode>>;

pub trait Transformer<A>: for<'a> SpecificLifetimeTransformer<'a, A> {}

pub trait SpecificLifetimeTransformer<'a, A> {
    type Prepared: PreparedTransformer + 'a;

    fn prepare(self, program: &'a Program, data: A) -> Result<Self::Prepared, CompileError>;
}

impl<T, A> Transformer<A> for T
    where T: for<'a> SpecificLifetimeTransformer<'a, A> {}

pub trait DefaultCallableTransformer: for<'a> DefaultCallableSpecificLifetimeTransformer<'a> {}

pub trait DefaultCallableSpecificLifetimeTransformer<'a> {
    type Prepared: PreparedTransformer + 'a;
    type ExtraData;
    type Classical: SpecificLifetimeTransformer<'a, Self::ExtraData, Prepared = Self::Prepared>;

    fn as_classical(self) -> Self::Classical;
    fn prepare_default(self, program: &'a Program) -> Result<Self::Prepared, CompileError>;
}

impl<T> DefaultCallableTransformer for T
    where T: for<'a> DefaultCallableSpecificLifetimeTransformer<'a> {}

pub trait PreparedTransformer {
    fn transform(self, program: &Program) -> Result<(), CompileError>;
}

pub trait ChainablePreparedTransformer<T>: PreparedTransformer
{
    fn transform_chained(self, program: &Program, next: T) -> Result<(), CompileError>;
}

impl<T, M> ChainablePreparedTransformer<T> for M
    where T: DefaultCallableTransformer, M: PreparedTransformer
{
    default fn transform_chained(self, program: &Program, next: T) -> Result<(), CompileError> {
        self.transform(program)?;
        next.prepare_default(program)?.transform(program);
        return Ok(());
    }
}

impl<'a, T> DefaultCallableSpecificLifetimeTransformer<'a> for T
    where T: SpecificLifetimeTransformer<'a, ()>
{
    type Prepared = T::Prepared;
    type ExtraData = ();
    type Classical = T;

    fn as_classical(self) -> Self::Classical {
        self
    }

    fn prepare_default(self, program: &'a Program) -> Result<Self::Prepared, CompileError> {
        self.as_classical().prepare(program, ())
    }
}

impl<'a> SpecificLifetimeTransformer<'a, ()> for ! {
    type Prepared = !;

    fn prepare(self, program: &'a Vec<RefCell<FunctionNode>>, data: ()) -> Result<Self::Prepared, CompileError> {
        Ok(self)
    }
}

impl PreparedTransformer for ! {
    fn transform(self, program: &Program) -> Result<(), CompileError> {
        Ok(())
    }
}

#[macro_use]
pub mod transformer_list;