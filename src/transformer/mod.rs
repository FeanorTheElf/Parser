use super::parser::prelude::*;
use std::cell::RefCell;

pub type Program = Vec<RefCell<FunctionNode>>;

pub trait Transformer<T>: for<'a> SpecificLifetimeTransformer<'a, T> {}

pub trait SpecificLifetimeTransformer<'a, T> {
    type Prepared: PreparedTransformer + 'a;

    fn prepare(self, program: &'a Vec<RefCell<FunctionNode>>, data: T) -> Self::Prepared;
}

pub trait PreparedTransformer {
    fn transform(self, program: &Program);
}