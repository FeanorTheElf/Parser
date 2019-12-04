use super::super::lexer::tokens::{Identifier, Literal};
use super::super::lexer::position::TextPosition;
use super::super::lexer::error::CompileError;

use super::ast::*;

pub trait Visitor<'a>
{
    fn enter(&mut self, node: &'a dyn Node) -> Result<(), CompileError>;
    fn exit(&mut self, node: &'a dyn Node) -> Result<(), CompileError>;
}

pub trait Transformer
{
    fn transform(&mut self, node: Box<dyn Node>) -> Box<dyn Node>;
}

pub trait Visitable 
{
    fn iterate<'a>(&'a self, visitor: &mut dyn Visitor<'a>) -> Result<(), CompileError>;
}

pub trait Transformable 
{
	// If the closure does not return a object (i.e. if it panics), there is
	// an unrecoverable state, and the program will terminate
    fn transform(&mut self, f: &mut dyn Transformer);
}

pub struct TypeVisitor<'a, T, F, G, D>
    where F: FnMut(&'a T) -> Result<(), CompileError>, 
        G: FnMut(&'a T) -> Result<(), CompileError>, 
        D: Visitor<'a>,
        T: Node
{
    enter: F,
    exit: G,
    delegate: D,
    phantom: std::marker::PhantomData<&'a T>
}

impl<'a, T, F, G, D> TypeVisitor<'a, T, F, G, D>
    where F: FnMut(&'a T) -> Result<(), CompileError>, 
        G: FnMut(&'a T) -> Result<(), CompileError>, 
        D: Visitor<'a>,
        T: Node
{
    fn new(enter: F, exit: G, delegate: D) -> TypeVisitor<'a, T, F, G, D>
    {
        TypeVisitor {
            enter: enter,
            exit: exit,
            delegate: delegate,
            phantom: std::marker::PhantomData
        }
    }

    fn terminating(enter: F, exit: G) -> TypeVisitor<'a, T, F, G, DoNothingVisitor>
    {
        TypeVisitor {
            enter: enter,
            exit: exit,
            delegate: DO_NOTHING,
            phantom: std::marker::PhantomData
        }
    }
}

impl<'a, T, F, G, D> Visitor<'a> for TypeVisitor<'a, T, F, G, D>
    where F: FnMut(&'a T) -> Result<(), CompileError>, 
        G: FnMut(&'a T) -> Result<(), CompileError>, 
        D: Visitor<'a>,
        T: Node
{
    fn enter(&mut self, node: &'a dyn Node) -> Result<(), CompileError>
    {
        if let Some(value) = node.dynamic().downcast_ref::<T>() {
            self.enter(value)?;
        }
        self.delegate.enter(node)
    }

    fn exit(&mut self, node: &'a dyn Node) -> Result<(), CompileError>
    {
        if let Some(value) = node.dynamic().downcast_ref::<T>() {
            self.exit(value)?;
        }
        self.delegate.exit(node)
    }
}

pub struct DoNothingVisitor {}

pub const DO_NOTHING: DoNothingVisitor = DoNothingVisitor {};

impl<'a> Visitor<'a> for DoNothingVisitor
{
    fn enter(&mut self, node: &'a dyn Node) -> Result<(), CompileError>
    {
        Ok(())
    }

    fn exit(&mut self, node: &'a dyn Node) -> Result<(), CompileError>
    {
        Ok(())
    }
}

pub struct TypeTransformer<T, F, D>
    where F: FnMut(Box<T>) -> Box<T>, 
        D: for<'a> Visitor<'a>,
        T: Node
{
    transform: F,
    visitor: D,
    phantom: std::marker::PhantomData<T>
}

impl<T, F, D> TypeTransformer<T, F, D>
    where F: FnMut(Box<T>) -> Box<T>,
        D: for<'a> Visitor<'a>,
        T: Node
{
    // Creates a new transformer that transforms objects of type T
    // using the given function. The given visitor is given the
    // state before the transformation and after the transformation
    fn new(transform: F, visitor: D) -> TypeTransformer<T, F, D>
    {
        TypeTransformer {
            transform: transform,
            visitor: visitor,
            phantom: std::marker::PhantomData
        }
    }

    fn terminating(transform: F) -> TypeTransformer<T, F, DoNothingVisitor>
    {
        TypeTransformer {
            transform: transform,
            visitor: DO_NOTHING,
            phantom: std::marker::PhantomData
        }
    }
}

impl<T, F, D> Transformer for TypeTransformer<T, F, D>
    where F: FnMut(Box<T>) -> Box<T>,
        D: for<'a> Visitor<'a>,
        T: Node
{
    fn transform(&mut self, node: Box<dyn Node>) -> Box<dyn Node>
    {
        match cast::<dyn Node, T>(node) {
            Ok(value) => {
                self.visitor.enter(&*value);
                let result = self.transform(value);
                self.visitor.exit(&*result);
                return result;
            },
            Err(mut value) => {
                self.visitor.enter(&*value);
                value.transform(self);
                self.visitor.exit(&*value);
                return value;
            }
        }
    }
}
