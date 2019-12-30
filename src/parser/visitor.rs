use super::super::lexer::error::CompileError;

use super::ast::*;

pub trait Visitor<'a>
{
    fn enter(&mut self, node: &'a dyn Node) -> Result<(), CompileError>;
    fn exit(&mut self, node: &'a dyn Node) -> Result<(), CompileError>;
}

pub trait Transformer
{
    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>;
    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>;
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
    fn enter(&mut self, _node: &'a dyn Node) -> Result<(), CompileError>
    {
        Ok(())
    }

    fn exit(&mut self, _node: &'a dyn Node) -> Result<(), CompileError>
    {
        Ok(())
    }
}

pub struct ExprTransformer<F>
    where F: FnMut(Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
{
    f: F
}

impl<F> ExprTransformer<F>
    where F: FnMut(Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
{
    fn new(f: F) -> ExprTransformer<F>
    {
        ExprTransformer {
            f: f
        }
    }
}

impl<F> Transformer for ExprTransformer<F>
    where F: FnMut(Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
{
    fn transform_expr(&mut self, expr: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        (self.f)(expr)
    }

    fn transform_stmt(&mut self, mut stmt: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        stmt.transform(self);
        stmt
    }
}

impl<F> Transformer for F
    where F: FnMut(Box<dyn StmtNode>) -> Box<dyn StmtNode>
{
    fn transform_expr(&mut self, mut expr: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        expr.transform(self);
        expr
    }

    fn transform_stmt(&mut self, stmt: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        (self)(stmt)
    }
}

#[cfg(test)]
use super::prelude::*;
#[cfg(test)]
use super::Parse;
#[cfg(test)]
use super::super::lexer::lexer::lex;

#[cfg(test)]
fn blockify(mut stmt: Box<dyn StmtNode>) -> Box<dyn StmtNode>
{
    stmt.transform(&mut blockify);
    Box::new(BlockNode::new(
        TextPosition::create(0, 0),
        vec![ stmt ]
    ))
}

#[test]
fn test_transform() {
    let mut ast = FunctionNode::parse(&mut lex("
        fn test(a: int, b: int,): int {
            if a < b {
                return a;
            } 
            return b;
        }
    ")).unwrap();
    println!("{:?}", ast);
    ast.transform(&mut blockify);
    assert_eq!(FunctionNode::parse(&mut lex("
        fn test(a: int, b: int,): int {
            {
                if a < b {
                    {
                        return a;
                    }
                } 
            }
            {
                return b;
            }
        }")).unwrap(), ast);
}