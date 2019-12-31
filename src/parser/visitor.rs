use super::super::lexer::error::CompileError;

use super::ast::*;

pub trait Visitor<'a>
{
    fn enter(&mut self, node: &'a dyn Node) -> Result<(), CompileError>;
    fn exit(&mut self, node: &'a dyn Node) -> Result<(), CompileError>;
}

/**
 * For any node, the events are called in this order:
 *  - before(current)
 *  - transform_stmt(child) / transform_expr(child) / before(child), ..., after(child)
 *  - after(current)
 */
pub trait Transformer
{
    fn before(&mut self, node: &dyn Node);
    fn after(&mut self, node: &dyn Node);
    fn transform_stmt(&mut self, node: Box<dyn StmtNode>) -> Box<dyn StmtNode>;
    fn transform_expr(&mut self, node: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>;
}

pub trait Visitable 
{
    fn iterate<'a>(&'a self, visitor: &mut dyn Visitor<'a>) -> Result<(), CompileError>;
}

pub trait Transformable 
{
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

impl<'a, F> Visitor<'a> for F
    where F: FnMut(&'a dyn Node) -> Result<(), CompileError>
{
    fn enter(&mut self, node: &'a dyn Node) -> Result<(), CompileError>
    {
        self(node)
    }

    fn exit(&mut self, _node: &'a dyn Node) -> Result<(), CompileError>
    {
        Ok(())
    }
}

impl<F> Transformer for F
    where F: FnMut(Box<dyn StmtNode>) -> Box<dyn StmtNode>
{
    fn before(&mut self, _node: &dyn Node)
    {

    }

    fn after(&mut self, _node: &dyn Node)
    {
        
    }

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

pub struct RecTransformer<F, G>
    where F: FnMut(Box<dyn StmtNode>) -> Box<dyn StmtNode>,
        G: FnMut(Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
{
    f: F,
    g: G
}

impl<F, G> RecTransformer<F, G>
    where F: FnMut(Box<dyn StmtNode>) -> Box<dyn StmtNode>,
        G: FnMut(Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
{
    pub fn new(f: F, g: G) -> RecTransformer<F, G>
    {
        RecTransformer {
            f, g
        }
    }
}

impl<F, G> Transformer for RecTransformer<F, G>
    where F: FnMut(Box<dyn StmtNode>) -> Box<dyn StmtNode>,
        G: FnMut(Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
{
    fn before(&mut self, _node: &dyn Node)
    {

    }

    fn after(&mut self, _node: &dyn Node)
    {
        
    }

    fn transform_expr(&mut self, expr: Box<dyn UnaryExprNode>) -> Box<dyn UnaryExprNode>
    {
        let mut result = (self.g)(expr);
        result.transform(self);
        return result;
    }

    fn transform_stmt(&mut self, stmt: Box<dyn StmtNode>) -> Box<dyn StmtNode>
    {
        let mut result = (self.f)(stmt);
        result.transform(self);
        return result;
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
    let mut ast = FunctionNode::parse(lex("
        fn test(a: int, b: int,): int {
            if a < b {
                return a;
            } 
            return b;
        }").expect_next(&Token::BOF).unwrap()).unwrap();
        
    ast.transform(&mut blockify);
    assert_eq!(FunctionNode::parse(lex("
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
        }").expect_next(&Token::BOF).unwrap()).unwrap(), ast);
}