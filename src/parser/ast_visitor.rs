use super::ast::*;
use super::super::lexer::error::CompileError;

type VisitorReturnType = Result<(), CompileError>;

pub trait Visitable<T: ?Sized> {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a T) -> VisitorReturnType, T: 'a;
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlOr {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlAnd {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlCmp {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlAdd {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlMult {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitable<dyn UnaryExprNode> for ExprNodeLvlIndex {
    fn iterate<'a, F>(&'a self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&'a dyn UnaryExprNode) -> VisitorReturnType 
    {
        f(&*self.head)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}
