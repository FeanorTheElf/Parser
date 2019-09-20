use super::ast::*;
use super::super::lexer::error::CompileError;

type VisitorReturnType = Result<(), CompileError>;

pub trait Visitor<T: ?Sized> {
    fn iterate<F>(&self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&T) -> VisitorReturnType;
}

impl Visitor<dyn UnaryExprNode> for ExprNodeLvlOr {
    fn iterate<F>(&self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitor<dyn UnaryExprNode> for ExprNodeLvlAnd {
    fn iterate<F>(&self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitor<dyn UnaryExprNode> for ExprNodeLvlCmp {
    fn iterate<F>(&self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitor<dyn UnaryExprNode> for ExprNodeLvlAdd {
    fn iterate<F>(&self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitor<dyn UnaryExprNode> for ExprNodeLvlMult {
    fn iterate<F>(&self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&dyn UnaryExprNode) -> VisitorReturnType 
    {
        self.head.iterate(f)?;
        for part in &self.tail {
            part.get_expr().iterate(f)?;
        }
        return Ok(());
    }
}

impl Visitor<dyn UnaryExprNode> for ExprNodeLvlIndex {
    fn iterate<F>(&self, f: &mut F) -> VisitorReturnType
        where F: FnMut(&dyn UnaryExprNode) -> VisitorReturnType 
    {
        f(&*self.head)?;
        for part in &self.tail {
            part.expr.iterate(f)?;
        }
        return Ok(());
    }
}
