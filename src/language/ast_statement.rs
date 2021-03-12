use super::position::TextPosition;
use super::error::CompileError;
use super::identifier::{Identifier, Name};
use super::ast::*;
use super::ast_expr::*;

pub trait StatementFuncs: AstNode {
    ///
    /// Iterates over all blocks contained directly in this statement.
    /// 
    /// # Details
    /// This will not yield nested blocks. If this statement is a block itself, it will only return itself.
    /// The order in which the blocks are returned is unspecified.
    /// 
    /// # Example
    /// ```
    /// let a = Block::parse(&mut fragment_lex("{{{}}{}}"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.subblocks().count(), 1); // returns only the top level block
    /// assert_eq!(a.subblocks().flat_map(|b| b.statements.iter()).flat_map(|s| s.subblocks()).count(), 2); // returns the proper subblocks, also non-nested
    /// ```
    /// 
    fn subblocks<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)>;
    ///
    /// See subblocks()
    /// 
    fn subblocks_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)>;
    ///
    /// Iterates over all expressions contained directly in this statement.
    /// 
    /// # Details
    /// This will not yield nested expressions. If this statement is only one expression, only the statement itself will be returned.
    /// The expressions will be returned in the order of execution. If this statement contains other statements that in turn contain
    /// expressions, these expressions will also not be returned.
    /// 
    /// # Example
    /// ```
    /// let a = Statement::parse(&mut fragment_lex("a = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.expressions().count(), 2); // returns only the top level expressions 'a' and 'b(c, 0,)', in this order
    /// ```
    /// 
    fn expressions<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)>;
    ///
    /// See expressions()
    /// 
    fn expressions_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)>;

    ///
    /// Iterates over all names contained in this statement.
    /// 
    /// # Details
    /// This will return all names (i.e. user-defined identifiers) in this statement, which includes uses of variables/functions/... and
    /// also declarations. Since names cannot be nested, all names are returned (as opposed to subblocks() and expressions()). The order
    /// in which the names are returned is unspecified.
    ///
    /// # Example
    /// ```
    /// let a = Statement::parse(&mut fragment_lex("let a: int = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.names().map(|n| n.name.as_str()).collect::<Vec<_>>(), vec!["a", "c", "b"]);
    /// ```
    /// 
    fn names<'a>(&'a self) -> Box<(dyn Iterator<Item = &'a Name> + 'a)> {
        Box::new(self.expressions().flat_map(|e| e.names()))
    }

    ///
    /// Iterates over all names contained in this statement.
    /// 
    /// # Details
    /// This will return all names (i.e. user-defined identifiers) in this statement, which includes uses of variables/functions/... and
    /// also declarations. Since names cannot be nested, all names are returned (as opposed to subblocks() and expressions()). The order
    /// in which the names are returned is unspecified.
    ///
    /// # Example
    /// ```
    /// let a = Statement::parse(&mut fragment_lex("let a: int = b(c, 0,);"), &mut TypeVec::new()).unwrap();
    /// assert_eq!(a.names().map(|n| n.name.as_str()).collect::<Vec<_>>(), vec!["a", "c", "b"]);
    /// ```
    /// 
    fn names_mut<'a>(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Name> + 'a)> {
        Box::new(self.expressions_mut().flat_map(|e| e.names_mut()))
    }

}

dynamic_subtrait!{ Statement: StatementFuncs; StatementDynCastable }

pub struct Block {
    
}