use super::ast::{Func, Stmts, Stmt, Type, Expr, Summand, ExprMult, ProductPart, ExprUn, BasicExprUn, ParamDeclaration, BaseType};
use super::tokens::{Identifier, Literal, Keyword, Token, Stream};

use std::vec::Vec;

impl Parse for BaseType {
	fn guess_match(stream: &Stream) -> bool {
		true
	}
	fn parse(stream: &mut Stream) -> Self {
		BaseType::Int
	}
}