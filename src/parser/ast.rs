use super::super::lexer::tokens::{Identifier, Literal};
use super::super::lexer::position::TextPosition;

type Annotation = TextPosition;

#[derive(Debug)]
pub enum Function {
	Function(Annotation, Box<Identifier>, Vec<ParameterDeclaration>, Box<TypeDecl>, Box<Stmts>)
}

#[derive(Debug)]
pub enum ParameterDeclaration {
	ParameterDeclaration(Annotation, Box<Identifier>, Box<TypeDecl>)
}

#[derive(Debug)]
pub enum Stmts {
	Stmts(Annotation, Vec<Stmt>)
}

#[derive(Debug)]
pub enum Stmt {
	Declaration(Annotation, Box<TypeDecl>, Box<Identifier>, Option<Box<Expr>>),
	Assignment(Annotation, Box<Expr>, Box<Expr>),
	Expr(Annotation, Box<Expr>),
	If(Annotation, Box<Expr>, Box<Stmts>),
	While(Annotation, Box<Expr>, Box<Stmts>),
	Block(Annotation, Box<Stmts>),
	Return(Annotation, Box<Expr>)
}

#[derive(Debug)]
pub enum TypeDecl {
	Arr(Annotation, Box<BaseType>, u8),
	Void(Annotation)
}

pub type Expr = ExprLvlOr;

#[derive(Debug)]
pub enum ExprLvlOr {
	Or(Annotation, Box<ExprLvlAnd>, Vec<OrPart>)
}

#[derive(Debug)]
pub enum OrPart {
	Expr(Annotation, Box<ExprLvlAnd>)
}

#[derive(Debug)]
pub enum ExprLvlAnd {
	And(Annotation, Box<ExprLvlCmp>, Vec<AndPart>)
}

#[derive(Debug)]
pub enum AndPart {
	Expr(Annotation, Box<ExprLvlCmp>)
}

#[derive(Debug)]
pub enum ExprLvlCmp {
	Cmp(Annotation, Box<ExprLvlAdd>, Vec<CmpPart>)
}

#[derive(Debug)]
pub enum CmpPart {
	Eq(Annotation, Box<ExprLvlAdd>),
	Neq(Annotation, Box<ExprLvlAdd>),
	Leq(Annotation, Box<ExprLvlAdd>),
	Geq(Annotation, Box<ExprLvlAdd>),
	Ls(Annotation, Box<ExprLvlAdd>),
	Gt(Annotation, Box<ExprLvlAdd>)
}

#[derive(Debug)]
pub enum ExprLvlAdd {
	Add(Annotation, Box<ExprLvlMult>, Vec<AddPart>)
}

#[derive(Debug)]
pub enum AddPart {
	Add(Annotation, Box<ExprLvlMult>),
	Subtract(Annotation, Box<ExprLvlMult>)
}

#[derive(Debug)]
pub enum ExprLvlMult {
	Mult(Annotation, Box<ExprLvlIndex>, Vec<MultPart>)
}

#[derive(Debug)]
pub enum MultPart {
	Mult(Annotation, Box<ExprLvlIndex>),
	Divide(Annotation, Box<ExprLvlIndex>)
}

#[derive(Debug)]
pub enum ExprLvlIndex {
	Index(Annotation, Box<UnaryExpr>, Vec<IndexPart>)
}

#[derive(Debug)]
pub enum IndexPart {
	Expr(Annotation, Box<Expr>)
}

#[derive(Debug)]
pub enum UnaryExpr {
	Brackets(Annotation, Box<Expr>),
	Literal(Annotation, Box<Literal>),
	Variable(Annotation, Box<Identifier>),
	Call(Annotation, Box<Identifier>, Vec<Expr>),
	New(Annotation, Box<BaseType>, Vec<IndexPart>)
}

#[derive(Debug)]
pub enum BaseType {
	Int(Annotation)
}

pub trait Node {
	fn dynamic_cast_function(&self) -> Option<&Function> {
		None
	}

	fn dynamic_cast_stmts(&self) -> Option<&Stmts> {
		None
	}

	fn dynamic_cast_stmt(&self) -> Option<&Stmt> {
		None
	}

	fn dynamic_cast_type(&self) -> Option<&TypeDecl> {
		None
	}

	fn dynamic_cast_expr(&self) -> Option<&Expr> {
		None
	}

	fn dynamic_cast_un_expr(&self) -> Option<&UnaryExpr> {
		None
	}
}

macro_rules! derive_node {
	($name:ident, $dynamic_cast_func_name:ident) => {
		impl Node for $name {
			fn $dynamic_cast_func_name (&self) -> Option<& $name> {
				Some(&self)
			}
		}
	};
	($name:ident) => {
		impl Node for $name {}
	}
}

derive_node!(Function, dynamic_cast_function);
derive_node!(ParameterDeclaration);
derive_node!(Stmts, dynamic_cast_stmts);
derive_node!(Stmt, dynamic_cast_stmt);
derive_node!(TypeDecl, dynamic_cast_type);
derive_node!(ExprLvlOr, dynamic_cast_expr);
derive_node!(OrPart);
derive_node!(ExprLvlAnd);
derive_node!(AndPart);
derive_node!(ExprLvlCmp);
derive_node!(CmpPart);
derive_node!(ExprLvlAdd);
derive_node!(AddPart);
derive_node!(ExprLvlMult);
derive_node!(MultPart);
derive_node!(ExprLvlIndex);
derive_node!(IndexPart);
derive_node!(UnaryExpr, dynamic_cast_un_expr);
derive_node!(BaseType);