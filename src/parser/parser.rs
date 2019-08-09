use super::ast::*;
use super::super::lexer::tokens::*;
#[macro_use]
use super::parser_gen::{ Parse, Flatten };

use std::vec::Vec;

impl_parse!{ Function -> Function(Token#Fn identifier Token#BracketOpen {ParameterDeclaration} Token#BracketClose Token#Colon Type Token#CurlyBracketOpen Stmts Token#CurlyBracketClose) }

impl_parse!{ ParameterDeclaration -> ParameterDeclaration(identifier Token#Colon Type Token#Comma) }

impl_parse!{ Stmts -> Stmts({Stmt}) }

impl Parse for Stmt {
	fn guess_can_parse(stream: &Stream) -> bool {
		Expr::guess_can_parse(stream) || stream.ends(&Token::If) || stream.ends(&Token::While) || stream.ends(&Token::CurlyBracketOpen) || stream.ends(&Token::Return) || stream.ends(&Token::Let)
	}

	fn parse(stream: &mut Stream) -> Self {
		let pos = stream.pos();
		if stream.ends(&Token::If) {
			stream.expect_next(&Token::If);
			let condition = Expr::parse(stream);
			stream.expect_next(&Token::CurlyBracketOpen);
			let stmts = Stmts::parse(stream);
			stream.expect_next(&Token::CurlyBracketClose);
			return Stmt::If(pos, Box::new(condition), Box::new(stmts));
		} else if stream.ends(&Token::While) {
			stream.expect_next(&Token::While);
			let condition = Expr::parse(stream);
			stream.expect_next(&Token::CurlyBracketOpen);
			let stmts = Stmts::parse(stream);
			stream.expect_next(&Token::CurlyBracketClose);
			return Stmt::While(pos, Box::new(condition), Box::new(stmts));
		} else if stream.ends(&Token::CurlyBracketOpen) {
			stream.expect_next(&Token::CurlyBracketOpen);
			let stmts = Stmts::parse(stream);
			stream.expect_next(&Token::CurlyBracketClose);
			return Stmt::Block(pos, Box::new(stmts));
		} else if stream.ends(&Token::Return) {
			stream.expect_next(&Token::Return);
			let expr = Expr::parse(stream);
			stream.expect_next(&Token::Semicolon);
			return Stmt::Return(pos, Box::new(expr));
		} else if stream.ends(&Token::Let) {
			stream.expect_next(&Token::Let);
			let name = stream.next().as_ident();
			stream.expect_next(&Token::Colon);
			let var_type = Type::parse(stream);
			if stream.ends(&Token::Assign) {
				stream.expect_next(&Token::Assign);
				let value = Expr::parse(stream);
				stream.expect_next(&Token::Semicolon);
				return Stmt::Declaration(pos, Box::new(var_type), Box::new(name), Some(Box::new(value)));
			} else {
				stream.expect_next(&Token::Semicolon);
				return Stmt::Declaration(pos, Box::new(var_type), Box::new(name), None);
			}
		} else if Expr::guess_can_parse(stream) {
			let expr = Expr::parse(stream);
			if stream.ends(&Token::Assign) {
				stream.expect_next(&Token::Assign);
				let new_val = Expr::parse(stream);
				stream.expect_next(&Token::Semicolon);
				return Stmt::Assignment(pos, Box::new(expr), Box::new(new_val));
			} else {
				stream.expect_next(&Token::Semicolon);
				return Stmt::Expr(pos, Box::new(expr));
			}
		} else {
			panic!("Expected statement, got {:?}", stream);
		}
	}
}

impl Parse for Type {
	fn guess_can_parse(stream: &Stream) -> bool {
		BaseType::guess_can_parse(stream) || stream.ends(&Token::Void)
	}

	fn parse(stream: &mut Stream) -> Self {
		let pos = stream.pos();
		if stream.ends(&Token::Void) {
			return Type::Void(pos);
		} else if BaseType::guess_can_parse(stream) {
			let base_type = BaseType::parse(stream);
			let mut dimensions: u8 = 0;
			while stream.ends(&Token::SquareBracketOpen) {
				stream.expect_next(&Token::SquareBracketOpen);
				stream.expect_next(&Token::SquareBracketClose);
			}
			return Type::Arr(pos, Box::new(base_type), dimensions);
		} else {
			panic!("Expected type, got {:?}", stream);
		}
	}
}

impl_parse!{ ExprLvlOr -> Or(ExprLvlAnd {OrPart}) }
impl_parse!{ OrPart -> Expr(Token#OpOr ExprLvlAnd) }

impl_parse!{ ExprLvlAnd -> And(ExprLvlCmp {AndPart}) }
impl_parse!{ AndPart -> Expr(Token#OpAnd ExprLvlCmp) }

impl_parse!{ ExprLvlCmp -> Cmp(ExprLvlAdd {CmpPart}) }
impl_parse!{ CmpPart -> Eq(Token#OpEqual ExprLvlAdd)
                      | Neq(Token#OpUnequal ExprLvlAdd)
                      | Leq(Token#OpLessEq ExprLvlAdd)
                      | Geq(Token#OpGreaterEq ExprLvlAdd)
                      | Ls(Token#OpLess ExprLvlAdd)
					  | Gt(Token#OpGreater ExprLvlAdd) }

impl_parse!{ ExprLvlAdd -> Add(ExprLvlMult {AddPart}) }
impl_parse!{ AddPart -> Add(Token#OpAdd ExprLvlMult)
                      | Subtract(Token#OpSubtract ExprLvlMult) }

impl_parse!{ ExprLvlMult -> Mult(ExprLvlIndex {MultPart}) }
impl_parse!{ MultPart -> Mult(Token#OpMult ExprLvlIndex)
                       | Divide(Token#OpDivide ExprLvlIndex) }

impl_parse!{ ExprLvlIndex -> Index(UnaryExpr {IndexPart}) }
impl_parse!{ IndexPart -> Expr(Token#SquareBracketOpen Expr Token#SquareBracketClose) }

impl Parse for UnaryExpr {
	fn guess_can_parse(stream: &Stream) -> bool {
		stream.ends_ident() || stream.ends_literal() || stream.ends(&Token::BracketOpen) || stream.ends(&Token::New)
	}

	fn parse(stream: &mut Stream) -> Self {
		let pos = stream.pos();
		if stream.ends(&Token::BracketOpen) {
			stream.expect_next(&Token::BracketOpen);
			let expr = Expr::parse(stream);
			stream.expect_next(&Token::BracketClose);
			return UnaryExpr::Brackets(pos, Box::new(expr));
		} else if stream.ends_ident() {
			let ident = stream.next().as_ident();
			if stream.ends(&Token::BracketOpen) {
				stream.expect_next(&Token::BracketOpen);
				let mut params: Vec<Expr> = vec![];
				if !stream.ends(&Token::BracketClose) {
					params.push(Expr::parse(stream));
					while stream.ends(&Token::Comma) {
						stream.expect_next(&Token::Comma);
						params.push(Expr::parse(stream));
					}
				}
				stream.expect_next(&Token::BracketClose);
				return UnaryExpr::Call(pos, Box::new(ident), params);
			} else {
				return UnaryExpr::Variable(pos, Box::new(ident));
			}
		} else if stream.ends_literal() {
			return UnaryExpr::Literal(pos, Box::new(stream.next().as_literal()));
		} else if stream.ends(&Token::New) {
			stream.expect_next(&Token::New);
			let base_type = BaseType::parse(stream);
			let mut dimensions = vec![];
			while IndexPart::guess_can_parse(stream) {
				dimensions.push(IndexPart::parse(stream));
			}
			return UnaryExpr::New(pos, Box::new(base_type), dimensions);
		} else {
			panic!("Expected 'new', '(', identifier or literal, got {:?}", stream);
		}
	}
}

impl_parse!{ BaseType -> Int(Token#Int) }
