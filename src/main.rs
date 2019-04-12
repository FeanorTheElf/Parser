#![allow(warnings)]

mod tokens;
mod ast;
mod lexer;
#[macro_use]
mod parser_gen;

use parser_gen::Parse;
use lexer::lex;
use tokens::{Identifier, Literal, Keyword, Token, Stream};
use ast::{Func, Stmts, Stmt, Type, Expr, Summand, ExprMult, ProductPart, ExprUn, BasicExprUn, ParamDeclaration, BaseType};

#[derive(Debug)]
enum Test {
	Var1((BaseType, (BaseType, ()))),
	Var2((Vec<Foo>, ())),
	Var3((Foo, (Vec<BaseType>, ())))
}

impl_parse!{ Test -> Var1(BaseType Token#OpAdd BaseType) 
                   | Var2(Token#OpMult {Foo})
				   | Var3(Token#Colon Foo Token#OpSmallerEq {BaseType}) }

#[derive(Debug)]
enum Foo {
	Only
}

impl Parse for Foo {
	fn guess_can_parse(stream: &Stream) -> bool {
		stream.ends(&Token::Keyword(Keyword::If))
	}

	fn parse(stream: &mut Stream) -> Foo {
		stream.expect_next(&Token::Keyword(Keyword::If));
		return Foo::Only;
	}
}

impl Parse for BaseType {
	fn guess_can_parse(stream: &Stream) -> bool {
		stream.ends(&Token::Keyword(Keyword::Int))
	}

	fn parse(stream: &mut Stream) -> BaseType {
		stream.expect_next(&Token::Keyword(Keyword::Int));
		return BaseType::Int;
	}
}

fn main() {
	let mut stream = lex("*if if".to_string());
	println!("{:?}", Test::parse(&mut stream));
}