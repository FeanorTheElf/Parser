use std::vec::Vec;
use std::string::String;
use std::fmt::Debug;

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Eq)]
pub struct Identifier {
	pub name: String
}

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Eq)]
pub struct Literal {
	pub value: i32
}

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Eq)]
pub enum Token {
	Literal(Literal),
	Identifier(Identifier),
	If,
	PFor,
	While,
	Int,
	Fn,
	Return,
	New,
	Let,
	Void,
	Assign,
	SquareBracketOpen,
	SquareBracketClose,
	BracketOpen,
	BracketClose,
	CurlyBracketOpen,
	CurlyBracketClose,
	Semicolon,
	Comma,
	Colon,
	OpOr,
	OpAnd,
	OpAdd,
	OpMult,
	OpDivide,
	OpSubtract,
	OpLess,
	OpGreater,
	OpLessEq,
	OpGreaterEq,
	OpEqual,
	OpUnequal,
	OpNot
}

impl Token {
	pub fn as_literal(self) -> Literal {
		match self {
			Token::Literal(lit) => lit,
			_value => panic!("Expected literal, got {:?}", _value)
		}
	}

	pub fn as_ident(self) -> Identifier {
		match self {
			Token::Identifier(ident) => ident,
			_value => panic!("Expected identifier, got {:?}", _value)
		}
	}
}

#[derive(Debug)]
pub struct Stream(Vec<Token>);

impl Stream {
	pub fn create(tokens: Vec<Token>) -> Stream {
		Stream(tokens)
	}

	pub fn peek(&self) -> Option<&Token> {
		self.0.last()
	}

	pub fn next(&mut self) -> Token {
		self.0.pop().unwrap()
	}

	pub fn expect_next(&mut self, token: &Token) {
		match self.0.pop() {
			Some(value) => assert_eq!(*token, value, "Expected token {:?}, but got token {:?}", token, value),
			None => panic!("Expected token {:?}, but got end of stream", token)
		}
	}

	pub fn ends(&self, token: &Token) -> bool {
		self.peek().is_some() && self.peek().unwrap() == token
	}

	pub fn ends_literal(&self) -> bool {
		match self.peek() {
			Some(Token::Literal(_val)) => true,
			_ => false
		}
	}

	pub fn ends_ident(&self) -> bool {
		match self.peek() {
			Some(Token::Identifier(_name)) => true,
			_ => false
		}
	}
}
