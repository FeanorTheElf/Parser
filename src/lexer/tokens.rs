use super::position::TextPosition;

use std::vec::Vec;
use std::string::String;
use std::fmt::Debug;

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Eq)]
pub struct Identifier {
	pub name: String,
	pub key: Option<i32>
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
pub struct PosToken {
	token: Token,
	pos: TextPosition
}

impl PosToken {
	pub fn create(token: Token, pos: TextPosition) -> Self {
		PosToken {
			token: token,
			pos: pos
		}
	}
}

#[derive(Debug)]
pub struct Stream {
	data: Vec<PosToken>
}

impl Stream {
	pub fn create(tokens: Vec<PosToken>) -> Stream {
		Stream {
			data: tokens
		}
	}

	pub fn peek(&self) -> Option<&Token> {
		self.data.last().map(|t|&t.token)
	}

	pub fn next(&mut self) -> Token {
		self.data.pop().unwrap().token
	}

	pub fn expect_next(&mut self, token: &Token) {
		match self.data.pop() {
			Some(value) => assert_eq!(*token, value.token, "Expected token {:?}, but got token {:?}", token, value.token),
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

	pub fn pos(&self) -> TextPosition {
		self.data.last().unwrap().pos.clone()
	}
}

impl Iterator for Stream {
	type Item = Token;

	fn next(&mut self) -> Option<Self::Item> {
		self.data.pop().map(|t|t.token)
	}
}
