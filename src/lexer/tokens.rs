use super::position::TextPosition;
use super::error::CompileError;

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
	OpNot,
	Wildcard
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

	pub fn next_literal(&mut self) -> Result<Literal, CompileError> {
		let pos = self.pos();
		match self.next() {
			Token::Literal(lit) => Ok(lit),
			_value => Err(CompileError::new(pos, format!("Expected literal, got {:?}", _value)))
		}
	}

	pub fn next_ident(&mut self) -> Result<Identifier, CompileError> {
		let pos = self.pos();
		match self.next() {
			Token::Identifier(id) => Ok(id),
			_value => Err(CompileError::new(pos, format!("Expected identifier, got {:?}", _value)))
		}
	}

	pub fn expect_next(&mut self, token: &Token) -> Result<(), CompileError> {
		let pos = self.pos();
		match self.data.pop() {
			Some(value) => if *token != value.token {
				Err(CompileError::new(pos, format!("Expected token {:?}, but got token {:?}", token, value.token)))
			} else {
				Ok(())
			},
			None => Err(CompileError::new(pos, format!("Expected token {:?}, but got end of stream", token)))
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
