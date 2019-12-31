use super::position::TextPosition;
use super::error::{ CompileError, ErrorType };

use std::fmt::{ Display, Formatter, Error };
use std::vec::Vec;
use std::string::String;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Identifier {
	Named(String), Auto(u32)
}

impl Identifier
{
	pub fn new(name: &str) -> Identifier
	{
		Identifier::Named(name.to_owned())
	}

	pub fn auto(id: u32) -> Identifier
	{
		Identifier::Auto(id)
	}

	#[cfg(test)]
	pub fn repr(&self) -> &str
	{
		match self {
			Identifier::Named(name) => name,
			Identifier::Auto(_id) => "auto"
		}
	}
}

impl Display for Identifier 
{
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> 
	{
        match self {
			Identifier::Named(name) => write!(f, "{}", name),
			Identifier::Auto(id) => write!(f, "auto_{}", id)
		}
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Literal {
	pub value: i32
}

impl Display for Literal 
{
	fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> 
	{
        write!(f, "{}", self.value)
    }
}

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Eq)]
pub enum Token {
	Literal(Literal),
	Identifier(Identifier),
	If,
	PFor,
	Read,
	Write,
	Native,
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
	Wildcard,
	View,
	EOF,
	BOF
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			Token::Literal(ref literal) => write!(f, "'{}'", literal.value),
			Token::Identifier(ref identifier) => write!(f, "'{}'", identifier),
			Token::If => write!(f, "'if'"),
			Token::PFor => write!(f, "'pfor'"),
			Token::Read => write!(f, "'read'"),
			Token::Write => write!(f, "'write'"),
			Token::Native => write!(f, "'native'"),
			Token::While => write!(f, "'while'"),
			Token::Int => write!(f, "'int'"),
			Token::Fn => write!(f, "'fn'"),
			Token::Return => write!(f, "'return'"),
			Token::New => write!(f, "'new'"),
			Token::Let => write!(f, "'let'"),
			Token::Void => write!(f, "'void'"),
			Token::Assign => write!(f, "'='"),
			Token::SquareBracketOpen => write!(f, "'['"),
			Token::SquareBracketClose => write!(f, "']'"),
			Token::BracketOpen => write!(f, "'('"),
			Token::BracketClose => write!(f, "')'"),
			Token::CurlyBracketOpen => write!(f, "'{{'"),
			Token::CurlyBracketClose => write!(f, "'}}'"),
			Token::Semicolon => write!(f, "';'"),
			Token::Comma => write!(f, "','"),
			Token::Colon => write!(f, "':'"),
			Token::OpOr => write!(f, "'||'"),
			Token::OpAnd => write!(f, "'&&'"),
			Token::OpAdd => write!(f, "'+'"),
			Token::OpMult => write!(f, "'*'"),
			Token::OpDivide => write!(f, "'/'"),
			Token::OpSubtract => write!(f, "'-'"),
			Token::OpLess => write!(f, "'<'"),
			Token::OpGreater => write!(f, "'>'"),
			Token::OpLessEq => write!(f, "'<='"),
			Token::OpGreaterEq => write!(f, "'>='"),
			Token::OpEqual => write!(f, "'=='"),
			Token::OpUnequal => write!(f, "'!='"),
			Token::OpNot => write!(f, "'!'"),
			Token::Wildcard => write!(f, "'?'"),
			Token::View => write!(f, "'&'"),
			Token::EOF => write!(f, "EOF"),
			Token::BOF => write!(f, "BOF")
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

	pub fn next_literal(&mut self) -> Result<Literal, CompileError> {
		let pos = self.pos();
		match self.next() {
			Token::Literal(lit) => Ok(lit),
			_value => Err(CompileError::new(pos, format!("Expected literal, got {:?}", _value), ErrorType::SyntaxError))
		}
	}

	pub fn next_ident(&mut self) -> Result<Identifier, CompileError> {
		let pos = self.pos();
		match self.next() {
			Token::Identifier(id) => Ok(id),
			_value => Err(CompileError::new(pos, format!("Expected identifier, got {:?}", _value), ErrorType::SyntaxError))
		}
	}

	pub fn expect_next(&mut self, token: &Token) -> Result<&mut Self, CompileError> {
		let pos = self.pos();
		match self.data.pop() {
			Some(value) => if *token != value.token {
				Err(CompileError::new(pos, format!("Expected token {:?}, but got token {:?}", token, value.token), ErrorType::SyntaxError))
			} else {
				Ok(self)
			},
			None => Err(CompileError::new(pos, format!("Expected token {:?}, but got end of stream", token), ErrorType::SyntaxError))
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
