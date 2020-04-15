use super::super::language::error::{ CompileError, ErrorType };
use super::super::language::position::TextPosition;

use std::vec::Vec;
use std::string::String;

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Eq)]
pub enum Token 
{
	Literal(i32),
	Identifier(String),
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
	With,
	In,
	This,
	As,
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
	Goto,
	Target,
	EOF,
	BOF
}

impl std::fmt::Display for Token 
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result 
	{
		match self {
			Token::Literal(ref literal) => write!(f, "'{}'", literal),
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
			Token::Goto => write!(f, "'goto'"),
			Token::With => write!(f, "'with'"),
			Token::This => write!(f, "'this'"),
			Token::As => write!(f, "'as'"),
			Token::In => write!(f, "'in'"),
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
			Token::Target => write!(f, "'@'"),
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
	pub data: Vec<PosToken>
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

	pub fn next_literal(&mut self) -> Result<i32, CompileError> {
		let pos = self.pos().clone();
		match self.next() {
			Token::Literal(lit) => Ok(lit),
			_value => Err(CompileError::new(&pos, format!("Expected literal, got {:?}", _value), ErrorType::SyntaxError))
		}
	}

	pub fn next_ident(&mut self) -> Result<String, CompileError> {
		let pos = self.pos().clone();
		match self.next() {
			Token::Identifier(id) => Ok(id),
			_value => Err(CompileError::new(&pos, format!("Expected identifier, got {:?}", _value), ErrorType::SyntaxError))
		}
	}

	pub fn skip_next(&mut self, token: &Token) -> Result<&mut Self, CompileError> {
		let pos = self.pos().clone();
		match self.data.pop() {
			Some(value) => if *token != value.token {
				Err(CompileError::new(&pos, format!("Expected token {:?}, but got token {:?}", token, value.token), ErrorType::SyntaxError))
			} else {
				Ok(self)
			},
			None => Err(CompileError::new(&pos, format!("Expected token {:?}, but got end of stream", token), ErrorType::SyntaxError))
		}
	}

	pub fn is_next(&self, token: &Token) -> bool {
		self.peek().is_some() && self.peek().unwrap() == token
	}

	pub fn is_next_literal(&self) -> bool {
		match self.peek() {
			Some(Token::Literal(_val)) => true,
			_ => false
		}
	}

	pub fn is_next_identifier(&self) -> bool {
		match self.peek() {
			Some(Token::Identifier(_name)) => true,
			_ => false
		}
	}

	pub fn pos(&self) -> &TextPosition {
		&self.data.last().unwrap().pos
	}
}

impl Iterator for Stream {
	type Item = Token;

	fn next(&mut self) -> Option<Self::Item> {
		self.data.pop().map(|t|t.token)
	}
}
