use super::super::parser::tokens::*;

use std::vec::Vec;
use std::string::String;

fn lex_op(string: &str) -> Option<Token> {
	match string {
		"" => None,
		"(" => Some(Token::BracketOpen),
		")" => Some(Token::BracketClose),
		"[" => Some(Token::SquareBracketOpen),
		"]" => Some(Token::SquareBracketClose),
		";" => Some(Token::Semicolon),
		"||" => Some(Token::OpOr),
		"&&" => Some(Token::OpAnd),
		"<=" => Some(Token::OpLessEq),
		"<" => Some(Token::OpLess),
		">" => Some(Token::OpGreater),
		"=" => Some(Token::Assign),
		">=" => Some(Token::OpGreaterEq),
		"==" => Some(Token::OpEqual),
		"!=" => Some(Token::OpUnequal),
		"!" => Some(Token::OpNot),
		"-" => Some(Token::OpSubtract),
		"+" => Some(Token::OpAdd),
		"*" => Some(Token::OpMult),
		"/" => Some(Token::OpDivide),
		"{" => Some(Token::CurlyBracketOpen),
		"}" => Some(Token::CurlyBracketClose),
		"," => Some(Token::Comma),
		":" => Some(Token::Colon),
		_ => None
	}
}

fn lex_keyword(string: &str) -> Option<Token> {
	match string {
		"if" => Some(Token::If),
		"while" => Some(Token::While),
		"pfor" => Some(Token::PFor),
		"fn" => Some(Token::Fn),
		"int" => Some(Token::Int),
		"return" => Some(Token::Return),
		"new" => Some(Token::New),
		"let" => Some(Token::Let),
		"void" => Some(Token::Void),
		_ => None
	}
}

fn lex_str(string: &str) -> Token {
	lex_op(string)
	.or_else(||lex_keyword(string))
	.unwrap_or_else(||match string.parse::<i32>() {
		Ok(value) => Token::Literal(Literal { value: value }),
		Err(_err) => Token::Identifier(Identifier { name: string.to_string() })
	})
}

fn is_whitespace(c: char) -> bool {
	c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

fn is_alphanumeric(c: char) -> bool {
	c == '_' || c.is_ascii_alphabetic()
}

pub fn lex(mut input: String) -> Stream {
	let mut result = vec![];
	let mut current = String::new();
	for c in input.drain(..) {
		let mut separator = false;
		if is_whitespace(c) {
			if !current.is_empty() {
				separator = true;
			}
		} else if is_alphanumeric(c) {
			if current.chars().next().map(|first|!is_alphanumeric(first)).unwrap_or(false) {
				separator = true;
			}
		} else {
			if current.chars().next().map(|first|is_alphanumeric(first)).unwrap_or(false) {
				separator = true;
			}
			current.push(c);
			let new_current_operator = lex_op(&current).is_some();
			current.pop();
			if lex_op(&current).is_some() && !new_current_operator {
				separator = true;
			}
		}
		if separator {
			result.push(lex_str(&current));
			current.clear();
		}
		if !is_whitespace(c) {
			current.push(c);
		}
	}
	if !current.is_empty() {
		result.push(lex_str(&current));
	}
	result.reverse();
	return Stream::create(result);
}