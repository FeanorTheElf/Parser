use super::tokens::{Identifier, Literal, Keyword, Token, Stream};

use std::vec::Vec;
use std::string::String;

fn lex_str(string: String) -> Option<Token> {
	match string.as_str() {
		"" => None,
		"if" => Some(Token::Keyword(Keyword::If)),
		"while" => Some(Token::Keyword(Keyword::While)),
		"pfor" => Some(Token::Keyword(Keyword::PFor)),
		"fn" => Some(Token::Keyword(Keyword::Fn)),
		"int" => Some(Token::Keyword(Keyword::Int)),
		"return" => Some(Token::Keyword(Keyword::Return)),
		"new" => Some(Token::Keyword(Keyword::New)),
		"<=" => Some(Token::OpSmallerEq),
		"<" => Some(Token::OpSmaller),
		">" => Some(Token::OpGreater),
		"=" => Some(Token::Assign),
		">=" => Some(Token::OpGreaterEq),
		"==" => Some(Token::OpEqual),
		"!=" => Some(Token::OpUnequal),
		"!" => Some(Token::OpNot),
		_ => Some(match string.parse::<i32>() {
			Ok(value) => Token::Literal(Literal { value: value }),
			Err(_err) => Token::Identifier(Identifier { name: string })
		})
	}
}

pub fn lex(mut input: String) -> Stream {
	let mut result = vec![];
	let mut current = String::new();
	for c in input.drain(..) {
		let mut separator_token = true;
		let parsed_c: Option<Token> = match c {
			' ' => None,
			'\t' => None,
			'\r' => None,
			'\n' => None,
			'(' => Some(Token::BracketOpen),
			')' => Some(Token::BracketClose),
			'[' => Some(Token::SquareBracketOpen),
			']' => Some(Token::SquareBracketClose),
			';' => Some(Token::Semicolon),
			'=' => {
				if current == "=" || current == "<" || current == ">" || current == "!" {
				} else if let Some(token) = lex_str(std::mem::replace(&mut current, String::new())) {
					result.push(token);
				}
				separator_token = false;
				current.push(c);
				None
			},
			'<' => {
				if let Some(token) = lex_str(std::mem::replace(&mut current, String::new())) {
					result.push(token);
				}
				separator_token = false;
				current.push(c);
				None
			},
			'>' => {
				if let Some(token) = lex_str(std::mem::replace(&mut current, String::new())) {
					result.push(token);
				}
				separator_token = false;
				current.push(c);
				None
			},
			'!' => {
				if let Some(token) = lex_str(std::mem::replace(&mut current, String::new())) {
					result.push(token);
				}
				separator_token = false;
				current.push(c);
				None
			},
			'-' => Some(Token::OpSubtract),
			'+' => Some(Token::OpAdd),
			'*' => Some(Token::OpMult),
			'/' => Some(Token::OpDivide),
			'{' => Some(Token::CurlyBracketOpen),
			'}' => Some(Token::CurlyBracketClose),
			',' => Some(Token::Comma),
			':' => Some(Token::Colon),
			_ => {
				current.push(c);
				separator_token = false;
				None
			}
		};
		if separator_token {
			if let Some(token) = lex_str(std::mem::replace(&mut current, String::new())) {
				result.push(token);
			}
		}
		if let Some(token) = parsed_c {
			result.push(token);
		}
	}
	if let Some(token) = lex_str(current) {
		result.push(token)
	}
	result.reverse();
	return Stream::create(result);
}