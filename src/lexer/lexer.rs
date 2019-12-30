use super::tokens::*;
use super::position::TextPosition;

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
		"?" => Some(Token::Wildcard),
		"&" => Some(Token::View),
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
		"read" => Some(Token::Read),
		"write" => Some(Token::Write),
		"native" => Some(Token::Native),
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

fn is_newline(c: char) -> bool {
	c == '\n'
}

fn is_alphanumeric(c: char) -> bool {
	c == '_' || c.is_ascii_alphanumeric()
}

fn is_first_char_alphanumeric(string: &String) -> Option<bool> {
	string.chars().next().map(|first| is_alphanumeric(first))
}

pub fn lex(input: &str) -> Stream {
	let mut result: Vec<PosToken> = vec![];
	let mut current = String::new();
	let mut current_pos = TextPosition::create(0, 0);
	let mut current_token_start_pos = current_pos.clone();
	for c in input.chars() {
		let mut separator = false;

		if is_whitespace(c) {
			if !current.is_empty() {
				separator = true;
			}
		} else if is_alphanumeric(c) {
			if !is_first_char_alphanumeric(&current).unwrap_or(true) {
				separator = true;
			}
		} else {
			if is_first_char_alphanumeric(&current).unwrap_or(false) {
				separator = true;
			} else {
				current.push(c);
				let new_current_operator = lex_op(&current).is_some();
				current.pop();
				if lex_op(&current).is_some() && !new_current_operator {
					separator = true;
				}
			}
		}
		if separator {
			result.push(PosToken::create(lex_str(&current), current_token_start_pos.clone()));
			current.clear();
		}
		if !is_whitespace(c) {
			if current.is_empty() {
				current_token_start_pos = current_pos.clone();
			}
			current.push(c);
		}

		if is_newline(c) {
			current_pos.next_line();
		} else {
			current_pos.add_column(1);
		}
	}
	if !current.is_empty() {
		result.push(PosToken::create(lex_str(&current), current_token_start_pos.clone()));
	}
	result.reverse();
	return Stream::create(result);
}

#[cfg(test)]
use std::iter::FromIterator;

#[test]
fn test_lex() {
	assert_eq!(vec![Token::Let, 
		Token::Identifier(Identifier { name: "test".to_owned() }),
		Token::Colon,
		Token::Int,
		Token::SquareBracketOpen,
		Token::SquareBracketClose,
		Token::Assign,
		Token::Identifier(Identifier { name: "a".to_owned() }),
		Token::SquareBracketOpen,
		Token::Literal(Literal { value: 2 }),
		Token::SquareBracketClose,
		Token::SquareBracketOpen,
		Token::Literal(Literal { value: 4 }),
		Token::OpEqual,
		Token::Literal(Literal { value: 1 }),
		Token::SquareBracketClose,
		Token::OpGreaterEq,
		Token::Identifier(Identifier { name: "b".to_owned() }),
		Token::OpAnd,
		Token::Identifier(Identifier { name: "c".to_owned() }),
		Token::OpAdd,
		Token::OpSubtract,
		Token::Identifier(Identifier { name: "d".to_owned() }),
		Token::Semicolon], Vec::from_iter(lex("let test: int[] = a[2][4 ==1]>=b&&c+-d;")));
}

#[test]
fn test_lex_position() {
	let mut stream = lex("let a = b[c+-1];\n{\n\ta>=1==2;\n}");
	assert_eq!(0, stream.pos().column());
	assert_eq!(0, stream.pos().line());
	assert_eq!(Token::Let, stream.next());
	assert_eq!("a".to_owned(), stream.next_ident().unwrap().name);
	assert_eq!(6, stream.pos().column());
	assert_eq!(Token::Assign, stream.next());
	assert_eq!("b".to_owned(), stream.next_ident().unwrap().name);
	assert_eq!(Token::SquareBracketOpen, stream.next());
	assert_eq!(10, stream.pos().column());
	for _i in 0..7 {
		stream.next();
	}
	assert_eq!(1, stream.pos().column());
	assert_eq!(2, stream.pos().line());
	assert_eq!("a".to_owned(), stream.next_ident().unwrap().name);
	stream.next();
	stream.next();
	assert_eq!(5, stream.pos().column());
	assert_eq!(Token::OpEqual, stream.next());
	stream.next();
	stream.next();
	assert_eq!(0, stream.pos().column());
	assert_eq!(3, stream.pos().line());
}