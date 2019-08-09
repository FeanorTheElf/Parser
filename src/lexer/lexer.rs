use super::tokens::*;
use super::position::TextPosition;

use std::iter::FromIterator;
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
		Err(_err) => Token::Identifier(Identifier { name: string.to_string(), key: None })
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

pub fn lex(mut input: String) -> Stream {
	let mut result: Vec<PosToken> = vec![];
	let mut current = String::new();
	let mut current_pos = TextPosition::create(0, 0);
	let mut current_token_start_pos = current_pos.clone();
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

#[test]
fn test_lex() {
	assert_eq!(vec![Token::Let, 
		Token::Identifier(Identifier { name: "test".to_owned(), key: None }),
		Token::Colon,
		Token::Int,
		Token::SquareBracketOpen,
		Token::SquareBracketClose,
		Token::Assign,
		Token::Identifier(Identifier { name: "a".to_owned(), key: None }),
		Token::SquareBracketOpen,
		Token::Literal(Literal { value: 2 }),
		Token::SquareBracketClose,
		Token::SquareBracketOpen,
		Token::Literal(Literal { value: 4 }),
		Token::OpEqual,
		Token::Literal(Literal { value: 1 }),
		Token::SquareBracketClose,
		Token::OpGreaterEq,
		Token::Identifier(Identifier { name: "b".to_owned(), key: None }),
		Token::OpAnd,
		Token::Identifier(Identifier { name: "c".to_owned(), key: None }),
		Token::OpAdd,
		Token::OpSubtract,
		Token::Identifier(Identifier { name: "d".to_owned(), key: None }),
		Token::Semicolon], Vec::from_iter(lex("let test: int[] = a[2][4 ==1]>=b&&c+-d;".to_owned())));
}

#[test]
fn test_lex_position() {
	let mut stream = lex("let a = b[c+-1];\n{\n\ta>=1==2;\n}".to_owned());
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