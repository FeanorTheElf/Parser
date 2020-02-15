use std::vec::Vec;
use std::string::String;

use super::super::language::position::TextPosition;
use super::tokens::*;

fn lex_op(string: &str) -> Option<Token> 
{
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

fn lex_keyword(string: &str) -> Option<Token> 
{
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

fn lex_str(string: &str) -> Token 
{
	lex_op(string)
	.or_else(||lex_keyword(string))
	.unwrap_or_else(||match string.parse::<i32>() {
		Ok(value) => Token::Literal(value),
		Err(_err) => Token::Identifier(string.to_owned())
	})
}

fn is_whitespace(c: char) -> bool 
{
	c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

fn is_newline(c: char) -> bool 
{
	c == '\n'
}

fn is_alphanumeric(c: char) -> bool 
{
	c == '_' || c == '#' || c.is_ascii_alphanumeric()
}

fn is_first_char_alphanumeric(string: &String) -> Option<bool> 
{
	string.chars().next().map(|first| is_alphanumeric(first))
}

fn is_sequence_valid_operator(current: &mut String, next_char: char) -> bool
{
	current.push(next_char);
	let new_current_operator = lex_op(&current).is_some();
	current.pop();
	return new_current_operator;
}

pub fn lex(input: &str) -> Stream
{
	let mut result: Vec<PosToken> = vec![];
	let mut current = String::new();
	let mut current_pos = TextPosition::create(0, 0);
	let mut current_token_start_pos = current_pos.clone();

	result.push(PosToken::create(Token::BOF, current_pos.clone()));

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
			} else if lex_op(&current).is_some() && !is_sequence_valid_operator(&mut current, c) {
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
		result.push(PosToken::create(lex_str(&current), current_token_start_pos));
	}
	result.push(PosToken::create(Token::EOF, current_pos));
	result.reverse();
	return Stream::create(result);
}

#[cfg(test)]
pub fn fragment_lex(input: &str) -> Stream
{
	let mut result = lex(input);
	result.data.remove(0);
	result.data.pop();
	return result;
}

#[cfg(test)]
use std::iter::FromIterator;

#[test]
fn test_lex() 
{
	assert_eq!(vec![Token::BOF,
		Token::Let, 
		Token::Identifier("test".to_owned()),
		Token::Colon,
		Token::Int,
		Token::SquareBracketOpen,
		Token::SquareBracketClose,
		Token::Assign,
		Token::Identifier("a".to_owned()),
		Token::SquareBracketOpen,
		Token::Literal(2),
		Token::SquareBracketClose,
		Token::SquareBracketOpen,
		Token::Literal(4),
		Token::OpEqual,
		Token::Literal(1),
		Token::SquareBracketClose,
		Token::OpGreaterEq,
		Token::Identifier("b".to_owned()),
		Token::OpAnd,
		Token::Identifier("c".to_owned()),
		Token::OpAdd,
		Token::OpSubtract,
		Token::Identifier("d".to_owned()),
		Token::Semicolon,
		Token::EOF], Vec::from_iter(lex("let test: int[] = a[2][4 ==1]>=b&&c+-d;")));
}

#[test]
fn test_lex_position() 
{
	let mut stream = lex("let a = b[c+-1];\n{\n\ta>=1==2;\n}");
	stream.skip_next(&Token::BOF).unwrap();
	assert_eq!(0, stream.pos().column());
	assert_eq!(0, stream.pos().line());
	assert_eq!(Token::Let, stream.next());
	assert_eq!("a", stream.next_ident().unwrap());
	assert_eq!(6, stream.pos().column());
	assert_eq!(Token::Assign, stream.next());
	assert_eq!("b", stream.next_ident().unwrap());
	assert_eq!(Token::SquareBracketOpen, stream.next());
	assert_eq!(10, stream.pos().column());
	for _i in 0..7 {
		stream.next();
	}
	assert_eq!(1, stream.pos().column());
	assert_eq!(2, stream.pos().line());
	assert_eq!("a", stream.next_ident().unwrap());
	stream.next();
	stream.next();
	assert_eq!(5, stream.pos().column());
	assert_eq!(Token::OpEqual, stream.next());
	stream.next();
	stream.next();
	assert_eq!(0, stream.pos().column());
	assert_eq!(3, stream.pos().line());
}