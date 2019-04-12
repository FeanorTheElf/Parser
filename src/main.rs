#![allow(warnings)]
//#![feature(trace_macros)]

//trace_macros!(true);

mod tokens;
mod ast;
mod lexer;
//mod parser;
//mod rust_backend;

use std::vec::Vec;
use lexer::lex;
use ast::{BaseType};
use tokens::{Stream, Token, Keyword};

macro_rules! rule_alt_parser {
    ($stream:ident;) => {
		()
	};
	($stream:ident; { $name:ident }) => {
        ({
			let mut els: Vec<$name> = Vec::new();
			while $name::guess_can_parse($stream) {
				els.push($name::parse($stream));
			}
			els
		}, ())
    };
	($stream:ident; Token#$token:ident $($tail:tt)*) => {
        {($stream).expect_next(&Token::$token); rule_alt_parser!($stream; $($tail)*)}
    };
    ($stream:ident; $name:ident $($tail:tt)*) => {
        ($name::parse($stream), rule_alt_parser!($stream; $($tail)*))
    };
}

macro_rules! rule_base_alt_parser {
	($stream:ident; $result:ident; $else_code:tt; $variant:ident(Token#$token:ident $($tail:tt)*)) => {
        if $stream.ends(&Token::$token) {
			$result::$variant(rule_alt_parser!($stream; Token#$token $($tail)*))
		} else {
			$else_code
		}
    };
    ($stream:ident; $result:ident; $else_code:tt; $variant:ident($name:ident $($tail:tt)*)) => {
        if $name::guess_can_parse($stream) {
			$result::$variant(rule_alt_parser!($stream; $name $($tail)*))
		} else {
			$else_code
		}
    };
}

macro_rules! rule_parser {
	($stream:ident; $result:ident; $variant:ident $alt:tt) => {
        rule_base_alt_parser!($stream; $result; {
			panic!("Unexpected tokens: {:?}", $stream)
		}; $variant $alt)
    };
    ($stream:ident; $result:ident; $variant:ident $alt:tt | $($tail:tt)*) => {
		rule_base_alt_parser!($stream; $result; {
			rule_parser!($stream; $result; $($tail)*)
		}; $variant $alt)
    };
}

macro_rules! rule_alt_guess_can_parse {
	($stream:ident; $variant:ident(Token#$token:ident $($tail:tt)*)) => {
        ($stream).ends(&Token::$token)
    };
    ($stream:ident; $variant:ident($name:ident $($tail:tt)*)) => {
        $name::guess_can_parse($stream)
    };
}

macro_rules! rule_guess_can_parse {
	($stream:ident; $variant:ident $alt:tt) => {
        rule_alt_guess_can_parse!($stream; $variant $alt)
    };
    ($stream:ident; $variant:ident $alt:tt | $($tail:tt)*) => {
		rule_alt_guess_can_parse!($stream; $variant $alt) || rule_guess_can_parse!($stream; $($tail)*)
    };
}

macro_rules! impl_parse {
	($result:ident -> $($tail:tt)*) => {
		impl Parse for $result {
			fn guess_can_parse(stream: &Stream) -> bool {
				rule_guess_can_parse!(stream; $($tail)*)
			}
			fn parse(stream: &mut Stream) -> Self {
				rule_parser!(stream; $result; $($tail)*)
			}
		}
	}
}

#[derive(Debug)]
enum Test {
	Var1((BaseType, (BaseType, ()))),
	Var2((Vec<Foo>, ()))
}

impl_parse!(Test -> Var1(BaseType Token#OpAdd BaseType)|Var2(Token#OpMult {Foo}));

trait Parse {
	fn guess_can_parse(stream: &Stream) -> bool;
	fn parse(stream: &mut Stream) -> Self;
}

#[derive(Debug)]
enum Foo {
	Only
}

impl Parse for Foo {
	fn guess_can_parse(stream: &Stream) -> bool {
		stream.ends(&Token::Keyword(Keyword::If))
	}

	fn parse(stream: &mut Stream) -> Foo {
		stream.expect_next(&Token::Keyword(Keyword::If));
		return Foo::Only;
	}
}

impl Parse for BaseType {
	fn guess_can_parse(stream: &Stream) -> bool {
		stream.ends(&Token::Keyword(Keyword::Int))
	}

	fn parse(stream: &mut Stream) -> BaseType {
		stream.expect_next(&Token::Keyword(Keyword::Int));
		return BaseType::Int;
	}
}

fn main() {
	let mut stream = lex("int*if if if".to_string());
	println!("{:?}", Test::parse(&mut stream));
}