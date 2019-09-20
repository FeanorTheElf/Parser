use super::super::lexer::tokens::{Stream, Token};
use super::super::lexer::error::CompileError;
use super::ast::AstVec;
use std::vec::Vec;

pub trait Parse {
	fn guess_can_parse(stream: &Stream) -> bool;
	fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError>;
}

pub trait Flatten {
	type Flattened;

	fn flatten(self) -> Self::Flattened;
}

impl<A> Flatten for (A, ()) {
	type Flattened = (A, );

	fn flatten(self) -> Self::Flattened {
		(self.0, )
	}
}

impl<A, B> Flatten for (A, (B,)) {
	type Flattened = (A, B);

	fn flatten(self) -> Self::Flattened {
		(self.0, (self.1).0)
	}
}

impl<A, B, C> Flatten for (A, (B, C)) {
	type Flattened = (A, B, C);

	fn flatten(self) -> Self::Flattened {
		(self.0, (self.1).0, (self.1).1)
	}
}

impl<A, B, C, D> Flatten for (A, (B, C, D)) {
	type Flattened = (A, B, C, D);

	fn flatten(self) -> Self::Flattened {
		(self.0, (self.1).0, (self.1).1, (self.1).2)
	}
}

impl<A, B, C, D, E> Flatten for (A, (B, C, D, E)) {
	type Flattened = (A, B, C, D, E);

	fn flatten(self) -> Self::Flattened {
		(self.0, (self.1).0, (self.1).1, (self.1).2, (self.1).3)
	}
}

macro_rules! rule_alt_parser {
    ($stream:ident;) => {
		()
	};
	($stream:ident; identifier $($tail:tt)*) => {
        (($stream).next_ident()?, rule_alt_parser!($stream; $($tail)*)).flatten()
    };
	($stream:ident; { $name:ident } $($tail:tt)*) => {
        ({
			let mut els = AstVec::new();
			while $name::guess_can_parse($stream) {
				els.push($name::parse($stream)?);
			}
			els
		}, rule_alt_parser!($stream; $($tail)*)).flatten()
    };
	($stream:ident; [ $name:ident ] $($tail:tt)*) => {
        ({
			if $name::guess_can_parse($stream) {
				Some($name::parse($stream)?)
			} else {
				None
			}
		}, rule_alt_parser!($stream; $($tail)*)).flatten()
    };
	($stream:ident; Token#$token:ident $($tail:tt)*) => {
        {($stream).expect_next(&Token::$token)?; rule_alt_parser!($stream; $($tail)*)}
    };
    ($stream:ident; $name:ident $($tail:tt)*) => {
        ($name::parse($stream)?, rule_alt_parser!($stream; $($tail)*)).flatten()
    };
}


macro_rules! rule_base_alt_parser {
	($stream:ident; $else_code:tt; $variant:ident(identifier $($tail:tt)*)) => {
        if $stream.ends_ident() {
			let pos = ($stream).pos();
			Ok(Box::new(($variant::new).call((pos, rule_alt_parser!($stream; identifier $($tail)*)).flatten())))
		} else {
			$else_code
		}
    };
	($stream:ident; $else_code:tt; $variant:ident({ $name:ident } $($tail:tt)*)) => {
        if $name::guess_can_parse($stream) {
			let pos = ($stream).pos();
			Ok(Box::new(($variant::new).call((pos, rule_alt_parser!($stream; { $name } $($tail)*)).flatten())))
		} else {
			$else_code
		}
    };
	($stream:ident; $else_code:tt; $variant:ident(Token#$token:ident $($tail:tt)*)) => {
        if $stream.ends(&Token::$token) {
			let pos = ($stream).pos();
			Ok(Box::new(($variant::new).call((pos, rule_alt_parser!($stream; Token#$token $($tail)*)).flatten())))
		} else {
			$else_code
		}
    };
    ($stream:ident; $else_code:tt; $variant:ident($name:ident $($tail:tt)*)) => {
        if $name::guess_can_parse($stream) {
			let pos = ($stream).pos();
			Ok(Box::new(($variant::new).call((pos, rule_alt_parser!($stream; $name $($tail)*)).flatten())))
		} else {
			$else_code
		}
    };
}

macro_rules! rule_parser {
	($stream:ident; $result:ty; $variant:ident $alt:tt) => {
        rule_base_alt_parser!($stream; {
			panic!("Unexpected tokens while parsing {} at position {}", stringify!($result), ($stream).pos())
		}; $variant $alt)
    };
    ($stream:ident; $result:ty; $variant:ident $alt:tt | $($tail:tt)*) => {
		rule_base_alt_parser!($stream; {
			rule_parser!($stream; $result; $($tail)*)
		}; $variant $alt)
    };
}

macro_rules! rule_alt_guess_can_parse {
	($stream:ident; $variant:ident(identifier $($tail:tt)*)) => {
        ($stream).ends_ident()
    };
	($stream:ident; $variant:ident({ $name:ident } $($tail:tt)*)) => {
        $name::guess_can_parse($stream)
    };
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
	($result:ty => $($tail:tt)*) => {
		impl Parse for $result {
			fn guess_can_parse(stream: &Stream) -> bool {
				rule_guess_can_parse!(stream; $($tail)*)
			}
			fn parse(stream: &mut Stream) -> Result<Box<Self>, CompileError> {
				rule_parser!(stream; $result; $($tail)*)
			}
		}
	}
}
