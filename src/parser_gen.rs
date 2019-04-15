use super::tokens::{Stream, Token};
use std::vec::Vec;

pub trait Parse {
	fn guess_can_parse(stream: &Stream) -> bool;
	fn parse(stream: &mut Stream) -> Self;
}

macro_rules! rule_alt_parser {
    ($stream:ident;) => {
		()
	};
	($stream:ident; identifier $($tail:tt)*) => {
        (Box::new(($stream).next().as_ident()), rule_alt_parser!($stream; $($tail)*))
    };
	($stream:ident; { $name:ident } $($tail:tt)*) => {
        ({
			let mut els: Vec<$name> = Vec::new();
			while $name::guess_can_parse($stream) {
				els.push($name::parse($stream));
			}
			els
		}, rule_alt_parser!($stream; $($tail)*))
    };
	($stream:ident; Token#$token:ident $($tail:tt)*) => {
        {($stream).expect_next(&Token::$token); rule_alt_parser!($stream; $($tail)*)}
    };
    ($stream:ident; $name:ident $($tail:tt)*) => {
        (Box::new($name::parse($stream)), rule_alt_parser!($stream; $($tail)*))
    };
}


macro_rules! rule_base_alt_parser {
	($stream:ident; $result:ident; $else_code:tt; $variant:ident(identifier $($tail:tt)*)) => {
        if $stream.ends_ident() {
			$result::$variant(rule_alt_parser!($stream; identifier $($tail)*))
		} else {
			$else_code
		}
    };
	($stream:ident; $result:ident; $else_code:tt; $variant:ident({ $name:ident } $($tail:tt)*)) => {
        if $name::guess_can_parse($stream) {
			$result::$variant(rule_alt_parser!($stream; { $name } $($tail)*))
		} else {
			$else_code
		}
    };
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
