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
	($stream:ident; { Token#$token:ident $name:ident } $($tail:tt)*) => {
        ({
			let mut els = AstVec::new();
			while ($stream).ends(&Token::$token) {
				($stream).expect_next(&Token::$token)?;
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
	($stream:ident; [ Token#$token:ident $name:ident ] $($tail:tt)*) => {
        ({
			if ($stream).ends(&Token::$token) {
				($stream).expect_next(&Token::$token)?;
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

macro_rules! rule_alt_expectation_as_printable {
	((identifier $($tail:tt)*)) => {
        "identifier"
    };
	(({ $name:ident } $($tail:tt)*)) => {
        stringify!($name)
    };
	((Token#$token:ident $($tail:tt)*)) => {
        Token::$token
    };
    (($name:ident $($tail:tt)*)) => {
        stringify!($name)
    };
}

macro_rules! rule_parser {
	($stream:ident; $result:ty; $expected_string:expr; $variant:ident $alt:tt) => {
        rule_base_alt_parser!($stream; {
			Err(CompileError::new(($stream.pos()), 
				format!("{} or {}, got {} while parsing {}", $expected_string, rule_alt_expectation_as_printable!($alt), ($stream).peek().unwrap(), stringify!($result)), ErrorType::SyntaxError))
		}; $variant $alt)
    };
    ($stream:ident; $result:ty; $expected_string:expr; $variant:ident $alt:tt | $($tail:tt)*) => {
		rule_base_alt_parser!($stream; {
			rule_parser!($stream; $result;  format!("{}, {}", $expected_string, rule_alt_expectation_as_printable!($alt)); $($tail)*)
		}; $variant $alt)
    };
}

macro_rules! rule_alt_guess_can_parse {
	($stream:ident; $variant:ident(identifier $($tail:tt)*)) => {
        ($stream).ends_ident()
    };
	// This does not work, as it does not correctly recognize zero repetitions
	//($stream:ident; $variant:ident({ $name:ident } $($tail:tt)*)) => {
    //};
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
				rule_parser!(stream; $result; "Expected "; $($tail)*)
			}
		}
	}
}
