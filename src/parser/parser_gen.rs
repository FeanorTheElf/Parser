
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

macro_rules! impl_grammar_variant_parse {
    ($stream:ident;) => 
    {
		()
    }
    ;
	($stream:ident; identifier $($tail:tt)*) => {
        (($stream).next_ident()?, impl_grammar_variant_parse!($stream; $($tail)*)).flatten()
    };
    ($stream:ident; { $name:ident } $($tail:tt)*) => 
    {
        ({
			let mut els = Vec::new();
			while $name::guess_can_parse($stream) {
				els.push($name::parse($stream)?);
			}
			els
		}, impl_grammar_variant_parse!($stream; $($tail)*)).flatten()
    };
    ($stream:ident; { Token#$token:ident $name:ident } $($tail:tt)*) => 
    {
        ({
			let mut els = Vec::new();
			while ($stream).ends(&Token::$token) {
				($stream).expect_next(&Token::$token)?;
				els.push($name::parse($stream)?);
			}
			els
		}, impl_grammar_variant_parse!($stream; $($tail)*)).flatten()
    };
    ($stream:ident; [ $name:ident ] $($tail:tt)*) => 
    {
        ({
			if $name::guess_can_parse($stream) {
				Some($name::parse($stream)?)
			} else {
				None
			}
		}, impl_grammar_variant_parse!($stream; $($tail)*)).flatten()
	};
    ($stream:ident; [ Token#$token:ident $name:ident ] $($tail:tt)*) => 
    {
        ({
			if ($stream).ends(&Token::$token) {
				($stream).expect_next(&Token::$token)?;
				Some($name::parse($stream)?)
			} else {
				None
			}
		}, impl_grammar_variant_parse!($stream; $($tail)*)).flatten()
    };
    ($stream:ident; Token#$token:ident $($tail:tt)*) => 
    {
        {($stream).expect_next(&Token::$token)?; impl_grammar_variant_parse!($stream; $($tail)*)}
    };
    ($stream:ident; $name:ident $($tail:tt)*) => 
    {
        ($name::parse($stream)?, impl_grammar_variant_parse!($stream; $($tail)*)).flatten()
    };
}

macro_rules! impl_grammar_variant_guess_can_parse 
{
    ($stream:ident; identifier $($tail:tt)*) => 
    {
        ($stream).ends_ident()
    };
	// This does not work, as it does not correctly recognize zero repetitions
	//($stream:ident; $variant:ident({ $name:ident } $($tail:tt)*)) => {
    //};
    ($stream:ident; Token#$token:ident $($tail:tt)*) => 
    {
        ($stream).ends(&Token::$token)
    };
    ($stream:ident; $name:ident $($tail:tt)*) => 
    {
        $name::guess_can_parse($stream)
    };
}

macro_rules! impl_grammar_rule_parse 
{
    ($stream:ident; $result:ty; $else_code:tt; $variant:ident) => 
    {
        if <$variant>::guess_can_parse($stream) {
			let pos = ($stream).pos();
			Ok(<$result>::build(pos, $variant::parse($stream)?))
		} else {
			$else_code
		}
    };
}

macro_rules! impl_grammar_rule_parse_wrapper 
{
    ($stream:ident; $result:ty; $expected_string:expr; $variant:ident) => 
    {
        impl_grammar_rule_parse!($stream; $result; {
			Err(CompileError::new(($stream.pos()), 
				format!("{} or {}, got {} while parsing {}", $expected_string, stringify!($variant), ($stream).peek().unwrap(), stringify!($result)), ErrorType::SyntaxError))
		}; $variant)
    };
    ($stream:ident; $result:ty; $expected_string:expr; $variant:ident | $($tail:tt)*) => 
    {
		impl_grammar_rule_parse!($stream; $result; {
			impl_grammar_rule_parse_wrapper!($stream; $result;  format!("{}, {}", $expected_string, stringify!($variant)); $($tail)*)
		}; $variant)
    };
}

macro_rules! impl_parse 
{
    ($result:ty => $($variant:ident)|*) => {
        impl Parse for $result 
        {
            fn guess_can_parse(stream: &Stream) -> bool 
            {
				$($variant::guess_can_parse(stream))||*
            }
            
            fn parse(stream: &mut Stream) -> Result<Self::OutputType, CompileError> 
            {
                impl_grammar_rule_parse_wrapper!(stream; $result; "Expected"; $($variant)|*)
			}
		}
    };
    ($result:ty => $($tail:tt)*) => 
    {
        impl Parse for $result 
        {
            fn guess_can_parse(stream: &Stream) -> bool 
            {
                impl_grammar_variant_guess_can_parse!(stream; $($tail)*)
            }
            
            fn parse(stream: &mut Stream) -> Result<Self::OutputType, CompileError> 
            {
                let pos = stream.pos();
			    Ok(Self::build(pos, impl_grammar_variant_parse!(stream; $($tail)*)))
			}
		}
	}
}

macro_rules! extract_grammar_variant_children_types_as_tupel 
{
    () => 
    {
		()
    }
    ;
    (identifier $($tail:tt)*) => 
    {
        <(String, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    ({ $name:ident } $($tail:tt)*) => 
    {
        <(Vec<<$name as Buildable>::OutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    ({ Token#$token:ident $name:ident } $($tail:tt)*) => 
    {
        <(Vec<<$name as Buildable>::OutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    ([ $name:ident ] $($tail:tt)*) => 
    {
        <(Option<<$name as Buildable>::OutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
	};
    ([ Token#$token:ident $name:ident ] $($tail:tt)*) => 
    {
        <(Option<<$name as Buildable>::OutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    (Token#$token:ident $($tail:tt)*) => 
    {
        extract_grammar_variant_children_types_as_tupel!($($tail)*)
    };
    ($name:ident $($tail:tt)*) => 
    {
        <(<$name as Buildable>::OutputType, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
}

macro_rules! generate_grammar_rule_temporary_node
{
    ($result:ident => $($variant:ident)|*) => {
        enum $result 
        {
            $($variant(<$variant as Buildable>::OutputType)),*
        }

        impl Buildable for $result
        {
            type OutputType = Self;
        }

        $(
            impl Build<<$variant as Buildable>::OutputType> for $result
            {
                fn build(_pos: TextPosition, params: <$variant as Buildable>::OutputType) -> Self::OutputType
                {
                    $result::$variant(params)
                }
            }
        )*
    };
    ($result:ident => $($tail:tt)*) => {
        struct $result(TextPosition, extract_grammar_variant_children_types_as_tupel!($($tail)*));

        impl Buildable for $result
        {
            type OutputType = Self;
        }

        impl Build<extract_grammar_variant_children_types_as_tupel!($($tail)*)> for $result
        {
            fn build(pos: TextPosition, params: extract_grammar_variant_children_types_as_tupel!($($tail)*)) -> Self::OutputType
            {
                Self(pos, params)
            }
        }
    };
}

macro_rules! grammar_rule {
    ($($content:tt)*) => {
        generate_grammar_rule_temporary_node!($($content)*);
        impl_parse!($($content)*);
    };
}