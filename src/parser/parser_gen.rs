pub trait Flatten {
    type Flattened;

    fn flatten(self) -> Self::Flattened;
}

impl<A> Flatten for (A, ()) {
    type Flattened = (A,);

    fn flatten(self) -> Self::Flattened {

        (self.0,)
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

macro_rules! debug_assert_at_most_one_of_applicable {
    ($stream:ident; $($variant:ident)|*) => {
        debug_assert!([$(
            <$variant>::is_applicable($stream)
        ),*].iter().filter(|b| **b).count() <= 1);
    };
}

macro_rules! impl_parse_ast_generation {
    ($stream:ident; $progcontainer:ident; ) =>
    {
		()
    };
    ($stream:ident; $progcontainer:ident; Token#$token:ident $($tail:tt)*) =>
    {
        {($stream).skip_next(&Token::$token)?; impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)}
    };
    ($stream:ident; $progcontainer:ident; $name:ident $($tail:tt)*) =>
    {
        ($name::parse($stream, $progcontainer)?, impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)).flatten()
    };
    // Repeated symbols: { <nonterminal> } or { <nonterminal> terminal } or { terminal }
    ($stream:ident; $progcontainer:ident; { Token#$token:ident } $($tail:tt)*) =>
    {
        ({
			let mut count: u32 = 0;
			while ($stream).is_next(&Token::$token) {
				($stream).skip_next(&Token::$token)?;
				count += 1;
			}
			count
		}, impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)).flatten()
    };
    ($stream:ident; $progcontainer:ident; { $name:ident Token#$token:ident } $($tail:tt)*) =>
    {
        ({
			let mut els = Vec::new();
			while $name::is_applicable($stream) {
                els.push($name::parse($stream, $progcontainer)?);
                ($stream).skip_next(&Token::$token)?;
			}
			els
		}, impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)).flatten()
    };
    ($stream:ident; $progcontainer:ident; { $name:ident } $($tail:tt)*) =>
    {
        ({
			let mut els = Vec::new();
			while $name::is_applicable($stream) {
				els.push($name::parse($stream, $progcontainer)?);
			}
			els
		}, impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)).flatten()
    };
    // Optional symbols: [ <nonterminal> ] or [ <nonterminal> terminal ] or [ terminal ]
    ($stream:ident; $progcontainer:ident; [ $name:ident ] $($tail:tt)*) =>
    {
        ({
			if $name::is_applicable($stream) {
				Some($name::parse($stream, $progcontainer)?)
			} else {
				None
			}
		}, impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)).flatten()
	};
    ($stream:ident; $progcontainer:ident; [ Token#$token:ident $name:ident ] $($tail:tt)*) =>
    {
        ({
			if ($stream).is_next(&Token::$token) {
                ($stream).skip_next(&Token::$token)?;
                Some($name::parse($stream, $progcontainer)?)
			} else {
				None
			}
		}, impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)).flatten()
    };
    ($stream:ident; $progcontainer:ident; [ Token#$token:ident ] $($tail:tt)*) =>
    {
        ({
			if ($stream).is_next(&Token::$token) {
                ($stream).skip_next(&Token::$token);
				true
			} else {
				false
			}
		}, impl_parse_ast_generation!($stream; $progcontainer; $($tail)*)).flatten()
	};
}

macro_rules! impl_is_applicable_predicate
{
    ($stream:ident; Token#$token:ident $($tail:tt)*) =>
    {
        ($stream).is_next(&Token::$token)
    };
    ($stream:ident; $name:ident $($tail:tt)*) =>
    {
        $name::is_applicable($stream)
    };
    ($stream:ident; ) =>
    {
        true
    };

    // Repeated symbols: { <nonterminal> } or { <nonterminal> terminal }
    ($stream:ident; { $name:ident } $($tail:tt)*) =>
    {
        $name::is_applicable($stream) || impl_is_applicable_predicate!($stream; $($tail)*)
    };
    ($stream:ident; { $name:ident Token#$token:ident } $($tail:tt)*) =>
    {
        $name::is_applicable($stream) || impl_is_applicable_predicate!($stream; $($tail)*)
    };

    // Optional symbols: [ <nonterminal> ] or [ <nonterminal> terminal ]
    ($stream:ident; [ $name:ident ] $($tail:tt)*) =>
    {
        $name::is_applicable($stream) || impl_is_applicable_predicate!($stream; $($tail)*)
    };
    ($stream:ident; [ $name:ident Token#$token:ident ] $($tail:tt)*) =>
    {
        $name::is_applicable($stream) || impl_is_applicable_predicate!($stream; $($tail)*)
    };
}

macro_rules! impl_parse_function
{
    ($stream:ident; $progcontainer:ident; $result:ty; $expected_string:expr; $variant:ident) =>
    {
        if <$variant>::is_applicable($stream) {
            let pos = ($stream).pos().clone();
            let parts = $variant::parse($stream, $progcontainer)?;
			Ok(<$result>::build(pos, $progcontainer, parts))
		} else {
			Err(CompileError::new(($stream.pos()),
				format!("{} or {}, got {} while parsing {}", $expected_string, stringify!($variant), ($stream).peek().unwrap(), stringify!($result)), ErrorType::SyntaxError))
		}
    };
    ($stream:ident; $progcontainer:ident; $result:ty; $expected_string:expr; $variant:ident | $($tail:tt)*) =>
    {
        if <$variant>::is_applicable($stream) {
            let pos = ($stream).pos().clone();
            let parts = $variant::parse($stream, $progcontainer)?;
			Ok(<$result>::build(pos, $progcontainer, parts))
		} else {
			impl_parse_function!($stream; $progcontainer; $result;  format!("{}, {}", $expected_string, stringify!($variant)); $($tail)*)
		}
    };
}

/**
 * Supported syntax:
 * - N := V1 ... Vn
 *   where V1, ..., Vn are either 'Token#<token>' or names of types implementing parse
 *         and N is a type implementing Build<(V1::OutputType, ..., Vn::OutputType)>
 *         (ignoring Vi == 'Token#<token>')
 * - N := V1 | ... | Vn
 *   where V1, ..., Vn are names of types implementing parse and N is a type implementing
 *         Build<Vi::OutputType> for all Vi
 */

macro_rules! impl_parse_trait
{
    ($result:ty := $($variant:ident)|*) => {
        impl Parser for $result
        {
            fn is_applicable(stream: &Stream) -> bool
            {
				$($variant::is_applicable(stream))||*
            }

            fn parse(stream: &mut Stream, types: &mut TypeVec) -> Result<Self::ParseOutputType, CompileError>
            {
                debug_assert_at_most_one_of_applicable!(stream; $($variant)|*);
                impl_parse_function!(stream; types; $result; "Expected"; $($variant)|*)
			}
		}
    };
    ($result:ty := $($tail:tt)*) =>
    {
        impl Parser for $result
        {
            fn is_applicable(stream: &Stream) -> bool
            {
                impl_is_applicable_predicate!(stream; $($tail)*)
            }

            fn parse(stream: &mut Stream, types: &mut TypeVec) -> Result<Self::ParseOutputType, CompileError>
            {
                let pos = stream.pos().clone();
                let parts = impl_parse_ast_generation!(stream; types; $($tail)*);
			    Ok(Self::build(pos, types, parts))
			}
		}
    }
}

macro_rules! extract_grammar_variant_children_types_as_tupel
{
    () =>
    {
		()
    };
    (Token#$token:ident $($tail:tt)*) =>
    {
        extract_grammar_variant_children_types_as_tupel!($($tail)*)
    };
    ($name:ident $($tail:tt)*) =>
    {
        <(<$name as Parseable>::ParseOutputType, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    ({ $name:ident } $($tail:tt)*) =>
    {
        <(Vec<<$name as Parseable>::ParseOutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    ({ Token#$token:ident } $($tail:tt)*) =>
    {
        <(u32, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    ({ $name:ident Token#$token:ident } $($tail:tt)*) =>
    {
        <(Vec<<$name as Parseable>::ParseOutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
    ([ $name:ident ] $($tail:tt)*) =>
    {
        <(Option<<$name as Parseable>::ParseOutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
	};
    ([ Token#$token:ident ] $($tail:tt)*) =>
    {
        <(bool, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
	};
    ([ Token#$token:ident $name:ident ] $($tail:tt)*) =>
    {
        <(Option<<$name as Buildable>::OutputType>, extract_grammar_variant_children_types_as_tupel!($($tail)*)) as Flatten>::Flattened
    };
}

macro_rules! generate_grammar_rule_temporary_node
{
    ($result:ident := $($(dyn)? $variant:ident)|*) => {
        #[derive(Debug, PartialEq, Eq)]
        enum $result
        {
            $($variant(<$variant as Parseable>::ParseOutputType)),*
        }

        impl Parseable for $result
        {
            type ParseOutputType = Self;
        }

        impl AstNode for $result { }

        impl AstNodeFuncs for $result
        {
            fn pos(&self) -> &TextPosition
            {
                match self {
                    $($result::$variant(concrete) => concrete.pos()),*
                }
            }
        }

        $(
            impl Build<<$variant as Parseable>::ParseOutputType> for $result
            {
                fn build(_pos: TextPosition, _types: &mut TypeVec, params: <$variant as Parseable>::ParseOutputType) -> Self::ParseOutputType
                {
                    $result::$variant(params)
                }
            }
        )*
    };
    ($result:ident := $($tail:tt)*) => {
        #[derive(Debug, Eq)]
        struct $result(TextPosition, extract_grammar_variant_children_types_as_tupel!($($tail)*));

        impl Parseable for $result
        {
            type ParseOutputType = Self;
        }

        impl AstNode for $result { }

        impl AstNodeFuncs for $result
        {
            fn pos(&self) -> &TextPosition
            {
                &self.0
            }
        }

        impl PartialEq for $result
        {
            fn eq(&self, rhs: &$result) -> bool
            {
                self.1 == rhs.1
            }
        }

        impl Build<extract_grammar_variant_children_types_as_tupel!($($tail)*)> for $result
        {
            fn build(pos: TextPosition, _types: &mut TypeVec, params: extract_grammar_variant_children_types_as_tupel!($($tail)*)) -> Self::ParseOutputType
            {
                Self(pos, params)
            }
        }
    };
    (box $result:ident := $($tail:tt)*) => {
        #[derive(Debug, Eq)]
        struct $result(TextPosition, extract_grammar_variant_children_types_as_tupel!($($tail)*));

        impl Parseable for $result
        {
            type ParseOutputType = Box<Self>;
        }

        impl PartialEq for $result
        {
            fn eq(&self, rhs: &$result) -> bool
            {
                self.1 == rhs.1
            }
        }

        impl AstNode for $result {}

        impl AstNodeFuncs for $result
        {
            fn pos(&self) -> &TextPosition
            {
                &self.0
            }
        }

        impl Build<extract_grammar_variant_children_types_as_tupel!($($tail)*)> for $result
        {
            fn build(pos: TextPosition, _types: &mut TypeVec, params: extract_grammar_variant_children_types_as_tupel!($($tail)*)) -> Self::ParseOutputType
            {
                Box::new(Self(pos, params))
            }
        }
    };
}

/**
 * Supported syntax:
 * - box? N := V1 ... Vn
 *   where V1, ..., Vn are either 'Token#<token>', names of types implementing parse
 *   or a repetition bracket (either [] or {}) with a single type name, a single token
 *   or a type name followed a token as content
 * - N := V1 | ... | Vn
 *   where V1, ..., Vn are names of types implementing parse
 */

macro_rules! grammar_rule {
    (box $result:ident := $($tail:tt)*) => {
        generate_grammar_rule_temporary_node!(box $result := $($tail)*);
        impl_parse_trait!($result := $($tail)*);
    };
    ($result:ident := $($tail:tt)*) => {
        generate_grammar_rule_temporary_node!($result := $($tail)*);
        impl_parse_trait!($result := $($tail)*);
    };
}
