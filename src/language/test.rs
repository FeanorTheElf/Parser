pub use super::gwaihir_writer::{DisplayWrapper, AstWriter};
pub use super::compiler::{StringWriter, CodeWriter};
pub use super::super::parser::{TopLevelParser, Parser, ParserContext};
pub use super::super::lexer::lexer::{lex_str, fragment_lex};

use super::prelude::*;

macro_rules! assert_ast_eq {
    ($expected:expr, $actual:expr) => {{
        let mut expected_str = String::new();
        let mut actual_str = String::new();
        ($expected).write(&mut CodeWriter::new(&mut StringWriter::new(&mut expected_str))).unwrap();
        ($actual).write(&mut CodeWriter::new(&mut StringWriter::new(&mut actual_str))).unwrap();
        assert_eq!(expected_str, actual_str);
    }};
}

macro_rules! assert_ast_frag_eq {
    ($expected:expr, $actual:expr) => {{
        let mut expected_str = String::new();
        let mut actual_str = String::new();
        ($expected).write(&mut CodeWriter::new(&mut StringWriter::new(&mut expected_str))).unwrap();
        ($actual).write(&mut CodeWriter::new(&mut StringWriter::new(&mut actual_str))).unwrap();
        assert_eq!(expected_str, actual_str);
    }};
}

#[derive(Debug, PartialEq)]
pub struct TestDefinition {
    name: Name
}

pub fn testdef(name: &str) -> TestDefinition {
    TestDefinition {
        name: Name::l(name)
    }
}

impl AstNodeFuncs for TestDefinition {

    fn pos(&self) -> &TextPosition {
        &TextPosition::NONEXISTING
    }
}

impl AstNode for TestDefinition {}

impl SymbolDefinitionFuncs for TestDefinition {

    fn get_name(&self) -> &Name {
        &self.name
    }

    fn get_name_mut(&mut self) -> &mut Name {
        &mut self.name
    }

    fn cast_statement_mut(&mut self) -> Option<&mut dyn Statement> {
        None
    }

    fn get_type(&self) -> &Type {
        unimplemented!()
    }
} 

impl SymbolDefinition for TestDefinition {}