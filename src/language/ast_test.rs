pub use super::gwaihir_writer::{DisplayWrapper, AstWriter};
pub use super::compiler::{StringWriter, CodeWriter};
pub use super::super::parser::TopLevelParser;
pub use super::super::lexer::lexer::lex_str;

macro_rules! assert_ast_eq {
    ($expected:expr, $actual:expr) => {{
        let mut expected_str = String::new();
        let mut actual_str = String::new();
        ($expected).write(($expected).lifetime(), &mut CodeWriter::new(&mut StringWriter::new(&mut expected_str))).unwrap();
        ($actual).write(($actual).lifetime(), &mut CodeWriter::new(&mut StringWriter::new(&mut actual_str))).unwrap();
        assert_eq!(expected_str, actual_str);
    }};
}

macro_rules! assert_ast_frag_eq {
    ($expected:expr, $actual:expr; $lifetime_left:expr, $lifetime_right:expr) => {{
        let mut expected_str = String::new();
        let mut actual_str = String::new();
        ($expected).write($lifetime_left, &mut CodeWriter::new(&mut StringWriter::new(&mut expected_str))).unwrap();
        ($actual).write($lifetime_right, &mut CodeWriter::new(&mut StringWriter::new(&mut actual_str))).unwrap();
        assert_eq!(expected_str, actual_str);
    }};
}
