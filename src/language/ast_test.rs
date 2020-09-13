pub use super::gwaihir_writer::{DisplayWrapper, AstWriter};
pub use super::compiler::{StringWriter, CodeWriter};

macro_rules! assert_ast_eq {
    ($expected:expr, $actual:expr) => {{
        let expected = $expected;
        let actual = $actual;
        assert!(
            expected == actual,
            "Expected two asts to be the same, but got:\n  left: `{}`\n right: `{}`",
            DisplayWrapper::from(&expected),
            DisplayWrapper::from(&actual)
        );
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
