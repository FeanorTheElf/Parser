pub use super::gwaihir_writer::{DisplayWrapper, AstWriter};

#[cfg(test)]
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

#[cfg(test)]
macro_rules! assert_ast_frag_eq {
    ($expected:expr, $actual:expr; $lifetime_left:expr, $lifetime_right:expr) => {{
        let expected = $expected;
        let actual = $actual;
        assert!(
            expected == actual,
            "Expected two asts to be the same, but got:\n  left: `{}`\n right: `{}`",
            DisplayWrapper::from((&expected, $lifetime_left)),
            DisplayWrapper::from((&actual, $lifetime_right))
        );
    }};
}
