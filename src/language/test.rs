
#[cfg(test)]
macro_rules! assert_ast_eq {
    ($expected:expr, $actual:expr) => {
        {
            let expected = $expected;
            let actual = $actual;
            assert!(expected == actual, "Expected two asts to be the same, but got:\n  left: `{}`\n right: `{}`", print_nazgul(&expected), print_nazgul(&actual));
        }
    };
}