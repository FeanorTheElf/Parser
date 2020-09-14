use super::super::language::prelude::*;
use super::super::analysis::defs_test::{EnvironmentBuilder, Definitions};
use super::ast::Writable;

pub fn output<T: ?Sized + Writable>(ast: &T) -> String {
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    ast.write(&mut writer).unwrap();
    return output;
}

pub fn mock_program(env: EnvironmentBuilder) -> (Program, Definitions) {
    let (types, defs) = env.destruct();
    (Program {
        items: vec![],
        types: types
    }, defs)
}
