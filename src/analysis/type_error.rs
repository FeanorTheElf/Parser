use super::super::language::prelude::*;

pub fn error_jump_label_var_type(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("JumpLabel not valid as variable or parameter type"), ErrorType::TypeError)
}

pub fn error_test_type(pos: &TextPosition) -> ! {
    panic!("TestType")
}

pub fn error_nested_view(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("Nested views are illegal"), ErrorType::TypeError)
}

pub fn error_array_value_parameter(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("Passing arrays by value is not supported"), ErrorType::ArrayParameterByValue)
}
