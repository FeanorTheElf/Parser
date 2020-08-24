use super::super::language::prelude::*;

pub fn error_jump_label_var_type(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("JumpLabel not valid as variable or parameter type"), ErrorType::TypeError)
}

pub fn error_test_type(_pos: &TextPosition) -> ! {
    panic!("TestType")
}

pub fn error_nested_view(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("Nested views are illegal"), ErrorType::TypeError)
}

pub fn error_array_value_parameter(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("Passing arrays by value is not supported, use copy() if you have to copy the array"), ErrorType::ArrayParameterByValue)
}

pub fn error_not_callable(pos: &TextPosition, ty: &Type) -> CompileError {
    CompileError::new(pos, format!("Type {} is not callable", ty), ErrorType::TypeError)
}

pub fn error_not_indexable(pos: &TextPosition, ty: &Type) -> CompileError {
    CompileError::new(pos, format!("Type {} cannot be indexed", ty), ErrorType::TypeError)
}

pub fn error_not_indexable_buildin_identifier(pos: &TextPosition, builtin_identifier: &BuiltInIdentifier) -> CompileError {
    CompileError::new(pos, format!("Operator {} cannot be indexed", builtin_identifier), ErrorType::TypeError)
}

pub fn error_rvalue_not_assignable(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("Assignments to RValues are illegal"), ErrorType::RValueAssignment)
}