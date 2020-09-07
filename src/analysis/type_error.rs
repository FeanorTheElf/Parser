use super::super::language::prelude::*;

pub fn error_jump_label_var_type(pos: &TextPosition) -> CompileError {

    CompileError::new(
        pos,
        format!("JumpLabel not valid as variable, parameter or return type"),
        ErrorType::TypeError,
    )
}

pub fn error_test_type(_pos: &TextPosition) -> ! {

    panic!("TestType")
}

pub fn error_nested_view(pos: &TextPosition) -> CompileError {

    CompileError::new(
        pos,
        format!("Nested views are illegal"),
        ErrorType::TypeError,
    )
}

pub fn error_array_value_parameter(pos: &TextPosition) -> CompileError {

    CompileError::new(
        pos,
        format!(
            "Passing arrays by value is not supported, use copy() if you have to copy the array"
        ),
        ErrorType::ArrayParameterByValue,
    )
}

pub fn error_not_indexable_buildin_identifier(
    pos: &TextPosition,
    builtin_identifier: &BuiltInIdentifier,
) -> CompileError {

    CompileError::new(
        pos,
        format!("Operator {} cannot be indexed", builtin_identifier),
        ErrorType::TypeError,
    )
}

pub fn error_rvalue_not_assignable(pos: &TextPosition) -> CompileError {

    CompileError::new(
        pos,
        format!("Assignments to RValues are illegal"),
        ErrorType::RValueAssignment,
    )
}

pub fn error_return_view(pos: &TextPosition) -> CompileError {

    CompileError::new(pos, format!("The function returns a view, which is illegal, as views could reference values owned by the function that are deleted when the function returns"), ErrorType::ViewReturnType)
}

pub fn error_undefined_symbol(var: &Variable) -> CompileError {

    CompileError::new(
        var.pos(),
        format!("Undefined symbol: {}", var.identifier.unwrap_name()),
        ErrorType::UndefinedSymbol,
    )
}
