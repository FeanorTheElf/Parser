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
        format!("Operator {:?} cannot be indexed", builtin_identifier),
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

pub fn error_wrong_parameter_count(pos: &TextPosition, function: Identifier, expected_count: usize) -> CompileError {
    CompileError::new(pos, format!("Function {:?} expects {} parameters", function, expected_count), ErrorType::TypeError)
}

pub fn error_return_view(pos: &TextPosition) -> CompileError {
    CompileError::new(pos, format!("The function returns a view, which is illegal, as views could reference values owned by the function that are deleted when the function returns"), ErrorType::ViewReturnType)
}

pub fn error_view_not_initialized(declaration: &LocalVariableDeclaration, type_lifetime: Lifetime) -> CompileError {
    CompileError::new(declaration.pos(), format!("Local variable {:?} of view type {} must be initialized where declared", declaration.declaration.variable, LifetimedType::bind(declaration.declaration.variable_type, type_lifetime)), ErrorType::TypeError)
}

pub fn error_type_not_convertable(pos: &TextPosition, src: TypePtr, dst: TypePtr, type_lifetime: Lifetime) -> CompileError {
    CompileError::new(pos, format!("Cannot convert type {} to type {}", LifetimedType::bind(src, type_lifetime), LifetimedType::bind(dst, type_lifetime)), ErrorType::TypeError)
}

pub fn error_undefined_symbol(var: &Variable) -> CompileError {
    CompileError::new(
        var.pos(),
        format!("Undefined symbol: {:?}", var.identifier.unwrap_name()),
        ErrorType::UndefinedSymbol,
    )
}
