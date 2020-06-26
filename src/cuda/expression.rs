use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::analysis::symbol::*;
use super::declaration::*;
use super::writer::*;

pub trait CudaWritableExpression {
    fn write_parameter_context(&self, param_type: &Type, out: &mut CodeWriter) -> Result<(), OutputError>;
}

fn get_level(op: BuiltInIdentifier) -> i32 {
    match op {
        BuiltInIdentifier::FunctionIndex => 4,
        BuiltInIdentifier::FunctionUnaryDiv => 3,
        BuiltInIdentifier::FunctionMul => 2,
        BuiltInIdentifier::FunctionAdd => 1,
        BuiltInIdentifier::FunctionUnaryNeg => 1,
        BuiltInIdentifier::FunctionEq => 0,
        BuiltInIdentifier::FunctionGeq => 0,
        BuiltInIdentifier::FunctionLeq => 0,
        BuiltInIdentifier::FunctionNeq => 0,
        BuiltInIdentifier::FunctionLs => 0,
        BuiltInIdentifier::FunctionGt => 0,
        BuiltInIdentifier::FunctionAnd => -1,
        BuiltInIdentifier::FunctionOr => -2,
    }
}

/// Writes to out all operands separated by the symbol associated with the given operator
fn write_op_expression<I>(op: BuiltInIdentifier, operands: I, out: &mut CodeWriter) -> Result<(), OutputError>
    where I: Iterator, I::Item: FnOnce(&mut CodeWriter) -> Result<(), OutputError>
{
    out.write_separated(operands, |out| write!(out, " {} ", op.get_symbol()).map_err(OutputError::from))
}

/// Writes to out the index expression with the given parameters, i.e. the expression for element 
/// accessing the first operand, where the indices are all the following operands
fn write_index_expression(op: BuiltInIdentifier, operands: &Vec<Expression>, out: &mut CodeWriter) -> Result<(), OutputError> {
    let mut param_iter = operands.iter();
    write_base_value_context(param_iter.next().unwrap(), out, get_level(op))?;
    write!(out, "[")?;
    out.write_separated(param_iter.map(|index| move |out: &mut CodeWriter| write_base_value_context(index, out, i32::MIN)), |out| write!(out, "][").map_err(OutputError::from))?;
    write!(out, "]")?;
    Ok(())
}

fn write_unary_operation_expression(op: BuiltInIdentifier, operand: &Expression, out: &mut CodeWriter) -> Result<(), OutputError> {
    match op {
        BuiltInIdentifier::FunctionUnaryDiv =>  write!(out, "1/")?,
        BuiltInIdentifier::FunctionUnaryNeg =>  write!(out, "-")?,
        _ => panic!("Not a unary operation: {}", op)
    };
    write_base_value_context(&operand, out, get_level(op))?;
    Ok(())
}

/// Writes to out the expression resulting from applying the given builtin operator on the given operands.
/// parent_expr_level is an integer that is the higher the stronger a potential parent operator binds and
/// is used to determine whether brackets are required. If no parent expression exists, this should be i32::MIN 
fn write_builtin_function_call_value(op: BuiltInIdentifier, operands: &Vec<Expression>, out: &mut CodeWriter, parent_expr_level: i32) -> Result<(), OutputError> {
    let current_level = get_level(op);
    if current_level <= parent_expr_level {
        write!(out, "(")?;
    }
    if op.is_unary_function() {
        debug_assert_eq!(1, operands.len());
        write_unary_operation_expression(op, &operands[0], out)?;
    } else if op == BuiltInIdentifier::FunctionIndex {
        write_index_expression(op, operands, out)?;
    } else {
        write_op_expression(op, operands.iter().map(|index| move |out: &mut CodeWriter| write_base_value_context(index, out, current_level)), out)?;
    }
    if current_level <= parent_expr_level {
        write!(out, ")")?;
    }
    Ok(())
}

/// Writes to out the expression resulting from applying the user defined function with the given name
/// to the given operands.
fn write_defined_function_call_value(func: &Name, operands: &Vec<Expression>, out: &mut CodeWriter) -> Result<(), OutputError> {
    func.write_base(out)?;
    write!(out, "(")?;
    out.write_comma_separated(operands.iter().map(|o: &Expression| move |out: &mut CodeWriter| write_base_value_context(o, out, i32::MIN)))?;
    write!(out, ")")?;
    Ok(())
}

/// Writes to out the given expression. For parent_expr_level, see `write_builtin_function_call_value`;
/// This function only makes sense for expressions that have types that can be represented with a single 
/// variable in the target language. Currently, these are primitive types but not arrays.
fn write_base_value_context(expr: &Expression, out: &mut CodeWriter, parent_expr_level: i32) -> Result<(), OutputError> {
    match expr {
        Expression::Call(call) => match &call.function {
            Expression::Call(_) => Err(OutputError::UnsupportedCode(expr.pos().clone(), "Calling dynamic expressions is not supported".to_owned())),
            Expression::Literal(_) => CompileError::new(expr.pos(), "Literal not callable".to_owned(), ErrorType::TypeError).throw(),
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(op) => write_builtin_function_call_value(*op, &call.parameters, out, parent_expr_level),
                Identifier::Name(name) => write_defined_function_call_value(&name, &call.parameters, out)
            }
        },
        Expression::Literal(literal) => write!(out, "{}", literal.value).map_err(OutputError::from),
        Expression::Variable(variable) => match &variable.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => name.write_base(out)
        }
    }
}

/// Asserts that a rvalue expression (i.e. not a single variable/literal, but the result of a 
/// builtin/defined function) may be passed to a parameter of the given type.
fn check_complex_expression_as_param(pos: &TextPosition, param_type: &Type) -> Result<(), OutputError> {
    match &param_type {
        Type::Primitive(_) => Ok(()),
        Type::Array(_, _) => panic!("An expression evaluating to an arry was passed, but it was not extracted earlier. The code generator cannot deal with that, as an array requires multiple variables to be represented: {}", pos),
        Type::Function(_, _) => Err(OutputError::UnsupportedCode(pos.clone(), "Passing functions as parameters is not supported".to_owned()))?,
        Type::JumpLabel => CompileError::new(pos, format!("JumpLabel not passable as parameter"), ErrorType::TypeError).throw(),
        Type::TestType => panic!("TestType"),
        Type::View(_) => CompileError::new(pos, format!("Cannot pass an rvalue per view"), ErrorType::TypeError).throw()
    }
}

/// Writes to out a list of comma-separated expressions that can be used to pass a variable as
/// parameter to a function. These values will be the content and potential array size parameters.
fn write_variable_parameter_context(pos: &TextPosition, variable: &Name, param_type: &Type, out: &mut CodeWriter) -> Result<(), OutputError> {
    let function_param_not_supported = || Err(OutputError::UnsupportedCode(pos.clone(), "Passing functions as parameters is not supported".to_owned())) as Result<(), OutputError>;
    let jump_label_param_illegal = || CompileError::new(pos, format!("JumpLabel not passable as parameter"), ErrorType::TypeError).throw();
    let test_type_illegal = || panic!("TestType");
    match &param_type {
        Type::Array(_, _) => Err(OutputError::UnsupportedCode(pos.clone(), "Passing arrays by value is not supported".to_owned())),
        Type::Function(_, _) => function_param_not_supported(),
        Type::JumpLabel => jump_label_param_illegal(),
        Type::TestType => test_type_illegal(),
        Type::Primitive(_) => {
            variable.write_base(out).map_err(OutputError::from)
        },
        Type::View(ty) => match &**ty {
            Type::View(_) => CompileError::new(pos, format!("Nested views are illegal"), ErrorType::TypeError).throw(),
            Type::JumpLabel => jump_label_param_illegal(),
            Type::Function(_, _) => function_param_not_supported(),
            Type::TestType => test_type_illegal(),
            Type::Array(PrimitiveType::Int, dim) => {
                variable.write_base(out)?;
                for d in 0..*dim {
                    variable.write_dim(d, out)?;
                }
                Ok(())
            },
            Type::Primitive(_) => {
                write!(out, "&")?;
                variable.write_base(out).map_err(OutputError::from)
            },
        }
    }
}

impl CudaWritableExpression for Expression {
    fn write_parameter_context(&self, param_type: &Type, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            Expression::Variable(variable) => match &variable.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(name) => write_variable_parameter_context(self.pos(), name, param_type, out)
            },
            expr => {
                check_complex_expression_as_param(expr.pos(), param_type)?;
                write_base_value_context(expr, out, i32::MIN)
            }
        }
    }
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;

#[test]
fn test_write_base_value_context() {
    let expr = Expression::parse(&mut fragment_lex("(a + b * c) / d[e,]")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    write_base_value_context(&expr, &mut writer, i32::MIN).unwrap();
    assert_eq!("(_a_ + _b_ * _c_) * 1/_d_[_e_]", output);
}