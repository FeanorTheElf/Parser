use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::declaration::*;
use super::writer::*;

pub trait CudaWritableExpression {
    /// Writes code in the target language to out that evaluates this expression and can
    /// be inserted as a parameter in a function call.
    fn write_parameter_context(&self, param_type: &Type, out: &mut CodeWriter) -> Result<(), OutputError>;
    /// Writes code in the target language to out that evaluates this expression and can
    /// be included in a context where the target language requires a value. Some types
    /// in the source language can only be represented by multiple variables in the target
    /// language, and therefore expressions yielding these types cannot be written using
    /// this function.
    fn write_value_context(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
    /// Asserts that the expression is valid in a context that requires a complex type (i.e. a type
    /// that is represented by multiple values in the target language, as e.g. an array).
    /// Currently, this are only variables, as function calls can only return a single value
    /// in the target language. Other return values are therefore returned via output parameter, and
    /// chaining functions with output parameters is both difficult and hard to understand.
    /// 
    /// If this is a variable, the corresponding identifier is returned.
    fn assert_expression_valid_as_array(&self) -> Result<&Name, OutputError>;
}

pub enum CudaExpression<'a> {
    Base(&'a Expression), KernelIndexVariableCalculation(/* kernel id */ u32, /* n-th index variable */ u32, /* total dims */ u32)
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
fn write_operator_expression<I>(op: BuiltInIdentifier, operands: I, out: &mut CodeWriter) -> Result<(), OutputError>
    where I: Iterator, I::Item: FnOnce(&mut CodeWriter) -> Result<(), OutputError>
{
    out.write_separated(operands, |out| write!(out, " {} ", op.get_symbol()).map_err(OutputError::from))
}

fn write_target_index_calculation<'a, I>(indexed_array_name: &Name, dimension_count: u32, indices: I, out: &mut CodeWriter) -> Result<(), OutputError>
    where I: Iterator<Item = &'a Expression>
{
    out.write_separated(indices.enumerate().map(|(dimension, index)| 
        move |out: &mut CodeWriter| {
            // the length in the last dimension is 1
            if dimension as u32 + 1 < dimension_count {
                indexed_array_name.write_dim(dimension as u32 + 1, out)?;
                write!(out, " * ")?;
            }
            write_value_context(&CudaExpression::Base(index), out, get_level(BuiltInIdentifier::FunctionMul))?;
            Ok(())
        }), 
        |out| write!(out, " + ").map_err(OutputError::from))
}

/// Writes to out the index expression with the given parameters, i.e. the expression for element 
/// accessing the first operand, where the indices are all the following operands
fn write_index_expression(op: BuiltInIdentifier, operands: &Vec<Expression>, out: &mut CodeWriter) -> Result<(), OutputError> {
    debug_assert_eq!(BuiltInIdentifier::FunctionIndex, op);
    let mut param_iter = operands.iter();
    let indexed_array = param_iter.next().unwrap();
    let indexed_array_name = indexed_array.assert_expression_valid_as_array()?;
    write_value_context(&CudaExpression::Base(indexed_array), out, get_level(op))?;
    write!(out, "[")?;
    write_target_index_calculation(indexed_array_name, operands.len() as u32 - 1, param_iter, out)?;
    write!(out, "]")?;
    return Ok(());
}

fn write_unary_operation_expression(op: BuiltInIdentifier, operand: &Expression, out: &mut CodeWriter) -> Result<(), OutputError> {
    match op {
        BuiltInIdentifier::FunctionUnaryDiv =>  write!(out, "1/")?,
        BuiltInIdentifier::FunctionUnaryNeg =>  write!(out, "-")?,
        _ => panic!("Not a unary operation: {}", op)
    };
    write_value_context(&CudaExpression::Base(operand), out, get_level(op))?;
    Ok(())
}

/// Writes to out code in the target language that calculates the expression resulting from applying the given builtin 
/// operator on the given operands. `parent_expr_level` is an integer that is the higher the stronger a potential parent operator binds and
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
        write_operator_expression(op, operands.iter().map(|index| move |out: &mut CodeWriter| write_value_context(&CudaExpression::Base(index), out, current_level)), out)?;
    }
    if current_level <= parent_expr_level {
        write!(out, ")")?;
    }
    Ok(())
}

/// Writes to out code in the target language that calculates the expression resulting from applying the user defined 
/// function with the given name to the given operands.
fn write_defined_function_call_value(func: &Name, operands: &Vec<Expression>, out: &mut CodeWriter) -> Result<(), OutputError> {
    func.write_base(out)?;
    write!(out, "(")?;
    out.write_comma_separated(operands.iter().map(|o: &Expression| move |out: &mut CodeWriter| write_value_context(&CudaExpression::Base(o), out, i32::MIN)))?;
    write!(out, ")")?;
    Ok(())
}

/// Writes to out code in the target language that calculates the given expression. 
/// For the parameter parent_expr_level, see `write_builtin_function_call_value`;
/// This function only makes sense for expressions that have types that can be represented with a single 
/// variable in the target language. Currently, these are primitive types but not arrays.
fn write_value_context(expr: &CudaExpression, out: &mut CodeWriter, parent_expr_level: i32) -> Result<(), OutputError> {
    match expr {
        CudaExpression::Base(Expression::Call(call)) => match &call.function {
            Expression::Call(_) => Err(OutputError::UnsupportedCode(call.pos().clone(), "Calling dynamic expressions is not supported".to_owned())),
            Expression::Literal(literal) => CompileError::new(literal.pos(), "Literal not callable".to_owned(), ErrorType::TypeError).throw(),
            Expression::Variable(var) => match &var.identifier {
                Identifier::BuiltIn(op) => write_builtin_function_call_value(*op, &call.parameters, out, parent_expr_level),
                Identifier::Name(name) => write_defined_function_call_value(&name, &call.parameters, out)
            }
        },
        CudaExpression::Base(Expression::Literal(literal)) => write!(out, "{}", literal.value).map_err(OutputError::from),
        CudaExpression::Base(Expression::Variable(variable)) => match &variable.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => name.write_base(out)
        },
        CudaExpression::KernelIndexVariableCalculation(kernel_id, dim, total_dim_count) => {
            // calculate the coordinates as queued thread from the one-dimension threadIdx.x
            if *dim > 0 {
                write!(out, "(static_cast<int>(threadIdx.x) % ")?;
                Name::write_thread_acc_count(*kernel_id, *dim, out)?;
                write!(out, ")");
            } else {
                write!(out, "static_cast<int>(threadIdx.x)")?;
            }
            if *dim + 1 != *total_dim_count {
                write!(out, " / ")?;
                Name::write_thread_acc_count(*kernel_id, *dim + 1, out)?;
            }
            // add the offset
            write!(out, " + ")?;
            Name::write_thread_offset(*kernel_id, *dim, out)?;
            Ok(())
        }
    }
}

/// Asserts that a rvalue expression (i.e. not a single variable/literal, but the result of a 
/// builtin/defined function) may be passed to a parameter of the given type.
fn check_complex_expression_as_param(pos: &TextPosition, param_type: &Type) -> Result<(), OutputError> {
    match &param_type {
        Type::Primitive(_) => Ok(()),
        Type::Array(_, _) => panic!("An expression evaluating to an array was passed, 
            but it was not extracted earlier. The code generator cannot deal with 
            that, as an array requires multiple variables to be represented: {}", pos),
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
    let nested_views_illegal = || CompileError::new(pos, format!("Nested views are illegal"), ErrorType::TypeError).throw();
    match &param_type {
        Type::Array(_, _) => Err(OutputError::UnsupportedCode(pos.clone(), "Passing arrays by value is not supported".to_owned())),
        Type::Function(_, _) => function_param_not_supported(),
        Type::JumpLabel => jump_label_param_illegal(),
        Type::TestType => test_type_illegal(),
        Type::Primitive(_) => variable.write_base(out).map_err(OutputError::from),
        Type::View(ty) => match &**ty {
            Type::View(_) => nested_views_illegal(),
            Type::JumpLabel => jump_label_param_illegal(),
            Type::Function(_, _) => function_param_not_supported(),
            Type::TestType => test_type_illegal(),
            Type::Array(PrimitiveType::Int, dim) => {
                variable.write_base(out)?;
                write!(out, ", ")?;
                out.write_comma_separated((0..*dim).map(|d| move |out: &mut CodeWriter| variable.write_dim(d, out)))?;
                Ok(())
            },
            Type::Primitive(_) => {
                write!(out, "&")?;
                variable.write_base(out).map_err(OutputError::from)?;
                Ok(())
            },
        }
    }
}

fn assert_expression_valid_as_array<'a>(expr: &CudaExpression<'a>) -> Result<&'a Name, OutputError> {
    let identifier = match expr {
        CudaExpression::Base(Expression::Call(call)) => panic!("Expecting the function call at {} to yield a value representable as
            a single value in the target language. The code generator cannot deal with intermediate values that
            are the result from function calls and require multiple variables to be represented, as these functions
            cannot return more than one value in the target language. Instead, this function call should have been
            extracted earlier into one temporary variable (in the source language) that corresponds to multiple
            temporary variables in the target language.", call.pos()),
        CudaExpression::Base(Expression::Literal(_)) => unimplemented!("Up to now, literals could not have types that have complex 
            representations in the target language, got"),
        CudaExpression::Base(Expression::Variable(variable)) => &variable.identifier,
        CudaExpression::KernelIndexVariableCalculation(_, _, _) => panic!("index variable expressions are not valid in places where an array is expected")
    };
    return match &identifier {
        Identifier::Name(name) => Ok(name),
        Identifier::BuiltIn(builtin) => panic!("Currently, no builtin identifiers that yield arrays exist, but got {}", builtin)
    };
}

impl<'a> CudaWritableExpression for CudaExpression<'a> {
    fn write_parameter_context(&self, param_type: &Type, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            CudaExpression::Base(Expression::Variable(variable)) => match &variable.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(name) => write_variable_parameter_context(variable.pos(), name, param_type, out)
            },
            CudaExpression::Base(expr) => {
                check_complex_expression_as_param(expr.pos(), param_type)?;
                self.write_value_context(out)
            },
            CudaExpression::KernelIndexVariableCalculation(_, _, _) => {
                self.write_value_context(out)
            }
        }
    }

    fn write_value_context(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write_value_context(self, out, i32::MIN)
    }
    
    fn assert_expression_valid_as_array(&self) -> Result<&Name, OutputError> {
        assert_expression_valid_as_array(self)
    }
}

impl CudaWritableExpression for Expression {
    fn write_parameter_context(&self, param_type: &Type, out: &mut CodeWriter) -> Result<(), OutputError> {
        CudaExpression::Base(self).write_parameter_context(param_type, out)
    }

    fn write_value_context(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        CudaExpression::Base(self).write_value_context(out)
    }

    fn assert_expression_valid_as_array(&self) -> Result<&Name, OutputError> {
        assert_expression_valid_as_array(&CudaExpression::Base(self))
    }
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;

#[test]
fn test_write_value_context_no_unnecessary_brackets() {
    let expr = Expression::parse(&mut fragment_lex("(a + b * c) / d[e,]")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    expr.write_value_context(&mut writer).unwrap();
    assert_eq!("(_a_ + _b_ * _c_) * 1/_d_[_e_]", output);
}

#[test]
fn test_write_value_context_function_call() {
    let expr = Expression::parse(&mut fragment_lex("foo[a(b,), c,]")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    expr.write_value_context(&mut writer).unwrap();
    assert_eq!("_foo_[d1_foo_ * _a_(_b_) + _c_]", output);
}

#[test]
fn test_write_value_parameter_context_array() {
    let expr = Expression::parse(&mut fragment_lex("a")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    expr.write_parameter_context(&Type::View(Box::new(Type::Array(PrimitiveType::Int, 2))), &mut writer).unwrap();
    assert_eq!("_a_, d0_a_, d1_a_", output);
}