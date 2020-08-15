use super::super::language::prelude::*;
use super::super::language::backend::OutputError;
use super::super::language::position::NONEXISTING;
use super::writer::*;
use super::variable::*;
use super::CudaContext;
use super::INDEX_TYPE;

pub trait CudaWritableExpression {
    /// Writes code in the target language to out that evaluates this expression and can
    /// be inserted as a parameter in a function call.
    fn write_as_parameter(&self, param_type: &Type, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError>;
    /// Writes code in the target language to out that evalutes this expression and assigns
    /// the result to the given target.
    fn write_assignment_to(&self, target_type: &Type, target: &Expression, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError>;
    /// Writes code in the target language to out that evaluates this expression and can
    /// be included in a context where the target language requires a value. Some types
    /// in the source language can only be represented by multiple variables in the target
    /// language, and therefore expressions yielding these types cannot be written using
    /// this function.
    fn write_as_value(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError>;
    /// Asserts that the expression is valid in a context that requires a complex type (i.e. a type
    /// that is represented by multiple values in the target language, currently an array).
    /// Currently, this are only variables, as function calls can only return a single value
    /// in the target language. Other return values are therefore returned via output parameter, and
    /// chaining functions with output parameters is both difficult and hard to understand.
    /// 
    /// If this is a variable, the corresponding identifier is returned.
    fn assert_expression_valid_as_array(&self) -> Result<CudaVariable, OutputError>;
}

#[derive(Debug, PartialEq)]
pub enum CudaExpression<'a> {
    Base(&'a Expression), KernelIndexVariableCalculation(/* kernel id */ u32, /* n-th index variable */ u32, /* total dims */ u32),
    CudaVariable(CudaVariable<'a>)
}

impl<'a> CudaExpression<'a> {
    fn calc_type(&self, context: &dyn CudaContext) -> Type {
        match self {
            CudaExpression::Base(base) => context.calculate_type(base),
            CudaExpression::KernelIndexVariableCalculation(_, _, _) => Type::Primitive(PrimitiveType::Int),
            CudaExpression::CudaVariable(var) => var.calc_type(context)
        }
    }
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

fn write_target_index_calculation<'a, I>(indexed_array_name: CudaVariable, indices: I, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError>
    where I: Iterator<Item = CudaExpression<'a>>
{
    let mut peekable_indices = indices.enumerate().peekable();
    while let Some((dimension, index)) = peekable_indices.next() {
        // the length in the last dimension is 1
        if let Some(_) = peekable_indices.peek() {
            indexed_array_name.write_dim(dimension as u32 + 1, out)?;
            write!(out, " * ")?;
        }
        write_value_context(&index, out, context, get_level(BuiltInIdentifier::FunctionMul))?;
        if let Some(_) = peekable_indices.peek() {
            write!(out, " + ")?;
        }
    }
    Ok(())
}

/// Writes to out the index expression with the given parameters, i.e. the expression for element 
/// accessing the first operand, where the indices are all the following operands
fn write_index_expression<'a, I>(op: BuiltInIdentifier, operands: I, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> 
    where I: Iterator<Item = CudaExpression<'a>>
{
    debug_assert_eq!(BuiltInIdentifier::FunctionIndex, op);
    let mut param_iter = operands;
    let indexed_array = param_iter.next().unwrap();
    let indexed_array_name = indexed_array.assert_expression_valid_as_array()?;
    write_value_context(&indexed_array, out, context, get_level(op))?;
    write!(out, "[")?;
    write_target_index_calculation(indexed_array_name, param_iter, out, context)?;
    write!(out, "]")?;
    Ok(())
}

fn write_unary_operation_expression(op: BuiltInIdentifier, operand: CudaExpression, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
    match op {
        BuiltInIdentifier::FunctionUnaryDiv =>  write!(out, "1/")?,
        BuiltInIdentifier::FunctionUnaryNeg =>  write!(out, "-")?,
        _ => panic!("Not a unary operation: {}", op)
    };
    write_value_context(&operand, out, context, get_level(op))?;
    Ok(())
}

/// Writes to out code in the target language that calculates the expression resulting from applying the given builtin 
/// operator on the given operands. `parent_expr_level` is an integer that is the higher the stronger a potential parent operator binds and
/// is used to determine whether brackets are required. If no parent expression exists, this should be i32::MIN 
fn write_builtin_function_call_value<'a, I>(op: BuiltInIdentifier, mut operands: I, out: &mut CodeWriter, context: &dyn CudaContext, parent_expr_level: i32) -> Result<(), OutputError> 
    where I: Iterator<Item = CudaExpression<'a>>
{
    let current_level = get_level(op);
    if current_level <= parent_expr_level {
        write!(out, "(")?;
    }
    if op.is_unary_function() {
        write_unary_operation_expression(op, operands.next().unwrap(), out, context)?;
        debug_assert_eq!(None, operands.next());
    } else if op == BuiltInIdentifier::FunctionIndex {
        write_index_expression(op, operands, out, context)?;
    } else {
        write_operator_expression(op, operands.map(|index| move |out: &mut CodeWriter| write_value_context(&index, out, context, current_level)), out)?;
    }
    if current_level <= parent_expr_level {
        write!(out, ")")?;
    }
    Ok(())
}

/// Writes to out code in the target language that calculates the expression resulting from applying the user defined 
/// function with the given name to the given operands.
fn write_defined_function_call_value<'a, I>(pos: &TextPosition, func: &Name, operands: I, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> 
    where I: Iterator<Item = CudaExpression<'a>>
{
    func.write_base(out)?;
    write!(out, "(")?;
    let (param_types, return_type) = match context.calculate_var_type(func) {
        Type::Function(params, return_type) => (params, return_type),
        ty => CompileError::new(pos, format!("Expression of type {} is not callable", ty), ErrorType::TypeError).throw()
    };
    let mut param_types_iter = param_types.into_iter();
    out.write_comma_separated(operands.map(|expr: CudaExpression<'a>| {
        let param_type = param_types_iter.next();
        let output_param_type = return_type.clone().map(|t| Type::View(t));
        move |out: &mut CodeWriter| {
            let ty = expr.calc_type(context);
            if let Some(param_type) = param_type {
                if !param_type.is_assignable_from(&ty) {
                    CompileError::new(pos, format!("Type mismatch: Expected {}, got {}", param_type, ty), ErrorType::TypeError).throw();
                }
                expr.write_as_parameter(&param_type, out, context)
            } else {
                // in this case, we have an output parameter
                if let Some(output_param_type) = output_param_type {
                    if !output_param_type.is_assignable_from(&ty) {
                        CompileError::new(pos, format!("Type mismatch: Output parameter has type {}, but the assigned to expression has type {}", output_param_type, ty), ErrorType::TypeError).throw();
                    }
                    expr.write_as_parameter(&output_param_type, out, context)
                } else {
                    CompileError::new(pos, format!("Type mismatch: function call parameter count exceeds parameter count, remaining parameters are considered output parameters, but the function returns no value"), ErrorType::TypeError).throw()
                }
            }
        }
    }))?;
    write!(out, ")")?;
    Ok(())
}

fn write_function_call_value<'a, I>(call: &'a FunctionCall, additional_params: I, out: &mut CodeWriter, context: &dyn CudaContext, parent_expr_level: i32) -> Result<(), OutputError> 
    where I: Iterator<Item = CudaExpression<'a>>
{
    let params = call.parameters.iter().map(|e: &'a Expression| CudaExpression::Base(e));
    match &call.function {
        Expression::Call(_) => Err(OutputError::UnsupportedCode(call.pos().clone(), "Calling dynamic expressions is not supported".to_owned())),
        Expression::Literal(literal) => CompileError::new(literal.pos(), "Literal not callable".to_owned(), ErrorType::TypeError).throw(),
        Expression::Variable(var) => match &var.identifier {
            Identifier::BuiltIn(op) => write_builtin_function_call_value(*op, params.chain(additional_params), out, context, parent_expr_level),
            Identifier::Name(name) => write_defined_function_call_value(call.pos(), &name, params.chain(additional_params), out, context)
        }
    }
}

/// Writes to out code in the target language that calculates the given expression. 
/// For the parameter parent_expr_level, see `write_builtin_function_call_value`;
/// This function only makes sense for expressions that have types that can be represented with a single 
/// variable in the target language. Currently, these are primitive types but not arrays.
fn write_value_context(expr: &CudaExpression, out: &mut CodeWriter, context: &dyn CudaContext, parent_expr_level: i32) -> Result<(), OutputError> {
    match expr {
        CudaExpression::Base(Expression::Call(call)) => write_function_call_value(call, std::iter::empty(), out, context, parent_expr_level),
        CudaExpression::Base(Expression::Literal(literal)) => write!(out, "{}", literal.value).map_err(OutputError::from),
        CudaExpression::Base(Expression::Variable(variable)) => match &variable.identifier {
            Identifier::BuiltIn(_) => unimplemented!(),
            Identifier::Name(name) => name.write_base(out)
        },
        CudaExpression::CudaVariable(variable) => variable.write_base(out),
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
fn write_variable_as_parameter(pos: &TextPosition, variable: &CudaVariable, param_type: &Type, out: &mut CodeWriter) -> Result<(), OutputError> {
    let function_param_not_supported = || Err(OutputError::UnsupportedCode(pos.clone(), "Passing functions as parameters is not supported".to_owned())) as Result<(), OutputError>;
    let jump_label_param_illegal = || CompileError::new(pos, format!("JumpLabel not passable as parameter"), ErrorType::TypeError).throw();
    let test_type_illegal = || panic!("TestType");
    let nested_views_illegal = || CompileError::new(pos, format!("Nested views are illegal"), ErrorType::TypeError).throw();
    let arrays_by_val_not_supported = || CompileError::new(pos, format!("Passing arrays by value is not supported"), ErrorType::ArrayParameterByValue).throw();

    match &param_type {
        Type::Array(_, _) => arrays_by_val_not_supported(),
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

fn write_temporary_call_array_result_variable(call: &FunctionCall, array_base_type: PrimitiveType, array_dim: u32, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
    debug_assert_eq!(PrimitiveType::Int, array_base_type);
    write!(out, "int* ")?;
    CudaVariable::TemporaryArrayResult(array_base_type, array_dim).write_base(out)?;
    write!(out, " = nullptr;")?;
    out.newline()?;
    for d in 0..array_dim {
        write!(out, "{} ", INDEX_TYPE)?;
        CudaVariable::TemporaryArrayResult(array_base_type, array_dim).write_dim(d, out)?;
        write!(out, " = 0;")?;
        out.newline()?;
    }
    write_function_call_value(call, std::iter::once(CudaExpression::CudaVariable(CudaVariable::TemporaryArrayResult(array_base_type, array_dim))), out, context, i32::MIN)?;
    write!(out, ";")?;
    Ok(())
}

fn write_device_memcpy(from: &CudaVariable, to: &CudaVariable, array_base_type: PrimitiveType, array_dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
    for d in 0..array_dim {
        write!(out, "assert(")?;
        from.write_dim(d, out)?;
        write!(out, " == ")?;
        to.write_dim(d, out)?;
        write!(out, ");")?;
        out.newline()?;
    }
    write!(out, "memcpy(")?;
    to.write_base(out)?;
    write!(out, ", ")?;
    from.write_base(out)?;
    write!(out, ", ")?;
    from.write_dim(0, out)?;
    debug_assert_eq!(PrimitiveType::Int, array_base_type);
    write!(out, " * sizeof(int));")?;
    Ok(())
}

fn write_host_memcpy(from: &CudaVariable, to: &CudaVariable, array_base_type: PrimitiveType, array_dim: u32, out: &mut CodeWriter) -> Result<(), OutputError> {
    for d in 0..array_dim {
        write!(out, "assert(")?;
        from.write_dim(d, out)?;
        write!(out, " == ")?;
        to.write_dim(d, out)?;
        write!(out, ");")?;
        out.newline()?;
    }
    write!(out, "checkCudaStatus(cudaMemcpy(")?;
    to.write_base(out)?;
    write!(out, ", ")?;
    from.write_base(out)?;
    write!(out, ", ")?;
    from.write_dim(0, out)?;
    debug_assert_eq!(PrimitiveType::Int, array_base_type);
    write!(out, " * sizeof(int), cudaMemcpyDeviceToDevice));")?;
    Ok(())
}

fn write_call_to_array_assignment(call: &FunctionCall, target: &Expression, array_base_type: PrimitiveType, array_dim: u32, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
    out.enter_block()?;
    write_temporary_call_array_result_variable(call, array_base_type, array_dim, out, context)?;
    out.newline()?;
    let name = match target.expect_identifier().map_err(CompileError::throw).unwrap() {
        Identifier::Name(name) => name,
        Identifier::BuiltIn(_) => unimplemented!()
    };
    if context.is_device_context() {
        write_device_memcpy(&CudaVariable::TemporaryArrayResult(array_base_type, array_dim), &CudaVariable::Base(name), array_base_type, array_dim, out)?;
        out.newline()?;
        write!(out, "free(")?;
        CudaVariable::TemporaryArrayResult(array_base_type, array_dim).write_base(out)?;
        write!(out, ");")?;
    } else {
        write_host_memcpy(&CudaVariable::TemporaryArrayResult(array_base_type, array_dim), &CudaVariable::Base(name), array_base_type, array_dim, out)?;
        out.newline()?;
        write!(out, "checkCudaStatus(cudaFree(")?;
        CudaVariable::TemporaryArrayResult(array_base_type, array_dim).write_base(out)?;
        write!(out, "));")?;
    }
    out.exit_block()?;
    Ok(())
}

fn write_var_to_array_assignment(var: &Variable, target: &Expression, array_dim: u32, array_base_type: PrimitiveType, out: &mut CodeWriter, device_context: bool) -> Result<(), OutputError> {
    let var_name = match &var.identifier {
        Identifier::Name(name) => name,
        Identifier::BuiltIn(_) => unimplemented!()
    };
    let target_name = match target.expect_identifier().map_err(CompileError::throw).unwrap() {
        Identifier::Name(name) => name,
        Identifier::BuiltIn(_) => unimplemented!()
    };
    if device_context {
        write_device_memcpy(&CudaVariable::Base(var_name), &CudaVariable::Base(target_name), array_base_type, array_dim, out)?;
    } else {
        write_host_memcpy(&CudaVariable::Base(var_name), &CudaVariable::Base(target_name), array_base_type, array_dim, out)?;
    }
    Ok(())
}

fn write_assignment_to(pos: &TextPosition, target_type: &Type, target: &Expression, value: &Expression, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
    let function_assignment_not_supported = || Err(OutputError::UnsupportedCode(pos.clone(), "Assigning to function variables is not supported".to_owned())) as Result<(), OutputError>;
    let jump_label_assignment_illegal = || CompileError::new(pos, format!("JumpLabels cannot be assigned to"), ErrorType::TypeError).throw();
    let test_type_illegal = || panic!("TestType");
    let nested_views_illegal = || CompileError::new(pos, format!("Nested views are illegal"), ErrorType::TypeError).throw();

    match &target_type {
        Type::Array(PrimitiveType::Int, dim) => match value {
            Expression::Call(call) => write_call_to_array_assignment(call, target, PrimitiveType::Int, *dim, out, context),
            Expression::Variable(var) => write_var_to_array_assignment(var, target, *dim, PrimitiveType::Int, out, context.is_device_context()),
            Expression::Literal(_) => unimplemented!()
        },
        Type::Function(_, _) => function_assignment_not_supported(),
        Type::JumpLabel => jump_label_assignment_illegal(),
        Type::TestType => test_type_illegal(),
        Type::Primitive(PrimitiveType::Int) => {
            target.write_as_value(out, context)?;
            write!(out, " = ")?;
            value.write_as_value(out, context)?;
            Ok(())
        },
        Type::View(ty) => match &**ty {
            Type::View(_) => nested_views_illegal(),
            Type::JumpLabel => jump_label_assignment_illegal(),
            Type::Function(_, _) => function_assignment_not_supported(),
            Type::TestType => test_type_illegal(),
            Type::Array(PrimitiveType::Int, dim) => match value {
                Expression::Call(call) => write_call_to_array_assignment(call, target, PrimitiveType::Int, *dim, out, context),
                Expression::Variable(var) => write_var_to_array_assignment(var, target, *dim, PrimitiveType::Int, out, context.is_device_context()),
                Expression::Literal(_) => unimplemented!()
            },
            Type::Primitive(PrimitiveType::Int) => {
                write!(out, "*")?;
                target.write_as_value(out, context)?;
                write!(out, " = ")?;
                value.write_as_value(out, context)?;
                Ok(())
            },
        }
    }
}

fn assert_expression_valid_as_array<'a>(expr: &CudaExpression<'a>) -> Result<CudaVariable<'a>, OutputError> {
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
        CudaExpression::KernelIndexVariableCalculation(_, _, _) => panic!("index variable expressions are not valid in places where an array is expected"),
        CudaExpression::CudaVariable(var) => return Ok(var.clone())
    };
    return match &identifier {
        Identifier::Name(name) => Ok(CudaVariable::Base(name)),
        Identifier::BuiltIn(builtin) => panic!("Currently, no builtin identifiers that yield arrays exist, but got {}", builtin)
    };
}

impl<'a> CudaWritableExpression for CudaExpression<'a> {
    fn write_as_parameter(&self, param_type: &Type, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        match self {
            CudaExpression::Base(Expression::Variable(variable)) => match &variable.identifier {
                Identifier::BuiltIn(_) => unimplemented!(),
                Identifier::Name(name) => write_variable_as_parameter(variable.pos(), &CudaVariable::Base(name), param_type, out)
            },
            CudaExpression::Base(expr) => {
                check_complex_expression_as_param(expr.pos(), param_type)?;
                self.write_as_value(out, context)
            },
            CudaExpression::KernelIndexVariableCalculation(_, _, _) => {
                self.write_as_value(out, context)
            },
            CudaExpression::CudaVariable(variable) => write_variable_as_parameter(&NONEXISTING, variable, param_type, out)
        }
    }

    fn write_as_value(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        write_value_context(self, out, context, i32::MIN)
    }
    
    fn assert_expression_valid_as_array(&self) -> Result<CudaVariable, OutputError> {
        assert_expression_valid_as_array(self)
    }
    
    fn write_assignment_to(&self, target_type: &Type, target: &Expression, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        debug_assert!(target.is_lvalue());
        match self {
            CudaExpression::Base(base) => {
                write_assignment_to(base.pos(), target_type, target, base, out, context)
            },
            CudaExpression::KernelIndexVariableCalculation(_, _, _) => {
                debug_assert_eq!(&Type::Primitive(PrimitiveType::Int), target_type);
                target.write_as_value(out, context)?;
                write!(out, " = ")?;
                self.write_as_value(out, context)?;
                Ok(())
            },
            CudaExpression::CudaVariable(_) => {
                unimplemented!()
            }
        }
    }
}

impl CudaWritableExpression for Expression {
    fn write_as_parameter(&self, param_type: &Type, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        CudaExpression::Base(self).write_as_parameter(param_type, out, context)
    }

    fn write_as_value(&self, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        CudaExpression::Base(self).write_as_value(out, context)
    }

    fn assert_expression_valid_as_array(&self) -> Result<CudaVariable, OutputError> {
        assert_expression_valid_as_array(&CudaExpression::Base(self))
    }

    fn write_assignment_to(&self, target_type: &Type, target: &Expression, out: &mut CodeWriter, context: &dyn CudaContext) -> Result<(), OutputError> {
        CudaExpression::Base(self).write_assignment_to(target_type, target, out, context)
    }
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::context::CudaContextImpl;

#[test]
fn test_write_value_context_no_unnecessary_brackets() {
    let expr = Expression::parse(&mut fragment_lex("(a + b * c) / d[e,]")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut counter: u32 = 0;
    let context = CudaContextImpl::new(&[], &mut counter);
    expr.write_as_value(&mut writer, &context).unwrap();
    assert_eq!("(_a_ + _b_ * _c_) * 1/_d_[_e_]", output);
}

#[test]
fn test_write_value_context_function_call() {
    let expr = Expression::parse(&mut fragment_lex("foo[a(b,), c,]")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut counter: u32 = 0;
    let mut context = CudaContextImpl::new(&[], &mut counter);
    let defs = [
        (Name::l("a"), Type::Function(vec![Box::new(Type::Primitive(PrimitiveType::Int))], Some(Box::new(Type::Primitive(PrimitiveType::Int))))),
        (Name::l("foo"), Type::Array(PrimitiveType::Int, 2)),
        (Name::l("b"), Type::Primitive(PrimitiveType::Int)),
        (Name::l("c"), Type::Primitive(PrimitiveType::Int))
    ];
    expr.write_as_value(&mut writer, &context.in_test_subscope(&defs)).unwrap();
    assert_eq!("_foo_[d1_foo_ * _a_(_b_) + _c_]", output);
}

#[test]
fn test_write_value_parameter_context_array() {
    let expr = Expression::parse(&mut fragment_lex("a")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut counter: u32 = 0;
    let context = CudaContextImpl::new(&[], &mut counter);
    expr.write_as_parameter(&Type::View(Box::new(Type::Array(PrimitiveType::Int, 2))), &mut writer, &context).unwrap();
    assert_eq!("_a_, d0_a_, d1_a_", output);
}

#[test]
fn test_write_assignment_to_array_var() {
    let expr = Expression::parse(&mut fragment_lex("a")).unwrap();
    let assignee = Expression::parse(&mut fragment_lex("b")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut counter: u32 = 0;
    let mut context = CudaContextImpl::new(&[], &mut counter);
    let defs = [(Name::l("a"), Type::Array(PrimitiveType::Int, 2))];
    expr.write_assignment_to(&Type::Array(PrimitiveType::Int, 2), &assignee, &mut writer, &context.in_test_subscope(&defs)).unwrap();
    assert_eq!("assert(d0_a_ == d0_b_);
assert(d1_a_ == d1_b_);
checkCudaStatus(cudaMemcpy(_b_, _a_, d0_a_ * sizeof(int), cudaMemcpyDeviceToDevice));", output);
}

#[test]
fn test_write_assignment_to_array_call() {
    let expr = Expression::parse(&mut fragment_lex("foo(a, c,)")).unwrap();
    let assignee = Expression::parse(&mut fragment_lex("b")).unwrap();
    let mut output = "".to_owned();
    let mut target = StringWriter::new(&mut output);
    let mut writer = CodeWriter::new(&mut target);
    let mut counter: u32 = 0;
    let mut context = CudaContextImpl::new(&[], &mut counter);
    let defs = [
        (Name::l("foo"), Type::Function(vec![
            Box::new(Type::View(Box::new(Type::Array(PrimitiveType::Int, 1)))),
            Box::new(Type::Primitive(PrimitiveType::Int))
        ], Some(Box::new(Type::Array(PrimitiveType::Int, 2))))),
        (Name::l("a"), Type::Array(PrimitiveType::Int, 1)),
        (Name::l("c"), Type::Primitive(PrimitiveType::Int))
    ];
    expr.write_assignment_to(&Type::Array(PrimitiveType::Int, 2), &assignee, &mut writer, &context.in_test_subscope(&defs)).unwrap();
    assert_eq!("{
    int* tmp_result = nullptr;
    unsigned int d0_tmp_result = 0;
    unsigned int d1_tmp_result = 0;
    _foo_(_a_, d0_a_, _c_, tmp_result, d0_tmp_result, d1_tmp_result);
    assert(d0_tmp_result == d0_b_);
    assert(d1_tmp_result == d1_b_);
    checkCudaStatus(cudaMemcpy(_b_, tmp_result, d0_tmp_result * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaStatus(cudaFree(tmp_result));
}
", output);
}