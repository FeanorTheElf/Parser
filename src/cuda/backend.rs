use super::super::la::prelude::*;
use super::super::la::rat::r64;
use super::super::language::backend::{Backend, OutputError};
use super::super::language::prelude::*;
use super::super::util::ref_eq::Ref;
use super::function_use_analyser::FunctionUse;
use super::parallel_for_variable_use::ParallelForData;

use std::collections::HashMap;
use std::fmt::Write;

const VAR_PREFIX: &'static str = "_";
const ARRAY_DIM_PREFIX: &'static str = "dim";
const RETURN_PARAM_PREFIX: &'static str = "out_";
const RETURN_PARAM_ARRAY_DIM_PREFIX: &'static str = "outdim";
const QUERIED_ITEMS_COUNT_PREFIX: &'static str = "queried_items_count";
const QUERIED_ITEMS_BEGIN_PREFIX: &'static str = "queried_items_begin";
const QUERIED_ITEMS_END_PREFIX: &'static str = "queried_items_end";
const KERNEL_PREFIX: &'static str = "kernel_";

const INDEX_VAR_BUFFER_NAME: &'static str = "index_var_buffer";
const GRID_DIM_NAME: &'static str = "griddim";
const BLOCK_DIM_NAME: &'static str = "blockdim";

struct OutputThread {
    out: String,
    indent: String,
    block_level: usize,
}

pub struct CudaBackend<'a> {
    output_threads: Vec<OutputThread>,
    function_uses: HashMap<&'a Name, FunctionUse>,
    parallel_for_data: HashMap<Ref<'a, ParallelFor>, ParallelForData<'a>>,
    create_dim_checks: bool,
    result: Vec<String>,
    kernel_index: usize,
}

fn function_not_supported(pos: &TextPosition) -> OutputError {
    OutputError::UnsupportedCode(
        pos.clone(),
        "Cuda backend: type function in this location is not supported".to_owned(),
    )
}

fn jump_label_not_allowed(pos: &TextPosition) -> ! {
    CompileError::new(
        pos,
        format!("Jump Label is not a valid parameter type"),
        ErrorType::TypeError,
    )
    .throw()
}

fn panic_test_type() -> ! {
    panic!("")
}

impl<'a> CudaBackend<'a> {
    fn out(&mut self) -> &mut impl std::fmt::Write {
        &mut self.output_threads.last_mut().unwrap().out
    }

    fn increase_indent(&mut self) {}

    fn decrease_indent(&mut self) {
        self.output_threads.last_mut().unwrap().indent.pop();
    }

    fn is_device_context(&self) -> bool {
        self.output_threads.len() > 1
    }

    fn builtin_string(&self, builtin: BuiltInIdentifier) -> &'static str {
        match builtin {
            BuiltInIdentifier::FunctionAdd => "ADD",
            BuiltInIdentifier::FunctionAnd => "AND",
            BuiltInIdentifier::FunctionEq => "EQ",
            BuiltInIdentifier::FunctionGeq => "GEQ",
            BuiltInIdentifier::FunctionGt => "GT",
            BuiltInIdentifier::FunctionLeq => "LEQ",
            BuiltInIdentifier::FunctionLs => "LS",
            BuiltInIdentifier::FunctionMul => "MUL",
            BuiltInIdentifier::FunctionNeq => "NEQ",
            BuiltInIdentifier::FunctionOr => "OR",
            BuiltInIdentifier::FunctionUnaryDiv => "INV",
            BuiltInIdentifier::FunctionUnaryNeg => "NEG",
            BuiltInIdentifier::FunctionIndex => unimplemented!(),
        }
    }

    ///
    /// We pass
    /// - int as int
    /// - &int as *int
    /// - int[,,] not at all
    /// - &int[,,] as *int, unsigned, unsigned
    ///
    /// We return
    /// - int as int (per return value)
    /// - int[,,] as **int, *unsigned, *unsigned (as parameter pointer)
    /// - &... not at all
    ///
    fn print_parameter_function_definition(
        pos: &TextPosition,
        param_name: &Name,
        param_type: &Type,
    ) -> Result<Vec<String>, OutputError> {
        match param_type {
            Type::Function(_, _) => Err(function_not_supported(pos)),
            Type::JumpLabel => jump_label_not_allowed(pos),
            Type::TestType => panic_test_type(),
            Type::Primitive(PrimitiveType::Int) => {
                Ok(vec![format!("int {}{}", VAR_PREFIX, param_name)])
            }
            Type::Array(PrimitiveType::Int, _) => CompileError::new(
                pos,
                format!("Arrays may only be passed per reference"),
                ErrorType::ArrayParameterPerValue,
            )
            .throw(),
            Type::View(viewn_type) => match &**viewn_type {
                Type::Primitive(PrimitiveType::Int) => {
                    Ok(vec![format!("int * {}{}", VAR_PREFIX, param_name)])
                }
                Type::Array(PrimitiveType::Int, dim) => Ok(std::iter::once(format!(
                    "int * {}{}",
                    VAR_PREFIX, param_name
                ))
                .chain(
                    (0..*dim).map(|i| format!("unsigned {}{}_{}", ARRAY_DIM_PREFIX, i, param_name)),
                )
                .collect::<Vec<String>>()),
                Type::Function(_, _) => Err(function_not_supported(pos)),
                Type::JumpLabel => jump_label_not_allowed(pos),
                Type::TestType => panic_test_type(),
                Type::View(_) => CompileError::new(
                    pos,
                    format!("Views on views are not allowed, use single view &... instead"),
                    ErrorType::ViewOnView,
                )
                .throw(),
            },
        }
    }

    fn print_all_parameters_function_definition<'b, I>(
        name: &Name,
        pos: &TextPosition,
        return_type: &Option<Type>,
        params: I,
    ) -> Result<String, OutputError>
    where
        I: Iterator<Item = (&'b Name, &'b Type)>,
    {
        let mut result = Vec::new();
        match return_type {
            Some(Type::Function(_, _)) => Err(function_not_supported(pos))?,
            Some(Type::JumpLabel) => jump_label_not_allowed(pos),
            Some(Type::TestType) => panic_test_type(),
            Some(Type::Primitive(PrimitiveType::Int)) => {}
            Some(Type::View(_)) => CompileError::new(
                pos,
                format!("Views are not allowed as return type"),
                ErrorType::ViewReturnType,
            )
            .throw(),
            Some(Type::Array(PrimitiveType::Int, dim)) => result.extend(
                std::iter::once(format!("int ** {}{}", RETURN_PARAM_PREFIX, name)).chain(
                    (0..*dim).map(|i| {
                        format!("unsigned * {}{}_{}", RETURN_PARAM_ARRAY_DIM_PREFIX, i, name)
                    }),
                ),
            ),
            None => {}
        }
        for (param_name, param_type) in params {
            result.extend(Self::print_parameter_function_definition(
                pos, param_name, param_type,
            )?);
        }
        return Ok(result.join(", "));
    }

    fn print_parameter_function_call(
        &self,
        expression: &Expression,
        param_type: &Type,
    ) -> Result<Vec<String>, OutputError> {
        let mut result = Vec::new();
        match expression {
            Expression::Call(call) => {
                assert_eq!(param_type, &Type::Primitive(PrimitiveType::Int), "Function argument is anonymus result of function call and not a primitive type, function call should have been extracted at {}", expression.pos());
                unimplemented!();
            }
            Expression::Literal(lit) => {
                result.push(format!("{}", lit.value));
            }
            Expression::Variable(var) => {
                let name = match &var.identifier {
                    Identifier::Name(name) => name,
                    Identifier::BuiltIn(_) => unimplemented!(),
                };
                match &param_type {
                    Type::Function(_, _) => Err(function_not_supported(expression.pos()))?,
                    Type::JumpLabel => jump_label_not_allowed(expression.pos()),
                    Type::TestType => panic_test_type(),
                    Type::Primitive(PrimitiveType::Int) => {
                        result.push(format!("{}{}", VAR_PREFIX, name));
                    }
                    Type::Array(PrimitiveType::Int, _) => CompileError::new(
                        expression.pos(),
                        format!("Arrays may only be passed per reference"),
                        ErrorType::ArrayParameterPerValue,
                    )
                    .throw(),
                    Type::View(viewn_type) => match &**viewn_type {
                        Type::Primitive(PrimitiveType::Int) => {
                            result.push(format!("&{}{}", VAR_PREFIX, name));
                        }
                        Type::Array(PrimitiveType::Int, dim) => {
                            result.push(format!("{}{}", VAR_PREFIX, name));
                            for i in 0..*dim {
                                result.push(format!("unsigned {}{}_{}", ARRAY_DIM_PREFIX, i, name));
                            }
                        }
                        Type::Function(_, _) => Err(function_not_supported(expression.pos()))?,
                        Type::JumpLabel => jump_label_not_allowed(expression.pos()),
                        Type::TestType => panic_test_type(),
                        Type::View(_) => CompileError::new(
                            expression.pos(),
                            format!("Views on views are not allowed, use single view &... instead"),
                            ErrorType::ViewOnView,
                        )
                        .throw(),
                    },
                }
            }
        };
        Ok(result)
    }

    fn print_all_parameters_function_call<'b, I>(
        &self,
        params: I,
    ) -> Result<Vec<String>, OutputError>
    where
        I: Iterator<Item = (&'b Expression, &'b Type)>,
    {
        let mut result = Vec::new();
        for (p, t) in params {
            result.extend(self.print_parameter_function_call(p, t)?);
        }
        return Ok(result);
    }

    fn print_function_call(
        &self,
        call: &FunctionCall,
        param_types: &Vec<Type>,
    ) -> Result<String, OutputError> {
        let function = match &call.function {
            Expression::Variable(var) => &var.identifier,
            _ => {
                return Err(OutputError::UnsupportedCode(call.pos().clone(), format!("Non callable expression. Note that higher order functions are not supported")));
            }
        };
        return Ok(match function {
            Identifier::Name(name) => format!(
                "{}({})",
                name,
                self.print_all_parameters_function_call(
                    call.parameters.iter().zip(param_types.iter())
                )?
                .join(", ")
            ),
            Identifier::BuiltIn(builtin) => format!(
                "{}({})",
                self.builtin_string(*builtin),
                self.print_all_parameters_function_call(
                    call.parameters.iter().zip(param_types.iter())
                )?
                .join(", ")
            ),
        });
    }

    fn newline(&mut self) -> Result<(), OutputError> {
        let last_entry: &mut OutputThread = self.output_threads.last_mut().unwrap();
        write!(&mut last_entry.out, "\r\n{}", last_entry.indent)?;
        Ok(())
    }
}

impl<'a> Backend for CudaBackend<'a> {
    fn print_function_header(&mut self, node: &Function) -> Result<(), OutputError> {
        let function_use = self.function_uses.get(&node.identifier).unwrap();
        let device_modifier = if function_use.device_called {
            "__device__"
        } else {
            ""
        };
        let host_modifier = if function_use.host_called {
            "__host__"
        } else {
            ""
        };
        // if a function returns anything but a primitive type, this is done via mutable parameter pointers
        let return_type = match node.return_type {
            Some(Type::Primitive(PrimitiveType::Int)) => "int",
            _ => "void",
        };
        write!(
            self.out(),
            "{} {} {} {}{}({})",
            device_modifier,
            host_modifier,
            return_type,
            VAR_PREFIX,
            node.identifier,
            Self::print_all_parameters_function_definition(
                &node.identifier,
                node.pos(),
                &node.return_type,
                node.params
                    .iter()
                    .map(|param| (&param.variable, &param.variable_type))
            )?
        )?;
        Ok(())
    }

    fn enter_block(&mut self) -> Result<(), OutputError> {
        write!(self.out(), "{{")?;
        let mut current_out = self.output_threads.last_mut().unwrap();
        current_out.indent.push('\t');
        current_out.block_level += 1;
        Ok(())
    }

    fn exit_block(&mut self) -> Result<(), OutputError> {
        self.newline()?;
        write!(self.out(), "}}")?;
        let is_device_context = self.is_device_context();
        let mut current_out = self.output_threads.last_mut().unwrap();
        current_out.indent.pop();
        if current_out.block_level > 1 || !is_device_context {
            current_out.block_level -= 1;
            return Ok(());
        } else {
            current_out.indent.pop();
            self.newline()?;
            write!(self.out(), "}}")?;
            // close current output thread
            let out = self.output_threads.pop().unwrap();
            self.result.push(out.out);
            return Ok(());
        }
    }

    fn print_parallel_for_header(&mut self, node: &ParallelFor) -> Result<(), OutputError> {
        self.enter_block()?;
        self.newline()?;

        // calculate workspace begin and end
        let workspace_dim = node.index_variables.len();

        let worksize_key_access_matrix = node.access_pattern[0].entry_accesses[0]
            .get_transformation_matrix(node.index_variables.iter().map(|v| &v.variable))
            .internal_error();
        let matrix_linear_part = <Matrix<r64> as From<MatRef<i32>>>::from(
            worksize_key_access_matrix.get((.., 1..worksize_key_access_matrix.cols())),
        );

        assert_eq!(matrix_linear_part.rows(), matrix_linear_part.cols());

        let inverse: Matrix<r64> = matrix_linear_part.get((.., ..)).invert()
            .expect("First array access is not injective, must be to determine the correct range of threads to start");

        let print_dim_linear_combination =
            |coeffs: VecRef<r64>, translation: VecRef<r64>, array: &Name, use_nonzero_corner: Vec<bool>| {
                let n = coeffs.len();
                coeffs
                    .get(0..(n - 1))
                    .into_iter()
                    .zip(translation.into_iter())
                    .enumerate()
                    .map(|(i, (coeff, trans))| {
                        if use_nonzero_corner[i] {
                            format!(
                                "{}./{}. * ({}{}_{} / {}{}_{} - {}./{}.)",
                                coeff.num(),
                                coeff.den(),
                                ARRAY_DIM_PREFIX,
                                i,
                                array,
                                ARRAY_DIM_PREFIX,
                                i + 1,
                                array,
                                trans.num(),
                                trans.den()
                            )
                        } else {
                            format!(
                                "{}./{}. * (- {}./{}.)",
                                coeff.num(),
                                coeff.den(),
                                trans.num(),
                                trans.den()
                            )
                        }
                    })
                    .chain(
                        std::iter::once(coeffs[n - 1])
                            .zip(std::iter::once(translation[n - 1]))
                            .map(|(coeff, trans)| {
                                if use_nonzero_corner[n - 1] {
                                    format!(
                                        "{}./{}. * ({}{}_{} - {}./{}.)",
                                        coeff.num(),
                                        coeff.den(),
                                        ARRAY_DIM_PREFIX,
                                        n - 1,
                                        array,
                                        trans.num(),
                                        trans.den()
                                    )
                                } else {
                                    format!(
                                        "{}./{}. * (-{}./{}.)",
                                        coeff.num(),
                                        coeff.den(),
                                        trans.num(),
                                        trans.den()
                                    )
                                }
                            }),
                    )
                    .collect::<Vec<String>>()
                    .join(" + ")
            };

        let worksize_key_array = match node.access_pattern[0]
            .array
            .expect_identifier()
            .internal_error()
        {
            Identifier::Name(name) => name,
            Identifier::BuiltIn(_) => panic!(""),
        };
        let translation = <Matrix<r64> as From<MatRef<i32>>>::from(worksize_key_access_matrix.get((.., 0..1))).into_column();
        for i in 0..workspace_dim {
            // TODO: taking only (0, ..., 0) and (dim_1, ..., dim_n) is not sufficient to estimate the size
            let expr = print_dim_linear_combination(inverse.get(i), translation.get(..), worksize_key_array, std::iter::repeat(true).take(workspace_dim).collect());
            self.newline()?;
            write!(
                self.out(),
                "const int {}{} = (int) min(0, {});",
                QUERIED_ITEMS_BEGIN_PREFIX,
                i,
                expr
            )?;
            self.newline()?;
            write!(
                self.out(),
                "const int {}{} = 1 + (int) max(0, {});",
                QUERIED_ITEMS_END_PREFIX,
                i,
                expr
            )?;
        }

        // calculate workspace size
        self.newline()?;
        write!(
            self.out(),
            "const int {}{} = 1;",
            QUERIED_ITEMS_COUNT_PREFIX,
            workspace_dim
        )?;
        for i in 1..=workspace_dim {
            self.newline()?;
            write!(
                self.out(),
                "const int {}{} = {}{} * ({}{} - {}{})",
                QUERIED_ITEMS_COUNT_PREFIX,
                (workspace_dim - i),
                QUERIED_ITEMS_COUNT_PREFIX,
                (workspace_dim - i + 1),
                QUERIED_ITEMS_END_PREFIX,
                workspace_dim - i,
                QUERIED_ITEMS_BEGIN_PREFIX,
                workspace_dim - i
            )?;
        }

        // initialize workspace size variables
        write!(self.out(), "dim3 {} = dim3();", GRID_DIM_NAME)?;
        self.newline()?;
        write!(self.out(), "dim3 {} = dim3();", BLOCK_DIM_NAME)?;
        self.newline()?;

        let kernel_num = self.kernel_index;
        self.kernel_index += 1;

        let variable_usage_info_in_pfor = self.parallel_for_data.get(&Ref::from(node)).unwrap();

        // get formal parameters
        let worksize_parameters = (0..workspace_dim)
            .map(|d| {
                (
                    Name::new(format!("{}{}", QUERIED_ITEMS_COUNT_PREFIX, d), 0),
                    Type::Primitive(PrimitiveType::Int),
                )
            })
            .chain((0..workspace_dim).map(|d| {
                (
                    Name::new(format!("{}{}", QUERIED_ITEMS_BEGIN_PREFIX, d), 0),
                    Type::Primitive(PrimitiveType::Int),
                )
            }));
        let formal_parameters: Vec<(Name, Type)> = variable_usage_info_in_pfor
            .used_outer_variables
            .iter()
            .map(|definition| (definition.get_name().clone(), definition.calc_type()))
            .chain(worksize_parameters)
            .collect();

        let formal_param_iter = formal_parameters
            .iter()
            .map(|(param, param_type)| (param, param_type));
        let formal_param_string = Self::print_all_parameters_function_definition(
            &Name::new("kernel".to_owned(), kernel_num as u32),
            node.pos(),
            &None,
            formal_param_iter,
        )?;

        // get actual parameters
        let actual_parameters: Vec<(Expression, Type)> = formal_parameters
            .into_iter()
            .map(|(param, param_type)| {
                (
                    Expression::Variable(Variable {
                        pos: node.pos().clone(),
                        identifier: Identifier::Name(param),
                    }),
                    param_type,
                )
            })
            .collect();
        let actual_param_iter = actual_parameters
            .iter()
            .map(|(param, param_type)| (param, param_type));
        let acutal_param_string = self
            .print_all_parameters_function_call(actual_param_iter)?
            .join(", ");

        // print function call
        write!(
            self.out(),
            "{}{}<<<griddim_, blockdim_>>>({});",
            KERNEL_PREFIX,
            kernel_num,
            acutal_param_string
        )?;

        self.exit_block();

        // now we continue in the kernel function
        self.output_threads.push(OutputThread {
            block_level: 0,
            indent: "".to_owned(),
            out: "".to_owned(),
        });

        // write function signature
        write!(
            self.out(),
            "__global__ void {}{}({})",
            KERNEL_PREFIX,
            kernel_num,
            formal_param_string
        )?;
        self.enter_block()?;

        // initialize index variables
        self.newline()?;
        write!(self.out(), "int {} = threadIdx.x;", INDEX_VAR_BUFFER_NAME)?;
        self.newline()?;
        write!(
            self.out(),
            "const int {}{} = 1;",
            QUERIED_ITEMS_COUNT_PREFIX,
            workspace_dim
        )?;
        for i in 0..workspace_dim {
            self.newline()?;
            write!(
                self.out(),
                "const int {}{} = {} / {}{} + {}{};",
                VAR_PREFIX,
                node.index_variables[i].variable,
                INDEX_VAR_BUFFER_NAME,
                QUERIED_ITEMS_COUNT_PREFIX,
                i + 1,
                QUERIED_ITEMS_BEGIN_PREFIX,
                i
            );
            self.newline()?;
            write!(
                self.out(),
                "{} -= {}{} -  {}{}",
                INDEX_VAR_BUFFER_NAME,
                VAR_PREFIX,
                node.index_variables[i].variable,
                QUERIED_ITEMS_BEGIN_PREFIX,
                i
            );
        }

        Ok(())
    }

    fn print_label(&mut self, node: &Label) -> Result<(), OutputError> {
        self.newline()?;
        write!(self.out(), "{}{}: ;", VAR_PREFIX, node.label)?;
        Ok(())
    }

    fn print_goto(&mut self, node: &Goto) -> Result<(), OutputError> {
        self.newline()?;
        write!(self.out(), "goto {}{};", VAR_PREFIX, node.target)?;
        Ok(())
    }

    fn print_if_header(&mut self, node: &If) -> Result<(), OutputError> {
        unimplemented!();
        Ok(())
    }

    fn print_while_header(&mut self, node: &While) -> Result<(), OutputError> {
        unimplemented!();
        Ok(())
    }

    fn print_return(&mut self, node: &Return) -> Result<(), OutputError> {
        unimplemented!();
        Ok(())
    }

    fn print_expression(&mut self, node: &Expression) -> Result<(), OutputError> {
        unimplemented!();
        Ok(())
    }

    fn print_assignment(&mut self, node: &Assignment) -> Result<(), OutputError> {
        unimplemented!();
        Ok(())
    }

    fn print_declaration(&mut self, node: &LocalVariableDeclaration) -> Result<(), OutputError> {
        unimplemented!();
        Ok(())
    }
}
