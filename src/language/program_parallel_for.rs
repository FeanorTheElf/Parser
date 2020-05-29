use super::backend::{Backend, OutputError, Printable};
use super::error::*;
use super::identifier::{BuiltInIdentifier, Identifier, Name};
use super::position::TextPosition;
use super::program::*;
use super::AstNode;

use super::super::la::prelude::*;
use super::super::util::iterable::LifetimeIterable;

use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::ops::MulAssign;

#[derive(Debug, Clone)]
pub struct ArrayEntryAccess {
    pub pos: TextPosition,
    indices: Vec<Expression>,
    pub alias: Option<Name>,
    pub write: bool,
    matrix_cache: RefCell<Option<Result<Matrix<i32>, CompileError>>>,
}

#[derive(Debug, Clone)]
pub struct ArrayAccessPattern {
    pub pos: TextPosition,
    pub entry_accesses: Vec<ArrayEntryAccess>,
    pub array: Expression,
}

#[derive(Debug, Clone)]
pub struct ParallelFor {
    pub pos: TextPosition,
    pub index_variables: Vec<Declaration>,
    pub access_pattern: Vec<ArrayAccessPattern>,
    pub body: Block,
}

impl AstNode for ParallelFor {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for ParallelFor {
    fn eq(&self, rhs: &ParallelFor) -> bool {
        self.index_variables == rhs.index_variables && self.access_pattern == rhs.access_pattern
    }
}

impl AstNode for ArrayAccessPattern {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for ArrayAccessPattern {
    fn eq(&self, rhs: &ArrayAccessPattern) -> bool {
        self.entry_accesses == rhs.entry_accesses && self.array == rhs.array
    }
}

impl AstNode for ArrayEntryAccess {
    fn pos(&self) -> &TextPosition {
        &self.pos
    }
}

impl PartialEq for ArrayEntryAccess {
    fn eq(&self, rhs: &ArrayEntryAccess) -> bool {
        self.indices == rhs.indices && self.alias == rhs.alias
    }
}

impl Statement for ParallelFor {
    fn dyn_clone(&self) -> Box<dyn Statement> {
        Box::new(self.clone())
    }
}

impl Printable for ParallelFor {
    fn print<'a>(&self, printer: &mut (dyn Backend + 'a)) -> Result<(), OutputError> {
        printer.print_parallel_for_header(self)?;
        self.body.print(printer)?;
        Ok(())
    }
}

impl<'a> LifetimeIterable<'a, Expression> for ParallelFor {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Expression> + 'a)> {
        Box::new(self.access_pattern.iter().map(|pattern| &pattern.array))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Expression> + 'a)> {
        Box::new(
            self.access_pattern
                .iter_mut()
                .map(|pattern| &mut pattern.array),
        )
    }
}

impl<'a> LifetimeIterable<'a, Block> for ParallelFor {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a Block> + 'a)> {
        Box::new(std::iter::once(&self.body))
    }

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut Block> + 'a)> {
        Box::new(std::iter::once(&mut self.body))
    }
}

const ACCESS_MATRIX_AFFINE_COLUMN: usize = 0;

impl ArrayEntryAccess {
    pub fn new(
        pos: TextPosition,
        indices: Vec<Expression>,
        alias: Option<Name>,
        write: bool,
    ) -> Self {
        ArrayEntryAccess {
            pos: pos,
            indices: indices,
            alias: alias,
            write: write,
            matrix_cache: RefCell::new(None),
        }
    }

    fn evaluate_constant(
        expression: &Expression,
    ) -> Result<Result<i32, TextPosition>, CompileError> {
        match expression {
            Expression::Call(call) => {
                if call.function == BuiltInIdentifier::FunctionAdd {
                    call.parameters
                        .iter()
                        .map(|param| Self::evaluate_constant(param))
                        .sum()
                } else if call.function == BuiltInIdentifier::FunctionMul {
                    call.parameters
                        .iter()
                        .map(|param| Self::evaluate_constant(param))
                        .product()
                } else if call.function == BuiltInIdentifier::FunctionUnaryNeg {
                    Self::evaluate_constant(&call.parameters[0])
                        .map(|result_x| result_x.map(|x| -x))
                } else {
                    Err(CompileError::new(
                        call.pos(),
                        format!("Only +, - and * are allowed operations in index expressions"),
                        ErrorType::IllegalPForIndexExpression,
                    ))
                }
            }
            Expression::Literal(lit) => Ok(Ok(lit.value)),
            Expression::Variable(var) => Ok(Err(var.pos().clone())),
        }
    }

    fn add_multiplication_transform_to_transformation_matrix<'a, I>(
        factors: I,
        index_variable_map: &HashMap<&Name, usize>,
        result: &mut VecRefMut<i32>,
    ) -> Result<(), CompileError>
    where
        I: Iterator<Item = &'a Expression>,
    {
        let mut transform = Vector::<i32>::zero(result.len());
        let mut f = 1;
        let mut transform_text_position: Option<TextPosition> = None;
        for factor in factors {
            match (
                Self::evaluate_constant(factor)?,
                transform_text_position.clone(),
            ) {
                (Ok(a), _) => f *= a,
                (Err(new_pos), None) => {
                    Self::add_expression_transform_to_transformation_matrix(
                        factor,
                        index_variable_map,
                        &mut transform.get_mut(..),
                    )?;
                    transform_text_position = Some(new_pos);
                }
                (Err(new_pos), Some(fst_pos)) => {
                    return Err(CompileError::new(&fst_pos,
                        format!("Only affine linear transformations are allowed in pfor index expressions, but the variables at {} and {} are multiplied", fst_pos, new_pos),
                        ErrorType::IllegalPForIndexExpression));
                }
            }
        }
        if transform_text_position.is_some() {
            transform.get_mut(..).mul_assign(f);
            *result += transform.get(..);
        } else {
            result[ACCESS_MATRIX_AFFINE_COLUMN] += f;
        }
        return Ok(());
    }

    fn add_function_application_to_transformation_matrix(
        call: &FunctionCall,
        index_variable_map: &HashMap<&Name, usize>,
        result: &mut VecRefMut<i32>,
    ) -> Result<(), CompileError> {
        if call.function == BuiltInIdentifier::FunctionAdd {
            for param in &call.parameters {
                Self::add_expression_transform_to_transformation_matrix(
                    param,
                    index_variable_map,
                    result,
                )?;
            }
        } else if call.function == BuiltInIdentifier::FunctionMul {
            Self::add_multiplication_transform_to_transformation_matrix(
                call.parameters.iter(),
                index_variable_map,
                result,
            )?;
        } else if call.function == BuiltInIdentifier::FunctionUnaryNeg {
            let mut transform = Vector::<i32>::zero(result.len());
            Self::add_expression_transform_to_transformation_matrix(
                &call.parameters[0],
                index_variable_map,
                &mut transform.get_mut(..),
            )?;
            transform.get_mut(..).mul_assign(-1);
            *result += transform.get(..);
        } else {
            return Err(CompileError::new(
                call.pos(),
                format!("Only +, - and * are allowed operations in index expressions"),
                ErrorType::IllegalPForIndexExpression,
            ));
        }
        return Ok(());
    }

    fn add_expression_transform_to_transformation_matrix(
        expression: &Expression,
        index_variable_map: &HashMap<&Name, usize>,
        result: &mut VecRefMut<i32>,
    ) -> Result<(), CompileError> {
        match expression {
            Expression::Call(call) => {
                Self::add_function_application_to_transformation_matrix(
                    call,
                    index_variable_map,
                    result,
                )?;
            }
            Expression::Literal(lit) => {
                result[ACCESS_MATRIX_AFFINE_COLUMN] += lit.value;
            }
            Expression::Variable(var) => {
                if let Identifier::Name(name) = &var.identifier {
                    if let Some(index) = index_variable_map.get(&name) {
                        let matrix_column = *index + 1;
                        result[matrix_column] += 1;
                    } else {
                        return Err(CompileError::new(var.pos(),
                            format!("Currently only index variables are allowed in pfor index expressions, found {}", name),
                            ErrorType::IllegalPForIndexExpression));
                    }
                } else {
                    return Err(CompileError::new(var.pos(),
                        format!("Currently only index variables are allowed in pfor array index expressions, found builtin identifier"),
                        ErrorType::IllegalPForIndexExpression));
                }
            }
        }
        return Ok(());
    }

    fn calculate_transformation_matrix<'a, I>(
        &self,
        index_variables: I,
    ) -> Result<Matrix<i32>, CompileError>
    where
        I: Iterator<Item = &'a Name>,
    {
        let mut index_variable_map: HashMap<&Name, usize> = HashMap::new();
        for (index, var) in index_variables.enumerate() {
            index_variable_map.insert(&var, index);
        }
        let variables_in = index_variable_map.len();
        let variables_out = self.indices.len();
        let mut result = Matrix::<i32>::zero(variables_out, variables_in + 1);
        for dimension in 0..variables_out {
            let index = &self.indices[dimension];
            Self::add_expression_transform_to_transformation_matrix(
                index,
                &index_variable_map,
                &mut result.get_mut(dimension),
            )?;
        }
        return Ok(result);
    }

    ///
    /// Returns the matrix that represents the affine linear transform from the index variable vector
    /// to the array index vector for a given thread. An error is returned, if the expression is not
    /// an affine linear transform. On success, the result is an array_dimension_count x (index_variable_count + 1)
    /// matrix A, where the column at ACCESS_MATRIX_AFFINE_COLUMN is the translation part and the
    /// rest is the linear transformation (columns are in the same order as the index variables).
    ///
    pub fn get_transformation_matrix<'a, 'b, I>(
        &'a self,
        index_variables: I,
    ) -> Result<Ref<'a, Matrix<i32>>, CompileError>
    where
        I: Iterator<Item = &'b Name>,
    {
        if self.matrix_cache.borrow().is_none() {
            let result = self.calculate_transformation_matrix(index_variables);
            self.matrix_cache.replace(Some(result));
        }
        if self
            .matrix_cache
            .borrow()
            .as_ref()
            .unwrap()
            .as_ref()
            .is_err()
        {
            return Err(self
                .matrix_cache
                .borrow()
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap_err()
                .clone());
        } else {
            return Ok(Ref::map(self.matrix_cache.borrow(), |cache_value| {
                cache_value.as_ref().unwrap().as_ref().unwrap()
            }));
        }
    }

    pub fn get_indices(&self) -> &Vec<Expression> {
        &self.indices
    }
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;
#[cfg(test)]
use super::position::BEGIN;

#[test]
fn test_get_transformation_matrix() {
    let array_entry_access = ArrayEntryAccess::new(
        BEGIN,
        vec![
            Expression::parse(&mut fragment_lex("a + b * 2 - c - 2")).unwrap(),
            Expression::parse(&mut fragment_lex("a * (-1) - (-1) * b + 2 * 3")).unwrap(),
            Expression::parse(&mut fragment_lex("(a - c) * (1 + 2 * (1 + 3))")).unwrap(),
        ],
        None,
        true,
    );

    #[rustfmt::skip]
    let expected: Matrix<i32> = Matrix::new(Box::new([-2, 1, 2, -1, 
                                                      6, -1, 1,  0, 
                                                      0,  9, 0, -9]), 3);
    assert_eq!(
        expected,
        *array_entry_access
            .get_transformation_matrix(
                vec![&Name::l("a"), &Name::l("b"), &Name::l("c")].into_iter()
            )
            .unwrap()
    );
}

#[test]
fn test_get_transformation_matrix_non_affine_transform() {
    let array_entry_access = ArrayEntryAccess::new(
        BEGIN,
        vec![Expression::parse(&mut fragment_lex("-1 + a * (2 + x)")).unwrap()],
        None,
        true,
    );
    assert!(array_entry_access
        .get_transformation_matrix(vec![&Name::l("a"), &Name::l("x")].into_iter())
        .is_err());
}

#[test]
fn test_get_transformation_matrix_non_index_variable() {
    let array_entry_access = ArrayEntryAccess::new(
        BEGIN,
        vec![Expression::parse(&mut fragment_lex("a")).unwrap()],
        None,
        true,
    );
    assert!(array_entry_access
        .get_transformation_matrix(vec![&Name::l("i")].into_iter())
        .is_err());
}
