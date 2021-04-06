use super::super::language::prelude::*;
use super::super::language::ast_pfor::*;
use super::super::language::gwaihir_writer::*;
use feanor_la::la::mat::*;
use feanor_la::algebra::rat::*;

pub fn check_program_pfor_data_races(program: &Program) -> Result<(), CompileError> {
    program.traverse_preorder(&mut |block: &Block, _scopes| {
        for statement in block.statements() {
            if let Some(pfor) = statement.any().downcast_ref::<ParallelFor>() {
                check_pfor_data_races(pfor)?;
            }
        }
        return Ok(());
    })
}

fn check_pfor_data_races(pfor: &ParallelFor) -> Result<(), CompileError> {
    let index_var_count = pfor.index_variables.len();
    for access_pattern in pfor.access_pattern()? {
        for (i, entry1) in access_pattern.accessed_entries().enumerate() {
            for (j, entry2) in access_pattern.accessed_entries().enumerate() {

                let one_write = entry1.writeable || entry2.writeable;
                let transform1 = entry1.get_transform();
                let transform2 = entry2.get_transform();

                if one_write {
                    let index_var_transform1 = transform1.linear_part.submatrix(.., ..index_var_count);
                    let index_var_transform2 = transform2.linear_part.submatrix(.., ..index_var_count);
                    let linear_part1 = expect_integral_matrix(index_var_transform1, entry1.pos())?;
                    let linear_part2 = expect_integral_matrix(index_var_transform2, entry2.pos())?;

                    let other_var_transform1 = transform1.linear_part.submatrix(.., index_var_count..);
                    let other_var_transform2 = transform2.linear_part.submatrix(.., index_var_count..);
                    let other_var_difference = expect_integral_matrix(
                        other_var_transform1 - other_var_transform2,
                        entry1.pos()
                    )?;

                    let affine_difference = expect_integral_matrix(
                        Matrix::col_vec(transform1.affine_part.as_ref() - transform2.affine_part.as_ref()), 
                        entry1.pos()
                    )?;

                    let collision = get_collision(
                        linear_part1.as_ref(), 
                        linear_part2.as_ref(),
                        other_var_difference,
                        affine_difference.col(0), 
                        i != j
                    );
                    if let Some((i1, i2, v)) = collision {
                        return Err(CompileError::pfor_collision(
                            entry1.pos(), entry2.pos(), i1, i2, v, pfor
                        ));
                    }
                }
            }
        }
    }
    return Ok(());
}

struct Joined<I>
    where I: Iterator + Clone, I::Item: std::fmt::Display
{
    it: I,
    separator: &'static str
}

impl<I> std::fmt::Display for Joined<I>
    where I: Iterator + Clone, I::Item: std::fmt::Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut it_copy = self.it.clone();
        if let Some(x) = it_copy.next() {
            write!(f, "{}", x)?;
        }
        while let Some(x) = it_copy.next() {
            write!(f, "{}{}", self.separator, x)?;
        }
        return Ok(());
    }
}

impl CompileError {

    fn no_integral_range(pos: &TextPosition, index: usize) -> CompileError {
        CompileError::new(
            pos, 
            format!("The {}-th index expression must be guaranteed to take integral values for all values of the referenced variables", index),
            ErrorType::IllegalPForIndexExpression
        )
    }

    fn pfor_collision<V, W, U>(
        access1: &TextPosition, 
        access2: &TextPosition, 
        vars1: Vector<V, i32>, 
        vars2: Vector<W, i32>, 
        shared_vars: Vector<U, i32>, 
        pfor: &ParallelFor
    ) -> CompileError 
        where V: VectorView<i32>, W: VectorView<i32>, U: VectorView<i32>
    {
        let var_assignment1 = pfor.index_variables.iter().enumerate()
            .map(|(i, variable)| format!("{} = {}", DisplayWrapper::from(&variable.name), *vars1.at(i)));

        let var_assignment2 = pfor.index_variables.iter().enumerate()
            .map(|(i, variable)| format!("{} = {}", DisplayWrapper::from(&variable.name), *vars2.at(i)));

        let shared_var_assignment = pfor.used_variables.iter().enumerate()
            .map(|(i, expr)| format!("{} = {}", DisplayWrapper::from(expr), *shared_vars.at(i)));

        CompileError::new(
            access1,
            format!(
                "This index access collides with the index access defined at {}, e.g. for variable assignment {} and {} in the context {{ {} }}", 
                access2,
                Joined { it: var_assignment1, separator: ", " },
                Joined { it: var_assignment2, separator: ", " },
                Joined { it: shared_var_assignment, separator: ", " }
            ),
            ErrorType::PForAccessCollision
        )
    }
}

fn expect_integral_matrix<M>(a: Matrix<M, r64>, pos: &TextPosition) -> Result<Matrix<MatrixOwned<i32>, i32>, CompileError>
    where M: MatrixView<r64>
{
    let mut error = None;
    let result = Matrix::from_fn(a.row_count(), a.col_count(), |i, j| {
        let mut value = *a.at(i, j);
        value.reduce();
        if value.den() != 1 {
            error = Some(CompileError::no_integral_range(pos, i));
        }
        return value.num() as i32;
    });
    if let Some(err) = error {
        return Err(err);
    } else {
        return Ok(result);
    }
}

type VariableAssignment = Vector<VectorOwned<i32>, i32>;

///
/// Tries to find a collision between two accesses, i.e. given
/// index var transform matrices A1, A2, other var transform matrices
/// B1, B2 and vectors c1, c2 such that the following holds:
/// 
/// A1 * i1 + B1 * v + c1 = A2 * i2 + B2 * v + c2
/// 
/// for integral index var assignments i1 resp. i2 and an integral
/// other var assignment v.
/// 
/// Instead of B1 and B2 resp. c1 and c2 this function accepts
/// their difference for simplicity.
/// 
/// The returned values are (i1, i2, v) resp. None if such an assignment
/// does not exist.
/// 
#[allow(non_snake_case)]
fn get_collision<M, N, P, V>(
    index_var_transform1: Matrix<M, i32>,
    index_var_transform2: Matrix<N, i32>,
    other_var_transform_difference: Matrix<P, i32>,
    affine_difference: Vector<V, i32>,
    same_index_collides: bool,
) -> Option<(VariableAssignment, VariableAssignment, VariableAssignment)> 
    where M: MatrixView<i32>, N: MatrixView<i32>, P: MatrixView<i32>, V: VectorView<i32>
{
    assert_eq!(index_var_transform1.col_count(), index_var_transform2.col_count());
    assert_eq!(index_var_transform1.row_count(), affine_difference.len());
    assert_eq!(index_var_transform1.row_count(), index_var_transform2.row_count());
    assert_eq!(index_var_transform1.row_count(), other_var_transform_difference.row_count());

    let index_variables_in = index_var_transform1.col_count();
    let other_variables_in = other_var_transform_difference.col_count();
    let variables_out = index_var_transform1.row_count();

    let degrees_of_freedom = 2 * index_variables_in + other_variables_in;

    let mut joined_transform = Matrix::zero(variables_out, degrees_of_freedom).to_owned();

    // fill the matrix correctly
    {
        let mut left_half = joined_transform.submatrix_mut(.., ..index_variables_in);
        left_half.assign(index_var_transform1);

        let mut right_half = joined_transform.submatrix_mut(.., index_variables_in..(2 * index_variables_in));
        right_half.assign(index_var_transform2);
        right_half *= -1;

        let mut other_var_part = joined_transform.submatrix_mut(.., (2 * index_variables_in)..);
        other_var_part.assign(other_var_transform_difference);
    }
    let mut joined_translate = Matrix::col_vec(affine_difference.to_owned());
    joined_translate *= -1;

    let mut iL = Matrix::<_, i32>::identity(variables_out, variables_out);
    let mut iR = Matrix::<_, i32>::identity(degrees_of_freedom, degrees_of_freedom);

    feanor_la::algebra::diophantine::partial_smith(
        joined_transform.as_mut(),
        iL.as_mut(),
        iR.as_mut(),
        0,
    );

    let x_mat = iL * joined_translate;
    let x = x_mat.col(0);
    let mut y = Vector::zero(degrees_of_freedom).to_owned();
    let mut free_dimensions: Vec<usize> = Vec::new();

    for i in 0..variables_out.min(degrees_of_freedom) {
        if *joined_transform.at(i, i) == 0 && *x.at(i) != 0 {
            return None;
        } else if *joined_transform.at(i, i) == 0 && *x.at(i) == 0 {
            // in this case, we can choose variable i as we want
            // if it is an index variable, use it for later different-index-collision
            // check, otherwise ignore it as we can choose the other variables always
            // arbitrarily
            if i <= 2 * index_variables_in {
                free_dimensions.push(i);
            }
            // we leave y[i] to be zero, though any value would be ok
        } else if *x.at(i) % *joined_transform.at(i, i) != 0 {
            return None;
        } else {
            *y.at_mut(i) = *x.at(i) / *joined_transform.at(i, i);
        }
    }

    // We are done, since we found a solution
    if same_index_collides {
        let result = iR.as_ref() * Matrix::col_vec(y);
        return Some((
            result.col(0).subvector(..index_variables_in).to_owned(),
            result.col(0).subvector(index_variables_in..(2 * index_variables_in)).to_owned(),
            result.col(0).subvector((2 * index_variables_in)..).to_owned()
        ));
    }

    // We have to check whether a solution space looks like (...a... ...a...) * Z^n + ... + (...c... ...c...) * Z^n,
    // so there is a solution but only one where the same thread collides with itself

    // the diagonal matrix has only up to out_variables many entries, so the entries for the other input
    // variables are implicitly zero, so free dimensions
    free_dimensions.extend(variables_out..degrees_of_freedom);

    // check all solution space basis vectors
    for free_dim in free_dimensions {
        *y.at_mut(free_dim) = 1;
        let basis_vector_mat = iR.as_ref() * Matrix::col_vec(y.as_ref());
        let basis_vector = basis_vector_mat.col(0);
        if basis_vector.subvector(..index_variables_in) != basis_vector.subvector(index_variables_in..(2 * index_variables_in)) {
            return Some((
                basis_vector.subvector(..index_variables_in).to_owned(),
                basis_vector.subvector(index_variables_in..(2 * index_variables_in)).to_owned(),
                basis_vector.subvector((2 * index_variables_in)..).to_owned()
            ));
        }
        *y.at_mut(free_dim) = 0;
    }

    return None;
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::{ Parser, ParserContext };

#[test]
fn test_check_collision() {
    let mut context = ParserContext::new();
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "pfor a: int, with write this[2 * a, ] as x, read this[2 * a + 1, ] as y, in array {}",
    ), &mut context)
    .unwrap();

    assert_eq!((), check_pfor_data_races(&pfor).unwrap());
}

#[test]
fn test_check_collision_with_collision() {
    let mut context = ParserContext::new();
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "pfor a: int, b: int, with write this[2 * a + b, ] as x, read this[2 * a + 1, ] as y, in array {}",
    ), &mut context)
    .unwrap();

    assert!(check_pfor_data_races(&pfor).is_err());
}

#[test]
fn test_check_collision_collision_through_outer_var() {
    let mut context = ParserContext::new();
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "pfor a: int, with write this[a, n,] as x, read this[a, 2,] as y, in array {}",
    ), &mut context)
    .unwrap();

    assert!(check_pfor_data_races(&pfor).is_err());
}

#[test]
fn test_check_collision_self_collision() {
    let mut context = ParserContext::new();
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "pfor a: int, b: int, with write this[a + b,] as x, in array {}",
    ), &mut context)
    .unwrap();

    assert!(check_pfor_data_races(&pfor).is_err());
}

#[test]
fn test_check_collision_no_self_collision() {
    let mut context = ParserContext::new();
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "pfor a: int, b: int, with write this[a + b, a,] as x, in array {}",
    ), &mut context)
    .unwrap();

    assert_eq!((), check_pfor_data_races(&pfor).unwrap());
}