use super::super::la::prelude::*;
use super::super::language::prelude::*;

#[allow(unused)]
pub fn check_program_pfor_data_races(program: &Program) -> Result<(), CompileError> {
    for item in &program.items {
        if let Some(body) = &item.body {
            call_for_pfor_in_block(body, &mut check_pfor_data_races)?;
        }
    }
    return Ok(());
}

fn call_for_pfor_in_block<F>(block: &Block, f: &mut F) -> Result<(), CompileError>
where
    F: FnMut(&ParallelFor) -> Result<(), CompileError>,
{
    for statement in &block.statements {
        if let Some(pfor) = statement.dynamic().downcast_ref::<ParallelFor>() {
            f(pfor)?;
        }
        for block in statement.iter() {
            call_for_pfor_in_block(&block, f)?;
        }
    }
    return Ok(());
}

fn check_pfor_data_races(pfor: &ParallelFor) -> Result<(), CompileError> {
    for access_pattern in &pfor.access_pattern {
        for i in 0..access_pattern.entry_accesses.len() {
            for j in i..access_pattern.entry_accesses.len() {
                let entry1 = &access_pattern.entry_accesses[i];
                let entry2 = &access_pattern.entry_accesses[j];
                let one_write = entry1.write || entry2.write;
                let transform1 = entry1.get_transformation_matrix(
                    pfor.index_variables.iter().map(|var| &var.variable),
                )?;
                let transform2 = entry2.get_transformation_matrix(
                    pfor.index_variables.iter().map(|var| &var.variable),
                )?;
                if one_write {
                    let collision =
                        get_collision(transform1.get((.., ..)), transform2.get((.., ..)), i != j);
                    if let Some((x, y)) = collision {
                        return Err(CompileError::new(entry1.pos(),
                            format!("Array index accesses collide, defined at {} and {}. Collision happens e.g. for index variable values {} and {}", entry1.pos(), entry2.pos(), x.get(..), y.get(..)),
                            ErrorType::PForAccessCollision));
                    }
                }
            }
        }
    }
    return Ok(());
}

// The first column is for the translation
#[allow(non_snake_case)]
fn get_collision(
    transform1: MatRef<i32>,
    transform2: MatRef<i32>,
    same_index_collides: bool,
) -> Option<(Vector<i32>, Vector<i32>)> {
    debug_assert_eq!(transform1.cols(), transform2.cols());
    debug_assert_eq!(transform1.rows(), transform2.rows());
    let variables_in = transform1.cols() - 1;
    let variables_out = transform1.rows();

    let mut joined_transform = Matrix::<i32>::zero(variables_out, 2 * variables_in);
    let mut left_half = joined_transform.get_mut((.., 0..variables_in));
    left_half.assign_copy(transform1.get((.., 1..(variables_in + 1))));
    let mut right_half = joined_transform.get_mut((.., variables_in..(2 * variables_in)));
    right_half.assign_copy(transform2.get((.., 1..variables_in + 1)));
    right_half *= -1;

    let mut joined_translate = Matrix::<i32>::zero(variables_out, 1);
    let mut translate = joined_translate.get_mut((.., ..));
    translate -= transform1.get((.., 0..1));
    translate += transform2.get((.., 0..1));

    let mut iL = Matrix::<i32>::identity(variables_out);
    let mut iR = Matrix::<i32>::identity(2 * variables_in);
    diophantine::smith(
        &mut joined_transform.get_mut((.., ..)),
        &mut iL.get_mut((.., ..)),
        &mut iR.get_mut((.., ..)),
        0,
    );

    let x = iL.get((.., ..)) * joined_translate.into_column().get(..);
    let mut y = Vector::<i32>::zero(2 * variables_in);
    let mut free_dimensions: Vec<usize> = Vec::new();
    for i in 0..variables_out.min(2 * variables_in) {
        if joined_transform[i][i] == 0 && x[i] != 0 {
            return None;
        } else if joined_transform[i][i] == 0 && x[i] == 0 {
            free_dimensions.push(i);
        // y[i] is already zero
        } else if x[i] % joined_transform[i][i] != 0 {
            return None;
        } else {
            y[i] = x[i] / joined_transform[i][i];
        }
    }
    // We are done, since we found a solution
    if same_index_collides {
        let result = iR.get((.., ..)) * y.get(..);
        return Some((
            result.get(0..variables_in).to_owned(),
            result.get(variables_in..(2 * variables_in)).to_owned(),
        ));
    }
    // We have to check whether a solution space looks like (...a... ...a...) * Z^n + ... + (...c... ...c...) * Z^n,
    // so there is a solution but only one where the same thread collides with itself
    free_dimensions.extend(variables_out..(2 * variables_in));
    // check all solution space basis vectors
    for free_dim in free_dimensions {
        y[free_dim] = 1;
        let basis_vector = iR.get((.., ..)) * y.get(..);
        if basis_vector.get(0..variables_in) != basis_vector.get(variables_in..(2 * variables_in)) {
            return Some((
                basis_vector.get(0..variables_in).to_owned(),
                basis_vector
                    .get(variables_in..(2 * variables_in))
                    .to_owned(),
            ));
        }
        y[free_dim] = 0;
    }
    return None;
}

#[cfg(test)]
use super::super::lexer::lexer::fragment_lex;
#[cfg(test)]
use super::super::parser::Parser;

#[test]
fn test_check_collision() {
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "pfor a: int, with write this[2 * a, ], read this[2 * a + 1, ], in array {}",
    ))
    .unwrap();
    assert_eq!((), check_pfor_data_races(&pfor).internal_error());
}

#[test]
fn test_check_collision_with_collision() {
    let pfor = ParallelFor::parse(&mut fragment_lex(
        "pfor a: int, b: int, with write this[2 * a + b, ], read this[2 * a + 1, ], in array {}",
    ))
    .unwrap();
    assert!(check_pfor_data_races(&pfor).is_err());
}
