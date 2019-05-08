pub mod vector;
pub mod matrix;

use vector::{ Indexed, IndexedMut };
use matrix::{ Matrix };
use std::ops::{ Index, IndexMut };
use std::vec::Vec;
use std::collections::HashMap;
use std::option::Option;

fn mult(vec: &mut [f64], factor: f64) {
	for i in 0..vec.len() {
		vec[i] *= factor;
	}
}

fn add(vec: &mut [f64], subtract: &[f64], factor: f64) {
	assert!(vec.len() == subtract.len(), "To subtract two vectors, they must have equal length. Got {} and {}", vec.len(), subtract.len());
	for i in 0..vec.len() {
		vec[i] += subtract[i] * factor;
	}
}

type Tableau = Matrix<f64>;
type TableauRow = [f64];
type BasicVars = Box<[usize]>;

fn simplex(table: &mut Tableau, basic_vars: &mut BasicVars) -> Result<(), ()> {
	while let Some(pivot_col) = find_pivot_col(&table[0]) {
		pivot(table, pivot_col, basic_vars)?;
	}
	return Ok(());
}

fn solve(table: &Tableau) -> Result<Box<[f64]>, ()> {
	let (mut matrix, mut basic_vars) = add_artificials(table);
	simplex(&mut matrix, &mut basic_vars);
	return Ok(extract_solution(&matrix, &basic_vars));
}

fn extract_solution(table: &Tableau, basic_vars: &BasicVars) -> Box<[f64]> {
	assert_eq!(basic_vars.len(), table.rows() - 1);
	let mut result: Box<[f64]> = {
		let mut vec = Vec::new();
		vec.resize(table.cols(), 0.0);
		vec.into_boxed_slice()
	};
	for row_index in 0..basic_vars.len() {
		result[basic_vars[row_index]] = table[row_index + 1][table.cols() - 1];
		debug_assert!(table[row_index + 1][basic_vars[row_index]] == 1.0);
	}
	result[table.cols() - 1] = -table[0][table.cols() - 1];
	return result;
}

fn pivot(table: &mut Tableau, pivot_col_index: usize, basic_vars: &mut BasicVars) -> Result<(), ()> {
	let pivot_row_index: usize = find_pivot_row(table, pivot_col_index)?;
	basic_vars[pivot_row_index - 1] = pivot_col_index;
	eliminate(table, pivot_row_index, pivot_col_index);
	return Ok(());
}

fn eliminate(table: &mut Tableau, row_index: usize, col_index: usize) {
	let pivot_value: f64 = table[row_index][col_index];
	assert!(pivot_value != 0.0);
	mult(&mut table[row_index], 1.0/pivot_value);
	for target_row_index in 0..table.rows() {
		if target_row_index != row_index {
			let factor = table[target_row_index][col_index];
			let (mut target_row, mut base_row) = table.get_mut((target_row_index, row_index));
			add(target_row.as_slice(), base_row.as_slice(), -factor);
		}
	}
}

fn find_pivot_row(table: &Tableau, pivot_col_index: usize) -> Result<usize, ()> {
	let last_col: usize = table.cols() - 1;
	let mut current_min: Option<(usize, f64)> = None;
	for row_index in 1..table.rows() {
		if table[row_index][pivot_col_index] > 0.0 {
			let row_value = table[row_index][last_col]/table[row_index][pivot_col_index];
			if current_min.map_or(true, |(_index, min)| min > row_value) {
				current_min = Some((row_index, row_value));
			}
		}
	}
	if let Some((result, _value)) = current_min {
		return Ok(result);
	} else {
		return Err(());
	}
}

fn find_pivot_col(row: &TableauRow) -> Option<usize> {
	for i in 0..row.len() {
		if row[i] > 0.0 {
			return Some(i);
		}
	}
	return None;
}

fn add_artificials(table: &Tableau) -> (Tableau, BasicVars) {
	let rows = table.rows();
	let cols = table.cols() + rows - 1;
	let mut basic_vars = {
		let mut vec = Vec::new();
		vec.resize(rows - 1, 0);
		vec.into_boxed_slice()
	};
	let mut result = Matrix::new({
		let mut vec = Vec::new();
		vec.resize(rows * cols, 0.0);
		vec.into_boxed_slice()
	}, rows);
	for row_index in 1..rows {
		for col_index in 0..(table.cols() - 1) {
			result[row_index][col_index] = table[row_index][col_index];
		}
		result[row_index][cols - 1] = table[row_index][table.cols() - 1];
		if result[row_index][cols - 1] < 0.0 {
			mult(&mut result[row_index], -1.0);
		}
		let (dst_row, cnt_row) = result.get_mut((0, row_index));
		add(dst_row.into_slice(), cnt_row.into_slice(), 1.0);
	}
	for row_index in 1..rows {
		let basic_var_col = table.cols() + row_index - 2;
		result[row_index][basic_var_col] = 1.0;
		basic_vars[row_index - 1] = basic_var_col;
	}
	result[0][cols - 1] = 0.0;
	return (result, basic_vars);
}

#[test]
fn test_simplex_no_artificials() {
	let mut basic_vars: Box<[usize]> = Box::new([4, 5]);
	let mut m = Matrix::new(Box::new([2.0, 3.0, 4.0, 0.0, 0.0, 0.0,
	                                  3.0, 2.0, 1.0, 1.0, 0.0, 10.0,
									  2.0, 5.0, 3.0, 0.0, 1.0, 15.0]), 3);
	assert_eq!(Ok(()), simplex(&mut m, &mut basic_vars));
	assert_eq!(&[-0.6666666666666669, -3.666666666666667, 0.0, 0.0, -1.3333333333333333, -20.0, 
	             2.333333333333334,   0.3333333333333333, 0.0, 1.0, -0.3333333333333334, 5.0,
				 0.6666666666666669,  1.6666666666666665, 1.0, 0.0, 0.33333333333333326, 5.0], m.data());
	assert_eq!(&[3, 2], &*basic_vars);
}

#[test]
fn test_extract_solution() {
	let mut m = Matrix::new(Box::new([-2.0, -11.0, 0.0, 0.0, -4.0, -60.0, 
	                                  7.0,  1.0,   0.0, 1.0, -1.0, 15.0,
				                      2.0,  5.0,   1.0, 0.0, 1.0,  10.0]), 3);
	let basic_vars: Box<[usize]> = Box::new([3, 2]);
	let solution = extract_solution(&m, &basic_vars);
	assert_eq!(&[0.0, 0.0, 10.0, 15.0, 0.0, 60.0], &*solution);
}

#[test]
fn test_add_artificials() {
	let m = Matrix::new(Box::new([1.0,  2.0,  3.0, 
	                              4.0,  5.0,  6.0, 
								  7.0,  8.0,  -9.0, 
								  10.0, 11.0, 12.0]), 4);
	let (result, basic_vars) = add_artificials(&m);
	assert_eq!(&[7.0,  8.0,  0.0, 0.0, 0.0, 0.0,
	             4.0,  5.0,  1.0, 0.0, 0.0, 6.0,
				 -7.0, -8.0, 0.0, 1.0, 0.0, 9.0,
				 10.0, 11.0, 0.0, 0.0, 1.0, 12.0], result.data());
	assert_eq!(&[2, 3, 4], &*basic_vars);
}

#[test]
fn test_solve() {
	let m = Matrix::new(Box::new([0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
	                              -1.0, 0.0,  1.0, 0.0, 0.0, -1.0, 
								  1.0,  1.0,  0.0, 1.0, 0.0, 4.0,
								  1.0,  -1.0, 0.0, 0.0, 1.0, 0.0]), 4);
	let solution = solve(&m);
	assert_eq!(&[2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], &*solution.unwrap());
}

pub fn experiment() {
	let m = Matrix::new(Box::new([0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
	                              -1.0, 0.0,  1.0, 0.0, 0.0, -1.0, 
								  1.0,  1.0,  0.0, 1.0, 0.0, 4.0,
								  1.0,  -1.0, 0.0, 0.0, 1.0, 0.0]), 4);
	println!("{:?}", solve(&m));
}