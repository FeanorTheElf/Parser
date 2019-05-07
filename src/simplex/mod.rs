use std::ops::{ Index, IndexMut };
use std::vec::Vec;
use std::option::Option;

#[derive(Debug)]
struct Matrix<T> {
	rows: usize,
	cols: usize,
	data: Box<[T]>
}

impl<T> Matrix<T> {

	fn new(data: Box<[T]>, rows: usize) -> Matrix<T> {
		assert!(data.len() % rows == 0, "Data length must be a multiple of row count, but got {} and {}", data.len(), rows);
		Matrix {
			rows: rows,
			cols: data.len() / rows,
			data: data
		}
	}

	fn assert_in_range(&self, row_index: usize) {
		assert!(row_index >= 0, "Expected row index {} to be greater than 0", row_index);
		assert!(row_index < self.rows, "Expected row index {} to be smaller than the row count {}", row_index, self.rows);
	}
	
	fn two_rows(&mut self, fst_index: usize, snd_index: usize) -> (&mut [T], &mut [T]) {
		self.assert_in_range(fst_index);
		self.assert_in_range(snd_index);
		assert!(fst_index != snd_index, "When borrowing two rows, their indices must be different, got {}", fst_index);
		if fst_index < snd_index {
			let part: &mut [T] = &mut self.data[(fst_index * self.cols)..((snd_index + 1) * self.cols)];
			let (fst_row, rest) = part.split_at_mut(self.cols);
			let second_row_start_index = rest.len() - self.cols;
			return (fst_row, &mut rest[second_row_start_index..]);
		} else {
			let part: &mut [T] = &mut self.data[(snd_index * self.cols)..((fst_index + 1) * self.cols)];
			let (snd_row, rest) = part.split_at_mut(self.cols);
			let first_row_start_index = rest.len() - self.cols;
			return (&mut rest[first_row_start_index..], snd_row);
		}
	}
}

impl<T> Index<usize> for Matrix<T> {
	type Output = [T];

	fn index(&self, row_index: usize) -> &[T] {
		self.assert_in_range(row_index);
		&self.data[(row_index * self.cols)..((row_index + 1) * self.cols)]
	}
}

impl<T> IndexMut<usize> for Matrix<T> {

	fn index_mut(&mut self, row_index: usize) -> &mut [T] {
		self.assert_in_range(row_index);
		&mut self.data[(row_index * self.cols)..((row_index + 1) * self.cols)]
	}
}

fn mult(vec: &mut [f64], factor: f64) {
	for i in 0..vec.len() {
		vec[i] *= factor;
	}
}

fn subtract(vec: &mut [f64], subtract: &[f64], factor: f64) {
	assert!(vec.len() == subtract.len(), "To subtract two vectors, they must have equal length. Got {} and {}", vec.len(), subtract.len());
	for i in 0..vec.len() {
		vec[i] -= subtract[i] * factor;
	}
}

type Tableau = Matrix<f64>;
type TableauRow = [f64];
type BasicVars = Vec<usize>;

fn simplex(table: &mut Tableau) -> Result<(), ()> {
	while let Some(pivot_col) = find_pivot_col(&table[0]) {
		pivot(table, pivot_col)?;
	}
	return Ok(());
}

fn pivot(table: &mut Tableau, pivot_col_index: usize) -> Result<(), ()> {
	let pivot_row_index: usize = find_pivot_row(table, pivot_col_index)?;
	eliminate(table, pivot_row_index, pivot_col_index);
	return Ok(());
}

fn eliminate(table: &mut Tableau, row_index: usize, col_index: usize) {
	let pivot_value: f64 = table[row_index][col_index];
	assert!(pivot_value != 0.0);
	mult(&mut table[row_index], 1.0/pivot_value);
	for target_row_index in 0..table.rows {
		if target_row_index != row_index {
			let factor = table[target_row_index][col_index];
			let (target_row, base_row) = table.two_rows(target_row_index, row_index);
			subtract(target_row, base_row, factor);		
		}
	}
}

fn find_pivot_row(table: &Tableau, pivot_col_index: usize) -> Result<usize, ()> {
	let last_col: usize = table.cols - 1;
	let mut current_min: Option<(usize, f64)> = None;
	for row_index in 1..table.rows {
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

#[test]
fn test_simplex_no_artificials() {
	let mut m = Matrix::new(Box::new([2.0, 3.0, 4.0, 0.0, 0.0, 0.0,
	                                  3.0, 2.0, 1.0, 1.0, 0.0, 10.0,
									  2.0, 5.0, 3.0, 0.0, 1.0, 15.0]), 3);
	assert_eq!(Ok(()), simplex(&mut m));
	assert_eq!(&[-0.6666666666666669, -3.666666666666667, 0.0, 0.0, -1.3333333333333333, -20.0, 
	             2.333333333333334,   0.3333333333333333, 0.0, 1.0, -0.3333333333333334, 5.0,
				 0.6666666666666669,  1.6666666666666665, 1.0, 0.0, 0.33333333333333326, 5.0], &*m.data);
}

#[test]
fn test_matrix_two_rows() {
	let mut m = Matrix::new(Box::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), 4);
	assert_eq!(3, m.cols);
	assert_eq!(4, m.rows);
	{
		let (fst_row, snd_row) = m.two_rows(0, 2);
		assert_eq!(&[1, 2, 3], fst_row);
		assert_eq!(&[7, 8, 9], snd_row);
	}
	{
		let (fst_row, snd_row) = m.two_rows(3, 2);
		assert_eq!(&[10, 11, 12], fst_row);
		assert_eq!(&[7, 8, 9], snd_row);
	}
}

pub fn experiment() {
//	let mut m = Matrix::new(Box::new([2.0, 3.0, 4.0, 0.0, 0.0, 0.0,
//	                                  3.0, 2.0, 1.0, 1.0, 0.0, 10.0,
//									  2.0, 5.0, 3.0, 0.0, 1.0, 15.0]), 3);
	let mut m = Matrix::new(Box::new([1.0,  0.0, 0.0,
	                                  2.0,  1.0, 10.0]), 2);
	println!("{:?}", simplex(&mut m));
	println!("{:?}", m);
}