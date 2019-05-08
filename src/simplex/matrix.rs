use std::ops::{ Index, IndexMut, Add, Mul, AddAssign, MulAssign, SubAssign };
use std::cmp::{ min, max };
use super::vector::{ Indexed, IndexedMut, FixedLen };

#[derive(Debug)]
pub struct Matrix<T> {
	rows: usize,
	cols: usize,
	data: Box<[T]>
}

#[derive(Debug)]
pub struct MRow<'a, T> {
	data: &'a [T]
}

#[derive(Debug)]
pub struct MRowMut<'a, T> {
	data: &'a mut [T]
}

impl<T> Matrix<T> {

	pub fn new(data: Box<[T]>, rows: usize) -> Matrix<T> {
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
	
	pub fn cols(&self) -> usize {
		self.cols
	}

	pub fn rows(&self) -> usize {
		self.rows
	}

	pub fn data(&self) -> &[T] {
		&*self.data
	}
}

impl<T> Index<usize> for Matrix<T> {
	type Output = [T];

	fn index(&self, row_index: usize) -> &[T] {
		self.get(row_index).get_slice()
	}
}

impl<T> IndexMut<usize> for Matrix<T> {

	fn index_mut(&mut self, row_index: usize) -> &mut [T] {
		self.get_mut(row_index).into_slice()
	}
}

impl<'a, T: 'a> Indexed<'a, usize> for Matrix<T> {
	type Output = MRow<'a, T>;

	fn get(&'a self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		let offset = self.cols * index;
		MRow {
			data: &self.data[offset..(offset + self.cols)]
		}
	}
}

impl<'a, T: 'a> IndexedMut<'a, usize> for Matrix<T> {
	type Output = MRowMut<'a, T>;

	fn get_mut(&'a mut self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		let offset = self.cols * index;
		MRowMut {
			data: &mut self.data[offset..(offset + self.cols)]
		}
	}
}

impl<'a, T: 'a> Indexed<'a, (usize, usize)> for Matrix<T> {
	type Output = (MRow<'a, T>, MRow<'a, T>);

	fn get(&'a self, indices: (usize, usize)) -> Self::Output {
		(self.get(indices.0), self.get(indices.1))
	}
}

impl<'a, T: 'a> IndexedMut<'a, (usize, usize)> for Matrix<T> {
	type Output = (MRowMut<'a, T>, MRowMut<'a, T>);

	fn get_mut(&'a mut self, indices: (usize, usize)) -> Self::Output {
		self.assert_in_range(indices.0);
		self.assert_in_range(indices.1);
		assert!(indices.0 != indices.1, "When borrowing two rows, their indices must be different, got {}", indices.0);

		if indices.0 < indices.1 {
			let part: &mut [T] = &mut self.data[(indices.0 * self.cols)..((indices.1 + 1) * self.cols)];
			let (fst_row, rest) = part.split_at_mut(self.cols);
			let snd_row_start = rest.len() - self.cols;
			return (MRowMut {
					data: fst_row 
				}, MRowMut {
					data: &mut rest[snd_row_start..]
				});
		} else {
			let part: &mut [T] = &mut self.data[(indices.1 * self.cols)..((indices.0 + 1) * self.cols)];
			let (snd_row, rest) = part.split_at_mut(self.cols);
			let fst_row_start = rest.len() - self.cols;
			return (MRowMut {
					data: &mut rest[fst_row_start..] 
				}, MRowMut {
					data: snd_row
				});
		}
	}
}

impl<'a, T: 'a> MRow<'a, T> {

	pub fn get_slice(&self) -> &'a [T] 
	{
		self.data
	}
}

impl<'a, T: 'a> MRowMut<'a, T> {

	pub fn as_slice<'b>(&'b mut self) -> &'b mut [T] 
		where 'a: 'b
	{
		self.data
	}

	pub fn into_slice(self) -> &'a mut [T] 
	{
		self.data
	}
}

impl<'a, T> FixedLen for MRow<'a, T> {

	fn len(&self) -> usize {
		self.data.len()
	}
}

impl<'a, T> FixedLen for MRowMut<'a, T> {

	fn len(&self) -> usize {
		self.data.len()
	}
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for MRow<'b, T> {
	type Output = &'a T;

	fn get(&'a self, index: usize) -> Self::Output {
		&self.data[index]
	}
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for MRowMut<'b, T> {
	type Output = &'a T;

	fn get(&'a self, index: usize) -> Self::Output {
		&self.data[index]
	}
}

impl<'a, 'b, T: 'a> IndexedMut<'a, usize> for MRowMut<'b, T> {
	type Output = &'a mut T;

	fn get_mut(&'a mut self, index: usize) -> Self::Output {
		&mut self.data[index]
	}
}


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

impl<'a, 'c, T, U> AddAssign<MRow<'c, U>> for MRowMut<'a, T>
	where for<'b> T: AddAssign<&'b U>
{
	fn add_assign(&mut self, other: MRow<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
        for i in 0..self.len() {
			(*self.get_mut(i)).add_assign(other.get(i).clone());
		}
    }
}

impl<'a, 'c, T, U> AddAssign<MRowMut<'c, U>> for MRowMut<'a, T>
	where for<'b> T: AddAssign<&'b U>
{
	fn add_assign(&mut self, other: MRowMut<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
        for i in 0..self.len() {
			(*self.get_mut(i)).add_assign(other.get(i).clone());
		}
    }
}

impl<'a, 'c, T, U> SubAssign<MRow<'c, U>> for MRowMut<'a, T>
	where for<'b> T: SubAssign<&'b U>
{
	fn sub_assign(&mut self, other: MRow<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
        for i in 0..self.len() {
			(*self.get_mut(i)).sub_assign(other.get(i).clone());
		}
    }
}

impl<'a, 'c, T, U> SubAssign<MRowMut<'c, U>> for MRowMut<'a, T>
	where for<'b> T: SubAssign<&'b U>
{
	fn sub_assign(&mut self, other: MRowMut<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
        for i in 0..self.len() {
			(*self.get_mut(i)).sub_assign(other.get(i).clone());
		}
    }
}

impl<'a, T, U> MulAssign<U> for MRowMut<'a, T>
	where T: MulAssign<U>, U: Copy
{
	fn mul_assign(&mut self, other: U) {
        for i in 0..self.len() {
			(*self.get_mut(i)).mul_assign(other);
		}
    }
}

#[test]
fn test_matrix_get_rows() {
	let mut m = Matrix::new(Box::new([1,  2,  3, 
	                                  4,  5,  6, 
								      7,  8,  9, 
								      10, 11, 12]), 4);
	assert_eq!(3, m.cols);
	assert_eq!(4, m.rows);
	{
		let (fst_row, snd_row) = m.get_mut((0, 2));
		assert_eq!(&[1, 2, 3], fst_row.into_slice());
		assert_eq!(&[7, 8, 9], snd_row.into_slice());
	}
	{
		let (mut fst_row, mut snd_row) = m.get_mut((3, 2));
		assert_eq!(&[10, 11, 12], fst_row.into_slice());
		assert_eq!(&[7, 8, 9], snd_row.into_slice());
	}
}