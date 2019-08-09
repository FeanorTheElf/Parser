use std::ops::{ Index, IndexMut, Add, Mul, AddAssign, MulAssign, SubAssign, Deref, Range };
use std::cmp::{ min, max };
use std::borrow::{ Borrow, BorrowMut };

#[derive(Debug)]
pub struct Matrix<T> {
	rows: usize,
	data: Box<[T]>
}

#[derive(Debug)]
pub struct RowRef<'a, T> {
	data: &'a [T]
}

#[derive(Debug)]
pub struct RowRefMut<'a, T> {
	data: &'a mut [T]
}

#[derive(Debug)]
pub struct MatRef<'a, T> {
	rows: Range<usize>,
	cols: Range<usize>,
	matrix: &'a Matrix<T>
}

#[derive(Debug)]
pub struct MatRefMut<'a, T> {
	rows: Range<usize>,
	cols: Range<usize>,
	matrix: &'a mut Matrix<T>
}

impl Matrix<f64> {

	pub fn zero(rows: usize, cols: usize) -> Matrix<f64> {
		let mut data: Vec<f64> = Vec::new();
		data.resize(rows * cols, 0.0);
		return Matrix::new(data.into_boxed_slice(), rows);
	}
}

impl<T> Matrix<T> {

	pub fn new(data: Box<[T]>, rows: usize) -> Matrix<T> {
		assert!(data.len() % rows == 0, "Data length must be a multiple of row count, but got {} and {}", data.len(), rows);
		Matrix {
			rows: rows,
			data: data
		}
	}
}

impl<T> Matrix<T> {
	
	fn assert_in_range(&self, row_index: usize) {
		assert!(row_index >= 0, "Expected row index {} to be greater than 0", row_index);
		assert!(row_index < self.rows, "Expected row index {} to be smaller than the row count {}", row_index, self.rows);
	}
	
	pub fn cols(&self) -> usize {
		self.data.len() / self.rows
	}

	pub fn rows(&self) -> usize {
		self.rows
	}

	pub fn data(&self) -> &[T] {
		&*self.data
	}

	pub fn sub_matrix<'a>(&'a self, rows: Range<usize>, cols: Range<usize>) -> MatRef<'a, T> {
		MatRef {
			rows: rows,
			cols: cols,
			matrix: self
		}
	}

	pub fn sub_matrix_mut<'a>(&'a mut self, rows: Range<usize>, cols: Range<usize>) -> MatRefMut<'a, T> {
		MatRefMut {
			rows: rows,
			cols: cols,
			matrix: self
		}
	}

	fn borrow<'a>(&'a self) -> MatRef<'a, T> {
		MatRef {
			cols: 0..self.cols(),
			rows: 0..self.rows(),
			matrix: self
		}
	}

	fn borrow_mut<'a>(&'a mut self) -> MatRefMut<'a, T> {
		MatRefMut {
			cols: 0..self.cols(),
			rows: 0..self.rows(),
			matrix: self
		}
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

pub trait Indexed<'a, T> {
	type Output;

	fn get(&'a self, index: T) -> Self::Output;
}

pub trait IndexedMut<'a, T> {
	type Output;

	fn get_mut(&'a mut self, index: T) -> Self::Output;
}

impl<'a, T: 'a> Indexed<'a, usize> for Matrix<T> {
	type Output = RowRef<'a, T>;

	fn get(&'a self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		let offset = self.cols() * index;
		RowRef {
			data: &self.data[offset..(offset + self.cols())]
		}
	}
}

impl<'a, T: 'a> IndexedMut<'a, usize> for Matrix<T> {
	type Output = RowRefMut<'a, T>;

	fn get_mut(&'a mut self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		let start = self.cols() * index;
		let end = start + self.cols();
		RowRefMut {
			data: &mut self.data[start..end]
		}
	}
}

impl<'a, T: 'a> Indexed<'a, (usize, usize)> for Matrix<T> {
	type Output = (RowRef<'a, T>, RowRef<'a, T>);

	fn get(&'a self, indices: (usize, usize)) -> Self::Output {
		(self.get(indices.0), self.get(indices.1))
	}
}

impl<'a, T: 'a> IndexedMut<'a, (usize, usize)> for Matrix<T> {
	type Output = (RowRefMut<'a, T>, RowRefMut<'a, T>);

	fn get_mut(&'a mut self, indices: (usize, usize)) -> Self::Output {
		self.assert_in_range(indices.0);
		self.assert_in_range(indices.1);
		assert!(indices.0 != indices.1, "When borrowing two rows, their indices must be different, got {}", indices.0);

		let cols = self.cols();
		if indices.0 < indices.1 {
			let part: &mut [T] = &mut self.data[(indices.0 * cols)..((indices.1 + 1) * cols)];
			let (fst_row, rest) = part.split_at_mut(cols);
			let snd_row_start = rest.len() - cols;
			return (RowRefMut {
					data: fst_row 
				}, RowRefMut {
					data: &mut rest[snd_row_start..]
				});
		} else {
			let part: &mut [T] = &mut self.data[(indices.1 * cols)..((indices.0 + 1) * cols)];
			let (snd_row, rest) = part.split_at_mut(cols);
			let fst_row_start = rest.len() - cols;
			return (RowRefMut {
					data: &mut rest[fst_row_start..] 
				}, RowRefMut {
					data: snd_row
				});
		}
	}
}

impl<'a, T> MatRef<'a, T> {
	
	fn assert_in_range(&self, row_index: usize) {
		assert!(row_index >= 0, "Expected row index {} to be greater than 0", row_index);
		assert!(row_index < self.rows(), "Expected row index {} to be smaller than the row count {}", row_index, self.rows());
	}
	
	pub fn cols(&self) -> usize {
		self.cols.end - self.cols.start
	}

	pub fn rows(&self) -> usize {
		self.rows.end - self.rows.start
	}
}

impl<'a, T> MatRefMut<'a, T> {
	
	fn assert_in_range(&self, row_index: usize) {
		assert!(row_index >= 0, "Expected row index {} to be greater than 0", row_index);
		assert!(row_index < self.rows(), "Expected row index {} to be smaller than the row count {}", row_index, self.rows());
	}
	
	pub fn as_const<'b>(&'b self) -> MatRef<'b, T> {
		MatRef {
			rows: self.rows.start..self.rows.end,
			cols: self.cols.start..self.cols.end,
			matrix: &self.matrix
		}
	}

	pub fn cols(&self) -> usize {
		self.cols.end - self.cols.start
	}

	pub fn rows(&self) -> usize {
		self.rows.end - self.rows.start
	}
}

impl<'a, T> Index<usize> for MatRef<'a, T> {
	type Output = [T];

	fn index(&self, row_index: usize) -> &[T] {
		self.get(row_index).get_slice()
	}
}

impl<'a, T> Index<usize> for MatRefMut<'a, T> {
	type Output = [T];

	fn index(&self, row_index: usize) -> &[T] {
		self.get(row_index).get_slice()
	}
}

impl<'a, T> IndexMut<usize> for MatRefMut<'a, T> {

	fn index_mut(&mut self, row_index: usize) -> &mut [T] {
		self.get_mut(row_index).into_slice()
	}
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for MatRef<'b, T> {
	type Output = RowRef<'a, T>;

	fn get(&'a self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		self.matrix.get(index + self.rows.start).range(&self.cols)
	}
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for MatRefMut<'b, T> {
	type Output = RowRef<'a, T>;

	fn get(&'a self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		self.matrix.get(index + self.rows.start).range(&self.cols)
	}
}

impl<'a, 'b, T: 'a> IndexedMut<'a, usize> for MatRefMut<'b, T> {
	type Output = RowRefMut<'a, T>;

	fn get_mut(&'a mut self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		self.matrix.get_mut(index + self.rows.start).range_mut(&self.cols)
	}
}

fn map_tuple<T, U, F: FnMut(T) -> U>(tuple: (T, T), mut f: F) -> (U, U) {
	(f(tuple.0), f(tuple.1))
}

impl<'a, 'b, T: 'a> Indexed<'a, (usize, usize)> for MatRef<'b, T> {
	type Output = (RowRef<'a, T>, RowRef<'a, T>);

	fn get(&'a self, index: (usize, usize)) -> Self::Output {
		self.assert_in_range(index.0);
		self.assert_in_range(index.1);
		map_tuple(self.matrix.get(map_tuple(index, |row_index|row_index + self.rows.start)), 
			|row_ref: RowRef<'a, T>|row_ref.range(&self.cols))
	}
}

impl<'a, 'b, T: 'a> Indexed<'a, (usize, usize)> for MatRefMut<'b, T> {
	type Output = (RowRef<'a, T>, RowRef<'a, T>);

	fn get(&'a self, index: (usize, usize)) -> Self::Output {
		self.assert_in_range(index.0);
		self.assert_in_range(index.1);
		map_tuple(self.matrix.get(map_tuple(index, |row_index|row_index + self.rows.start)), 
			|row_ref: RowRef<'a, T>|row_ref.range(&self.cols))
	}
}

impl<'a, 'b, T: 'a> IndexedMut<'a, (usize, usize)> for MatRefMut<'b, T> {
	type Output = (RowRefMut<'a, T>, RowRefMut<'a, T>);

	fn get_mut(&'a mut self, index: (usize, usize)) -> Self::Output {
		self.assert_in_range(index.0);
		self.assert_in_range(index.1);
		let cols = &self.cols;
		map_tuple(self.matrix.get_mut(map_tuple(index, |row_index|row_index + self.rows.start)), 
			|row_ref: RowRefMut<'a, T>|row_ref.range_mut(cols))
	}
}

impl<'a, T: 'a> RowRef<'a, T> {

	pub fn len(&self) -> usize {
		self.data.len()
	}

fn range(self, range: &Range<usize>) -> RowRef<'a, T> {
		RowRef {
			data: &self.data[range.start..range.end]
		}
	}

	pub fn get_slice(&self) -> &'a [T] {
		self.data
	}
}

impl<'a, T: 'a> RowRefMut<'a, T> {

	pub fn len(&self) -> usize {
		self.data.len()
	}

	fn range(self, range: &Range<usize>) -> RowRef<'a, T> {
		RowRef {
			data: &self.data[range.start..range.end]
		}
	}

	fn range_mut(mut self, range: &Range<usize>) -> RowRefMut<'a, T> {
		RowRefMut {
			data: &mut self.data[range.start..range.end]
		}
	}

	pub fn as_const<'b>(&'b self) -> RowRef<'b, T> {
		RowRef {
			data: self.data
		}
	}

	pub fn as_slice<'b>(&'b mut self) -> &'b mut [T] 
		where 'a: 'b
	{
		self.data
	}

	pub fn into_slice(self) -> &'a mut [T] 
	{
		self.data
	}

	pub fn add_multiple<'c, V, U>(&mut self, other: RowRef<'c, U>, mult: V) 
		where V: Copy,
			U: Mul<V> + Clone,
			T: AddAssign<<U as Mul<V>>::Output>
	{
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
    for i in 0..self.len() {
			(*self.get_mut(i)).add_assign(other.get(i).clone() * mult);
		}
	}
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for RowRef<'b, T> {
	type Output = &'a T;

	fn get(&'a self, index: usize) -> Self::Output {
		&self.data[index]
	}
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for RowRefMut<'b, T> {
	type Output = &'a T;

	fn get(&'a self, index: usize) -> Self::Output {
		&self.data[index]
	}
}

impl<'a, 'b, T: 'a> IndexedMut<'a, usize> for RowRefMut<'b, T> {
	type Output = &'a mut T;

	fn get_mut(&'a mut self, index: usize) -> Self::Output {
		&mut self.data[index]
	}
}

impl<'a, 'c, T, U> AddAssign<RowRef<'c, U>> for RowRefMut<'a, T>
	where for<'b> T: AddAssign<&'b U>
{
	fn add_assign(&mut self, other: RowRef<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
    for i in 0..self.len() {
			(*self.get_mut(i)).add_assign(other.get(i));
		}
  }
}

impl<'a, 'c, T, U> AddAssign<RowRefMut<'c, U>> for RowRefMut<'a, T>
	where for<'b> T: AddAssign<&'b U>
{
	fn add_assign(&mut self, other: RowRefMut<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
    for i in 0..self.len() {
			(*self.get_mut(i)).add_assign(other.get(i));
		}
  }
}

impl<'a, 'c, T, U> SubAssign<RowRef<'c, U>> for RowRefMut<'a, T>
	where for<'b> T: SubAssign<&'b U>
{
	fn sub_assign(&mut self, other: RowRef<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
    for i in 0..self.len() {
			(*self.get_mut(i)).sub_assign(other.get(i));
		}
  }
}

impl<'a, 'c, T, U> SubAssign<RowRefMut<'c, U>> for RowRefMut<'a, T>
	where for<'b> T: SubAssign<&'b U>
{
	fn sub_assign(&mut self, other: RowRefMut<'c, U>) {
		assert_eq!(self.len(), other.len(), "Expected the lengths of summed vectors to be equal, but got {} and {}", self.len(), other.len());
    for i in 0..self.len() {
			(*self.get_mut(i)).sub_assign(other.get(i));
		}
  }
}

impl<'a, T, U> MulAssign<U> for RowRefMut<'a, T>
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
	assert_eq!(3, m.cols());
	assert_eq!(4, m.rows());
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

#[test]
fn test_matrix_submatrix() {
	let mut m = Matrix::new(Box::new([1,  2,  3,  7, 
	                                  4,  5,  6,  11, 
								      							7,  8,  9,  2, 
								      							10, 11, 12, 4]), 4);
	assert_eq!(4, m.cols());
	assert_eq!(4, m.rows());

	let mut n = m.sub_matrix_mut(1..3, 1..3);
	assert_eq!(5, n[0][0]);
	assert_eq!(9, n[1][1]);
	assert_eq!(2, n.rows());
	assert_eq!(2, n.cols());

	{
		let (mut r1, r2) = n.get_mut((1, 0));
		r1 += r2;
	}

	assert_eq!(7, m[2][0]);
	assert_eq!(13, m[2][1]);
	assert_eq!(15, m[2][2]);
	assert_eq!(2, m[2][3]);

	assert_eq!(2, m[0][1]);
	assert_eq!(5, m[1][1]);
	assert_eq!(11, m[3][1]);
}