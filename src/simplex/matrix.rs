use std::ops::{ Index, IndexMut, Add, Mul, AddAssign, MulAssign, SubAssign, Deref };
use std::cmp::{ min, max };

#[derive(Debug)]
pub struct Matrix<T> {
	rows: usize,
	cols: usize,
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
		let offset = self.cols * index;
		RowRef {
			data: &self.data[offset..(offset + self.cols)]
		}
	}
}

impl<'a, T: 'a> IndexedMut<'a, usize> for Matrix<T> {
	type Output = RowRefMut<'a, T>;

	fn get_mut(&'a mut self, index: usize) -> Self::Output {
		self.assert_in_range(index);
		let offset = self.cols * index;
		RowRefMut {
			data: &mut self.data[offset..(offset + self.cols)]
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

		if indices.0 < indices.1 {
			let part: &mut [T] = &mut self.data[(indices.0 * self.cols)..((indices.1 + 1) * self.cols)];
			let (fst_row, rest) = part.split_at_mut(self.cols);
			let snd_row_start = rest.len() - self.cols;
			return (RowRefMut {
					data: fst_row 
				}, RowRefMut {
					data: &mut rest[snd_row_start..]
				});
		} else {
			let part: &mut [T] = &mut self.data[(indices.1 * self.cols)..((indices.0 + 1) * self.cols)];
			let (snd_row, rest) = part.split_at_mut(self.cols);
			let fst_row_start = rest.len() - self.cols;
			return (RowRefMut {
					data: &mut rest[fst_row_start..] 
				}, RowRefMut {
					data: snd_row
				});
		}
	}
}

impl<'a, T: 'a> RowRef<'a, T> {

	fn len(&self) -> usize {
		self.data.len()
	}

	pub fn get_slice(&self) -> &'a [T] {
		self.data
	}
}

impl<'a, T: 'a> RowRefMut<'a, T> {

	fn len(&self) -> usize {
		self.data.len()
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