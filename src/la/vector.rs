use super::indexed::{ Indexed, IndexedMut };
use std::ops::{ Index, IndexMut, Add, Mul, AddAssign, MulAssign, SubAssign, Deref, Range };
use std::cmp::{ min, max };
use std::borrow::{ Borrow, BorrowMut };

#[derive(Debug)]
pub struct Vector<T> {
    data: Box<[T]>
}

impl<T> Vector<T> {

	pub fn new(data: Box<[T]>) -> Vector<T> {
		assert!(data.len() > 0, "Cannot create vector with zero elements");
		Vector { data: data }
	}

	pub fn len(&self) -> usize {
		self.data.len()
	}

	pub fn data(&self) -> &[T] {
		&*self.data
	}
}

impl Vector<f64> {
	
    pub fn zero(len: usize) -> Self {
        let mut data = Vec::new();
        data.resize(len, 0.0);
        return Vector { data: data.into_boxed_slice() };
    }
}

impl Vector<i32> {

    pub fn zero(len: usize) -> Self {
        let mut data = Vec::new();
        data.resize(len, 0);
        return Vector { data: data.into_boxed_slice() };
    }
}

impl<'a, T: 'a> Indexed<'a, usize> for Vector<T> {
    type Output = &'a T;

	fn get(&'a self, index: usize) -> Self::Output {
		&self.data[index]
	}
}


impl<'a, T: 'a> IndexedMut<'a, usize> for Vector<T> {
    type Output = &'a mut T;

	fn get_mut(&'a mut self, index: usize) -> Self::Output {
		&mut self.data[index]
	}
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

	fn index(&self, index: usize) -> &T {
		self.get(index)
	}
}

impl<T> IndexMut<usize> for Vector<T> {

	fn index_mut(&mut self, index: usize) -> &mut T {
		self.get_mut(index)
	}
}
