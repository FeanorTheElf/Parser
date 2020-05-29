use super::arith::*;
use super::indexed::{Indexed, IndexedMut};
use std::mem::swap;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    Sub, SubAssign,
};

///
/// Represents a mxn matrix with elements of type T. Typical matrix operations
/// are not optimized much, so this type is only suitable for small matrices.
/// Instead, the focus lies on a convenient interface and a generic design.
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    rows: usize,
    data: Box<[T]>,
}

#[derive(Debug)]
pub struct MatRef<'a, T> {
    rows_begin: usize,
    rows_end: usize,
    cols_begin: usize,
    cols_end: usize,
    matrix: &'a Matrix<T>,
}

#[derive(Debug)]
pub struct MatRefMut<'a, T> {
    rows: Range<usize>,
    cols: Range<usize>,
    matrix: &'a mut Matrix<T>,
}

#[derive(Debug)]
pub struct Vector<T> {
    data: Box<[T]>,
}

#[derive(Debug)]
pub struct VecRef<'a, T> {
    data: &'a [T],
}

#[derive(Debug)]
pub struct VecRefMut<'a, T> {
    data: &'a mut [T],
}

// ===============================================================================================================
// Impls
// ===============================================================================================================

fn assert_legal_subrange(len: usize, range: &Range<usize>) {
    assert!(
        range.end > range.start,
        "Subrange must have a positive number of elements"
    );
    assert!(range.end <= len, "Subrange must be included in total range");
}

impl<T> Matrix<T>
where
    T: Zero + Clone,
{
    pub fn zero(rows: usize, cols: usize) -> Matrix<T> {
        let mut data: Vec<T> = Vec::new();
        data.resize(rows * cols, T::zero());
        return Matrix::new(data.into_boxed_slice(), rows);
    }
}

impl<T> Matrix<T>
where
    T: Zero + One + Clone,
{
    pub fn identity(size: usize) -> Matrix<T> {
        let mut result = Matrix::<T>::zero(size, size);
        for i in 0..size {
            result[i][i] = T::one();
        }
        return result;
    }
}

impl<T> Matrix<T> {
    pub fn new(data: Box<[T]>, rows: usize) -> Matrix<T> {
        assert!(data.len() > 0, "Cannot create matrix with zero elements");
        assert!(
            data.len() % rows == 0,
            "Data length must be a multiple of row count, but got {} and {}",
            data.len(),
            rows
        );
        Matrix {
            rows: rows,
            data: data,
        }
    }

    fn assert_row_in_range(&self, row_index: usize) {
        assert!(
            row_index < self.rows,
            "Expected row index {} to be smaller than the row count {}",
            row_index,
            self.rows
        );
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

    pub fn into_column(self) -> Vector<T> {
        assert_eq!(1, self.cols());
        Vector { data: self.data }
    }

    pub fn into_row(self) -> Vector<T> {
        assert_eq!(1, self.rows());
        Vector { data: self.data }
    }

    pub fn from<U>(value: Matrix<U>) -> Self
    where
        T: From<U>,
    {
        let rows = value.rows();
        let data: Vec<T> = value
            .data
            .into_vec()
            .into_iter()
            .map(|d| T::from(d))
            .collect();
        return Matrix {
            data: data.into_boxed_slice(),
            rows: rows,
        };
    }
}

impl<'a, T> MatRef<'a, T> {
    fn offset_col_range(&self, range: Range<usize>) -> Range<usize> {
        assert_legal_subrange(self.cols(), &range);
        (self.cols_begin + range.start)..(self.cols_begin + range.end)
    }

    fn offset_row_range(&self, range: Range<usize>) -> Range<usize> {
        assert_legal_subrange(self.rows(), &range);
        (self.rows_begin + range.start)..(self.rows_begin + range.end)
    }

    fn assert_row_in_range(&self, row_index: usize) {
        assert!(
            row_index < self.rows(),
            "Expected row index {} to be smaller than the row count {}",
            row_index,
            self.rows()
        );
    }

    pub fn cols(&self) -> usize {
        self.cols_end - self.cols_begin
    }

    pub fn rows(&self) -> usize {
        self.rows_end - self.rows_begin
    }
}

impl<'a, T> MatRef<'a, T>
where
    T: Clone,
{
    pub fn owned(&self) -> Matrix<T> {
        let cols = self.cols();
        let data: Vec<T> = (0..(self.rows() * cols))
            .map(|index: usize| self.get(index / cols).get(index % cols).clone())
            .collect();
        Matrix::new(data.into_boxed_slice(), self.rows())
    }
}

impl<'a, T> MatRef<'a, T>
where
    T: Clone
        + PartialEq<T>
        + AddAssign<T>
        + MulAssign<T>
        + Neg<Output = T>
        + Zero
        + One
        + Div<T, Output = T>
        + Mul<T, Output = T>,
    <T as Div<T>>::Output: Clone,
{
    ///
    /// Calculates the inverse of this matrix. Use only for small matrices, this
    /// is just simple gaussian elimination, and is neither very performant nor
    /// numerically stable!
    ///
    pub fn invert(&self) -> Result<Matrix<T>, ()> {
        assert_eq!(self.rows(), self.cols());
        let n: usize = self.rows();

        let mut result: Matrix<T> = Matrix::identity(n);
        let mut work: Matrix<T> = self.owned();

        // just simple gaussian elimination
        for i in 0..n {
            let not_null_index = (i..n).find(|r| work[*r][i] != T::zero()).ok_or(())?;
            if not_null_index != i {
                result.get_mut((.., ..)).swap_rows(i, not_null_index);
                work.get_mut((.., ..)).swap_rows(i, not_null_index);
            }
            for j in (i + 1)..n {
                let (row1, mut row2) = work.get_mut((i, j));
                let factor = -row2[i].clone() / row1[i].clone();
                row2.add_product(row1.as_const(), factor.clone());
                let (res1, mut res2) = result.get_mut((i, j));
                res2.add_product(res1.as_const(), factor);
            }
        }
        // now we have an upper triangle matrix
        for i in 1..n {
            for j in 0..i {
                let (row1, mut row2) = work.get_mut((i, j));
                let factor = -row2[i].clone() / row1[i].clone();
                row2.add_product(row1.as_const(), factor.clone());
                let (res1, mut res2) = result.get_mut((i, j));
                res2.add_product(res1.as_const(), factor);
            }
        }
        // now we have a diagonal matrix
        for i in 0..n {
            result.get_mut(i).mul_assign(T::one() / work[i][i].clone());
        }
        return Ok(result);
    }
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
        return Vector {
            data: data.into_boxed_slice(),
        };
    }
}

impl Vector<i32> {
    pub fn zero(len: usize) -> Self {
        let mut data = Vec::new();
        data.resize(len, 0);
        return Vector {
            data: data.into_boxed_slice(),
        };
    }
}

impl<'a, T> MatRefMut<'a, T> {
    fn offset_col_range(&self, range: Range<usize>) -> Range<usize> {
        assert_legal_subrange(self.cols(), &range);
        (self.cols.start + range.start)..(self.cols.start + range.end)
    }

    fn offset_row_range(&self, range: Range<usize>) -> Range<usize> {
        assert_legal_subrange(self.rows(), &range);
        (self.rows.start + range.start)..(self.rows.start + range.end)
    }

    fn assert_row_in_range(&self, row_index: usize) {
        assert!(
            row_index < self.rows(),
            "Expected row index {} to be smaller than the row count {}",
            row_index,
            self.rows()
        );
    }

    fn assert_col_in_range(&self, col_index: usize) {
        assert!(
            col_index < self.cols(),
            "Expected column index {} to be smaller than the column count {}",
            col_index,
            self.cols()
        );
    }

    pub fn as_const<'b>(&'b self) -> MatRef<'b, T> {
        self.matrix.get((self.rows.clone(), self.cols.clone()))
    }

    pub fn cols(&self) -> usize {
        self.cols.end - self.cols.start
    }

    pub fn rows(&self) -> usize {
        self.rows.end - self.rows.start
    }

    pub fn sub_matrix(self, rows: Range<usize>, cols: Range<usize>) -> MatRefMut<'a, T> {
        assert_legal_subrange(self.rows(), &rows);
        assert_legal_subrange(self.cols(), &cols);
        self.matrix.get_mut((rows, cols))
    }

    pub fn swap_cols(&mut self, fst: usize, snd: usize) {
        self.assert_col_in_range(fst);
        self.assert_col_in_range(snd);
        if fst != snd {
            for row in self.rows.clone() {
                self.matrix[row].swap(fst + self.cols.start, snd + self.cols.start);
            }
        }
    }

    pub fn swap_rows(&mut self, fst: usize, snd: usize) {
        if fst != snd {
            let cols = self.cols();
            let (mut fst_row, mut snd_row) = self.get_mut((fst, snd));
            for col in 0..cols {
                swap(fst_row.get_mut(col), snd_row.get_mut(col));
            }
        }
    }
}

impl<'a, T> MatRef<'a, T>
where
    T: Copy,
{
    pub fn to_owned(&self) -> Matrix<T> {
        let mut data = Vec::with_capacity(self.rows() * self.cols());
        for row in 0..self.rows() {
            data.extend_from_slice(&self[row]);
        }
        Matrix::new(data.into_boxed_slice(), self.rows())
    }
}

impl<'a, T> MatRefMut<'a, T>
where
    T: Copy,
{
    pub fn assign_copy(&mut self, data: MatRef<T>)
    where
        T: Copy,
    {
        assert_eq!(self.rows(), data.rows());
        assert_eq!(self.cols(), data.cols());
        for row in 0..self.rows() {
            self[row].copy_from_slice(&data[row]);
        }
    }
}

impl<'a, T> MatRefMut<'a, T>
where
    T: Add<T, Output = T> + Copy + Mul<T, Output = T>,
{
    ///
    /// Let T be the identity matrix (mxm where this matrix is mxn), in which the entries
    /// [fst,fst], [fst, snd], [snd, fst], [snd, snd] are replaced by the values in transform.
    /// This function performs the multiplication A' := T * A, where A is this matrix
    ///
    pub fn transform_two_dims_left(&mut self, fst: usize, snd: usize, transform: &[T; 4]) {
        assert!(fst < snd);
        self.assert_row_in_range(fst);
        self.assert_row_in_range(snd);
        for col in 0..self.cols() {
            let b = self[fst][col];
            self[fst][col] = self[fst][col] * transform[0] + self[snd][col] * transform[1];
            self[snd][col] = b * transform[2] + self[snd][col] * transform[3];
        }
    }

    ///
    /// Let T be the identity matrix (nxn where this matrix is mxn), in which
    /// the entries [fst,fst], [fst, snd], [snd, fst], [snd, snd] are replaced by the
    /// values in transform.
    /// This function performs the multiplication A' := A * T, where A is this matrix
    ///
    pub fn transform_two_dims_right(&mut self, fst: usize, snd: usize, transform: &[T; 4]) {
        assert!(fst < snd);
        self.assert_col_in_range(fst);
        self.assert_col_in_range(snd);
        for row in 0..self.rows() {
            let b = self[row][fst];
            self[row][fst] = self[row][fst] * transform[0] + self[row][snd] * transform[2];
            self[row][snd] = b * transform[1] + self[row][snd] * transform[3];
        }
    }
}

impl<'a, T: 'a> VecRef<'a, T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

impl<'a, T: 'a> VecRef<'a, T>
where
    T: Copy,
{
    pub fn to_owned(&self) -> Vector<T> {
        let mut data = Vec::with_capacity(self.len());
        data.extend_from_slice(&self.data);
        Vector::new(data.into_boxed_slice())
    }
}

impl<'a, 'b, T> Mul<VecRef<'b, T>> for VecRef<'a, T>
where
    T: Add<T, Output = T> + Copy + Mul<T, Output = T> + 'a,
{
    type Output = T;

    fn mul(self, rhs: VecRef<'b, T>) -> T {
        assert!(self.len() == rhs.len(), "Can only multiply vectors of same length, got row ref of length {} and vector of length {}", self.len(), rhs.len());
        (1..self.len())
            .map(|index: usize| *self.get(index) * *rhs.get(index))
            .fold(*self.get(0) * *rhs.get(0), |acc: T, item: T| acc + item)
    }
}

impl<'a, T: 'a> VecRefMut<'a, T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_const<'b>(&'b self) -> VecRef<'b, T> {
        VecRef { data: self.data }
    }

    pub fn as_slice<'b>(&'b mut self) -> &'b mut [T]
    where
        'a: 'b,
    {
        self.data
    }

    pub fn into_slice(self) -> &'a mut [T] {
        self.data
    }

    pub fn sub_range(self, range: Range<usize>) -> VecRefMut<'a, T> {
        VecRefMut {
            data: &mut self.data[range],
        }
    }

    pub fn add_product<'c, V, U>(&mut self, other: VecRef<'c, U>, mult: V)
    where
        V: Clone,
        U: Mul<V> + Clone,
        T: AddAssign<<U as Mul<V>>::Output>,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "Expected the lengths of summed vectors to be equal, but got {} and {}",
            self.len(),
            other.len()
        );
        for i in 0..self.len() {
            (*self.get_mut(i)).add_assign(other.get(i).clone() * mult.clone());
        }
    }
}

impl<'a, T> Copy for VecRef<'a, T> {}

impl<'a, T> Clone for VecRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

// ===============================================================================================================
// Traits for Matrix
// ===============================================================================================================

impl<'a, T, U: 'a> From<MatRef<'a, U>> for Matrix<T>
where
    T: From<&'a U>,
{
    fn from(value: MatRef<'a, U>) -> Self {
        let data: Vec<T> = (value.rows_begin..value.rows_end)
            .flat_map(|row| {
                (value.cols_begin..value.cols_end)
                    .map(move |col| &value.matrix[row][col])
                    .map(|d| T::from(d))
            })
            .collect();
        return Matrix {
            data: data.into_boxed_slice(),
            rows: value.rows(),
        };
    }
}

impl<'a, T: 'a> Indexed<'a, (RangeFull, RangeFull)> for Matrix<T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, _: (RangeFull, RangeFull)) -> Self::Output {
        self.get((0..self.rows(), 0..self.cols()))
    }
}

impl<'a, T: 'a> IndexedMut<'a, (RangeFull, RangeFull)> for Matrix<T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, _: (RangeFull, RangeFull)) -> Self::Output {
        self.get_mut((0..self.rows(), 0..self.cols()))
    }
}

impl<'a, T: 'a> Indexed<'a, (RangeFull, Range<usize>)> for Matrix<T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, index: (RangeFull, Range<usize>)) -> Self::Output {
        self.get((0..self.rows(), index.1))
    }
}

impl<'a, T: 'a> Indexed<'a, (Range<usize>, RangeFull)> for Matrix<T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, index: (Range<usize>, RangeFull)) -> Self::Output {
        self.get((index.0, 0..self.cols()))
    }
}

impl<'a, T: 'a> IndexedMut<'a, (RangeFull, Range<usize>)> for Matrix<T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, index: (RangeFull, Range<usize>)) -> Self::Output {
        self.get_mut((0..self.rows(), index.1))
    }
}

impl<'a, T: 'a> IndexedMut<'a, (Range<usize>, RangeFull)> for Matrix<T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, index: (Range<usize>, RangeFull)) -> Self::Output {
        self.get_mut((index.0, 0..self.cols()))
    }
}

impl<'a, T: 'a> Indexed<'a, (Range<usize>, Range<usize>)> for Matrix<T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, index: (Range<usize>, Range<usize>)) -> Self::Output {
        assert_legal_subrange(self.rows(), &index.0);
        assert_legal_subrange(self.cols(), &index.1);
        MatRef {
            rows_begin: index.0.start,
            rows_end: index.0.end,
            cols_begin: index.1.start,
            cols_end: index.1.end,
            matrix: self,
        }
    }
}

impl<'a, T: 'a> IndexedMut<'a, (Range<usize>, Range<usize>)> for Matrix<T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, index: (Range<usize>, Range<usize>)) -> Self::Output {
        assert_legal_subrange(self.rows(), &index.0);
        assert_legal_subrange(self.cols(), &index.1);
        MatRefMut {
            rows: index.0,
            cols: index.1,
            matrix: self,
        }
    }
}

impl<'a, T: 'a> Indexed<'a, usize> for Matrix<T> {
    type Output = VecRef<'a, T>;

    fn get(&'a self, index: usize) -> Self::Output {
        self.assert_row_in_range(index);
        let offset = self.cols() * index;
        VecRef {
            data: &self.data[offset..(offset + self.cols())],
        }
    }
}

impl<'a, T: 'a> IndexedMut<'a, usize> for Matrix<T> {
    type Output = VecRefMut<'a, T>;

    fn get_mut(&'a mut self, index: usize) -> Self::Output {
        self.assert_row_in_range(index);
        let start = self.cols() * index;
        let end = start + self.cols();
        VecRefMut {
            data: &mut self.data[start..end],
        }
    }
}

impl<'a, T: 'a> Indexed<'a, (usize, usize)> for Matrix<T> {
    type Output = (VecRef<'a, T>, VecRef<'a, T>);

    fn get(&'a self, indices: (usize, usize)) -> Self::Output {
        (self.get(indices.0), self.get(indices.1))
    }
}

impl<'a, T: 'a> IndexedMut<'a, (usize, usize)> for Matrix<T> {
    type Output = (VecRefMut<'a, T>, VecRefMut<'a, T>);

    fn get_mut(&'a mut self, indices: (usize, usize)) -> Self::Output {
        self.assert_row_in_range(indices.0);
        self.assert_row_in_range(indices.1);
        assert!(
            indices.0 != indices.1,
            "When borrowing two rows, their indices must be different, got {}",
            indices.0
        );

        let cols = self.cols();
        if indices.0 < indices.1 {
            let part: &mut [T] = &mut self.data[(indices.0 * cols)..((indices.1 + 1) * cols)];
            let (fst_row, rest) = part.split_at_mut(cols);
            let snd_row_start = rest.len() - cols;
            return (
                VecRefMut { data: fst_row },
                VecRefMut {
                    data: &mut rest[snd_row_start..],
                },
            );
        } else {
            let part: &mut [T] = &mut self.data[(indices.1 * cols)..((indices.0 + 1) * cols)];
            let (snd_row, rest) = part.split_at_mut(cols);
            let fst_row_start = rest.len() - cols;
            return (
                VecRefMut {
                    data: &mut rest[fst_row_start..],
                },
                VecRefMut { data: snd_row },
            );
        }
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, row_index: usize) -> &[T] {
        self.get(row_index).as_slice()
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, row_index: usize) -> &mut [T] {
        self.get_mut(row_index).into_slice()
    }
}

impl<'a, T> Copy for MatRef<'a, T> {}

impl<'a, T> Clone for MatRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

// ===============================================================================================================
// Traits for MatRef
// ===============================================================================================================

impl<'a, 'b, T> Mul<VecRef<'b, T>> for MatRef<'a, T>
where
    T: Add<T, Output = T> + Copy + Mul<T, Output = T>,
{
    type Output = Vector<T>;

    fn mul(self, rhs: VecRef<'b, T>) -> Self::Output {
        let data: Vec<T> = (0..self.rows())
            .map(|row: usize| self.get(row) * rhs)
            .collect();
        Vector::new(data.into_boxed_slice())
    }
}

impl<'a, 'b, T> Mul<MatRef<'b, T>> for MatRef<'a, T>
where
    T: Add<T, Output = T> + Copy + Mul<T, Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: MatRef<'b, T>) -> Self::Output {
        assert_eq!(self.cols(), rhs.rows());
        let cols = rhs.cols();
        let data: Vec<T> = (0..(cols * self.rows()))
            .map(|index: usize| {
                let row = index / cols;
                let col = index % cols;
                (1..self.cols())
                    .map(|k: usize| self[row][k].clone() * rhs[k][col].clone())
                    .fold(
                        self[row][0].clone() * rhs[0][col].clone(),
                        |acc: T, el: T| acc + el,
                    )
            })
            .collect();
        Matrix::new(data.into_boxed_slice(), self.rows())
    }
}

impl<'a, 'b, T> Add<MatRef<'b, T>> for MatRef<'a, T>
    where T: Clone + for<'c> AddAssign<&'c T>
{
    type Output = Matrix<T>;

    fn add(self, rhs: MatRef<'b, T>) -> Self::Output {
        let mut result: Matrix<T> = self.owned();
        result.get_mut((.., ..)).add_assign(rhs);
        return result;
    }
}

impl<'a, 'b, T> Sub<MatRef<'b, T>> for MatRef<'a, T>
    where T: Clone + for<'c> SubAssign<&'c T>
{
    type Output = Matrix<T>;

    fn sub(self, rhs: MatRef<'b, T>) -> Self::Output {
        let mut result: Matrix<T> = self.owned();
        result.get_mut((.., ..)).sub_assign(rhs);
        return result;
    }
}

impl<'a, T> Index<usize> for MatRef<'a, T> {
    type Output = [T];

    fn index(&self, row_index: usize) -> &[T] {
        self.get(row_index).as_slice()
    }
}

impl<'a, 'b, T: 'b> Indexed<'a, usize> for MatRef<'b, T> {
    type Output = VecRef<'b, T>;

    fn get(&'a self, index: usize) -> Self::Output {
        self.assert_row_in_range(index);
        self.matrix
            .get(index + self.rows_begin)
            .get(self.cols_begin..self.cols_end)
    }
}

impl<'a, 'b, T: 'b> Indexed<'a, (usize, usize)> for MatRef<'b, T> {
    type Output = (VecRef<'b, T>, VecRef<'b, T>);

    fn get(&'a self, index: (usize, usize)) -> Self::Output {
        self.assert_row_in_range(index.0);
        self.assert_row_in_range(index.1);
        (self.get(index.0), self.get(index.1))
    }
}

impl<'a, 'b, T: 'b> Indexed<'a, (RangeFull, RangeFull)> for MatRef<'b, T> {
    type Output = MatRef<'b, T>;

    fn get(&'a self, _: (RangeFull, RangeFull)) -> Self::Output {
        *self
    }
}

impl<'a, 'b, T: 'b> Indexed<'a, (RangeFull, Range<usize>)> for MatRef<'b, T> {
    type Output = MatRef<'b, T>;

    fn get(&'a self, index: (RangeFull, Range<usize>)) -> Self::Output {
        self.matrix.get((.., self.offset_col_range(index.1)))
    }
}

impl<'a, 'b, T: 'b> Indexed<'a, (Range<usize>, RangeFull)> for MatRef<'b, T> {
    type Output = MatRef<'b, T>;

    fn get(&'a self, index: (Range<usize>, RangeFull)) -> Self::Output {
        self.matrix.get((self.offset_row_range(index.0), ..))
    }
}

impl<'a, 'b, T: 'b> Indexed<'a, (Range<usize>, Range<usize>)> for MatRef<'b, T> {
    type Output = MatRef<'b, T>;

    fn get(&'a self, index: (Range<usize>, Range<usize>)) -> Self::Output {
        self.matrix.get((
            self.offset_row_range(index.0),
            self.offset_col_range(index.1),
        ))
    }
}

// ===============================================================================================================
// Traits for MatRefMut
// ===============================================================================================================

impl<'a, T, U> MulAssign<U> for MatRefMut<'a, T>
where
    T: MulAssign<U>,
    U: Clone,
{
    fn mul_assign(&mut self, rhs: U) {
        for row in self.rows.clone() {
            self.get_mut(row).mul_assign(rhs.clone());
        }
    }
}

impl<'a, 'b, T, U> AddAssign<MatRef<'b, U>> for MatRefMut<'a, T>
where
    T: for<'c> AddAssign<&'c U>,
{
    fn add_assign(&mut self, rhs: MatRef<'b, U>) {
        for row in self.rows.clone() {
            self.get_mut(row).add_assign(rhs.get(row));
        }
    }
}

impl<'a, 'b, T, U> SubAssign<MatRef<'b, U>> for MatRefMut<'a, T>
where
    T: for<'c> SubAssign<&'c U>,
{
    fn sub_assign(&mut self, rhs: MatRef<'b, U>) {
        for row in self.rows.clone() {
            self.get_mut(row).sub_assign(rhs.get(row));
        }
    }
}

impl<'a, T> Index<usize> for MatRefMut<'a, T> {
    type Output = [T];

    fn index(&self, row_index: usize) -> &[T] {
        self.get(row_index).as_slice()
    }
}

impl<'a, T> IndexMut<usize> for MatRefMut<'a, T> {
    fn index_mut(&mut self, row_index: usize) -> &mut [T] {
        self.get_mut(row_index).into_slice()
    }
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for MatRefMut<'b, T> {
    type Output = VecRef<'a, T>;

    fn get(&'a self, index: usize) -> Self::Output {
        self.as_const().get(index)
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, usize> for MatRefMut<'b, T> {
    type Output = VecRefMut<'a, T>;

    fn get_mut(&'a mut self, index: usize) -> Self::Output {
        self.assert_row_in_range(index);
        self.matrix
            .get_mut(index + self.rows.start)
            .sub_range(self.cols.clone())
    }
}

fn map_tuple<T, U, F: FnMut(T) -> U>(tuple: (T, T), mut f: F) -> (U, U) {
    (f(tuple.0), f(tuple.1))
}

impl<'a, 'b, T: 'a> Indexed<'a, (usize, usize)> for MatRefMut<'b, T> {
    type Output = (VecRef<'a, T>, VecRef<'a, T>);

    fn get(&'a self, index: (usize, usize)) -> Self::Output {
        self.as_const().get(index)
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, (usize, usize)> for MatRefMut<'b, T> {
    type Output = (VecRefMut<'a, T>, VecRefMut<'a, T>);

    fn get_mut(&'a mut self, index: (usize, usize)) -> Self::Output {
        self.assert_row_in_range(index.0);
        self.assert_row_in_range(index.1);
        let cols = &self.cols;
        map_tuple(
            self.matrix
                .get_mut(map_tuple(index, |row_index| row_index + self.rows.start)),
            |row_ref: VecRefMut<'a, T>| row_ref.sub_range(cols.clone()),
        )
    }
}

impl<'a, 'b, T: 'a> Indexed<'a, (RangeFull, RangeFull)> for MatRefMut<'b, T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, _: (RangeFull, RangeFull)) -> Self::Output {
        self.as_const()
    }
}

impl<'a, 'b, T: 'a> Indexed<'a, (RangeFull, Range<usize>)> for MatRefMut<'b, T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, index: (RangeFull, Range<usize>)) -> Self::Output {
        self.as_const().get(index)
    }
}

impl<'a, 'b, T: 'a> Indexed<'a, (Range<usize>, RangeFull)> for MatRefMut<'b, T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, index: (Range<usize>, RangeFull)) -> Self::Output {
        self.as_const().get(index)
    }
}

impl<'a, 'b, T: 'a> Indexed<'a, (Range<usize>, Range<usize>)> for MatRefMut<'b, T> {
    type Output = MatRef<'a, T>;

    fn get(&'a self, index: (Range<usize>, Range<usize>)) -> Self::Output {
        self.as_const().get(index)
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, (RangeFull, RangeFull)> for MatRefMut<'b, T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, _: (RangeFull, RangeFull)) -> Self::Output {
        MatRefMut {
            matrix: self.matrix,
            cols: self.cols.clone(),
            rows: self.rows.clone(),
        }
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, (RangeFull, Range<usize>)> for MatRefMut<'b, T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, index: (RangeFull, Range<usize>)) -> Self::Output {
        self.matrix.get_mut((.., self.offset_col_range(index.1)))
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, (Range<usize>, RangeFull)> for MatRefMut<'b, T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, index: (Range<usize>, RangeFull)) -> Self::Output {
        self.matrix.get_mut((self.offset_row_range(index.0), ..))
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, (Range<usize>, Range<usize>)> for MatRefMut<'b, T> {
    type Output = MatRefMut<'a, T>;

    fn get_mut(&'a mut self, index: (Range<usize>, Range<usize>)) -> Self::Output {
        self.matrix.get_mut((
            self.offset_row_range(index.0),
            self.offset_col_range(index.1),
        ))
    }
}

// ===============================================================================================================
// Traits for VecRef, VecRefMut
// ===============================================================================================================

impl<'a, T> IntoIterator for VecRef<'a, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, 'c, T, U> AddAssign<VecRef<'c, U>> for VecRefMut<'a, T>
where
    T: for<'b> AddAssign<&'b U>,
{
    fn add_assign(&mut self, other: VecRef<'c, U>) {
        assert_eq!(
            self.len(),
            other.len(),
            "Expected the lengths of summed vectors to be equal, but got {} and {}",
            self.len(),
            other.len()
        );
        for i in 0..self.len() {
            (*self.get_mut(i)).add_assign(other.get(i));
        }
    }
}

impl<'a, 'c, T, U> SubAssign<VecRef<'c, U>> for VecRefMut<'a, T>
where
    T: for<'b> SubAssign<&'b U>,
{
    fn sub_assign(&mut self, other: VecRef<'c, U>) {
        assert_eq!(
            self.len(),
            other.len(),
            "Expected the lengths of summed vectors to be equal, but got {} and {}",
            self.len(),
            other.len()
        );
        for i in 0..self.len() {
            (*self.get_mut(i)).sub_assign(other.get(i));
        }
    }
}

impl<'a, T, U> MulAssign<U> for VecRefMut<'a, T>
where
    T: MulAssign<U>,
    U: Clone,
{
    fn mul_assign(&mut self, other: U) {
        for i in 0..self.len() {
            (*self.get_mut(i)).mul_assign(other.clone());
        }
    }
}

impl<'a, 'c, T, U> PartialEq<VecRef<'c, U>> for VecRef<'a, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &VecRef<'c, U>) -> bool {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            if self[i] != rhs[i] {
                return false;
            }
        }
        return true;
    }
}

impl<'a, T> Eq for VecRef<'a, T> where T: Eq {}

impl<'a, 'b, T: 'b> Indexed<'a, usize> for VecRef<'b, T> {
    type Output = &'b T;

    fn get(&'a self, index: usize) -> Self::Output {
        &self.data[index]
    }
}

impl<'a, 'b, T: 'b> Indexed<'a, Range<usize>> for VecRef<'b, T> {
    type Output = VecRef<'b, T>;

    fn get(&'a self, index: Range<usize>) -> Self::Output {
        assert_legal_subrange(self.len(), &index);
        VecRef {
            data: &self.data[index],
        }
    }
}

impl<'a, 'b, T: 'a> Indexed<'a, usize> for VecRefMut<'b, T> {
    type Output = &'a T;

    fn get(&'a self, index: usize) -> Self::Output {
        self.as_const().get(index)
    }
}

impl<'a, 'b, T: 'a> Indexed<'a, Range<usize>> for VecRefMut<'b, T> {
    type Output = VecRef<'a, T>;

    fn get(&'a self, index: Range<usize>) -> Self::Output {
        self.as_const().get(index)
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, usize> for VecRefMut<'b, T> {
    type Output = &'a mut T;

    fn get_mut(&'a mut self, index: usize) -> Self::Output {
        &mut self.data[index]
    }
}

impl<'a, 'b, T: 'a> IndexedMut<'a, Range<usize>> for VecRefMut<'b, T> {
    type Output = VecRefMut<'a, T>;

    fn get_mut(&'a mut self, index: Range<usize>) -> Self::Output {
        assert_legal_subrange(self.len(), &index);
        VecRefMut {
            data: &mut self.data[index],
        }
    }
}

impl<'a, T: 'a> Index<usize> for VecRef<'a, T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        self.get(i)
    }
}

impl<'a, T: 'a> Index<usize> for VecRefMut<'a, T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        self.get(i)
    }
}

impl<'a, T: 'a> IndexMut<usize> for VecRefMut<'a, T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        self.get_mut(i)
    }
}

impl<'a, T> std::fmt::Display for VecRef<'a, T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..(self.len() - 1) {
            write!(f, "{}, ", self.data[i])?;
        }
        if self.len() > 0 {
            write!(f, "{}]", self.data[self.len() - 1])
        } else {
            write!(f, "]")
        }
    }
}

// ===============================================================================================================
// Traits for Vector
// ===============================================================================================================

impl<'a, T: 'a> Indexed<'a, usize> for Vector<T> {
    type Output = &'a T;

    fn get(&'a self, index: usize) -> Self::Output {
        &self.data[index]
    }
}

impl<'a, T: 'a> Indexed<'a, Range<usize>> for Vector<T> {
    type Output = VecRef<'a, T>;

    fn get(&'a self, index: Range<usize>) -> Self::Output {
        VecRef {
            data: &self.data[index],
        }
    }
}

impl<'a, T: 'a> Indexed<'a, RangeFull> for Vector<T> {
    type Output = VecRef<'a, T>;

    fn get(&'a self, _: RangeFull) -> Self::Output {
        VecRef { data: &self.data }
    }
}

impl<'a, T: 'a> IndexedMut<'a, usize> for Vector<T> {
    type Output = &'a mut T;

    fn get_mut(&'a mut self, index: usize) -> Self::Output {
        &mut self.data[index]
    }
}

impl<'a, T: 'a> IndexedMut<'a, Range<usize>> for Vector<T> {
    type Output = VecRefMut<'a, T>;

    fn get_mut(&'a mut self, index: Range<usize>) -> Self::Output {
        VecRefMut {
            data: &mut self.data[index],
        }
    }
}

impl<'a, T: 'a> IndexedMut<'a, RangeFull> for Vector<T> {
    type Output = VecRefMut<'a, T>;

    fn get_mut(&'a mut self, _: RangeFull) -> Self::Output {
        VecRefMut {
            data: &mut self.data,
        }
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

#[cfg(test)]
use super::rat::*;

#[test]
fn test_matrix_get_rows() {
    #[rustfmt::skip]
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
        let (fst_row, snd_row) = m.get_mut((3, 2));
        assert_eq!(&[10, 11, 12], fst_row.into_slice());
        assert_eq!(&[7, 8, 9], snd_row.into_slice());
    }
}

#[test]
fn test_matrix_submatrix() {
    #[rustfmt::skip]
	let mut m = Matrix::new(Box::new([1,  2,  3,  7,
	                                  4,  5,  6,  11,
								      7,  8,  9,  2,
								      10, 11, 12, 4]), 4);
    assert_eq!(4, m.cols());
    assert_eq!(4, m.rows());

    let mut n = m.get_mut((1..3, 1..3));
    assert_eq!(5, n[0][0]);
    assert_eq!(9, n[1][1]);
    assert_eq!(2, n.rows());
    assert_eq!(2, n.cols());

    {
        let (mut r1, r2) = n.get_mut((1, 0));
        r1 += r2.as_const();
    }

    assert_eq!(7, m[2][0]);
    assert_eq!(13, m[2][1]);
    assert_eq!(15, m[2][2]);
    assert_eq!(2, m[2][3]);

    assert_eq!(2, m[0][1]);
    assert_eq!(5, m[1][1]);
    assert_eq!(11, m[3][1]);
}

#[test]
fn test_matrix_transform_two_dims_left() {
    #[rustfmt::skip]
	let mut m = Matrix::new(Box::new([1., 2., 3.,
	                                  4., 5., 6.,
								      7., 8., 9.]), 3);
    m.get_mut((.., ..))
        .transform_two_dims_left(0, 2, &[0., 1., 1., 0.]);

    #[rustfmt::skip]
	assert_eq!(&[7., 8., 9.,
	             4., 5., 6.,
				 1., 2., 3.], m.data());
    m.get_mut((.., ..))
        .transform_two_dims_left(1, 2, &[0.5, 0.5, 1.0, 0.5]);

    #[rustfmt::skip]
	assert_eq!(&[7.,  8.,  9.,
	             2.5, 3.5, 4.5,
				 4.5, 6.0, 7.5], m.data());
}

#[test]
fn test_matmul() {
    #[rustfmt::skip]
    let a = Matrix::new(Box::new([1, 2,
                                  3, 2]), 2);

    #[rustfmt::skip]
    let b = Matrix::new(Box::new([1, 2, 3,
                                  3, 4, 2]), 2);

    #[rustfmt::skip]
    assert_eq!(&[7, 10, 7, 
                 9, 14, 13], (a.get((.., ..)) * b.get((.., ..))).data());
}

#[test]
fn test_invert() {
    #[rustfmt::skip]
    let a = Matrix::new(Box::new([1., 2.,
                                  2., 0.]), 2);

    #[rustfmt::skip]
    assert_eq!(&[0.,  0.5,
                 0.5, -0.25], a.get((.., ..)).invert().unwrap().data());
}
