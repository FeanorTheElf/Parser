use super::matrix::{ Matrix, MatRef, MatRefMut, Vector };
use std::mem::swap;

type Item = i32;
type Mat<'a> = MatRefMut<'a, Item>;

fn eea(fst: i32, snd: i32) -> (i32, i32) 
{
    let (mut a, mut b) = if fst > snd { (fst, snd) } else { (snd, fst) };
    let (mut sa, mut ta) = (1, 0);
    let (mut sb, mut tb) = (0, 1);
    while b != 0 {
        ta -= a / b * tb;
        sa -= a / b * sb;
        a = a % b;
        swap(&mut a, &mut b);
        swap(&mut sa, &mut sb);
        swap(&mut ta, &mut tb);
    }
    return if fst > snd { (sa, ta) } else { (ta, sa) };
}

fn gcd(a: i32, b: i32) -> i32 {
    let (s, t) = eea(a, b);
    return s * a + t * b;
}

pub fn diophantine_solve<'a>(A: &MatRef<'a, Item>, b: &Vector<Item>) -> Option<Vector<Item>> {
    let mut smith_A = A.clone();
    let mut iL = Matrix::<Item>::identity(A.rows());
    let mut iR = Matrix::<Item>::identity(A.cols());
    smith(&mut smith_A.borrow_mut(), &mut iL.borrow_mut(), &mut iR.borrow_mut(), 0);
    // x is solution of (L * smith_A) x = b, get result through r := R^-1 * x
    let mut x = Vector::<Item>::zero(A.cols());
    let c = iL.borrow() * b.borrow();
    for i in 0..usize::min(x.len(), A.rows()) {
        let entry = smith_A[i][i];
        if entry == 0 && c[i] != 0 {
            return None;
        } else if entry != 0 && c[i] % entry != 0 {
            return None;
        } else if entry != 0 {
            x[i] = c[i] / entry;
        } 
    }
    return Some(iR.borrow() * x.borrow());
}

/*
 * Transforms the matrix A into diagonal form and 
 * changes L, R so that L' * A' * R' = L * A * R 
 * and |det L'| = |det L|, |det R'| = |det R| holds
 * Instead of L and R, this function works on their
 * inverses iL and iR
 */
fn smith<'a>(A: &mut Mat<'a>, iL: &mut Mat<'a>, iR: &mut Mat<'a>, pivot: usize) 
{
    if pivot == A.rows() || pivot == A.cols() {
        return;
    }
    let is_zero_matrix = swap_pivot_entry_if_zero(A, iL, iR, pivot);
    // pivot must be != 0
    if is_zero_matrix {
        return;
    }
    // pivot must divide all entries on pivot row and pivot column
    let mut changed = true;
    while changed {
        changed = transform_pivot_gcd_col(A, iL, pivot) || transform_pivot_gcd_row(A, iR, pivot);
    }
    // eliminate the entries on pivot row and pivot column
    eliminate_col(A, iL, pivot);
    eliminate_row(A, iR, pivot);
    smith(A, iL, iR, pivot + 1);
}

fn eliminate_col<'a>(A: &mut Mat<'a>, iL: &mut Mat<'a>, pivot: usize) {
    for row in (pivot + 1)..A.rows() {
        let transform = [1, 0, -A[row][pivot] / A[pivot][pivot], 1];
        A.transform_two_dims_left(pivot, row, &transform);
        iL.transform_two_dims_left(pivot, row, &transform);
    }
}

fn eliminate_row<'a>(A: &mut Mat<'a>, iR: &mut Mat<'a>, pivot: usize) {
    for col in (pivot + 1)..A.cols() {
        let transform = [1, -A[pivot][col] / A[pivot][pivot], 0, 1];
        A.transform_two_dims_right(pivot, col, &transform);
        iR.transform_two_dims_left(pivot, col, &transform);
    }
}

fn transform_pivot_gcd_col<'a>(A: &mut Mat<'a>, iL: &mut Mat<'a>, pivot: usize)  -> bool {
    let pivot_row = pivot;
    let pivot_col = pivot;
    let mut current = find_smallest_gcd_entry_in_pivot_col(&A.sub_matrix(pivot..A.rows(), pivot..A.cols()));
    if current == 0 {
        return false;
    }
    while current != 0 {
        let (a, b) = (A[pivot_row][pivot_col], A[pivot_row + current][pivot_col]);
        let (s, t) = eea(a, b);
        let gcd = s * a + t * b;
        let transform = [s, t, -b/gcd, a/gcd];
        A.transform_two_dims_left(pivot_row, pivot_row + current, &transform);
        iL.transform_two_dims_left(pivot_row, pivot_row + current, &transform);
        current = find_smallest_gcd_entry_in_pivot_col(&A.sub_matrix(pivot..A.rows(), pivot..A.cols()));
    }
    return true;
}

fn transform_pivot_gcd_row<'a>(A: &mut Mat<'a>, iR: &mut Mat<'a>, pivot: usize)  -> bool {
    let pivot_row = pivot;
    let pivot_col = pivot;
    let mut current = find_smallest_gcd_entry_in_pivot_row(&A.sub_matrix(pivot..A.rows(), pivot..A.cols()));
    if current == 0 {
        return false;
    }
    while current != 0 {
        let (a, b) = (A[pivot_row][pivot_col], A[pivot_row][pivot_col + current]);
        let (s, t) = eea(a, b);
        let gcd = s * a + t * b;
        let transform = [s, -b/gcd, t, a/gcd];
        A.transform_two_dims_right(pivot_col, pivot_col + current, &transform);
        iR.transform_two_dims_left(pivot_col, pivot_col + current, &transform);
        current = find_smallest_gcd_entry_in_pivot_row(&A.sub_matrix(pivot..A.rows(), pivot..A.cols()));
    }
    return true;
}

fn find_smallest_gcd_entry<'a>(A: &Mat<'a>) -> (usize, usize) {
    let result_row = find_smallest_gcd_entry_in_pivot_col(A);
    if result_row == 0 {
        return (0, find_smallest_gcd_entry_in_pivot_row(A));
    } else {
        return (result_row, 0);
    }
}

fn find_min<T, I, F>(mut it: I, mut f: F) -> Option<T>
    where I: Iterator<Item = T>,
        F: FnMut(&T) -> i32
{
    let mut result: T = it.next()?;
    let mut current_val: i32 = f(&result);
    for item in it {
        let value = f(&item);
        if value < current_val {
            result = item;
            current_val = value;
        }
    }
    return Some(result);
}

fn find_smallest_gcd_entry_in_pivot_row<'a>(A: &Mat<'a>) -> usize {
    find_min(0..A.cols(), |col: &usize| gcd(A[0][0], A[0][*col])).unwrap()
}

fn find_smallest_gcd_entry_in_pivot_col<'a>(A: &Mat<'a>) -> usize {
    find_min(0..A.rows(), |row: &usize| gcd(A[0][0], A[*row][0])).unwrap()
}

fn swap_pivot_entry_if_zero<'a>(A: &mut Mat<'a>, iL: &mut Mat<'a>, iR: &mut Mat<'a>, pivot: usize) -> bool {
    let pivot_row = pivot;
    let pivot_col = pivot;
    if let Some((row, col)) = find_not_zero(&mut A.sub_matrix(pivot_row..A.rows(), pivot_col..A.cols())) {
        A.swap_rows(pivot_row, row + pivot_row);
        iL.swap_rows(pivot_row, row + pivot_row);
        A.swap_cols(pivot_col, col + pivot_col);
        iR.swap_cols(pivot_col, col + pivot_col);
        return false;
    } else {
        return true;
    }
}

fn find_not_zero<'a>(mat: &mut Mat<'a>) -> Option<(usize, usize)> {
    for row in 0..mat.rows() {
        for col in 0..mat.cols() {
            if mat[row][col] != 0 {
                return Some((row, col));
            }
        }
    }
    return None;
}

#[test]
fn test_eea() 
{
    assert_eq!((-1, 1), eea(6, 8));
    assert_eq!((-2, 3), eea(16, 11));
    assert_eq!((1, -2), eea(10, 4));
    assert_eq!((0, 1), eea(0, 42));
    assert_eq!((1, 0), eea(42, 42));
}

#[test]
fn test_eea_neg() 
{
    assert_eq!((-1, 1), eea(-6, -8));
    assert_eq!((2, 1), eea(15, -27));
    assert_eq!((2, 1), eea(-15, 27));
    assert_eq!((2, -1), eea(-15, -27));
    assert_eq!((1, 0), eea(-15, 0));
    assert_eq!((0, 1), eea(0, -15));
}

#[test]
fn test_diophantine() {
    let A = Matrix::new(Box::new([15, 10, 6, 7]), 2);
    let b = Vector::new(Box::new([195, 87]));
    let x = diophantine_solve(&A.borrow(), &b);
    assert_eq!(&[11, 3], x.unwrap().data());
}

#[test]
fn test_diophantine_no_solution() {
    let A = Matrix::new(Box::new([2, -2]), 1);
    let b = Vector::new(Box::new([1]));
    let x = diophantine_solve(&A.borrow(), &b);
    assert!(x.is_none());
}

#[test]
fn test_diophantine_no_solution_three_dim() {
    let A = Matrix::new(Box::new([1, 2, 0, 1, 0, 2]), 2);
    let b = Vector::new(Box::new([2, 1]));
    let x = diophantine_solve(&A.borrow(), &b);
    assert!(x.is_none());
}

#[test]
fn test_diophantine_three_dim() {
    let A = Matrix::new(Box::new([1, 2, 0, 1, 0, 2]), 2);
    let b = Vector::new(Box::new([2, 4]));
    let x = diophantine_solve(&A.borrow(), &b);
    assert_eq!(&[4, -1, 0], x.unwrap().data());
}

#[test]
fn test_diophantine_unnecessary_conditions() {
    let A = Matrix::new(Box::new([1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]), 4);
    let b = Vector::new(Box::new([2, 2, 2, 4]));
    let x = diophantine_solve(&A.borrow(), &b);
    assert_eq!(&[4, -1, 0], x.unwrap().data());
}

#[test]
fn test_diophantine_no_rational_solutions() {
    let A = Matrix::new(Box::new([1, 2, 0, 1, 2, 0, 1, 0, 2]), 3);
    let b = Vector::new(Box::new([2, 3, 4]));
    let x = diophantine_solve(&A.borrow(), &b);
    assert!(x.is_none());
}
