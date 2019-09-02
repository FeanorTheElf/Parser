use super::super::util::find_min;
use super::indexed::{ Indexed, IndexedMut };
use super::vector::Vector;
use super::matrix::{ Matrix, MatRef, MatRefMut };
use std::mem::swap;
use std::ops::{ Index, IndexMut, MulAssign, AddAssign };
use std::vec::Vec;

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

/*
 * Transforms the matrix A into diagonal form so that
 * L' * A' * R' = L * A * R and |det L'| = |det L|,
 * |det R'| = |det R|
 */
fn smith<'a>(A: &'a mut Mat<'a>, L: &'a mut Mat<'a>, R: &'a mut Mat<'a>, pivot: usize) 
{
    let is_zero_matrix = swap_pivot_entry_if_zero(A, L, R, pivot);
    // pivot must be != 0
    if is_zero_matrix {
        return;
    }
    // pivot must divide all entries on pivot row and pivot column
    let mut changed = true;
    while changed {
        changed = transform_pivot_gcd_col(A, L, pivot) || transform_pivot_gcd_row(A, R, pivot);
    }
    // eliminate the entries on pivot row and pivot column
}

fn transform_pivot_gcd_col<'a>(A: &mut Mat<'a>, L: &mut Mat<'a>, pivot: usize)  -> bool {
    let pivot_row = pivot;
    let pivot_col = pivot;
    let mut current = find_smallest_gcd_entry_in_pivot_col(A);
    if current == 0 {
        return false;
    }
    while current != 0 {
        let (a, b) = (A[pivot_row][pivot_col], A[pivot_row + current][pivot_col]);
        let (s, t) = eea(a, b);
        let gcd = s * a + t * b;
        A.transform_two_dims_left(pivot_row, pivot_row + current, &[s, t, -b/gcd, a/gcd]);
        L.transform_two_dims_right(pivot_row, pivot_row + current, &[a/gcd, -t, b/gcd, s]);
        current = find_smallest_gcd_entry_in_pivot_col(A);
    }
    return true;
}

fn transform_pivot_gcd_row<'a>(A: &mut Mat<'a>, R: &mut Mat<'a>, pivot: usize)  -> bool {
    let pivot_row = pivot;
    let pivot_col = pivot;
    let mut current = find_smallest_gcd_entry_in_pivot_row(A);
    if current == 0 {
        return false;
    }
    while current != 0 {
        let (a, b) = (A[pivot_row][pivot_col], A[pivot_row][pivot_col + current]);
        let (s, t) = eea(a, b);
        let gcd = s * a + t * b;
        A.transform_two_dims_right(pivot_col, pivot_col + current, &[s, -b/gcd, t, a/gcd]);
        R.transform_two_dims_left(pivot_col, pivot_col + current, &[a/gcd, b/gcd, -t, s]);
        current = find_smallest_gcd_entry_in_pivot_row(A);
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

fn find_smallest_gcd_entry_in_pivot_row<'a>(A: &Mat<'a>) -> usize {
    find_min(0..A.cols(), &|col: &usize| gcd(A[0][0], A[0][*col])).unwrap()
}

fn find_smallest_gcd_entry_in_pivot_col<'a>(A: &Mat<'a>) -> usize {
    find_min(0..A.rows(), &|row: &usize| gcd(A[0][0], A[*row][0])).unwrap()
}

fn swap_pivot_entry_if_zero<'a>(A: &mut Mat<'a>, L: &mut Mat<'a>, R: &mut Mat<'a>, pivot: usize) -> bool {
    let pivot_row = pivot;
    let pivot_col = pivot;
    if let Some((row, col)) = find_not_zero(&mut A.sub_matrix(pivot_row..A.rows(), pivot_col..A.cols())) {
        A.swap_rows(pivot_row, row + pivot_row);
        L.swap_cols(pivot_row, row + pivot_row);
        A.swap_cols(pivot_col, col + pivot_col);
        R.swap_rows(pivot_col, col + pivot_col);
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

pub fn experiment() {
    let mut a = Matrix::new(Box::new([15, 10, 6, 7]), 2);
    let mut l = Matrix::<i32>::identity(2);
    let mut r = Matrix::<i32>::identity(2);
    smith(&mut a.borrow_mut(), &mut l.borrow_mut(), &mut r.borrow_mut(), 0);
    println!("{:?}", a);
    println!("{:?}", l);
    println!("{:?}", r);
    assert!(false);
}