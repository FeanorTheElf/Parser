use std::ops::{ Add, Mul, AddAssign, MulAssign, Div, DivAssign, Sub, SubAssign };
use std::convert::From;
use std::fmt::{ Debug, Display, Formatter };

#[derive(Clone)]
#[derive(Copy)]
struct r64 {
    numerator: i64,
    denominator: u64
}

fn perfect_abs(value: i64) -> u64 {
    if value.is_negative() {
        ((!value) as u64) + 1
    } else {
        value as u64
    }
}

impl r64 {

    fn new(numerator: i64, denominator: u64) -> r64 {
        r64 {
            numerator: numerator,
            denominator: denominator
        }
    }

    fn must_reduce_for_add(s1: &r64, s2: &r64) -> bool {
        s1.denominator.leading_zeros() + s2.denominator.leading_zeros() < 64 ||
        // we have to make a signed multiply (lz >= 65) and an addition (lz >= 66)
        s1.numerator.leading_zeros() + s2.denominator.leading_zeros() < 66 ||
        s2.numerator.leading_zeros() + s1.denominator.leading_zeros() < 66
    }

    fn must_reduce_for_eq(s1: &r64, s2: &r64) -> bool {
        // we have to make a signed multiply (lz >= 65)
        s1.numerator.leading_zeros() + s2.denominator.leading_zeros() < 65 ||
        s2.numerator.leading_zeros() + s1.denominator.leading_zeros() < 65
    }

    fn must_reduce_for_mult(s1: &r64, s2: &r64) -> bool {
        s1.denominator.leading_zeros() + s2.denominator.leading_zeros() < 64 ||
        // we have to make a signed multiply (lz >= 65)
        s1.numerator.leading_zeros() + s2.numerator.leading_zeros() < 65
    }

    fn ggT(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let c = a % b;
            a = b;
            b = c;
        }
        return a;
    }

    fn reduce(&mut self) {
        let ggT: u64 = r64::ggT(perfect_abs(self.numerator), self.denominator);
        // TODO: if self.numerator = ggT = -min_value => overflow
        self.numerator /= ggT as i64;
        self.denominator /= ggT;
    }
}

impl From<i64> for r64 {

    fn from(value: i64) -> Self {
        r64::new(value, 1)
    }
}

impl AddAssign<r64> for r64 {

    fn add_assign(&mut self, rhs: r64) {
        // inf + inf or -inf + -inf
        if self.denominator == 0 && rhs.denominator == 0 && (self.numerator > 0 && rhs.numerator > 0 || self.numerator < 0 && rhs.numerator < 0) {
            return;
        }
        if r64::must_reduce_for_add(self, &rhs) {
            self.reduce();
            assert!(!r64::must_reduce_for_add(self, &rhs), "Rational arithmetic overflow: {:?} + {:?}", self, rhs);
        }
        self.numerator *= (rhs.denominator as i64);
        self.numerator += rhs.numerator * (self.denominator as i64);
        self.denominator *= rhs.denominator
    }
}

impl MulAssign<r64> for r64 {

    fn mul_assign(&mut self, rhs: r64) {
        if r64::must_reduce_for_mult(self, &rhs) {
            self.reduce();
            assert!(!r64::must_reduce_for_add(self, &rhs), "Rational arithmetic overflow: {:?} * {:?}", self, rhs);
        }
        self.numerator *= rhs.numerator;
        self.denominator *= rhs.denominator;
    }
}

impl PartialEq for r64 {
    fn eq(&self, rhs: &r64) -> bool {
        if r64::must_reduce_for_eq(self, rhs) {
            let mut reduced = *self;
            reduced.reduce();
            return *rhs == reduced;
        } else {
            return (self.denominator as i64) * rhs.numerator == (rhs.denominator as i64) * self.numerator;
        }
    }
}

impl Debug for r64 {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "({}/{})", self.numerator, self.denominator)
    }
}

impl Display for r64 {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let mut reduced: r64 = *self;
        reduced.reduce();
        return write!(f, "({}/{})", reduced.numerator, reduced.denominator);
    }
}

#[test]
fn test_ggT() {
    assert_eq!(3, r64::ggT(15, 6));
    assert_eq!(3, r64::ggT(6, 15));

    assert_eq!(7, r64::ggT(0, 7));
    assert_eq!(7, r64::ggT(7, 0));
    assert_eq!(0, r64::ggT(0, 0));

    assert_eq!(1, r64::ggT(9, 1));
    assert_eq!(1, r64::ggT(1, 9));

    assert_eq!(1, r64::ggT(13, 300));
    assert_eq!(1, r64::ggT(300, 13));
}

#[test]
fn test_add_assign() {
    let mut test: r64 = r64::from(1);
    assert_eq!(r64::from(1), test);

    test += test;
    assert_eq!(r64::from(2), test);
}