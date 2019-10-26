#![allow(non_camel_case_types)]

use std::ops::{ Add, Mul, AddAssign, MulAssign, Div, DivAssign, Sub, SubAssign };
use std::convert::From;
use std::fmt::{ Debug, Display, Formatter };
use std::cell::Cell;

#[derive(Clone, Copy)]
pub struct r64 {
    numerator: i64,
    denominator: i64
}

impl r64 {

    pub fn new(numerator: i64, denominator: i64) -> r64 {
        r64 {
            numerator: numerator,
            denominator: denominator
        }
    }

    // a < 0 => ggT(a, b) < 0
    // a > 0 => ggT(a, b) > 0
    // sign of b is irrelevant
    fn ggT(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let c = a % b;
            if c == 0 {
                return if (a ^ b) & i64::min_value() != 0 { -b } else { b };
            }
            b = b % c;
            a = c;
        }
        return a;
    }

    pub fn reduce(&mut self) {
        let ggT: i64 = r64::ggT(self.denominator, self.numerator);
        self.denominator /= ggT;
        self.numerator/= ggT;
    }
}

impl From<i64> for r64 {

    fn from(value: i64) -> Self {
        r64::new(value, 1)
    }
}

macro_rules! assign_or_reduce_on_failure {
    ($e:expr => $target:expr; $reduce_fst:stmt; $reduce_snd:stmt; $reduce_trd:stmt; $reduce_fth:stmt) => {
        if let Some(value) = $e {
            $target = value;
        } else {
            $reduce_fst;
            if let Some(value) = $e {
                $target = value;
            } else {
                $reduce_snd;
                if let Some(value) = $e {
                    $target = value;
                } else {
                    $reduce_trd;
                    if let Some(value) = $e {
                        $target = value;
                    } else {
                        $reduce_fth;
                        $target = ($e).unwrap();
                    }
                }
            }
        }
    };
    ($e:expr => $target:expr; $reduce_fst:stmt; $reduce_snd:stmt) => {
        if let Some(value) = $e {
            $target = value;
        } else {
            $reduce_fst;
            if let Some(value) = $e {
                $target = value;
            } else {
                $reduce_snd;
                $target = ($e).unwrap();
            }
        }
    };
    ($e:expr => $target:expr; $reduce_fst:expr) => {
        if let Some(value) = $e {
            $target = value;
        } else {
            $reduce_fst;
            $target = ($e).unwrap();
        }
    };
}

impl AddAssign<r64> for r64 {

    fn add_assign(&mut self, mut rhs: r64) {
        // inf + inf or -inf + -inf
        if self.denominator == 0 && rhs.denominator == 0 && (self.numerator > 0 && rhs.numerator > 0 || self.numerator < 0 && rhs.numerator < 0) {
            return;
        }
        assign_or_reduce_on_failure!(self.numerator.checked_mul(rhs.denominator) => self.numerator; self.reduce(); rhs.reduce());
        assign_or_reduce_on_failure!(self.denominator.checked_mul(rhs.numerator) => rhs.numerator; self.reduce(); {
            let ggT = r64::ggT(rhs.denominator, rhs.numerator);
            rhs.denominator /= ggT;
            rhs.numerator /= ggT;
            self.numerator /= ggT;
        });
        assign_or_reduce_on_failure!(self.numerator.checked_add(rhs.numerator) => self.numerator; {
            let ggT = r64::ggT(self.denominator, self.numerator);
            self.denominator /= ggT;
            self.numerator /= ggT;
            rhs.numerator /= ggT;
        }; {
            let ggT = r64::ggT(rhs.denominator, rhs.numerator);
            rhs.denominator /= ggT;
            rhs.numerator /= ggT;
            self.numerator /= ggT;
        });
        assign_or_reduce_on_failure!(self.denominator.checked_mul(rhs.denominator) => self.denominator; self.reduce(); {
            let ggT = r64::ggT(self.numerator, rhs.denominator);
            self.numerator /= ggT;
            rhs.denominator /= ggT;
        });
    }
}

impl MulAssign<r64> for r64 {

    fn mul_assign(&mut self, mut rhs: r64) {
        assign_or_reduce_on_failure!(self.numerator.checked_mul(rhs.numerator) => self.numerator; self.reduce(); rhs.reduce());
        assign_or_reduce_on_failure!(self.denominator.checked_mul(rhs.denominator) => self.denominator; self.reduce(); {
            let ggT = r64::ggT(rhs.numerator, rhs.denominator);
            rhs.denominator /= ggT;
            self.numerator /= ggT;
        });
    }
}

impl DivAssign<r64> for r64 {

    fn div_assign(&mut self, mut rhs: r64) {
        self.mul_assign(r64::new(rhs.denominator, rhs.numerator));
    }
}

impl PartialEq for r64 {
    fn eq(&self, rhs: &r64) -> bool {
        let mut this = *self;
        this.reduce();
        return rhs.numerator % self.numerator == 0 && (rhs.numerator / self.numerator) * self.denominator == rhs.denominator;
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

    assert_eq!(-3, r64::ggT(-15, 6));
    assert_eq!(3, r64::ggT(6, -15));
    assert_eq!(-3, r64::ggT(-6, -15));
}

#[test]
fn test_add_assign() {
    let mut test: r64 = r64::from(1);
    assert_eq!(r64::from(1), test);

    test += test;
    assert_eq!(r64::from(2), test);
}

#[test]
fn test_add_assign_overflow() {
    let mut a = r64::new(1 << 60, 81);
    let mut b = r64::new(1 << 60, 1 << 55);
    a += b;
    assert_eq!(r64::new((1 << 5) * 81 + (1 << 60), 81), a);
    
    a = r64::new((1 << 62) / 50 /* divisible by 3 */, 81);
    b = r64::new((1 << 62) / 81, 50);
    a += b;
    assert_eq!(r64::new(((1 << 62) / 150) * 50 + ((1 << 62) / 81) * 27, 27 * 50), a);
}

#[bench]
fn benchmark_combined_add_mult_eq(bencher: &mut test::Bencher) {
    let numerator_a = (1 << 62) / 50 /* divisible by 3 */;
    let numerator_b = (1 << 62) / 81;
    let result_numerator = 44211 * 69540552025927 + 1350;
    let not_optimized: i64 = (std::time::Instant::now().elapsed().as_secs() / 3600) as i64;
    bencher.iter(|| {
        let mut a = r64::new(numerator_a + not_optimized, 81);
        let b = r64::new(numerator_b + not_optimized, 50);
        a += b;
        let c = r64::new(1 + not_optimized, 44211);
        a *= c;
        assert_eq!(r64::new(69540552025927 + not_optimized, 1350), a);
        if (b == a) {
            a /= r64::from(100);
        } else {
            a /= c;
        }
        let mut d = r64::new(32 + not_optimized, 1024);
        d *= d;
        d *= d;
        d *= d;
        d *= c;
        if r64::new(1 + not_optimized, (1 << 40) * 44211) == d {
            a += r64::from(1);
        }
        assert_eq!(r64::new(result_numerator, 1350), a);
    });
}