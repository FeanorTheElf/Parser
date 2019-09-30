pub mod push_iter;
pub mod ref_eq;
pub mod dyn_clone;

use std::ops::{ Div };

fn approx_eq(lhs: &[f64], rhs: &[f64], delta: f64) -> bool {
	lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).map(f64::abs).all(|a| a < delta)
}

macro_rules! assert_approx_eq {
	($left:expr, $right:expr, $delta:expr) => {
		if !($left).iter().zip(($right).iter()).map(|(a, b)| a - b).map(f64::abs).all(|a| a < $delta) {
			panic!(r#"assertion failed: `(left == right +-{:?})`
  left: `{:?}`,
 right: `{:?}`"#, $delta, $left, $right);
		}
	};
}

pub fn find_min<I, F>(it: I, f: &F) -> Option<I::Item>
	where I: IntoIterator, F: Fn(&I::Item) -> i32
{
	let mut current_item = None;
	let mut current_value = i32::max_value();
	for item in it {
		let value = f(&item);
		if current_item.is_none() || value < current_value {
			current_item = Some(item);
			current_value = value;
		}
	}
	return current_item;
}