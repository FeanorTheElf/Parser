pub mod push_iter;
pub mod ref_eq;
pub mod dynamic;
#[macro_use]
pub mod macros;

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

pub fn equal_ignore_order<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> bool
	where T: PartialEq<T>
{
	if lhs.len() != rhs.len() {
		return false;
	}
	for el in lhs.iter() {
		if lhs.iter().filter(|t| t == &el).count() != rhs.iter().filter(|t| t == &el).count() {
			return false;
		}
	}
	return true;
}