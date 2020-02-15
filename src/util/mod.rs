pub mod push_iter;
pub mod ref_eq;
pub mod dynamic;
#[macro_use]
pub mod macros;
pub mod iterable;

#[allow(unused)]
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