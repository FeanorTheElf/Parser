pub mod push_iter;
pub mod ref_eq;
pub mod dyn_eq;
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