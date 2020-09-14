pub mod cmp;
pub mod dynamic;
pub mod iterable;
pub mod ref_eq;
pub mod dyn_lifetime;
pub mod ref_handling;
#[macro_use] pub mod singleton;

#[allow(unused)]

pub fn equal_ignore_order<T>(lhs: &Vec<T>, rhs: &Vec<T>) -> bool
where
    T: PartialEq<T>,
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

pub fn get_all_mut<'a, T, I>(vec: &'a mut Vec<T>, it: I) -> impl 'a + Iterator<Item = &'a mut T>
where
    I: Iterator<Item = usize> + 'a,
{

    it.scan((&mut vec[..], 0), |state: &mut (&'a mut [T], usize), index| {
		assert!(index >= state.1 && index < state.1 + state.0.len(), "The given index operator must yield a strictly ascending sequence bounded by the vector length");
		let mut result: Option<&'a mut T> = None;
		take_mut::take(state, |(current_slice, current_index)| {
			let (head, tail): (&'a mut [T], &'a mut [T]) = current_slice.split_at_mut(index - current_index + 1);
			result = Some(&mut head[head.len() - 1]);
			return (tail, index + 1);
		});
		return result;
	})
}

pub struct SkippingLast<I>
    where I: Iterator
{
    iter: std::iter::Peekable<I>
}

impl<I> Iterator for SkippingLast<I>
    where I: Iterator
{
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        let result = self.iter.next();
        if self.iter.peek().is_some() {
            result
        } else {
            None
        }
    }
}

pub fn skip_last<I>(it: I) -> SkippingLast<I>
    where I: Iterator
{
    SkippingLast {
        iter: it.peekable()
    }
}