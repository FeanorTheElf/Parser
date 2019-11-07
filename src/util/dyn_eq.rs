use std::any::Any;

pub trait DynEq {
	fn dyn_eq(&self, rhs: &dyn Any) -> bool;
}

impl<T: Any + PartialEq<T>> DynEq for T {
	fn dyn_eq(&self, rhs: &dyn Any) -> bool {
		if let Some(rhs_as_t) = rhs.downcast_ref::<T>() {
			self == rhs_as_t
		} else {
			false
		}
	}
}