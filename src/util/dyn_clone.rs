use std::any::Any;

pub trait DynClone<T: ?Sized> {
    fn dyn_clone(&self) -> Box<T>;
}

impl<T: Clone + Sized> DynClone<T> for T {
    fn dyn_clone(&self) -> Box<T> {
        Box::new(self.clone())
    }
}
