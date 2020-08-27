use std::any::Any;

pub trait DynEq {
    fn dyn_eq(&self, rhs: &dyn Any) -> bool;
}

pub trait Dynamic {
    fn dynamic(&self) -> &dyn Any;

    fn dynamic_box(self: Box<Self>) -> Box<dyn Any>;

    fn dynamic_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any> Dynamic for T {
    fn dynamic(&self) -> &dyn Any {

        self
    }

    fn dynamic_box(self: Box<Self>) -> Box<dyn Any> {

        self
    }

    fn dynamic_mut(&mut self) -> &mut dyn Any {

        self
    }
}

pub trait DynClone {
    fn dyn_clone(&self) -> Box<dyn Any>;
}

impl<T: Any + Clone> DynClone for T {
    fn dyn_clone(&self) -> Box<dyn Any> {

        Box::new(self.clone())
    }
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
