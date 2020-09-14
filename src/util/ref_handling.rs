use std::cell::{Ref};
use std::borrow::Borrow;

pub trait OptionRefTranspose<'a, T> {
    fn transpose(self) -> Option<Ref<'a, T>>;
}

impl<'a, T> OptionRefTranspose<'a, T> for Ref<'a, Option<T>> {
    fn transpose(self) -> Option<Ref<'a, T>> {
        if self.borrow().is_some() {
            Some(Ref::map(self, |opt| opt.as_ref().unwrap()))
        } else {
            None
        }
    }
}