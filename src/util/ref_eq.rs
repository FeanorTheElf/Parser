use std::borrow::Borrow;
use std::cmp::{Eq, PartialEq};
use std::convert::From;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

#[derive(Debug)]
pub struct Ref<'a, T: ?Sized> {
    data: RefEq<'a, T>,
}

#[derive(Debug)]
pub struct RefEq<'a, T>
where
    T: ?Sized,
{
    data: &'a T,
}

impl<'a, T: ?Sized> Deref for Ref<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data.data
    }
}

impl<'a, 'b: 'a, T> Borrow<RefEq<'a, T>> for Ref<'b, T>
where
    T: ?Sized,
{
    fn borrow(&self) -> &RefEq<'a, T> {
        &self.data
    }
}

impl<'a, T> From<&'a T> for Ref<'a, T>
where
    T: ?Sized,
{
    fn from(reference: &'a T) -> Self {
        Ref {
            data: RefEq { data: reference },
        }
    }
}

impl<'a, T> From<&'a T> for RefEq<'a, T>
where
    T: ?Sized,
{
    fn from(reference: &'a T) -> Self {
        RefEq { data: reference }
    }
}

impl<'a, 'b, T> PartialEq<RefEq<'b, T>> for RefEq<'a, T>
where
    T: ?Sized,
{
    fn eq(&self, other: &RefEq<'b, T>) -> bool {
        self.data as *const T as *const () == other.data as *const T as *const ()
    }
}

impl<'a, 'b, T> Eq for RefEq<'a, T> where T: ?Sized {}

impl<'a, T> Hash for RefEq<'a, T>
where
    T: ?Sized,
{
    fn hash<H: Hasher>(&self, h: &mut H) {
        (self.data as *const T as *const ()).hash(h);
    }
}
impl<'a, 'b, T> PartialEq<Ref<'b, T>> for Ref<'a, T>
where
    T: ?Sized,
{
    fn eq(&self, other: &Ref<'b, T>) -> bool {
        self.data == other.data
    }
}

impl<'a, 'b, T> Eq for Ref<'a, T> where T: ?Sized {}

impl<'a, T> Hash for Ref<'a, T>
where
    T: ?Sized,
{
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.data.hash(h);
    }
}
