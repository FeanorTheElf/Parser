use std::borrow::Borrow;
use std::cmp::{Eq, PartialEq};
use std::convert::From;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

#[derive(Debug)]

pub struct Ptr<'a, T: ?Sized> {
    data: RefEq<'a, T>,
}

#[derive(Debug)]

pub struct RefEq<'a, T>
where
    T: ?Sized,
{
    data: &'a T,
}

impl<'a, T: ?Sized> Clone for Ptr<'a, T> {
    fn clone(&self) -> Self {

        *self
    }
}

impl<'a, T: ?Sized> Copy for Ptr<'a, T> {}

impl<'a, T: ?Sized> Clone for RefEq<'a, T> {
    fn clone(&self) -> Self {

        *self
    }
}

impl<'a, T: ?Sized> Copy for RefEq<'a, T> {}

impl<'a, T: ?Sized> Deref for Ptr<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {

        &self.data.data
    }
}

impl<'a, 'b: 'a, T> Borrow<RefEq<'a, T>> for Ptr<'b, T>
where
    T: ?Sized,
{
    fn borrow(&self) -> &RefEq<'a, T> {

        &self.data
    }
}

impl<'a, T> From<&'a T> for Ptr<'a, T>
where
    T: ?Sized,
{
    fn from(reference: &'a T) -> Self {

        Ptr {
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

impl<'a, 'b, T> PartialEq<Ptr<'b, T>> for Ptr<'a, T>
where
    T: ?Sized,
{
    fn eq(&self, other: &Ptr<'b, T>) -> bool {

        self.data == other.data
    }
}

impl<'a, 'b, T> Eq for Ptr<'a, T> where T: ?Sized {}

impl<'a, T> Hash for Ptr<'a, T>
where
    T: ?Sized,
{
    fn hash<H: Hasher>(&self, h: &mut H) {

        self.data.hash(h);
    }
}
