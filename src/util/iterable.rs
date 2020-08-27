pub trait LifetimeIterable<'a, T: ?Sized + 'a> {
    fn iter(&'a self) -> Box<(dyn Iterator<Item = &'a T> + 'a)>;

    fn iter_mut(&'a mut self) -> Box<(dyn Iterator<Item = &'a mut T> + 'a)>;
}

pub trait Iterable<T: ?Sized>: for<'a> LifetimeIterable<'a, T> {}

impl<T, U: ?Sized> Iterable<U> for T where T: for<'a> LifetimeIterable<'a, U> {}
