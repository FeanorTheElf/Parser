use super::*;

pub trait TypedList 
    where Self: Sized
{
    type Head;
    type Tail: TypedList;

    fn parts(self) -> (Option<Self::Head>, Self::Tail);
    fn push_front<Next>(self, next: Next) -> TypedListImpl<Next, Self>;
}

pub struct TypedListImpl<Head, Tail: TypedList> {
    head: Head,
    tail: Tail
}

impl<Head, Tail: TypedList> TypedList for TypedListImpl<Head, Tail> {
    type Head = Head;
    type Tail = Tail;

    fn parts(self) -> (Option<Self::Head>, Self::Tail) {
        (Some(self.head), self.tail)
    }

    fn push_front<Next>(self, next: Next) -> TypedListImpl<Next, Self> {
        TypedListImpl {
            head: next,
            tail: self
        }
    }
}

impl TypedList for () {
    type Head = !;
    type Tail = ();

    fn parts(self) -> (Option<Self::Head>, Self::Tail) {
        (None, ())
    }

    fn push_front<Next>(self, next: Next) -> TypedListImpl<Next, Self> {
        TypedListImpl {
            head: next,
            tail: self
        }
    }
}

macro_rules! typedlist {
    ($expr:expr) => {
        TypedList::new($expr)
    };
    ($expr:expr, $($tail:tt)*) => {
        (typedlist!($($tail)*)).push_front($expr)
    };
}