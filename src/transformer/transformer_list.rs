use super::*;

pub trait TransformerList 
    where Self: Sized
{
    type Head: DefaultCallableTransformer;
    type Tail: TransformerList;

    fn parts(self) -> (Option<Self::Head>, Self::Tail);
    fn push_front<Next>(self, next: Next) -> TransformerListImpl<Next, Self>;
}

pub struct TransformerListImpl<Head, Tail: TransformerList> {
    head: Head,
    tail: Tail
}

impl<Head, Tail: TransformerList> TransformerList for TransformerListImpl<Head, Tail> 
    where Head: DefaultCallableTransformer
{
    type Head = Head;
    type Tail = Tail;

    fn parts(self) -> (Option<Self::Head>, Self::Tail) {
        (Some(self.head), self.tail)
    }

    fn push_front<Next>(self, next: Next) -> TransformerListImpl<Next, Self> {
        TransformerListImpl {
            head: next,
            tail: self
        }
    }
}

impl TransformerList for () {
    type Head = !;
    type Tail = ();

    fn parts(self) -> (Option<Self::Head>, Self::Tail) {
        (None, ())
    }

    fn push_front<Next>(self, next: Next) -> TransformerListImpl<Next, Self> {
        TransformerListImpl {
            head: next,
            tail: self
        }
    }
}

macro_rules! typedlist {
    () => {
        ()
    };
    ($expr:expr) => {
        ().push_front($expr)
    };
    ($expr:expr; $($tail:tt)*) => {
        (typedlist!($($tail)*)).push_front($expr)
    };
}