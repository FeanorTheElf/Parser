use std::cell::Cell;

pub trait PIter {
    type Element;

    fn iterate<F>(self, f: F) 
        where F: FnMut(Self::Element) -> bool;

    #[inline]
    fn map<F, T>(self, f: F) -> Map<Self, F, T>
        where F: FnMut(Self::Element) -> T, Self: Sized
    {
        Map {
            iter: self,
            f: f
        }
    }

    #[inline]
    fn filter<F>(self, f: F) -> Filter<Self, F>
        where F: FnMut(&Self::Element) -> bool, Self: Sized
    {
        Filter {
            iter: self,
            f: f
        }
    }

    #[inline]
    fn concat<J>(self, iter: J) -> Concat<Self, J>
        where J: PIter<Element = Self::Element>, Self: Sized
    {
        Concat {
            fst_iter: self,
            snd_iter: iter
        }
    }

    #[inline]
    fn for_each<F>(self, mut f: F)
        where F: FnMut(Self::Element), Self: Sized
    {
        self.iterate(|el| {
            f(el); 
            return false;
        });
    }

    #[inline]
    fn fold<F, B>(self, init: B, mut f: F) -> B
        where F: FnMut(B, Self::Element) -> B, Self: Sized
    {
        let mut current: Option<B> = Some(init);
        self.iterate(|el| {
            let b = std::mem::replace(&mut current, None);
            std::mem::replace(&mut current, Some(f(b.unwrap(), el)));
            return false;
        });
        return current.unwrap();
    }
    
    #[inline]
    fn reduce<F,>(self, mut f: F) -> Option<Self::Element>
        where F: FnMut(Self::Element, Self::Element) -> Self::Element, Self: Sized
    {
        let mut current: Option<Self::Element> = None;
        self.iterate(|el| {
            let b = std::mem::replace(&mut current, None);
            if let Some(b_content) = b {
                std::mem::replace(&mut current, Some(f(b_content, el)));
            } else {
                std::mem::replace(&mut current, Some(el));
            }
            return false;
        });
        return current;
    }
}

pub struct Map<I, F, T> 
    where I: PIter, F: FnMut(I::Element) -> T
{
    iter: I,
    f: F
}

impl<I, F, T> PIter for Map<I, F, T> 
    where I: PIter, F: FnMut(I::Element) -> T
{
    type Element = T;

    fn iterate<G>(self, mut g: G) 
        where G: FnMut(Self::Element) -> bool 
    {
        let mut map_fn = self.f;
        self.iter.iterate(|el| g(map_fn(el)));
    }
}

pub struct Filter<I, F> 
    where I: PIter, F: FnMut(&I::Element) -> bool
{
    iter: I,
    f: F
}

impl<I, F> PIter for Filter<I, F> 
    where I: PIter, F: FnMut(&I::Element) -> bool
{
    type Element = I::Element;

    fn iterate<G>(self, mut g: G) 
        where G: FnMut(Self::Element) -> bool 
    {
        let mut filter_fn = self.f;
        self.iter.iterate(|el| if filter_fn(&el) { g(el) } else { false });
    }
}

pub struct Concat<I, J>
    where I: PIter, J: PIter<Element = I::Element> 
{
    fst_iter: I,
    snd_iter: J
}

impl<I, J> PIter for Concat<I, J>
    where I: PIter, J: PIter<Element = I::Element>
{
    type Element = I::Element;

    fn iterate<G>(self, mut g: G)
        where G: FnMut(Self::Element) -> bool
    {
        self.fst_iter.iterate(&mut g);
        self.snd_iter.iterate(g);
    }
}