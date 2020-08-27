
pub trait PLifetimeIter<'a> {
    type Element: 'a;
}

pub trait PIter: for<'a> PLifetimeIter<'a> {

    fn iterate<F>(self, f: F) 
        where F: for<'a> FnMut(<Self as PLifetimeIter<'a>>::Element) -> bool;

    #[inline]
    fn map<F>(self, f: F) -> Map<Self, F>
        where F: for<'b> FnMut<(<Self as PLifetimeIter<'b>>::Element,)>, Self: Sized
    {
        Map {
            iter: self,
            f: f
        }
    }

    #[inline]
    fn filter<F>(self, f: F) -> Filter<Self, F>
        where F: for<'b> FnMut<(&'b <Self as PLifetimeIter<'b>>::Element,), Output = bool>, Self: Sized
    {
        Filter {
            iter: self,
            f: f
        }
    }

    #[inline]
    fn concat<J>(self, iter: J) -> Concat<Self, J>
        where J: PIter, J: for<'b> PLifetimeIter<'b, Element = <Self as PLifetimeIter<'b>>::Element>, Self: Sized
    {
        Concat {
            fst_iter: self,
            snd_iter: iter
        }
    }

    #[inline]
    fn for_each<G>(self, mut f: G)
        where G: for<'a> FnMut<(<Self as PLifetimeIter<'a>>::Element,), Output = bool>, Self: Sized
    {
        self.iterate(|el| {
            f(el); 
            return false;
        });
    }

    #[inline]
    fn fold<G, B>(self, init: B, mut f: G) -> B
        where G: for<'a> FnMut<(B, <Self as PLifetimeIter<'a>>::Element,), Output = B>, Self: Sized
    {
        let mut current = init;
        self.iterate(|el| {
            take_mut::take(&mut current, |c| f(c, el));
            return false;
        });
        return current;
    }
}

pub struct Map<I, F> 
    where I: PIter, F: for<'b> FnMut<(<I as PLifetimeIter<'b>>::Element,)>
{
    iter: I,
    f: F
}

impl<'a, I, F> PLifetimeIter<'a> for Map<I, F> 
    where I: PIter, 
        F: 'static,
        F: for<'b> FnMut<(<I as PLifetimeIter<'b>>::Element,)>, 
        <F as FnOnce<(<I as PLifetimeIter<'a>>::Element,)>>::Output: 'a
{
    type Element = <F as FnOnce<(<I as PLifetimeIter<'a>>::Element,)>>::Output;
}

impl<I, F> PIter for Map<I, F> 
    where I: PIter, 
        F: 'static,
        F: for<'b> FnMut<(<I as PLifetimeIter<'b>>::Element,)>, 
        for<'b> <F as FnOnce<(<I as PLifetimeIter<'b>>::Element,)>>::Output: 'b
{
    fn iterate<G>(self, f: G) 
        where G: for<'a> FnMut<(<Self as PLifetimeIter<'a>>::Element,), Output = bool>
    {
        let map_fn = self.f;
        self.iter.iterate(move |el| {
            let next_el = map_fn(el);
            return f(next_el);
        })
    }
}

pub struct Filter<I, F> 
    where I: PIter, F: for<'b> FnMut<(&'b <I as PLifetimeIter<'b>>::Element,), Output = bool>
{
    iter: I,
    f: F
}

impl<'a, I, F> PLifetimeIter<'a> for Filter<I, F> 
    where I: PIter, F: for<'b> FnMut<(&'b <I as PLifetimeIter<'b>>::Element,), Output = bool>
{
    type Element = <I as PLifetimeIter<'a>>::Element;
}

impl<I, F> PIter for Filter<I, F> 
    where I: PIter, F: for<'b> FnMut<(&'b <I as PLifetimeIter<'b>>::Element,), Output = bool>
{
    fn iterate<G>(self, mut g: G) 
        where G: for<'b> FnMut<(<Self as PLifetimeIter<'b>>::Element,), Output = bool> 
    {
        let mut filter_fn = self.f;
        self.iter.iterate(|el| if filter_fn(&el) { g(el) } else { false });
    }
}

pub struct Concat<I, J>
    where I: PIter, J: PIter, J: for<'b> PLifetimeIter<'b, Element = <I as PLifetimeIter<'b>>::Element>
{
    fst_iter: I,
    snd_iter: J
}

impl<'a, I, J> PLifetimeIter<'a> for Concat<I, J>
    where I: PIter, J: PIter, J: for<'b> PLifetimeIter<'b, Element = <I as PLifetimeIter<'b>>::Element>
{
    type Element = <I as PLifetimeIter<'a>>::Element;
}

impl<I, J> PIter for Concat<I, J>
    where I: PIter, J: PIter, J: for<'b> PLifetimeIter<'b, Element = <I as PLifetimeIter<'b>>::Element>
{
    fn iterate<G>(self, mut g: G)
        where G: for<'a> FnMut<(<Self as PLifetimeIter<'a>>::Element,), Output = bool>
    {
        self.fst_iter.iterate(&mut g);
        self.snd_iter.iterate(g);
    }
}