
pub trait PIter {
    type Element;

    fn iterate<F>(f: F) 
        where F: FnMut(Self::Element) -> bool;
}