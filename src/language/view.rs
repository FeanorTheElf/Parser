use super::types::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceView {

}

impl ConcreteViewFuncs for ReferenceView {
    fn identifier(&self) -> String {
        format!("r")
    }
}

impl ConcreteView for ReferenceView {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroView {

}

impl ZeroView {
    pub fn new() -> ZeroView {
        ZeroView { }
    }
}

impl ConcreteViewFuncs for ZeroView {

    fn identifier(&self) -> String {
        format!("0")
    }
}

impl ConcreteView for ZeroView {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    id: usize
}

impl Template {
    pub fn new(id: usize) -> Template {
        Template {
            id: id
        }
    }
}

impl ConcreteViewFuncs for Template {
    fn identifier(&self) -> String {
        format!("t{}", self.id)
    }
}

impl ConcreteView for Template {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexView {
    original_array_dims: usize
}

impl ConcreteViewFuncs for IndexView {
    fn identifier(&self) -> String {
        format!("i{}", self.original_array_dims)
    }
}

impl ConcreteView for IndexView {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteIndexView {
}

impl CompleteIndexView {
    pub fn new() -> Self {
        CompleteIndexView { }
    }
}

impl ConcreteViewFuncs for CompleteIndexView {
    fn identifier(&self) -> String {
        format!("i")
    }
}

impl ConcreteView for CompleteIndexView {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedView {
    chain: Vec<Box<dyn ConcreteView>>
}

impl ConcreteViewFuncs for ComposedView {

    fn identifier(&self) -> String {
        format!("c{}", self.chain.iter().fold(String::new(), |s, n| s + n.identifier().as_str()))
    }
}

impl ConcreteView for ComposedView {}

impl ComposedView {
    pub fn compose<F: ?Sized, S: ?Sized>(first: Box<F>, second: Box<S>) -> ComposedView
        where F: ConcreteView, S: ConcreteView
    {
        match (first.dynamic_box().downcast_box::<ComposedView>(), second.dynamic_box().downcast_box::<ComposedView>()) {
            (Ok(mut first), Ok(mut second)) => {
                first.chain.extend(second.chain.drain(..));
                return *first;
            },
            (Ok(mut first), Err(second)) => {
                first.chain.push(second);
                return *first;
            },
            (Err(first), Ok(mut second)) => {
                second.chain.insert(0, first);
                return *second;
            },
            (Err(first), Err(second)) => {
                return ComposedView {
                    chain: vec![first, second]
                };
            }
        };
    }
}