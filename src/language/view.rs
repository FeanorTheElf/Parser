use super::types::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceView {

}

impl ReferenceView {
    pub fn new() -> Self {
        ReferenceView {}
    }
}

impl ConcreteViewFuncs for ReferenceView {
    fn identifier(&self) -> String {
        format!("r")
    }

    fn hash(&self) -> u32 {
        HASH_ReferenceView << 24
    }

    fn replace_templated(self: Box<Self>, _value: Template, _target: &dyn ConcreteView) -> Box<dyn ConcreteView> {
        self
    }

    fn contains_templated(&self) -> bool {
        false
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

    fn hash(&self) -> u32 {
        HASH_ZeroView << 24
    }

    fn replace_templated(self: Box<Self>, _value: Template, _target: &dyn ConcreteView) -> Box<dyn ConcreteView> {
        self
    }
    
    fn contains_templated(&self) -> bool {
        false
    }
}

impl ConcreteView for ZeroView {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexView {
    original_array_dims: usize
}

impl IndexView {
    pub fn new(original_array_dims: usize) -> IndexView {
        IndexView {
            original_array_dims: original_array_dims
        }
    }
}

impl ConcreteViewFuncs for IndexView {

    fn identifier(&self) -> String {
        format!("i{}", self.original_array_dims)
    }

    fn hash(&self) -> u32 {
        (HASH_IndexView << 24) | (self.original_array_dims as u32 & 0xFFFFFF)
    }

    fn replace_templated(self: Box<Self>, _value: Template, _target: &dyn ConcreteView) -> Box<dyn ConcreteView> {
        self
    }

    fn contains_templated(&self) -> bool {
        false
    }
}

impl ConcreteView for IndexView {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedView {
    chain: Vec<Box<dyn ConcreteView>>
}

impl ConcreteViewFuncs for ComposedView {

    fn identifier(&self) -> String {
        format!("c{}", self.chain.iter().fold(String::new(), |s, n| s + n.identifier().as_str()))
    }

    fn hash(&self) -> u32 {
        (HASH_ComposedView << 24) | (self.chain.iter().map(|view| view.hash()).fold(0, |x, y| (x << 13) ^ y) & 0xFFFFFF)
    }
    
    fn replace_templated(mut self: Box<Self>, value: Template, target: &dyn ConcreteView) -> Box<dyn ConcreteView> {
        // only the last view can be a template
        assert!(self.chain.iter().rev().skip(1).all(|view| &**view == &value as &dyn ConcreteView));
        if &**self.chain.last().unwrap() == &value as &dyn ConcreteView {
            self.chain.pop();
            return Box::new(ComposedView::compose(self, target.dyn_clone()));
        } else {
            return self;
        }
    }

    fn contains_templated(&self) -> bool {
        self.chain.last().unwrap().contains_templated()
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