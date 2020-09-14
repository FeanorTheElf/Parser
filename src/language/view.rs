use super::types::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceView {

}

impl ConcreteView for ReferenceView {
    fn clone(&self) -> Box<dyn ConcreteView> {
        Box::new(<ReferenceView as Clone>::clone(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroView {

}

impl ConcreteView for ZeroView {
    fn clone(&self) -> Box<dyn ConcreteView> {
        Box::new(<ZeroView as Clone>::clone(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {

}

impl ConcreteView for Template {
    fn clone(&self) -> Box<dyn ConcreteView> {
        Box::new(<Template as Clone>::clone(self))
    }
}