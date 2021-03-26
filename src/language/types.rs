

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Static(StaticType),
    View(ViewType)
}