use super::types::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewZeros {}

impl ViewFuncs for ViewZeros {}
impl View for ViewZeros {}

pub const VIEW_ZEROS: ViewZeros = ViewZeros {};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewIndex {}

impl ViewFuncs for ViewIndex {}
impl View for ViewIndex {}

pub const VIEW_INDEX: ViewIndex = ViewIndex {};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewReference {}

impl ViewFuncs for ViewReference {}
impl View for ViewReference {}

pub const VIEW_REFERENCE: ViewReference = ViewReference {};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewComposed {
    parts: Vec<Box<dyn View>>
}

impl ViewFuncs for ViewComposed {}
impl View for ViewComposed {}

impl ViewComposed {

    pub fn compose_dyn(fst: Box<dyn View>, rest: Box<dyn View>) -> Box<dyn View> {
        match (fst.downcast_box::<ViewComposed>(), rest.downcast_box::<ViewComposed>()) {
            (Ok(mut a), Ok(b)) => {
                a.parts.extend(b.parts.into_iter());
                a
            },
            (Ok(mut a), Err(b)) => {
                a.parts.push(b);
                a
            },
            (Err(a), Ok(mut b)) => {
                b.parts.insert(0, a);
                b
            },
            (Err(a), Err(b)) => {
                Box::new(ViewComposed {
                    parts: vec![a, b]
                })
            }
        }
    }

    pub fn compose<V: View>(fst: V, snd: Box<dyn View>) -> Box<dyn View> {
        Self::compose_dyn(Box::new(fst), snd)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewTemplate {
    index: usize
}

impl ViewFuncs for ViewTemplate {}
impl View for ViewTemplate {}

impl ViewTemplate {

    pub fn new(index: usize) -> ViewTemplate {
        ViewTemplate {
            index
        }
    }
}