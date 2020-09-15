
pub trait DynEq: std::any::Any {
    fn dyn_eq(&self, rhs: &dyn std::any::Any) -> bool;
}

impl<T: PartialEq + std::any::Any> DynEq for T {
    fn dyn_eq(&self, rhs: &dyn std::any::Any) -> bool {
        if let Some(val) = rhs.downcast_ref::<T>() {
            self == val
        } else {
            false
        }
    }
}

pub trait Dynamic: std::any::Any {
    fn dynamic(&self) -> &dyn std::any::Any;
    fn dynamic_mut(&mut self) -> &mut dyn std::any::Any;
    fn dynamic_box(self: Box<Self>) -> Box<dyn std::any::Any>;
}

impl<T: std::any::Any> Dynamic for T {
    fn dynamic(&self) -> &dyn std::any::Any { self }
    fn dynamic_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn dynamic_box(self: Box<Self>) -> Box<dyn std::any::Any> { self }
}

macro_rules! dynamic_trait {
    ($name:ident: $supertrait:ident; $dyn_castable_name:ident) => {
        pub trait $name: std::any::Any + $dyn_castable_name + $supertrait {
        }

        impl dyn $name {
            pub fn downcast_box<T: $name>(self: Box<Self>) -> Result<Box<T>, Box<dyn $name>> {
                if self.any().is::<T>() {
                    Ok(Box::<dyn 'static + std::any::Any>::downcast::<T>(self.any_box()).unwrap())
                } else {
                    Err(self)
                }
            }

            pub fn downcast<T: $name>(&self) -> Option<&T> {
                self.any().downcast_ref::<T>()
            }

            pub fn downcast_mut<T: $name>(&mut self) -> Option<&mut T> {
                self.any_mut().downcast_mut::<T>()
            }
        }

        pub trait $dyn_castable_name {
            fn dynamic_box(self: Box<Self>) -> Box<dyn $name>;
            fn dynamic(&self) -> &dyn $name;
            fn dynamic_mut(&mut self) -> &mut dyn $name;
            fn any_box(self: Box<Self>) -> Box<dyn std::any::Any>;
            fn any(&self) -> &dyn std::any::Any;
            fn any_mut(&mut self) -> &mut dyn std::any::Any;
            fn dyn_clone(&self) -> Box<dyn $name>;
        }

        impl<T: $name + Sized + Clone> $dyn_castable_name for T {
            fn dynamic_box(self: Box<Self>) -> Box<dyn $name> { self }
            fn dynamic(&self) -> &dyn $name { self }
            fn dynamic_mut(&mut self) -> &mut dyn $name { self }
            fn any_box(self: Box<Self>) -> Box<dyn std::any::Any> { self }
            fn any(&self) -> &dyn std::any::Any { self }
            fn any_mut(&mut self) -> &mut dyn std::any::Any { self }
            fn dyn_clone(&self) -> Box<dyn $name> { Box::new(<T as Clone>::clone(self)) }
        }
    };
}
