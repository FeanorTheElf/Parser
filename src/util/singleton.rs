
macro_rules! singleton {
    ({$expr:expr}: $ty:ty) => {
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            static mut OBJECT: std::sync::atomic::AtomicPtr<$ty> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
            ONCE.call_once(|| unsafe {
                let ptr = Box::leak(Box::new({$expr})) as *mut $ty;
                OBJECT.store(ptr, std::sync::atomic::Ordering::Release)
            });

            fn access<T>() -> &'static T 
                where T: std::marker::Sync
            {
                unsafe {
                    (OBJECT.load(std::sync::atomic::Ordering::Acquire) as *const T).as_ref::<'static>().unwrap()
                }
            }
            access::<$ty>()
        }
    };
}

use std::cell::RefCell;

#[test]
fn test_singleton() {
    fn foo() -> &'static str {
        singleton!({"test".to_owned()}: String).as_str()
    }
    let a = foo();
    let b = foo();
    assert_eq!(a as *const str, b as *const str);
}