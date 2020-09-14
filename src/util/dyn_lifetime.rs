use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use std::ops::{Index, IndexMut, Deref};
use std::iter::FromIterator;

///
/// Object representing the dynamic lifetime of a concrete container with static lifetime at least 'a.
/// 
/// In other words, containers that allow dynamic lifetime references have a dynamic lifetime, represented by an
/// object of this type, where the lifetime parameter is at least the lifetime of the container. Therefore, each
/// reference with this dynamic lifetime is valid for at least lifetime 'a.
/// 
/// An instance of this object can also be thought of a proof that references with a certain lifetime id are
/// valid for 'a.
/// 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime<'a> {
    lifetime_id: u32,
    phantom: PhantomData<&'a ()>
}

impl<'a> Lifetime<'a> {
    pub fn lifetime_cast<T: ?Sized>(self, r: DynRef<T>) -> Option<&'a T> {
        if r.id != self.lifetime_id && r.id != STATIC_LIFETIME.lifetime_id {
            return None;
        }
        return Some(unsafe { r.target.as_ref() }.unwrap());
    }

    pub fn cast<T: ?Sized>(self, r: DynRef<T>) -> &'a T {
        self.lifetime_cast(r).unwrap()
    }
}

///
/// Reference to an object with dynamically managed lifetime. To access the object, you need 
/// to cast it to a specific lifetime, represented by a lifetime object that can be get from
/// the container containing the objects.
/// 
#[derive(Debug, PartialEq, Eq)]
pub struct DynRef<T: ?Sized> {
    id: u32,
    target: *const T
}

impl<T: ?Sized> Clone for DynRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized> Copy for DynRef<T> {}

impl<T: ?Sized> DynRef<T> {
    ///
    /// Returns the target of this reference, as raw pointer.
    /// 
    /// Safety: The only assumption that may be made on the result ptr is
    /// that it is not null. Apart from this, when dereferencing, the caller must
    /// ensure that the object pointed to has not been dropped and that there
    /// is no mutable reference on the target object at the same time.
    /// 
    pub fn get_target(&self) -> *const T {
        self.target
    }

    ///
    /// Given a reference that is valid for the static lifetime, this constructs
    /// a corresponding dynamic reference with dynamic lifetime STATIC_LIFETIME
    /// 
    pub fn from_static(data: &'static T) -> DynRef<T> {
        DynRef {
            id: STATIC_LIFETIME.lifetime_id,
            target: data
        }
    }
}

unsafe impl<T: ?Sized + Sync> Send for DynRef<T> {}
unsafe impl<T: ?Sized + Sync> Sync for DynRef<T> {}

///
/// Internal data of a container supporting dynamic references. This contains
/// an id that uniquely describes the lifetime of objects in this container.
/// 
#[derive(Debug)]
pub struct DynRefTargetData {
    lifetime_id: u32
}

static LIFETIME_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

pub const STATIC_LIFETIME: Lifetime<'static> = Lifetime {
    lifetime_id: 0,
    phantom: PhantomData
};

impl DynRefTargetData {

    ///
    /// Creates a new DynRefTargetData object with a new, program-wide unique
    /// lifetime id. If more than u32::MAX lifetime ids are queried while the program
    /// is running, the lifetime identifier length is not sufficient anymore to 
    /// uniquely describe each lifetime, and in this case, this function will terminate
    /// the program.
    /// 
    pub fn new() -> Self {
        let id = LIFETIME_ID_COUNTER.fetch_add(1, Ordering::AcqRel);
        if id == u32::MAX {
            std::process::exit(1);
        }
        DynRefTargetData {
            lifetime_id: id
        }
    }

    ///
    /// Returns the lifetime object describing the lifetime of the entries in this container
    /// and can be used to access dynamic references with this lifetime.
    /// 
    /// Safety: This function must not be called anymore after the target object of any reference passed 
    /// to get_ref() is invalidated. In particular, it is safe if self is dropped before any of these
    /// targets become invalid.
    /// 
    pub unsafe fn get_lifetime<'a>(&'a self) -> Lifetime<'a> {
        Lifetime { lifetime_id: self.lifetime_id, phantom: PhantomData }
    }

    ///
    /// Returns a dynamic reference for a given reference to an object in the container.
    /// 
    /// Safety: The target of the given reference must be valid until the last call of get_lifetime()
    /// on self. In particular, it is safe if self is dropped (potentially replaced) before the target 
    /// is invalidated.
    /// 
    pub unsafe fn get_ref<T: ?Sized>(&self, data: &T) -> DynRef<T> {
        DynRef { id: self.lifetime_id, target: data as *const T }
    }
}

#[derive(Debug)]
pub struct DynRefVec<T: ?Sized> {
    data: Vec<Box<T>>,
    ref_target: DynRefTargetData
}

impl<T: Clone> Clone for DynRefVec<T> {
    fn clone(&self) -> Self {
        DynRefVec {
            data: self.data.clone(),
            ref_target: DynRefTargetData::new()
        }
    }
}

impl<T: ?Sized> DynRefVec<T> {

    pub fn new() -> DynRefVec<T> {
        DynRefVec {
            data: Vec::new(),
            ref_target: DynRefTargetData::new()
        }
    }

    ///
    /// This function clears the old lifetime of this container and assigns a new one. Call this whenever
    /// a dynamic reference to this container might be invalidated by an operation (i.e. whenever an entry
    /// is removed).
    /// 
    /// Details on how this works: To access a dynamic lifetime
    /// 
    fn new_lifetime(&mut self) {
        self.ref_target = DynRefTargetData::new();
    }

    pub fn push(&mut self, obj: T) -> DynRef<T>
        where T: Sized
    {
        self.push_box(Box::new(obj))
    }

    pub fn push_from<S>(&mut self, obj: S) -> DynRef<T>
        where T: From<S>
    {
        self.push(T::from(obj))
    }

    pub fn push_box(&mut self, obj: Box<T>) -> DynRef<T> {
        // we have Box<T> as entries, so even a reallocation does not cause references to be invalidated
        self.data.push(obj);
        self.at(self.data.len() - 1)
    }

    pub fn insert(&mut self, i: usize, obj: Box<T>) -> DynRef<T> {
        // we have Box<T> as entries, so even a reallocation does not cause references to be invalidated
        self.data.insert(i, obj);
        self.at(i)
    }

    pub fn at(&self, i: usize) -> DynRef<T> {
        unsafe { self.ref_target.get_ref(self.index(i)) }
    }

    pub fn get_lifetime<'a>(&'a self) -> Lifetime<'a> {
        unsafe { self.ref_target.get_lifetime() }
    }

    pub fn drain<'b, R: std::ops::RangeBounds<usize>>(&'b mut self, r: R) -> std::vec::Drain<'b, Box<T>> {
        self.new_lifetime();
        return self.data.drain(r);
    }

    pub fn iter_mut<'b>(&'b mut self) -> std::slice::IterMut<'b, Box<T>> {
        // we do not require a new_lifetime() here, for the reason why see impl of IndexMut
        self.data.iter_mut()
    }

    pub fn iter<'b>(&'b self) -> std::slice::Iter<'b, Box<T>> {
        self.data.iter()
    }

    pub fn remove(&mut self, i: usize) -> Box<T> {
        self.new_lifetime();
        self.data.remove(i)
    }

    pub fn into_vec(self) -> Vec<Box<T>> {
        self.data
    }
}

impl<T: ?Sized> Index<usize> for DynRefVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: ?Sized> IndexMut <usize> for DynRefVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        // a user will not be able to completly remove this entry through a &mut reference, so 
        // all dynamic references stay valid (they cannot be accessed while self is mutably borrowed,
        // so there is no mutability problem)
        &mut self.data[index]
    }
}

impl<T: ?Sized> Deref for DynRefVec<T> {
    type Target = Vec<Box<T>>;

    fn deref<'b>(&'b self) -> &'b Vec<Box<T>> {
        &self.data
    }
}

impl<T: ?Sized> From<Vec<Box<T>>> for DynRefVec<T> {
    fn from(data: Vec<Box<T>>) -> Self {
        let mut result = Self::new();
        result.data = data;
        return result;
    }
}

impl<T> FromIterator<T> for DynRefVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(iter.into_iter().map(Box::new).collect::<Vec<_>>())
    }
}

impl<T: ?Sized> FromIterator<Box<T>> for DynRefVec<T> {
    fn from_iter<I: IntoIterator<Item = Box<T>>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

#[test]
fn test_ref_add_get_positive() {
    let mut vec: DynRefVec<i32> = (0..10).collect();
    let r0 = vec.at(9);
    let r1 = vec.push(11);
    let r2 = vec.push(13);
    assert_eq!(9, *vec.get_lifetime().cast(r0));
    assert_eq!(11, *vec.get_lifetime().cast(r1));
    assert_eq!(13, *vec.get_lifetime().cast(r2));
}

#[test]
fn test_ref_drain_get_negative() {
    let mut vec: DynRefVec<i32> = (0..10).collect();
    let r0 = vec.at(9);
    let r1 = vec.push(11);
    assert_eq!(Some(&9), vec.get_lifetime().lifetime_cast(r0));
    vec.drain(..);
    assert_eq!(None, vec.get_lifetime().lifetime_cast(r0));
    assert_eq!(None, vec.get_lifetime().lifetime_cast(r1));
}

#[test]
fn test_ref_modify_get_positive() {
    let mut vec: DynRefVec<i32> = (0..10).collect();
    let r0 = vec.at(1);
    assert_eq!(Some(&1), vec.get_lifetime().lifetime_cast(r0));
    vec[1] = 42;
    assert_eq!(Some(&42), vec.get_lifetime().lifetime_cast(r0));
}

#[test]
fn test_ref_remove_get_negative() {
    let mut vec: DynRefVec<i32> = (0..10).collect();
    let r0 = vec.at(1);
    assert_eq!(Some(&1), vec.get_lifetime().lifetime_cast(r0));
    vec.remove(1);
    assert_eq!(None, vec.get_lifetime().lifetime_cast(r0));
}

#[cfg(test)]
fn _test_compile_errors() {
    let mut vec: DynRefVec<i32> = DynRefVec::new();

    let r0 = vec.push(0);
    let lifetime = vec.get_lifetime();
    let mut iter_mut = vec.iter_mut();

    let r_mut = iter_mut.next().unwrap();

    // These lines should yield compiler errors

    //let r = vec.get_lifetime().cast(r0);
    //let r = lifetime.cast(r0);
    //assert_eq!(**r_mut, *r)
}