pub struct Comparing<T, F> {
    value: T,
    comparator: F,
}

impl<T, F> std::ops::Deref for Comparing<T, F> {
    type Target = T;

    fn deref(&self) -> &Self::Target {

        &self.value
    }
}

impl<T, F> Comparing<T, F> {
    pub fn new(value: T, f: F) -> Self {

        Comparing {
            value: value,
            comparator: f,
        }
    }
}

impl<T, F> std::convert::From<T> for Comparing<T, F>
where
    F: std::default::Default,
{
    fn from(value: T) -> Self {

        Comparing {
            value: value,
            comparator: F::default(),
        }
    }
}

impl<T, F> std::cmp::PartialEq for Comparing<T, F>
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    fn eq(&self, other: &Self) -> bool {

        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl<T, F> std::cmp::Eq for Comparing<T, F> where F: Fn(&T, &T) -> std::cmp::Ordering {}

impl<T, F> std::cmp::PartialOrd for Comparing<T, F>
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {

        Some(self.cmp(other))
    }
}

impl<T, F> std::cmp::Ord for Comparing<T, F>
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {

        debug_assert_eq!(
            (self.comparator)(&self.value, &other.value),
            (other.comparator)(&self.value, &other.value)
        );

        (self.comparator)(&self.value, &other.value)
    }
}
