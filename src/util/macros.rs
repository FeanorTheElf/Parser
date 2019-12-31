macro_rules! cmp_attributes {
	($fst:ident; $snd:ident; $attr:ident) => {
		&(($fst).$attr) == &(($snd).$attr)
	};
}

// This will implement the partial equality trait, abiding to this contract:
// If the given type is a concrete one (i.e. no 'dyn'), compare all given attributes for equality
// If the given type is a generic one, assert that both instances are of the same concrete type
//   and that they match according to the equality relation on this concrete type
macro_rules! impl_partial_eq {
	($type:ident; ) => {
		impl PartialEq<$type> for $type 
		{
			fn eq(&self, _rhs: &$type) -> bool 
			{
				true
			}
		}
	};
	($type:ident; $fst_attr:ident) => {
		impl PartialEq<$type> for $type 
		{
			fn eq(&self, rhs: &$type) -> bool 
			{
				cmp_attributes!(self; rhs; $fst_attr)
			}
		}
	};
	($type:ident; $fst_attr:ident, $($attr:ident),*) => {
		impl PartialEq<$type> for $type 
		{
			fn eq(&self, rhs: &$type) -> bool 
			{
				cmp_attributes!(self; rhs; $fst_attr) && $(cmp_attributes!(self; rhs; $attr))&&*
			}
		}
	};
	(dyn $type:ident) => {
		impl PartialEq<dyn $type> for dyn $type 
		{
			fn eq(&self, rhs: &dyn $type) -> bool 
			{
				(*self).dyn_eq(rhs.dynamic())
			}
		}
	}
}

fn approx_eq(lhs: &[f64], rhs: &[f64], delta: f64) -> bool {
	lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).map(f64::abs).all(|a| a < delta)
}

#[allow(unused)]
macro_rules! assert_approx_eq {
	($left:expr, $right:expr, $delta:expr) => {
		if !($left).iter().zip(($right).iter()).map(|(a, b)| a - b).map(f64::abs).all(|a| a < $delta) {
			panic!(r#"assertion failed: `(left == right +-{:?})`
  left: `{:?}`,
 right: `{:?}`"#, $delta, $left, $right);
		}
	};
}
