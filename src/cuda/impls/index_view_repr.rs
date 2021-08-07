use super::super::super::language::prelude::*;
use super::super::super::language::concrete_views::*;
use super::super::repr_utils::*;
use super::super::gwh_str::*;
use super::array_repr::*;
use super::super::code_gen::*;
use super::super::repr::*;

#[derive(Debug, Clone)]
pub struct IndexViewRepr {
    ty: Type,
    on_device: bool
}

impl IndexViewRepr {

    pub fn new(ty: Type, on_device: bool) -> Self {
        assert!(ty.is_view());
        assert!(ty.as_view().unwrap().get_concrete().downcast::<ViewIndex>().is_some());
        IndexViewRepr { ty, on_device }
    }

    pub fn write_init_from_index(
        &self, 
        name: &str, 
        arr_name: &str, 
        arr_repr: &dyn VariableStorage, 
        indices: Vec<OutExpression>, 
        g: &mut dyn BlockGenerator
    ) -> OutResult {
        assert_eq!(arr_repr.get_dims(), indices.len());
        if let Some(repr) = arr_repr.downcast::<ArrayRepr>() {
            g.write_variable_declaration(
                name.to_owned(), 
                self.get_out_type(), 
                Some(OutExpression::index_offset(
                    repr.get_data_type(), 
                    OutExpression::Symbol(arr_name.to_owned()), 
                    OutExpression::sum(
                        indices.into_iter().enumerate()
                            .map(|(i, index)| OutExpression::prod(vec![
                                index, repr.get_aggregated_len(arr_name, i)
                            ].into_iter()))
                    )
                ))
            )
        } else {
            unimplemented!()
        }
    }

    fn view_type(&self) -> &ViewType {
        self.ty.as_view().unwrap()
    }
}

impl VariableStorageFuncs for IndexViewRepr {

    fn write_struct(&self, g: &mut dyn CodeGenerator) -> OutResult {
        return Ok(())
    }

    fn write_init_from(&self, name: &str, rhs_name: &str, rhs: &dyn VariableStorage, g: &mut dyn BlockGenerator) -> OutResult {
        panic!("only creatable from index access")
    }

    fn write_copy_from(&self, _name: &str, _rhs_name: &str, _rhs: &dyn VariableStorage, _g: &mut dyn BlockGenerator) -> OutResult {
        panic!("immutable")
    }

    fn get_type(&self) -> &Type {
        &self.ty
    }
    
    fn get_out_type(&self) -> OutType {
        if self.on_device {
            OutType {
                storage: OutStorage::PtrDevice,
                base: convert_base_type(self.view_type().view_onto.base),
                mutable: self.view_type().view_onto.mutable
            }
        } else {
            OutType {
                storage: OutStorage::PtrHost,
                base: convert_base_type(self.view_type().view_onto.base),
                mutable: self.view_type().view_onto.mutable
            }
        }
    }

    fn get_aggregated_len(&self, name: &str, index: usize) -> OutExpression {
        panic!("no array")
    }

    fn get_entry_at(&self, name: &str, indices: Vec<OutExpression>) -> OutExpression {
        panic!("no array")
    }

    fn get_dims(&self) -> usize {
        0
    }
}

impl VariableStorage for IndexViewRepr {}