use super::super::super::language::prelude::*;
use super::super::super::language::concrete_views::*;
use super::super::repr_utils::*;
use super::super::gwh_str::*;
use super::super::code_gen::*;
use super::super::repr::*;

#[derive(Debug, Clone)]
pub struct ZeroViewRepr {
    ty: Type,
    struct_name: String
}

impl ZeroViewRepr {

    pub fn new(ty: Type) -> Self {
        assert!(ty.is_view());
        assert!(ty.as_view().unwrap().get_concrete().downcast::<ViewZeros>().is_some());
        ZeroViewRepr {
            struct_name: format!("{}{}", GWH_ZERO_VIEW_STRUCT_NAME_PREFIX, ty.as_view().unwrap().view_onto.dims),
            ty: ty
        }
    }

    pub fn write_init_from_zeros(&self, name: &str, lengths: Vec<OutExpression>, g: &mut dyn BlockGenerator) -> OutResult {
        assert_eq!(self.get_dims(), lengths.len());
        g.write_variable_declaration(
            name.to_owned(), 
            self.get_out_type(), 
            Some(OutExpression::StructLiteral(
                (0..self.get_dims()).map(|i| aggregate_lengths(&lengths, i)).collect()
            )
        ))?;
        return Ok(());
    }

    fn view_type(&self) -> &ViewType {
        self.ty.as_view().unwrap()
    }
}

impl VariableStorageFuncs for ZeroViewRepr {

    fn write_struct(&self, g: &mut dyn CodeGenerator) -> OutResult {
        let mut vars = Vec::new();
        for i in 0..self.get_dims() {
            vars.push((OutType {
                base: OutPrimitiveType::SizeT,
                storage: OutStorage::Value,
                mutable: false
            }, format!("agg_len{}", i)));
        }
        g.write_struct(
            self.struct_name.clone(),
            vars
        )
    }

    fn write_init_from(&self, _name: &str, _rhs_name: &str, _rhs: &dyn VariableStorage, _g: &mut dyn BlockGenerator) -> OutResult {
        panic!("only creatable from zeros-fn")
    }

    fn write_copy_from(&self, _name: &str, _rhs_name: &str, _rhs: &dyn VariableStorage, _g: &mut dyn BlockGenerator) -> OutResult {
        panic!("immutable")
    }

    fn get_type(&self) -> &Type {
        &self.ty
    }
    
    fn get_out_type(&self) -> OutType {
        OutType {
            storage: OutStorage::Value,
            base: OutPrimitiveType::Struct(self.struct_name.to_owned()),
            mutable: false
        }
    }

    fn get_aggregated_len(&self, name: &str, index: usize) -> OutExpression {
        assert!(index < self.get_dims());
        OutExpression::StructMember(
            Box::new(OutExpression::Symbol(name.to_owned())),
            format!("agg_len{}", index)
        )
    }

    fn get_entry_at(&self, _name: &str, indices: Vec<OutExpression>) -> OutExpression {
        assert_eq!(self.get_dims(), indices.len());
        OutExpression::Literal(0)
    }

    fn get_dims(&self) -> usize {
        self.view_type().view_onto.dims
    }
}

impl VariableStorage for ZeroViewRepr {}