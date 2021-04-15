use super::super::language::prelude::*;
use super::repr_utils::*;
use super::gwh_str::*;
use super::code_gen::*;
use super::repr::*;

#[derive(Debug, Clone)]
pub struct ScalarRepr {
    ty: Type
}

impl ScalarRepr {

    pub fn new(ty: Type) -> Self {
        assert!(ty.is_scalar());
        ScalarRepr { ty }
    }

    fn scalar_type(&self) -> &StaticType {
        self.ty.as_static().unwrap()
    }
}

impl TypeRepresentationFuncs for ScalarRepr {

    fn write_struct(&self, _g: &mut dyn CodeGenerator) -> OutResult {
        Ok(())
    }

    fn write_init_from(&self, name: &str, rhs_name: &str, rhs: &dyn TypeRepresentation, g: &mut dyn BlockGenerator) -> OutResult {
        assert_eq!(0, rhs.get_dims());
        assert_eq!(0, self.get_dims());

        if rhs.get_out_type().base != self.get_out_type().base {
            g.write_variable_declaration(
                name.to_owned(), 
                OutType{
                    base: convert_base_type(self.scalar_type().base),
                    mutable: self.scalar_type().mutable,
                    storage: OutStorage::Value
                }, 
                Some(OutExpression::StaticCast(
                    self.get_out_type(), 
                    Box::new(OutExpression::Symbol(rhs_name.to_owned()))
                ))
            )?;
        } else {
            g.write_variable_declaration(
                name.to_owned(), 
                OutType{
                    base: convert_base_type(self.scalar_type().base),
                    mutable: self.scalar_type().mutable,
                    storage: OutStorage::Value
                }, 
                Some(OutExpression::Symbol(rhs_name.to_owned()))
            )?;
        }

        return Ok(());
    }

    fn write_copy_from(&self, name: &str, rhs_name: &str, rhs: &dyn TypeRepresentation, g: &mut dyn BlockGenerator) -> OutResult {
        assert_eq!(0, rhs.get_dims());
        assert_eq!(0, self.get_dims());

        if rhs.get_out_type().base != self.get_out_type().base {
            g.write_value_assign(
                self.get_out_type(), 
                OutExpression::Symbol(name.to_owned()), 
                OutExpression::StaticCast(
                    self.get_out_type(),
                    Box::new(OutExpression::Symbol(rhs_name.to_owned()))
                )
            )?;
        } else {
            g.write_value_assign(
                self.get_out_type(), 
                OutExpression::Symbol(name.to_owned()), 
                OutExpression::Symbol(rhs_name.to_owned())
            )?;
        }

        return Ok(());
    }

    fn get_out_type(&self) -> OutType {
        OutType {
            base: convert_base_type(self.scalar_type().base),
            storage: OutStorage::Value,
            mutable: self.scalar_type().mutable
        }
    }

    fn get_type(&self) -> &Type {
        &self.ty
    }

    fn get_dims(&self) -> usize {
        0
    }

    fn get_aggregated_len(&self, name: &str, index: usize) -> OutExpression {
        panic!("no array")
    }

    fn get_entry_at(&self, _name: &str, _indices: Vec<OutExpression>) -> OutExpression {
        panic!("no array")
    }
}

impl TypeRepresentation for ScalarRepr {}