use super::super::language::prelude::*;
use super::repr_utils::*;
use super::gwh_str::*;
use super::code_gen::*;
use super::repr::*;

#[derive(Debug, Clone)]
pub struct ArrayRepr {
    ty: Type,
    on_device: bool,
    struct_name: String
}

impl ArrayRepr {

    fn get_data_storage(&self) -> OutStorage {
        if self.on_device {
            OutStorage::SmartPtrDevice
        } else {
            OutStorage::SmartPtrHost
        }
    }

    fn get_data_type(&self) -> OutType {
        OutType {
            base: convert_base_type(self.array_type().base),
            storage: self.get_data_storage(),
            mutable: true
        }
    }

    fn array_type(&self) -> &StaticType {
        self.ty.as_static().unwrap()
    }
}

fn write_for_loop_copy_recursive(
    target: &ArrayRepr, 
    target_name: &str, 
    source: &dyn TypeRepresentation, 
    source_name: &str, 
    index: usize, 
    dims: usize, 
    g: &mut dyn BlockGenerator
) -> OutResult {
    g.write_integer_for(
        format!("{}{}", GWH_COPY_FOR_LOOP_INDEX_PREFIX, index), 
        OutExpression::Literal(0), 
        target.get_aggregated_len(target_name, index), 
        if index == 0 {
            OutExpression::Literal(1)
        } else {
            target.get_aggregated_len(target_name, index - 1)
        }, Box::new(|mut g| {
            if index < dims {
                write_for_loop_copy_recursive(target, target_name, source, source_name, index + 1, dims, &mut *g)
            } else {
                let indices = (0..dims)
                    .map(|i| OutExpression::Symbol(format!("{}{}", GWH_COPY_FOR_LOOP_INDEX_PREFIX, i)))
                    .collect::<Vec<_>>();
                g.write_entry_assign(
                    target.get_data_type(),
                    OutExpression::StructMember(
                        Box::new(OutExpression::Symbol(target_name.to_owned())),
                        format!("data")
                    ),
                    OutExpression::Sum(
                        indices.clone()
                    ),
                    source.get_entry_at(source_name, indices)
                )
            }
        }))
}

impl TypeRepresentationFuncs for ArrayRepr {

    fn write_struct(&self, g: &mut dyn CodeGenerator) -> OutResult {
        let mut vars = Vec::new();
        vars.push((self.get_data_type(), format!("data")));
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

    fn write_init_from(&self, name: &str, rhs_name: &str, rhs: &dyn TypeRepresentation, g: &mut dyn BlockGenerator) -> OutResult {
        assert_eq!(self.get_dims(), rhs.get_dims());
        assert!(self.get_dims() > 0);

        let mut struct_members = Vec::new();
        struct_members.push(OutExpression::Nullptr);
        for i in 0..self.get_dims() {
            struct_members.push(rhs.get_aggregated_len(rhs_name, i));
        }

        g.write_variable_declaration(
            name.to_owned(), OutType{
                base: OutPrimitiveType::Struct(self.struct_name.clone()),
                mutable: true,
                storage: OutStorage::Value
            }, 
            Some(OutExpression::StructLiteral(struct_members))
        )?;

        g.write_value_assign(
            self.get_data_type(), 
            OutExpression::StructMember(
                Box::new(OutExpression::Symbol(name.to_owned())),
                format!("data")
            ),
            OutExpression::Allocate(
                self.get_data_type(), 
                Box::new(OutExpression::StructMember(
                    Box::new(OutExpression::Symbol(name.to_owned())), 
                    format!("agg_len{}", self.get_dims() - 1)
                ))
            )
        )?;

        self.write_copy_from(name, rhs_name, rhs, g)?;
        return Ok(());
    }

    fn write_copy_from(&self, name: &str, rhs_name: &str, rhs: &dyn TypeRepresentation, g: &mut dyn BlockGenerator) -> OutResult {
        if let Some(rhs_cast) = rhs.downcast::<ArrayRepr>() {
            g.write_range_assign(
                self.get_data_type(),
                OutExpression::StructMember(
                    Box::new(OutExpression::Symbol(name.to_owned())),
                    format!("data")
                ),
                rhs_cast.get_data_type(), 
                OutExpression::StructMember(
                    Box::new(OutExpression::Symbol(name.to_owned())),
                    format!("data")
                ), 
                OutExpression::StructMember(
                    Box::new(OutExpression::Symbol(name.to_owned())), 
                    format!("agg_len{}", self.get_dims() - 1)
                )
            )?;
        } else {
            write_for_loop_copy_recursive(
                self, name, rhs, rhs_name, 0, self.get_dims(), g
            )?;
        }
        return Ok(());
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

    fn get_entry_at(&self, name: &str, indices: Vec<OutExpression>) -> OutExpression {
        assert_eq!(self.get_dims(), indices.len());
        OutExpression::IndexRead(
            self.get_data_type(),
            Box::new(OutExpression::StructMember(
                Box::new(OutExpression::Symbol(name.to_owned())),
                format!("data")
            )),
            Box::new(OutExpression::Sum(
                indices.into_iter().enumerate()
                    .map(|(i, index)| OutExpression::Prod(vec![
                        self.get_aggregated_len(name, i), index
                    ])).collect()
            ))
        )
    }

    fn get_dims(&self) -> usize {
        self.array_type().dims
    }
}

impl TypeRepresentation for ArrayRepr {}