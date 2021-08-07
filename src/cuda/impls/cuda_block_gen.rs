use super::super::code_gen::*;
use super::super::gwh_str::*;
use super::super::super::language::compiler::CodeWriter;

use std::io::Write;

pub struct CudaHostBlockGenerator<'a> {
    out: CodeWriter<'a>,
    global_out: CodeWriter<'a>,
    unique_identifier: &'a mut usize
}

fn get_base_type_str(ty: &OutType) -> &str {
    match &ty.base {
        OutPrimitiveType::Int => "int",
        OutPrimitiveType::Float => "float",
        OutPrimitiveType::Double => "double",
        OutPrimitiveType::Bool => "bool",
        OutPrimitiveType::Long => "long",
        OutPrimitiveType::UInt => "unsigned int",
        OutPrimitiveType::SizeT => "size_t",
        OutPrimitiveType::Struct(name) => name.as_str()
    }
}

fn write_type(out: &mut CodeWriter, ty: OutType) -> OutResult {
    let base_str = if ty.mutable {
        get_base_type_str(&ty).to_owned()
    } else {
        "const".to_owned() + get_base_type_str(&ty)
    };
    match ty.storage {
        OutStorage::Value => {
            write!(out, "{}", base_str)?;
        },
        OutStorage::PtrDevice | OutStorage::PtrHost => {
            write!(out, "{}*", base_str)?;
        },
        OutStorage::SmartPtrDevice => {
            write!(out, "std::unique_ptr<{}, {}<{}>>", base_str, GWH_DELETER_DEVICE, base_str)?;
        },
        OutStorage::SmartPtrHost => {
            write!(out, "std::unique_ptr<{}, {}<{}>>", base_str, GWH_DELETER_HOST, base_str)?;
        }
    };
    return Ok(());
}

fn write_expr_value<'a, 'b>(out: &'a mut CodeWriter<'b>, expr: OutExpression, is_host: bool) -> OutResult {
    match expr {
        OutExpression::Sum(summands) => {
            out.write_separated(
                summands.into_iter().map(|s| move |out: &mut CodeWriter| write_expr_value(out, s, is_host)),
                |out| write!(out, " + ").map_err(Box::<dyn OutError>::from)
            )?;
        },
        OutExpression::Prod(factors) => {
            out.write_separated(
                factors.into_iter().map(|s| move |out: &mut CodeWriter| write_expr_value(out, s, is_host)),
                |out| write!(out, " * ").map_err(Box::<dyn OutError>::from)
            )?;
        },
        OutExpression::Call(function, params) => {
            write_expr_value(out, *function, is_host)?;
            write!(out, "(")?;
            out.write_comma_separated(
                params.into_iter().map(|s| move |out: &mut CodeWriter| write_expr_value(out, s, is_host))
            )?;
            write!(out, ")")?;
        },
        OutExpression::Symbol(sym) => {
            write!(out, "{}", sym)?;
        },
        OutExpression::Literal(lit) => {
            write!(out, "{}", lit)?;
        },
        OutExpression::ThreadIndex(_) => {
            panic!("Thread index not available in host code");
        },
        OutExpression::Allocate(ty, len) => {
            assert!(is_host, "Can only allocate memory on the host!");
            match ty.storage {
                OutStorage::Value | OutStorage::PtrDevice | OutStorage::PtrHost => {
                    panic!("Can only allocate smart pointer types!")
                },
                OutStorage::SmartPtrDevice => {
                    write!(out, "{}<{}>(", GWH_ALLOCATE_DEVICE, get_base_type_str(&ty))?;
                    write_expr_value(out, *len, is_host)?;
                    write!(out, ")")?;
                },
                OutStorage::SmartPtrHost => {
                    write!(out, "{}<{}>(", GWH_ALLOCATE_HOST, get_base_type_str(&ty))?;
                    write_expr_value(out, *len, is_host)?;
                    write!(out, ")")?;
                }
            }
        },
        OutExpression::BracketExpr(expr) => {
            write!(out, "(")?;
            write_expr_value(out, *expr, is_host)?;
            write!(out, ")")?;
        },
        OutExpression::IndexRead(ty, arr, index) => {
            match ty.storage {
                OutStorage::Value => {
                    panic!("Cannot index into scalar types")
                },
                OutStorage::SmartPtrDevice => {
                    assert!(is_host);
                    write!(out, "{}<{}>((", GWH_READ_AT, get_base_type_str(&ty))?;
                    write_expr_value(out, *arr, is_host)?;
                    write!(out, ").get(), ")?;
                    write_expr_value(out, *index, is_host)?;
                    write!(out, ")")?;
                },
                OutStorage::PtrDevice if is_host => {
                    write!(out, "{}<{}>(", GWH_READ_AT, get_base_type_str(&ty))?;
                    write_expr_value(out, *arr, is_host)?;
                    write!(out, "), ")?;
                    write_expr_value(out, *index, is_host)?;
                    write!(out, ")")?;
                },
                OutStorage::PtrDevice => {
                    write_expr_value(out, *arr, is_host)?;
                    write!(out, "[")?;
                    write_expr_value(out, *index, is_host)?;
                    write!(out, "]")?;
                },
                OutStorage::SmartPtrHost | OutStorage::PtrHost => {
                    assert!(is_host);
                    write_expr_value(out, *arr, is_host)?;
                    write!(out, "[")?;
                    write_expr_value(out, *index, is_host)?;
                    write!(out, "]")?;
                }
            }
        },
        OutExpression::StructLiteral(values) => {
            write!(out, "{{")?;
            out.write_comma_separated(
                values.into_iter().map(|v| move |out: &mut CodeWriter| write_expr_value(out, v, is_host)
            ))?;
            write!(out, "}}")?;
        },
        OutExpression::StructMember(accessed_struct, member_name) => {
            write_expr_value(out, *accessed_struct, is_host)?;
            write!(out, ".{}", member_name)?;
        },
        OutExpression::Nullptr => {
            write!(out, "nullptr")?;
        },
        OutExpression::StaticCast(ty, expr) => {
            write!(out, "static_cast<")?;
            write_type(out, ty)?;
            write!(out, ">(")?;
            write_expr_value(out, *expr, is_host)?;
            write!(out, ")")?;
        }
        OutExpression::IndexOffset(ty, arr, index) => {
            write_expr_value(out, *arr, is_host)?;
            if ty.storage.is_owned() {
                write!(out, ".get() + ")?;
            } else {
                write!(out, " + ")?;
            }
            write_expr_value(out, *index, is_host)?;
        },
    };
    return Ok(());
}

fn write_expr_value_host(out: &mut CodeWriter, expr: OutExpression) -> OutResult {
    write_expr_value(out, expr, true)
}

fn write_expr_value_device(out: &mut CodeWriter, expr: OutExpression) -> OutResult {
    write_expr_value(out, expr, false)
}

impl<'c> BlockGenerator for CudaHostBlockGenerator<'c> {

    fn write_value_assign(&mut self, ty: OutType, assignee: OutExpression, val: OutExpression) -> OutResult {
        assert!(ty.storage == OutStorage::Value);
        write_expr_value_host(&mut self.out, assignee)?;
        write!(self.out, " = ")?;
        write_expr_value_host(&mut self.out, val)?;
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_range_assign(&mut self, target_ty: OutType, target: OutExpression, source_ty: OutType, source: OutExpression, len: OutExpression) -> OutResult {
        assert!(target_ty.base == source_ty.base);
        assert!(target_ty.mutable);
        assert!(target_ty.storage != OutStorage::Value);
        assert!(source_ty.storage != OutStorage::Value);
        let copy_param = match (target_ty.storage.is_device(), source_ty.storage.is_device()) {
            (false, false) => "cudaMemcpyHostToHost",
            (false, true) => "cudaMemcpyDeviceToHost",
            (true, false) => "cudaMemcpyHostToDevice",
            (true, true) => "cudaMemcpyDeviceToDevice"
        };
        write!(self.out, "{}(cudaMemcpy(", GWH_CHECK)?;
        write_expr_value_host(&mut self.out, target)?;
        if target_ty.storage.is_owned() {
            write!(self.out, ".get()")?;
        }
        write!(self.out, ", ")?;
        write_expr_value_host(&mut self.out, source)?;
        if target_ty.storage.is_owned() {
            write!(self.out, ".get()")?;
        }
        write!(self.out, ", ")?;
        write_expr_value_host(&mut self.out, len)?;
        write!(self.out, ", {});", copy_param)?;
        self.out.newline()?;
        return Ok(());
    }
    
    fn write_assert(&mut self, value: OutExpression) -> OutResult {
        write!(self.out, "assert(")?;
        write_expr_value_host(&mut self.out, value)?;
        write!(self.out, ");")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_variable_declaration(&mut self, name: String, ty: OutType, value: Option<OutExpression>) -> OutResult {
        write_type(&mut self.out, ty)?;
        write!(self.out, " {}", name)?;
        if let Some(v) = value {
            write!(self.out, " = ")?;
            write_expr_value_host(&mut self.out, v)?;
        }
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_return(&mut self, value: Option<OutExpression>) -> OutResult {
        write!(self.out, "return")?;
        if let Some(v) = value {
            write!(self.out, " ")?;
            write_expr_value_host(&mut self.out, v)?;
        }
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_expr_statement(&mut self, expr: OutExpression) -> OutResult {
        write_expr_value_host(&mut self.out, expr)?;
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_entry_assign(&mut self, ty: OutType, arr: OutExpression, index: OutExpression, val: OutExpression) -> OutResult {
        match ty.storage {
            OutStorage::Value => {
                panic!("Cannot index into scalar types")
            },
            OutStorage::SmartPtrDevice => {
                write!(self.out, "{}<{}>((", GWH_WRITE_AT, get_base_type_str(&ty))?;
                write_expr_value_host(&mut self.out, arr)?;
                write!(self.out, ").get(), ")?;
                write_expr_value_host(&mut self.out, index)?;
                write!(self.out, ", ")?;
                write_expr_value_host(&mut self.out, val)?;
                write!(self.out, ")")?;
            },
            OutStorage::PtrDevice => {
                write!(self.out, "{}<{}>(", GWH_WRITE_AT, get_base_type_str(&ty))?;
                write_expr_value_host(&mut self.out, arr)?;
                write!(self.out, ", ")?;
                write_expr_value_host(&mut self.out, index)?;
                write!(self.out, ", ")?;
                write_expr_value_host(&mut self.out, val)?;
                write!(self.out, ")")?;
            },
            OutStorage::SmartPtrHost | OutStorage::PtrHost => {
                write_expr_value_host(&mut self.out, arr)?;
                write!(self.out, "[")?;
                write_expr_value_host(&mut self.out, index)?;
                write!(self.out, "] = ")?;
                write_expr_value_host(&mut self.out, val)?;
            }
        }
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_integer_for<'b>(
        &mut self, 
        name: String,
        init: OutExpression,
        limit: OutExpression,
        increment: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        write!(self.out, "for (size_t {} = ", name)?;
        write_expr_value_host(&mut self.out, init)?;
        write!(self.out, "; {} < ", name)?;
        write_expr_value_host(&mut self.out, limit)?;
        write!(self.out, "; {} += ", name)?;
        write_expr_value_host(&mut self.out, increment)?;
        write!(self.out, ") ")?;
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }

    fn write_if<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        write!(self.out, "if (")?;
        write_expr_value_host(&mut self.out, condition)?;
        write!(self.out, ") ")?;
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }
    
    fn write_block<'b>(
        &mut self, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }

    fn write_while<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        write!(self.out, "while (")?;
        write_expr_value_host(&mut self.out, condition)?;
        write!(self.out, ") ")?;
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }

    fn write_parallel_code<'b>(
        &mut self, 
        thread_count: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>, OutExpression) -> OutResult>,
        used_outer_vars: Vec<(OutType, String)>
    ) -> OutResult {

        let kernel_index = *self.unique_identifier;
        *self.unique_identifier += 1;
        let kernel_name = format!("{}{}", GWH_KERNEL_PREFIX, kernel_index);

        self.out.enter_block()?;
        write!(self.out, "size_t {} = static_cast<size_t>(", GWH_THREADCOUNT)?;
        write_expr_value_host(&mut self.out, thread_count)?;
        write!(self.out, ");")?;
        self.out.newline()?;

        write!(self.out, "dim3 {}(512);", GWH_BLOCKSIZE)?;
        self.out.newline()?;
        write!(self.out, "dim3 {}(({} - 1) / 512 + 1);", GWH_GRIDSIZE, GWH_THREADCOUNT)?;
        self.out.newline()?;
        write!(self.out, "{}<<< {}, {} >>>(", kernel_name, GWH_GRIDSIZE, GWH_BLOCKSIZE)?;
        self.out.enter_indented_level()?;

        self.global_out.newline()?;
        self.global_out.newline()?;
        write!(self.global_out, "__global__ void {}(", kernel_name)?;


        write!(self.out, "{}", GWH_THREADCOUNT)?;
        write!(self.global_out, "size_t {}", GWH_THREADCOUNT)?;
        for (var_type, used_var) in used_outer_vars {
            match var_type.storage {
                OutStorage::Value | OutStorage::PtrDevice => {
                    write!(self.out, ", {}", used_var)?;

                    write!(self.global_out, ", ")?;
                    write_type(&mut self.global_out, var_type)?;
                    write!(self.global_out, " {}", used_var)?;
                },
                OutStorage::SmartPtrDevice => {
                    write!(self.out, ", {}.get()", used_var)?;

                    write!(self.global_out, ", ")?;
                    let mut device_type = var_type;
                    device_type.storage = OutStorage::PtrDevice;
                    write_type(&mut self.global_out, device_type)?;
                    write!(self.global_out, " {}", used_var)?;
                },
                OutStorage::SmartPtrHost | OutStorage::PtrHost => {
                    panic!("Cannot pass host ptr to device");
                }
            }
        }

        self.out.exit_indented_level()?;
        write!(self.out, ");")?;
        self.out.exit_block()?;

        write!(self.global_out, ") ")?;
        self.global_out.enter_block()?;
        write!(self.global_out, "if (threadIdx.x < {}) ", GWH_THREADCOUNT)?;
        self.global_out.enter_block()?;

        body(Box::new(CudaDeviceBlockGenerator {
            out: CodeWriter::new(&mut self.global_out),
            unique_identifier: self.unique_identifier
        }), OutExpression::Symbol("threadIdx.x".to_owned()))?;

        self.global_out.exit_block()?;
        self.global_out.exit_block()?;

        return Ok(());
    }
}

pub struct CudaDeviceBlockGenerator<'a> {
    out: CodeWriter<'a>,
    unique_identifier: &'a mut usize
}

impl<'c> BlockGenerator for CudaDeviceBlockGenerator<'c> {

    fn write_value_assign(&mut self, ty: OutType, assignee: OutExpression, val: OutExpression) -> OutResult {
        assert!(ty.storage == OutStorage::Value);
        write_expr_value_device(&mut self.out, assignee)?;
        write!(self.out, " = ")?;
        write_expr_value_device(&mut self.out, val)?;
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_assert(&mut self, value: OutExpression) -> OutResult {
        write!(self.out, "assert(")?;
        write_expr_value_device(&mut self.out, value)?;
        write!(self.out, ");")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_range_assign(&mut self, target_ty: OutType, target: OutExpression, source_ty: OutType, source: OutExpression, len: OutExpression) -> OutResult {
        assert!(target_ty.storage == OutStorage::PtrDevice);
        assert!(source_ty.storage == OutStorage::PtrDevice);
        assert!(target_ty.base == source_ty.base);
        assert!(target_ty.mutable);
        write!(self.out, "std::memcpy(")?;
        write!(self.out, "static_cast<void*>(")?;
        write_expr_value_device(&mut self.out, target)?;
        write!(self.out, "), static_cast<void*>(")?;
        write_expr_value_device(&mut self.out, source)?;
        write!(self.out, "), sizeof({}) * (", get_base_type_str(&target_ty))?;
        write_expr_value_device(&mut self.out, len)?;
        write!(self.out, "));")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_variable_declaration(&mut self, name: String, ty: OutType, value: Option<OutExpression>) -> OutResult {
        write_type(&mut self.out, ty)?;
        write!(self.out, " {}", name)?;
        if let Some(v) = value {
            write!(self.out, " = ")?;
            write_expr_value_device(&mut self.out, v)?;
        }
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_return(&mut self, value: Option<OutExpression>) -> OutResult {
        write!(self.out, "return")?;
        if let Some(v) = value {
            write!(self.out, " ")?;
            write_expr_value_device(&mut self.out, v)?;
        }
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_expr_statement(&mut self, expr: OutExpression) -> OutResult {
        write_expr_value_device(&mut self.out, expr)?;
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_entry_assign(&mut self, ty: OutType, arr: OutExpression, index: OutExpression, val: OutExpression) -> OutResult {
        assert!(ty.storage == OutStorage::PtrDevice);
        write_expr_value_device(&mut self.out, arr)?;
        write!(self.out, "[")?;
        write_expr_value_device(&mut self.out, index)?;
        write!(self.out, "] = ")?;
        write_expr_value_device(&mut self.out, val)?;
        write!(self.out, ";")?;
        self.out.newline()?;
        return Ok(());
    }

    fn write_integer_for<'b>(
        &mut self, 
        name: String,
        init: OutExpression,
        limit: OutExpression,
        increment: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        write!(self.out, "for (size_t {} = ", name)?;
        write_expr_value_host(&mut self.out, init)?;
        write!(self.out, "; {} < ", name)?;
        write_expr_value_host(&mut self.out, limit)?;
        write!(self.out, "; {} += ", name)?;
        write_expr_value_host(&mut self.out, increment)?;
        write!(self.out, ") ")?;
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }

    fn write_if<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        write!(self.out, "if (")?;
        write_expr_value_host(&mut self.out, condition)?;
        write!(self.out, ") ")?;
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }
    
    fn write_block<'b>(
        &mut self,
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }
    
    fn write_while<'b>(
        &mut self, 
        condition: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>) -> OutResult>
    ) -> OutResult {
        write!(self.out, "while (")?;
        write_expr_value_host(&mut self.out, condition)?;
        write!(self.out, ") ")?;
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator))?;

        self.out.exit_block()?;
        return Ok(());
    }

    fn write_parallel_code<'b>(
        &mut self, 
        thread_count: OutExpression, 
        body: Box<dyn 'b + for<'a> FnOnce(Box<dyn 'a + BlockGenerator>, OutExpression) -> OutResult>,
        _used_outer_vars: Vec<(OutType, String)>
    ) -> OutResult {

        let index_var_index = *self.unique_identifier;
        *self.unique_identifier += 1;
        let index_var = format!("{}{}", GWH_FOR_LOOP_INDEX_PREFIX, index_var_index);

        let thread_count_index = *self.unique_identifier;
        *self.unique_identifier += 1;
        let thread_count_var = format!("{}{}", GWH_FOR_LOOP_THREADCOUNT_PREFIX, thread_count_index);

        self.out.enter_block()?;
        write!(self.out, "size_t {} = static_cast<size_t>(", thread_count_var)?;
        write_expr_value_device(&mut self.out, thread_count)?;
        write!(self.out, ");")?;
        self.out.newline()?;

        write!(self.out, "for (size_t {} = 0; {} < {}; ++{})", index_var, index_var, thread_count_var, index_var)?;
        self.out.enter_block()?;

        body(Box::new(self as &mut dyn BlockGenerator), OutExpression::Symbol(index_var))?;

        self.out.exit_block()?;
        
        return Ok(());
    }
}

#[cfg(test)]
use super::super::super::language::compiler::StringWriter;

#[test]
fn test_cuda_block_gen() {
    let mut out = "".to_owned();
    let mut global_out = "".to_owned();
    let mut unique_identifier = 0;

    {
        let mut out_writer = StringWriter::new(&mut out);
        let mut global_out_writer = StringWriter::new(&mut global_out);
        let mut main_gen = CudaHostBlockGenerator {
            out: CodeWriter::new(&mut out_writer),
            global_out: CodeWriter::new(&mut global_out_writer),
            unique_identifier: &mut unique_identifier
        };

        let ty = OutType {
            base: OutPrimitiveType::Int,
            storage: OutStorage::SmartPtrDevice,
            mutable: true
        };
        main_gen.write_variable_declaration(
            "foo".to_owned(), 
            ty.clone(), 
            Some(OutExpression::Allocate(ty.clone(), Box::new(OutExpression::Literal(10))))
        ).unwrap();
        let mut device_ty = ty.clone();
        device_ty.storage = OutStorage::PtrDevice;
        main_gen.write_parallel_code(
            OutExpression::Literal(10), 
            Box::new(|mut g: Box<dyn BlockGenerator>, tid| {
                g.write_entry_assign(
                    device_ty,
                    OutExpression::Symbol("foo".to_owned()),
                    tid.clone(),
                    tid
                )?;
                return Ok(());
            }), vec![(ty, "foo".to_owned())]
        ).unwrap();
    }

    assert_eq!("

__global__ void gwh_kernel_0(size_t gwh_thread_count, int* foo) {
    if (threadIdx.x < gwh_thread_count) {
        foo[threadIdx.x] = threadIdx.x;

    }
}", global_out);

assert_eq!("std::unique_ptr<int, gwh_deleter_device<int>> foo = gwh_allocate_device<int>(10);
{
    size_t gwh_thread_count = static_cast<size_t>(10);
    dim3 gwh_blocksize(512);
    dim3 gwh_gridsize((gwh_thread_count - 1) / 512 + 1);
    gwh_kernel_0<<< gwh_gridsize, gwh_blocksize >>>(
        gwh_thread_count, foo.get()
    );
}", out);
}