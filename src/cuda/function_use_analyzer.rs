use super::super::language::prelude::*;

use std::collections::HashMap;

pub struct FunctionUse {
    pub host_called: bool,
    pub device_called: bool
}

pub fn collect_function_use_info<'a>(program: &'a Program) -> HashMap<&'a Name, FunctionUse> {
    let mut data = HashMap::new();
    for item in &program.items {
        data.insert(&item.identifier, FunctionUse {
            host_called: false,
            device_called: false
        });
    }
    for item in &program.items {
        if let Some(body) = &item.body {
            collect_function_use_info_in_block(body, &mut data, false);
        }
    }
    return data;
}

fn collect_function_use_info_in_block<'a>(block: &'a Block, data: &mut HashMap<&'a Name, FunctionUse>, is_device_context: bool) {
    for statement in &block.statements {
        for expression in statement.iter() {
            collect_function_use_info_in_expression(expression, data, is_device_context);
        }
        let is_parallel_for = statement.dynamic().is::<ParallelFor>();
        for subblock in statement.iter() {
            collect_function_use_info_in_block(subblock, data, is_device_context || is_parallel_for);
        }
    }
}

fn collect_function_use_info_in_expression<'a>(expr: &'a Expression, data: &mut HashMap<&'a Name, FunctionUse>, is_device_context: bool) {
    match expr {
        Expression::Call(call) => {
            if let Identifier::Name(name) = call.function.expect_identifier().internal_error() {
                data.get_mut(name).unwrap().host_called |= !is_device_context;
                data.get_mut(name).unwrap().device_called |= is_device_context;
            }
            for parameter in &call.parameters {
                collect_function_use_info_in_expression(parameter, data, is_device_context);
            }
        },
        _ => {}
    }
}