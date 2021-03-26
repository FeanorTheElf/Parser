use super::prelude::*;
use super::ast_assignment::*;
use super::ast_ifwhile::*;
use super::ast_return::*;
use super::compiler::*;
use super::super::util::cmp::Comparing;

trait SimpleWriter {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

pub trait AstWriter {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError>;
}

fn get_priority(expr: &Expression) -> i32 {
    match expr {
        Expression::Call(call) => match &call.function {
            Expression::Call(_subcall) => get_priority(&call.function),
            Expression::Literal(_) => panic!("Literal not callable"),
            Expression::Variable(var) => match &var.identifier {
                Identifier::Name(_name) => i32::MAX,
                Identifier::BuiltIn(BuiltInIdentifier::ViewZeros) => i32::MAX,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionIndex) => i32::MAX,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionMul) | 
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryDiv) => 1,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionAdd) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionUnaryNeg) => 0,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionEq) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionNeq) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionLs) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionGt) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionLeq) |
                    Identifier::BuiltIn(BuiltInIdentifier::FunctionGeq) => -1,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionAnd) => -2,
                Identifier::BuiltIn(BuiltInIdentifier::FunctionOr) => -3,
            }
        },
        Expression::Literal(_) => i32::MAX,
        Expression::Variable(_) => i32::MAX
    }
}

fn write_non_builtin_call(call: &FunctionCall, out: &mut CodeWriter) -> Result<(), OutputError> {
    write_expression(&call.function, i32::MAX - 1, out)?;
    write!(out, "(")?;
    for param in &call.parameters {
        write_expression(param, i32::MIN, out)?;
        write!(out, ", ")?;
    }
    write!(out, ")")?;
    return Ok(());
}

fn write_builtin_call(call: &FunctionCall, priority: i32, op: BuiltInIdentifier, out: &mut CodeWriter) -> Result<(), OutputError> {
    match op {
        BuiltInIdentifier::FunctionIndex => {
            write_expression(&call.parameters[0], i32::MAX - 1, out)?;
            write!(out, "[")?;
            for index in call.parameters.iter().skip(1) {
                write_expression(index, i32::MIN, out)?;
                write!(out, ", ")?;
            }
            write!(out, "]")?;
        },
        BuiltInIdentifier::FunctionUnaryDiv | BuiltInIdentifier::FunctionUnaryNeg => {
            let symbol = op.get_symbol();
            write!(out, "{}", symbol)?;
            debug_assert_eq!(call.parameters.len(), 1);
            write_expression(&call.parameters[0], priority, out)?;
        },
        BuiltInIdentifier::ViewZeros => {
            write!(out, "zeros(")?;
            for index in call.parameters.iter() {
                write_expression(index, i32::MIN, out)?;
                write!(out, ", ")?;
            }
            write!(out, ")")?;
        }
        op => {
            let symbol = op.get_symbol();
            out.write_separated(call.parameters.iter().map(|p| move |out: &mut CodeWriter| write_expression(p, priority, out)), |out| write!(out, "{}", symbol).map_err(OutputError::from))?;
        }
    };
    return Ok(());
}

fn write_expression(expr: &Expression, parent_priority: i32, out: &mut CodeWriter) -> Result<(), OutputError> {
    if get_priority(expr) <= parent_priority {
        write!(out, "(")?;
    }
    match expr {
        Expression::Call(call) => match &call.function {
            Expression::Variable(var) => match &var.identifier {
                Identifier::Name(_) => write_non_builtin_call(&**call, out)?,
                Identifier::BuiltIn(op) => write_builtin_call(call, get_priority(expr), *op, out)?
            },
            _func => write_non_builtin_call(&**call, out)?
        },
        Expression::Literal(lit) => write!(out, "{}", lit.value)?,
        Expression::Variable(var) => var.identifier.write(out)?
    };
    if get_priority(expr) <= parent_priority {
        write!(out, ")")?;
    }
    return Ok(());
}

impl SimpleWriter for Name {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        if self.extra_data.len() == 0 {
            write!(out, "{}", self.name)?;
        } else {
            write!(out, "{}__{}", self.name, self.extra_data[0])?;
            for d in self.extra_data.iter().skip(1) {
                write!(out, "_{}", d)?;
            }
        }
        if self.id != 0 {
            write!(out, "#{}", self.id)?;
        }
        return Ok(());
    }
}

impl SimpleWriter for Identifier {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        match self {
            Identifier::Name(name) => name.write(out)?,
            Identifier::BuiltIn(builtin_op) => write!(out, "{}", builtin_op.get_symbol())?
        };
        return Ok(());
    }
}

impl AstWriter for Expression {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write_expression(self, i32::MIN, out)
    }
}

impl AstWriter for If {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "if (")?;
        write_expression(&self.cond, i32::MIN, out)?;
        write!(out, ")")?;
        self.body.write(out)?;
        return Ok(());
    }
}

impl AstWriter for While {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "while (")?;
        write_expression(&self.cond, i32::MIN, out)?;
        write!(out, ")")?;
        self.body.write(out)?;
        return Ok(());
    }
}

impl AstWriter for Return {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "return")?;
        if let Some(val) = &self.value {
            write!(out, " ")?;
            write_expression(val, i32::MIN, out)?;
        }
        write!(out, ";")?;
        return Ok(());
    }
}

impl AstWriter for ParallelFor {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "pfor ")?;
        for index_var in &self.index_variables {
            index_var.variable.write(out)?;
            write!(out, ": int, ")?;
        }
        for access_pattern in &self.access_pattern {
            write!(out, "with ")?;
            for entry_access in &access_pattern.entry_accesses {
                write!(out, "this[")?;
                for index in &entry_access.indices {
                    index.write(out)?;
                    write!(out, ", ")?;
                }
                write!(out, "]")?;
                if let Some(alias) = &entry_access.alias {
                    write!(out, " as ")?;
                    alias.write(out)?;
                }
                write!(out, ", ")?;
            }
        }
        self.body.write(out)?;
        return Ok(());
    }
}

impl AstWriter for dyn Statement {

    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        if let Some(statement) = self.any().downcast_ref::<If>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<While>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<Block>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<Return>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<LocalVariableDeclaration>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<Assignment>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<Goto>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<Label>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<ParallelFor>() {
            statement.write(out)
        } else if let Some(statement) = self.any().downcast_ref::<Expression>() {
            statement.write(out)?;
            write!(out, ";").map_err(OutputError::from)
        } else {
            panic!("Unknown statement type: {:?}", self)
        }
    }
}

impl AstWriter for Block {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        out.enter_block()?;
        out.write_separated(self.statements.iter().map(|s| move |out: &mut CodeWriter| s.write(out)), |out| out.newline().map_err(OutputError::from))?;
        out.exit_block()?;
        return Ok(());
    }
}

impl AstWriter for LocalVariableDeclaration {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "let ")?;
        self.declaration.variable.write(out)?;
        self.declaration.var_type.write(out)?;
        if let Some(val) = &self.value {
            write!(out, " = ")?;
            write_expression(val, i32::MIN, out)?;
        }
        write!(out, ";")?;
        return Ok(());
    }
}

impl AstWriter for Assignment {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write_expression(&self.assignee, i32::MIN, out)?;
        write!(out, " = ")?;
        write_expression(&self.value, i32::MIN, out)?;
        return Ok(());
    }
}

impl AstWriter for Label {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "@")?;
        self.label.write(out)?;
        write!(out, ":")?;
        return Ok(());
    }
}

impl AstWriter for Goto {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "goto ")?;
        self.target.write(out)?;
        write!(out, ";")?;
        return Ok(());
    }
}

impl AstWriter for Function {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        write!(out, "fn ", )?;
        self.identifier.write(out)?;
        write!(out, "(")?;
        for param in &self.params {
            param.variable.write(out)?;
            write!(out, ": {}, ", prog_lifetime.cast(param.variable_type))?;
        }
        write!(out, ")")?;
        if let Some(return_type) = &self.get_type(prog_lifetime).return_type(prog_lifetime) {
            write!(out, ": {} ", return_type)?;
        } else {
            write!(out, " ")?;
        }
        if let Some(body) = &self.body {
            body.write(out)?;
        } else {
            write!(out, "native;")?;
        }
        return Ok(());
    }
}

impl AstWriter for Program {
    fn write(&self, out: &mut CodeWriter) -> Result<(), OutputError> {
        let cmp_by_name = |lhs: &&Function, rhs: &&Function| lhs.identifier.cmp(&rhs.identifier);
        let mut sorted_items = self.items.iter().map(|f| Comparing::new(&**f, cmp_by_name)).collect::<Vec<_>>();
        sorted_items.sort();
        for item in sorted_items {
            item.write(self.types.get_lifetime(), out)?;
            out.newline()?;
        }
        return Ok(());
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayWrapper<'a, T: AstWriter + ?Sized> {
    content: &'a T
}

impl<'a, T: AstWriter + ?Sized> std::fmt::Display for DisplayWrapper<'a, T> {
    fn fmt<'b, 'c>(&self, fmt: &'b mut std::fmt::Formatter<'c>) -> std::fmt::Result {
        let mut writer = FormatterWriter::new(fmt);
        let mut out: CodeWriter = CodeWriter::new(&mut writer);
        self.content.write(&mut out).map_err(|_| std::fmt::Error)?;
        return Ok(());
    }
}

impl<'a, T: AstWriter + ?Sized> From<&'a T> for DisplayWrapper<'a, T> {
    fn from(writable: &'a T) -> DisplayWrapper<'a, T> {
        DisplayWrapper {
            content: writable
        }
    }
}
