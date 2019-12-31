use super::super::parser::prelude::*;

pub trait RenameAutoVars
{
    fn rename_auto_vars(&mut self);
}

impl<T: ?Sized> RenameAutoVars for T
    where T: Transformable
{
    fn rename_auto_vars(&mut self)
    {
        self.transform(&mut RecTransformer::new(|stmt| {
            match cast::<dyn StmtNode, VariableDeclarationNode>(stmt) {
                Ok(mut decl) => {
                    decl.ident = Identifier::new(&format!("{}", decl.ident));
                    decl
                },
                Err(node) => node
            }
        }, |expr| {
            match cast::<dyn UnaryExprNode, VariableNode>(expr) {
                Ok(mut var) => {
                    var.identifier = Identifier::new(&format!("{}", var.identifier));
                    var
                },
                Err(node) => node
            }
        }));
    }
}