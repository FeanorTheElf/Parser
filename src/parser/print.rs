
pub trait Format
{
    fn format(&self, f: &mut std::fmt::Formatter, line_prefix: &str) -> std::fmt::Result;
}

