use std::{borrow::Cow, str::Utf8Error};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum JinjaTemplateError {
    #[error("decode: {0}")]
    Decode(#[from] Utf8Error),
    #[error("parse: {0}")]
    Parse(minijinja::Error),
    #[error("render: {0}")]
    Render(minijinja::Error),
}

#[derive(Clone, Debug, Error)]
#[error("invalid field for a tool: {0}")]
pub struct ParseToolError(pub(crate) Cow<'static, str>);
