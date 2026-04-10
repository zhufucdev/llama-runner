pub mod error;
mod hf;
#[cfg(feature = "mcp")]
pub mod mcp;
mod runner;
pub mod sample;
pub mod template;

#[cfg(feature = "mcp")]
pub use minijinja;
pub use runner::*;

