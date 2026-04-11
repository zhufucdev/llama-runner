mod ext;
mod gemma3;
mod gemma4;
mod msg;
mod req;
mod stream;

use std::sync::LazyLock;

pub use ext::*;
pub use gemma3::*;
pub use gemma4::*;
use llama_cpp_2::llama_backend::LlamaBackend;
pub use msg::*;
pub use req::*;
pub use stream::*;

use crate::{error::GenericRunnerError, template::ChatTemplate};

static LLAMA_BACKEND: LazyLock<LlamaBackend> = LazyLock::new(|| {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
    LlamaBackend::init().unwrap()
});

pub trait TextLmRunner<'s, 'req, Tmpl>
where
    Tmpl: ChatTemplate,
{
    fn stream_lm_response(
        &'s self,
        request: GenericTextLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>>;
}

pub trait VisionLmRunner<'s, 'req, Tmpl>
where
    Tmpl: ChatTemplate,
{
    fn stream_vlm_response(
        &'s self,
        request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>>;
}
