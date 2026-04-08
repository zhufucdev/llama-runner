use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, GrammarError, LlamaContextLoadError,
    LlamaModelLoadError, StringToTokenError, TokenToStringError,
    llama_batch::BatchAddError,
    mtmd::{MtmdEvalError, MtmdInitError, MtmdTokenizeError},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CreateLlamaCppRunnerError {
    #[error("hf hub: {0}")]
    HfHub(#[from] hf_hub::api::tokio::ApiError),
    #[error("load model: {0}")]
    LoadModel(#[from] LlamaModelLoadError),
    #[error("load mtmd: {0}")]
    LoadMtmd(#[from] MtmdInitError),
    #[error("load chat template: {0}")]
    LoadChatTemplate(#[from] ChatTemplateError),
}

#[derive(Debug, Error)]
pub enum GenericRunnerError<TmplErr> {
    #[error("load context: {0}")]
    LoadContext(#[from] LlamaContextLoadError),
    #[error("apply chat template: {0}")]
    ApplyChatTemplate(TmplErr),
    #[error("mtmd tokenize: {0}")]
    MtmdTokenize(#[from] MtmdTokenizeError),
    #[error("token-string conversion: {0}")]
    TokenToString(#[from] TokenToStringError),
    #[error("string-token conversion: {0}")]
    StringToToken(#[from] StringToTokenError),
    #[error("batch add: {0}")]
    BatchAdd(#[from] BatchAddError),
    #[error("mtmd eval: {0}")]
    MtmdEval(#[from] MtmdEvalError),
    #[error("batch decode: {0}")]
    BatchDecode(#[from] DecodeError),
    #[error("llguidance: {0}")]
    Llguidance(#[from] GrammarError),
}

pub type RunnerError = GenericRunnerError<ApplyChatTemplateError>;
