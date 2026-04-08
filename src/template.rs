use llama_cpp_2::{
    ApplyChatTemplateError,
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel},
};

pub trait ChatTemplate {
    type Error;
    fn apply_template(
        &self,
        model: &LlamaModel,
        model_tmpl: &LlamaChatTemplate,
        messages: &[LlamaChatMessage],
    ) -> Result<String, Self::Error>;
}

#[derive(Clone, Default)]
pub struct ModelChatTemplate {
    pub template_name: Option<String>,
}

impl ChatTemplate for ModelChatTemplate {
    fn apply_template(
        &self,
        model: &LlamaModel,
        model_tmpl: &LlamaChatTemplate,
        messages: &[LlamaChatMessage],
    ) -> Result<String, Self::Error> {
        return model.apply_chat_template(model_tmpl, messages, true);
    }

    type Error = ApplyChatTemplateError;
}
