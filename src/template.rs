use llama_cpp_2::{
    ApplyChatTemplateError,
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel},
};

use crate::MessageRole;

pub trait ChatTemplate {
    type Error;
    fn apply_template(
        &self,
        model: &LlamaModel,
        model_tmpl: &LlamaChatTemplate,
        messages: &[(MessageRole, String)],
    ) -> Result<String, Self::Error>;
}

#[derive(Clone, Default)]
pub struct ModelChatTemplate;

impl ChatTemplate for ModelChatTemplate {
    fn apply_template(
        &self,
        model: &LlamaModel,
        model_tmpl: &LlamaChatTemplate,
        messages: &[(MessageRole, String)],
    ) -> Result<String, Self::Error> {
        let llama_msg = messages
            .iter()
            .map(|(role, cnt)| LlamaChatMessage::new(role.to_string(), cnt.clone()))
            .collect::<Result<Vec<_>, _>>()
            .expect("message preprocessing failed");
        return model.apply_chat_template(model_tmpl, &llama_msg, true);
    }

    type Error = ApplyChatTemplateError;
}
