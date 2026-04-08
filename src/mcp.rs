use std::{str::Utf8Error, sync::Arc};

use llama_cpp_2::model::{LlamaChatTemplate, LlamaModel};
use minijinja::{Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
pub use rmcp::*;
use serde::Serialize;
use thiserror::Error;

use crate::{MessageRole, template::ChatTemplate};

#[derive(Clone)]
pub struct Qwen3ChatTemplate<'s> {
    env: Arc<minijinja::Environment<'s>>,
}

impl Qwen3ChatTemplate<'_> {
    pub fn new(tools: impl AsRef<[model::Tool]>) -> Self {
        let mut env = minijinja::Environment::new();
        env.add_global("tools", Value::from_serialize(tools.as_ref().to_vec()));
        env.set_unknown_method_callback(unknown_method_callback);
        Self { env: Arc::new(env) }
    }
}

impl ChatTemplate for Qwen3ChatTemplate<'_> {
    type Error = Qwen3ChatTemplateError;

    fn apply_template(
        &self,
        _model: &LlamaModel,
        model_tmpl: &LlamaChatTemplate,
        messages: &[(MessageRole, String)],
    ) -> Result<String, Self::Error> {
        let template = self.env.template_from_str(model_tmpl.to_str()?)?;
        #[derive(Serialize)]
        struct MessageProxy {
            role: String,
            content: String,
        }
        let msg_proxies = messages
            .iter()
            .map(|(role, cnt)| MessageProxy {
                role: role.to_string(),
                content: cnt.clone(),
            })
            .collect::<Vec<_>>();
        Ok(template.render(context! { messages => msg_proxies })?)
    }
}

impl Default for Qwen3ChatTemplate<'_> {
    fn default() -> Self {
        Self {
            env: Arc::new(Default::default()),
        }
    }
}

#[derive(Debug, Error)]
pub enum Qwen3ChatTemplateError {
    #[error(transparent)]
    Jinja(#[from] minijinja::Error),
    #[error("template decode error: {0}")]
    Template(#[from] Utf8Error),
}

#[cfg(test)]
mod test {
    use super::{handler::server::tool::schema_for_type, model::Tool, *};
    use crate::*;

    #[tokio::test]
    async fn test_qwen3_mcp() {
        let runner = Gemma3VisionRunner::default().await.unwrap();
        let call_me_tool = Tool::new(
            "call_me",
            "This is a test for your MCP tool calling capability. You SHOULD call this tool once seeing it.",
            schema_for_type::<rmcp::model::EmptyObject>(),
        );
        let answer = runner
            .get_vlm_response(GenericVisionLmRequest {
                tmpl: Qwen3ChatTemplate::new([call_me_tool]),
                messages: vec![(
                    MessageRole::User,
                    ImageOrText::Text("Please call the `call_me` tool to continue"),
                )],
                ..Default::default()
            })
            .unwrap();
        assert!(answer.contains("<tool_call>"));
        assert!(answer.contains("<function=call_me>"));
    }
}
