use llama_cpp_2::model::{LlamaChatTemplate, LlamaModel};
use minijinja::{Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use rmcp::model;
use serde::Serialize;

use crate::{MessageRole, mcp::error::JinjaTemplateError, template::ChatTemplate};

#[derive(Clone)]
pub struct Qwen3ChatTemplate<'s> {
    env: minijinja::Environment<'s>,
}

impl Qwen3ChatTemplate<'_> {
    pub fn new(tools: impl AsRef<[model::Tool]>) -> Self {
        let mut env = minijinja::Environment::new();
        env.add_global("tools", Value::from_serialize(tools.as_ref().to_vec()));
        env.set_unknown_method_callback(unknown_method_callback);
        Self { env }
    }

    pub fn with_thinking(mut self) -> Self {
        self.env.add_global("enable_thinking", true);
        self
    }

    pub fn without_thinking(mut self) -> Self {
        self.env.add_global("enable_thinking", false);
        self
    }
}

impl ChatTemplate for Qwen3ChatTemplate<'_> {
    type Error = JinjaTemplateError;

    fn apply_template(
        &self,
        _model: &LlamaModel,
        model_tmpl: &LlamaChatTemplate,
        messages: &[(MessageRole, String)],
    ) -> Result<String, Self::Error> {
        let template = self
            .env
            .template_from_str(model_tmpl.to_str()?)
            .map_err(JinjaTemplateError::Parse)?;
        #[derive(Serialize)]
        struct MessageProxy<'s> {
            role: &'s str,
            content: &'s str,
        }
        let msg_proxies = messages
            .iter()
            .map(|(role, cnt)| MessageProxy {
                role: role.as_ref(),
                content: cnt.as_str(),
            })
            .collect::<Vec<_>>();
        Ok(template
            .render(context! { messages => msg_proxies, add_generation_prompt => true })
            .map_err(JinjaTemplateError::Render)?)
    }
}

impl Default for Qwen3ChatTemplate<'_> {
    fn default() -> Self {
        Self::new([])
    }
}

#[cfg(test)]
mod test {
    use super::{super::handler::server::tool::schema_for_type, model::Tool, *};
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
                tmpl: Qwen3ChatTemplate::new([call_me_tool]).with_thinking(),
                messages: vec![(
                    MessageRole::User,
                    ImageOrText::Text("Please call the `call_me` tool to continue"),
                )],
                ..Default::default()
            })
            .unwrap();
        assert!(answer.contains("</think>"));
        let (reasoning, rest) = answer.split_once("</think>").unwrap();
        assert!(!reasoning.is_empty());

        assert!(rest.contains("<tool_call>"));
        assert!(rest.contains("<function=call_me>"));
    }

    #[tokio::test]
    async fn test_qwen3_mcp_no_thinking() {
        let runner = Gemma3VisionRunner::default().await.unwrap();
        let call_me_tool = Tool::new(
            "call_me",
            "This is a test for your MCP tool calling capability. You SHOULD call this tool once seeing it.",
            schema_for_type::<rmcp::model::EmptyObject>(),
        );
        let answer = runner
            .get_vlm_response(GenericVisionLmRequest {
                tmpl: Qwen3ChatTemplate::new([call_me_tool]).without_thinking(),
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
