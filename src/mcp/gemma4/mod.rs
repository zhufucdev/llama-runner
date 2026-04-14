use llama_cpp_2::model::{LlamaChatTemplate, LlamaModel};
use minijinja::{Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::Serialize;

use crate::{
    Gemma4ApplicableChatTemplate, MessageRole, mcp::error::JinjaTemplateError,
    template::ChatTemplate,
};

mod convert;
mod tool;
pub use tool::*;

#[derive(Clone)]
pub struct Gemma4ChatTemplate<'s> {
    env: minijinja::Environment<'s>,
}

impl Gemma4ChatTemplate<'_> {
    pub fn new(tools: impl AsRef<[Gemma4Tool]>) -> Self {
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.add_global("tools", Value::from_serialize(tools.as_ref().to_vec()));
        env.set_unknown_method_callback(unknown_method_callback);
        Self { env }
    }

    pub fn with_thinking(mut self) -> Self {
        self.env.add_global("enable_thinking", true);
        self
    }
}

impl ChatTemplate for Gemma4ChatTemplate<'_> {
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
        struct MsgProxy<'s> {
            role: &'s str,
            content: &'s str,
        }
        let msg_proxies = messages
            .iter()
            .map(|(role, cnt)| MsgProxy {
                role: role.as_ref(),
                content: cnt.as_str(),
            })
            .collect::<Vec<_>>();
        Ok(template
            .render(context! { messages => msg_proxies, add_generation_prompt => true })
            .map_err(JinjaTemplateError::Render)?)
    }
}

impl Gemma4ApplicableChatTemplate for Gemma4ChatTemplate<'_> {}

impl Default for Gemma4ChatTemplate<'_> {
    fn default() -> Self {
        Self::new([])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::*;

    fn call_me_tool() -> Gemma4Tool {
        Gemma4Tool {
            function: Gemma4ToolFunction {
                name: "call_me".into(),
                description: "This is a test for your MCP tool calling capability. You SHOULD call this tool once seeing it.".into(),
                parameters: Some(Gemma4ToolFunctionParams {
                    properties: [("favourite_number".into(), Gemma4ToolFunctionParamsProp {
                        description: "Give me your favourite number from 0 to 99".into(),
                        type_: Gemma4ToolFunctionParamsPropType::Number,
                        nullable: false,
                        ..Default::default()
                    })].into_iter().collect(),
                    required: vec!["favourite_number".into()]
                }),
                ..Default::default()
            }
        }
    }

    #[tokio::test]
    async fn test_gemma4_mcp() {
        let runner = Gemma4VisionRunner::default().await.unwrap();
        let answer = runner
            .get_vlm_response(GenericVisionLmRequest {
                tmpl: Gemma4ChatTemplate::new([call_me_tool()]).with_thinking(),
                messages: vec![(
                    MessageRole::User,
                    ImageOrText::Text("Please call the `call_me` tool to continue"),
                )],
                ..Default::default()
            })
            .unwrap();
        println!("{answer}");
        assert!(answer.contains("<|tool_call>"));
        assert!(answer.contains("call:call_me"));
        assert!(answer.contains("favourite_number"));
    }

    #[tokio::test]
    async fn test_gemma4_mcp_no_thinking() {
        let runner = Gemma4VisionRunner::default().await.unwrap();
        let answer = runner
            .get_vlm_response(GenericVisionLmRequest {
                tmpl: Gemma4ChatTemplate::new([call_me_tool()]),
                messages: vec![(
                    MessageRole::User,
                    ImageOrText::Text("Please call the `call_me` tool to continue"),
                )],
                ..Default::default()
            })
            .unwrap();
        println!("{answer}");
        assert!(answer.contains("<|tool_call>"));
        assert!(answer.contains("call:call_me"));
        assert!(answer.contains("favourite_number"));
    }
}
