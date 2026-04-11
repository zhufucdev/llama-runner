use llama_cpp_2::model::{LlamaChatTemplate, LlamaModel};

use crate::{
    Gemma3Stream, Gemma3VisionRunner, GenericVisionLmRequest, MessageRole,
    RunnerWithRecommendedSampling, VisionLmRunner,
    error::{CreateLlamaCppRunnerError, GenericRunnerError},
    sample::SimpleSamplingParams,
    template::ChatTemplate,
};

pub const GEMMA_4_E2B_GUFF_MODEL_ID: &str = "unsloth/gemma-4-E2B-it-GGUF";
pub const GEMMA_4_E2B_GUFF_MODEL_FILENAME: &str = "gemma-4-E2B-it-Q4_0.gguf";
pub const GEMMA_4_E2B_GUFF_MULTIMODEL_FILENAME: &str = "mmproj-F16.gguf";

#[repr(transparent)]
pub struct Gemma4VisionRunner(Gemma3VisionRunner);

impl Gemma4VisionRunner {
    pub async fn default() -> Result<RunnerWithRecommendedSampling<Self>, CreateLlamaCppRunnerError>
    {
        let inner = Gemma3VisionRunner::new(
            GEMMA_4_E2B_GUFF_MODEL_ID,
            GEMMA_4_E2B_GUFF_MODEL_FILENAME,
            GEMMA_4_E2B_GUFF_MULTIMODEL_FILENAME,
            128_000u32.try_into().unwrap(),
        )
        .await?;
        Ok(RunnerWithRecommendedSampling {
            inner: Gemma4VisionRunner(inner),
            default_sampling: SimpleSamplingParams {
                top_p: Some(0.8f32),
                top_k: Some(20),
                temperature: Some(0.7f32),
                presence_penalty: Some(1.5),
                repetition_penalty: Some(1.0),
                seed: None,
            },
        })
    }
}

pub trait Gemma4ApplicableChatTemplate: ChatTemplate {}

impl<'s, 'req, Tmpl: Gemma4ApplicableChatTemplate> VisionLmRunner<'s, 'req, Tmpl>
    for Gemma4VisionRunner
{
    fn stream_vlm_response(
        &'s self,
        request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>> {
        self.0.stream_vlm_response(request)
    }
}

#[cfg(test)]
mod test {
    use crate::{mcp::Gemma4ChatTemplate, *};

    #[tokio::test]
    #[cfg(feature = "mcp")]
    async fn test_vlm() {
        let runner = Gemma4VisionRunner::default().await.unwrap();
        let eiffel_tower_im =
            image::load_from_memory(include_bytes!("../../assets/eiffel-tower.jpg")).unwrap();
        let answer = runner
            .get_vlm_response(GenericRunnerRequest {
                messages: vec![
                    (
                        MessageRole::User,
                        ImageOrText::Text("Which city is this building in?"),
                    ),
                    (MessageRole::User, ImageOrText::Image(&eiffel_tower_im)),
                ],
                tmpl: Gemma4ChatTemplate::default(),
                ..Default::default()
            })
            .unwrap();
        assert!(answer.contains("Paris"));
    }
}
