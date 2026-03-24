pub mod error;
mod runner;
pub mod sample;

pub use runner::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lm() {
        let runner = Gemma3TextRunner::default().await.unwrap();
        let answer = runner
            .get_lm_response(TextLmRequest {
                messages: vec![(
                    MessageRole::User,
                    "What is the capital of France?".to_string(),
                )],
                ..Default::default()
            })
            .unwrap();
        assert!(answer.contains("Paris"));
    }

    #[tokio::test]
    async fn test_vlm() {
        let runner = Gemma3VisionRunner::default().await.unwrap();
        let answer = runner
            .get_vlm_response(VisionLmRequest {
                messages: vec![
                    (
                        MessageRole::User,
                        ImageOrText::Text("Which city is this building in?".into()),
                    ),
                    (
                        MessageRole::User,
                        ImageOrText::Image(
                            image::load_from_memory(include_bytes!("../assets/eiffel-tower.jpg"))
                                .unwrap(),
                        ),
                    ),
                ],
                ..Default::default()
            })
            .unwrap();
        assert!(answer.contains("Paris"));
    }
}
