# Llama Runner

A straightforward Rust library for running
[llama.cpp](https://github.com/ggerganov/llama.cpp) models locally on device.

## Example

```rust
// Download and run Gemma3 1B QAT
let runner = Gemma3TextRunner::default().await.unwrap();
let answer = runner
    .get_lm_response(TextLmRequest {
        messages: vec![(
            MessageRole::User,
            "What is the capital of France?",
        )],
        ..Default::default()
    })
    .unwrap();
assert!(answer.contains("Paris"));
```

```rust
// Download and run Qwen 3.5 4B QAT
// Note that `Gemma3VisionRunner` merely means it's capable
// of running Gemma3 vision models, not necessarily Gemma though
// Configurable using the ::new constructor
let runner = Gemma3VisionRunner::default().await.unwrap();
let eiffel_tower_im =
    image::load_from_memory(include_bytes!("../assets/eiffel-tower.jpg")).unwrap();
let answer = runner
    .get_vlm_response(VisionLmRequest {
        messages: vec![
            (
                MessageRole::User,
                ImageOrText::Text("Which city is this building in?"),
            ),
            (MessageRole::User, ImageOrText::Image(&eiffel_tower_im)),
        ],
        ..Default::default()
    })
    .unwrap();
assert!(answer.contains("Paris"));
```

## Credits

- [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs/tree/main):
  this library is bascially a higher level wrapper around it
