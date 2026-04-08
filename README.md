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

```rust
// MCP support is bare minimum
let runner = Gemma3VisionRunner::default().await.unwrap();
let call_me_tool = Tool::new(
    "call_me",
    "This is a test for your MCP tool calling capability. You SHOULD call this tool once seeing it.",
    schema_for_type::<rmcp::model::EmptyObject>(),
);
let answer = runner
    .get_vlm_response(GenericVisionLmRequest {
        // The default `VisionLmRequest` uses `ModelChatTemplate`.
        // Here you want to use something more generic.
        // You may have to implement your own template
        // for models other than the Qwen 3 series
        tmpl: Qwen3ChatTemplate::new([call_me_tool]),
        messages: vec![(
            MessageRole::User,
            ImageOrText::Text("Please call the `call_me` tool to continue"),
        )],
        ..Default::default()
    })
    .unwrap();
// Need to parse model tool call yourself
assert!(answer.contains("<tool_call>"));
assert!(answer.contains("<function=call_me>"));
```

## Credits

- [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs/tree/main):
  this library is bascially a higher level wrapper around it
- [minijinja](https://github.com/mitsuhiko/minijinja): Qwen3 chat template implementation
- [rmcp](https://github.com/modelcontextprotocol/rust-sdk/): syndication
