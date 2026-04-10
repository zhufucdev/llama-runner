use strum::Display;

#[derive(Debug, Clone, Display, PartialEq, Eq)]
pub enum MessageRole {
    #[strum(to_string = "assistant")]
    Assistant,
    #[strum(to_string = "user")]
    User,
    #[strum(to_string = "system")]
    System,
    #[strum(to_string = "{0}")]
    Custom(&'static str),
}

#[derive(Debug, Clone)]
pub enum ImageOrText<'a> {
    Text(&'a str),
    Image(&'a image::DynamicImage),
}
