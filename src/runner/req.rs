use crate::{
    ImageOrText, MessageRole,
    sample::{LlguidanceSamplingParams, SimpleSamplingParams},
    template::ModelChatTemplate,
};

#[derive(Debug, Clone)]
pub struct GenericRunnerRequest<MsgCt, Tmpl> {
    pub messages: Vec<(MessageRole, MsgCt)>,
    pub sampling: SimpleSamplingParams,
    pub llguidance: Option<LlguidanceSamplingParams>,
    pub max_seq: usize,
    pub prefill: Option<String>,
    pub tmpl: Tmpl,
}

pub type GenericTextLmRequest<'a, Tmpl> = GenericRunnerRequest<&'a str, Tmpl>;
pub type GenericVisionLmRequest<'a, Tmpl> = GenericRunnerRequest<ImageOrText<'a>, Tmpl>;

pub type RunnerRequest<'a, MsgCnt> = GenericRunnerRequest<MsgCnt, ModelChatTemplate>;
pub type TextLmRequest<'a> = RunnerRequest<'a, &'a str>;
pub type VisionLmRequest<'a> = RunnerRequest<'a, ImageOrText<'a>>;

impl<M, T> Default for GenericRunnerRequest<M, T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            messages: vec![],
            sampling: Default::default(),
            llguidance: None,
            max_seq: usize::MAX,
            prefill: None,
            tmpl: Default::default(),
        }
    }
}
