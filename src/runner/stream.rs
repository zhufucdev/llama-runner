use encoding_rs::Decoder;
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};

use crate::{
    GenericRunnerRequest, GenericTextLmRequest, GenericVisionLmRequest, TextLmRunner,
    VisionLmRunner, error::GenericRunnerError, sample::SimpleSamplingParams,
    template::ChatTemplate,
};

pub struct Gemma3Stream<'a, Message, Runner, Tmpl: ChatTemplate> {
    pub(super) ctx_source: Option<Result<LlamaContext<'a>, GenericRunnerError<Tmpl::Error>>>,
    pub(super) ctx: Option<LlamaContext<'a>>,
    pub(super) req: GenericRunnerRequest<Message, Tmpl>,
    pub(super) runner: &'a Runner,
    pub(super) model: &'a LlamaModel,
    pub(super) runtime: Option<Runtime<'a>>,
    pub(super) done: bool,
}

pub(super) struct Runtime<'a> {
    pub(super) sampler: LlamaSampler,
    pub(super) decoder: Decoder,
    pub(super) batch: LlamaBatch<'a>,
    pub(super) n_past: i32,
    pub(super) step: usize,
}

pub(super) trait PrepareRun<TmplErr> {
    fn prepare(&mut self) -> Result<(), GenericRunnerError<TmplErr>>;
}

pub struct RunnerWithRecommendedSampling<Inner> {
    pub inner: Inner,
    pub default_sampling: SimpleSamplingParams,
}

impl<'a, Message, Runner, Tmpl> Iterator for Gemma3Stream<'a, Message, Runner, Tmpl>
where
    Tmpl: ChatTemplate,
    Self: PrepareRun<Tmpl::Error>,
{
    type Item = Result<String, GenericRunnerError<Tmpl::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if let Some(result) = self.ctx_source.take() {
            match result {
                Ok(ctx) => self.ctx = Some(ctx),
                Err(err) => {
                    self.done = true;
                    return Some(Err(err));
                }
            }
        }

        if self.runtime.is_none()
            && let Err(err) = self.prepare()
        {
            self.done = true;
            return Some(Err(err));
        }
        let Runtime {
            sampler,
            decoder,
            batch,
            n_past,
            step,
        } = self.runtime.as_mut().unwrap();

        if *step >= self.req.max_seq {
            self.done = true;
            return None;
        }

        // Sample response token
        let ctx = self.ctx.as_mut().unwrap();
        let model = self.model;
        let sample_idx = batch.n_tokens() - 1;
        let mut sample = |token: LlamaToken,
                          sampler: &mut LlamaSampler,
                          ctx: &mut LlamaContext<'a>,
                          step: usize|
         -> Result<Option<String>, GenericRunnerError<Tmpl::Error>> {
            sampler.accept(token);
            if model.is_eog_token(token) {
                return Ok(None);
            }
            batch.clear();
            batch.add(token, *n_past + (step as i32), &[0], true)?;

            ctx.decode(batch)?;

            let piece = model.token_to_piece(token, decoder, true, None)?;
            Ok(Some(piece))
        };
        if let Some(prefill) = self.req.prefill.take() {
            log::debug!(target: "gemma", "prefill: {}", prefill);
            let tokens = match model.str_to_token(&prefill, AddBos::Never) {
                Ok(tokens) => tokens,
                Err(err) => {
                    return Some(Err(err.into()));
                }
            };
            log::debug!(target: "gemma", "prefill tokens: {:?}", tokens.iter().map(|t| t.0).collect::<Vec<_>>());
            for token in tokens {
                match sample(token, sampler, ctx, *step) {
                    Ok(_) => {}
                    Err(err) => return Some(Err(err.into())),
                }
                *step += 1;
            }
            Some(Ok(prefill))
        } else {
            let token = sampler.sample(ctx, sample_idx);
            match sample(token, sampler, ctx, *step) {
                Ok(Some(piece)) => {
                    *step += 1;
                    return Some(Ok(piece));
                }
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Err(err) => {
                    self.done = true;
                    return Some(Err(err));
                }
            }
        }
    }
}

impl<'s, Message, Runner, Tmpl> Gemma3Stream<'s, Message, Runner, Tmpl>
where
    Tmpl: ChatTemplate,
{
    pub(crate) fn new(
        source: Result<LlamaContext<'s>, GenericRunnerError<Tmpl::Error>>,
        req: GenericRunnerRequest<Message, Tmpl>,
        runner: &'s Runner,
        model: &'s LlamaModel,
    ) -> Self {
        Self {
            ctx_source: Some(source),
            ctx: None,
            req,
            runner,
            model,
            runtime: None,
            done: false,
        }
    }
}

impl<'a, Inner> RunnerWithRecommendedSampling<Inner> {
    fn get_preprocessed_simple_sampling(
        &self,
        sampling: SimpleSamplingParams,
    ) -> SimpleSamplingParams {
        let mut sampling = sampling;
        if sampling.top_k.is_none() {
            sampling.top_k = self.default_sampling.top_k;
        }
        if sampling.top_p.is_none() {
            sampling.top_p = self.default_sampling.top_p;
        }
        if sampling.temperature.is_none() {
            sampling.temperature = self.default_sampling.temperature;
        }
        sampling
    }
}

impl<'s, 'req, Inner, Tmpl> VisionLmRunner<'s, 'req, Tmpl> for RunnerWithRecommendedSampling<Inner>
where
    Inner: VisionLmRunner<'s, 'req, Tmpl>,
    Tmpl: ChatTemplate,
{
    fn stream_vlm_response(
        &'s self,
        mut request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>> {
        request.sampling = self.get_preprocessed_simple_sampling(request.sampling);
        self.inner.stream_vlm_response(request)
    }
}

impl<'s, 'req, Inner, Tmpl> TextLmRunner<'s, 'req, Tmpl> for RunnerWithRecommendedSampling<Inner>
where
    Inner: TextLmRunner<'s, 'req, Tmpl>,
    Tmpl: ChatTemplate,
{
    fn stream_lm_response(
        &'s self,
        mut request: GenericTextLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>>
    where
        Tmpl: ChatTemplate,
    {
        request.sampling = self.get_preprocessed_simple_sampling(request.sampling);
        self.inner.stream_lm_response(request)
    }
}

impl<Inner> From<Inner> for RunnerWithRecommendedSampling<Inner> {
    fn from(value: Inner) -> Self {
        Self {
            inner: value,
            default_sampling: SimpleSamplingParams::default(),
        }
    }
}
