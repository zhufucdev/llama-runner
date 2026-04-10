use std::{
    io::IsTerminal,
    num::NonZeroU32,
    path::{Path, PathBuf},
    str::FromStr,
};

use encoding_rs::UTF_8;
use hf_hub::api::tokio::ApiBuilder;
use llama_cpp_2::{
    LlamaContextLoadError,
    context::{LlamaContext, params::LlamaContextParams},
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatTemplate, LlamaModel},
    mtmd::{self, MtmdBitmap, MtmdContext, MtmdInputText},
    sampling::LlamaSampler,
};

use crate::{
    GenericTextLmRequest, GenericVisionLmRequest, ImageOrText, MessageRole, TextLmRunner,
    VisionLmRunner,
    error::{CreateLlamaCppRunnerError, GenericRunnerError},
    runner::{Gemma3Stream, LLAMA_BACKEND, PrepareRun, Runtime},
    sample::SimpleSamplingParams,
    template::ChatTemplate,
};

pub const QWEN_3D5_4B_GUFF_MODEL_ID: &str = "unsloth/Qwen3.5-4B-GGUF";
pub const QWEN_3D5_4B_GUFF_MODDEL_FILENAME: &str = "Qwen3.5-4B-Q4_K_M.gguf";
pub const QWEN_3D5_4B_GUFF_MULTIMODEL_FILENAME: &str = "mmproj-F16.gguf";

pub const GEMMA_3_1B_GUFF_MODEL_ID: &str = "google/gemma-3-1b-it-qat-q4_0-gguf";
pub const GEMMA_3_1B_GUFF_MODEL_FILENAME: &str = "gemma-3-1b-it-q4_0.gguf";

pub struct Gemma3TextRunner {
    model: LlamaModel,
    llama_template: LlamaChatTemplate,
    ctx_size: NonZeroU32,
}

impl Gemma3TextRunner {
    pub async fn new(
        model_id: impl ToString,
        model_file: impl AsRef<str>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let repo = build_hf_api()?.model(model_id.to_string());
        Self::from_file(repo.get(model_file.as_ref()).await?, ctx_size)
    }

    pub fn recommend_sampling() -> SimpleSamplingParams {
        SimpleSamplingParams {
            top_p: Some(0.95f32),
            top_k: Some(64),
            temperature: Some(1f32),
            ..Default::default()
        }
    }

    pub fn from_file(
        model_file: impl AsRef<Path>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, model_file, &Default::default())?;

        let chat_template = model.chat_template(None)?;
        Ok(Self {
            model,
            llama_template: chat_template,
            ctx_size,
        })
    }

    pub async fn default() -> Result<RunnerWithRecommendedSampling<Self>, CreateLlamaCppRunnerError>
    {
        let inner = Self::new(
            GEMMA_3_1B_GUFF_MODEL_ID,
            GEMMA_3_1B_GUFF_MODEL_FILENAME,
            32_000.try_into().unwrap(),
        )
        .await?;
        Ok(RunnerWithRecommendedSampling {
            inner,
            default_sampling: Self::recommend_sampling(),
        })
    }
}

impl<'s, 'req> TextLmRunner<'s, 'req> for Gemma3TextRunner {
    fn stream_lm_response<Tmpl>(
        &'s self,
        request: GenericTextLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>>
    where
        Tmpl: ChatTemplate,
    {
        let ctx = self
            .model
            .new_context(
                &LLAMA_BACKEND,
                LlamaContextParams::default().with_n_ctx(Some(self.ctx_size)),
            )
            .map_err(|err| GenericRunnerError::from(err));
        Gemma3Stream::new(ctx, request, self, &self.model)
    }
}

pub struct Gemma3VisionRunner {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    mtmd_ctx: MtmdContext,
    ctx_size: NonZeroU32,
}

impl Gemma3VisionRunner {
    pub async fn new(
        repo_id: impl ToString,
        model_file: impl AsRef<str>,
        multimodel_file: impl AsRef<str>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let repo = build_hf_api()?.model(repo_id.to_string());
        let model = LlamaModel::load_from_file(
            &LLAMA_BACKEND,
            repo.get(model_file.as_ref()).await?,
            &Default::default(),
        )?;

        let mtmd_ctx = MtmdContext::init_from_file(
            repo.get(multimodel_file.as_ref()).await?.to_str().unwrap(),
            &model,
            &Default::default(),
        )?;

        let chat_template = model.chat_template(None)?;

        Ok(Self {
            model,
            mtmd_ctx,
            chat_template,
            ctx_size,
        })
    }

    pub fn from_files(
        model_file: impl AsRef<Path>,
        multimodel_file: impl AsRef<Path>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError> {
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, model_file, &Default::default())?;
        let mtmd_ctx = MtmdContext::init_from_file(
            multimodel_file.as_ref().as_os_str().to_str().unwrap(),
            &model,
            &Default::default(),
        )?;

        let chat_template = model.chat_template(None)?;

        Ok(Self {
            model,
            mtmd_ctx,
            chat_template,
            ctx_size,
        })
    }

    fn new_context_window(&self) -> Result<LlamaContext<'_>, LlamaContextLoadError> {
        self.model.new_context(
            &LLAMA_BACKEND,
            LlamaContextParams::default().with_n_ctx(Some(self.ctx_size)),
        )
    }

    pub async fn default() -> Result<RunnerWithRecommendedSampling<Self>, CreateLlamaCppRunnerError>
    {
        let inner = Self::new(
            QWEN_3D5_4B_GUFF_MODEL_ID,
            QWEN_3D5_4B_GUFF_MODDEL_FILENAME,
            QWEN_3D5_4B_GUFF_MULTIMODEL_FILENAME,
            16384u32.try_into().unwrap(),
        )
        .await?;
        Ok(RunnerWithRecommendedSampling {
            inner: inner,
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

impl<'s, 'req> VisionLmRunner<'s, 'req> for Gemma3VisionRunner {
    fn stream_vlm_response<Tmpl>(
        &'s self,
        request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>>
    where
        Tmpl: ChatTemplate,
    {
        let ctx = self
            .new_context_window()
            .map_err(|err| GenericRunnerError::from(err));
        Gemma3Stream::new(ctx, request, self, &self.model)
    }
}

impl<'s, 'req> TextLmRunner<'s, 'req> for Gemma3VisionRunner {
    fn stream_lm_response<Tmpl>(
        &'s self,
        request: GenericTextLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>>
    where
        Tmpl: ChatTemplate,
    {
        self.stream_vlm_response(request.into())
    }
}

impl<'a, Tmpl> From<GenericTextLmRequest<'a, Tmpl>> for GenericVisionLmRequest<'a, Tmpl> {
    fn from(value: GenericTextLmRequest<'a, Tmpl>) -> Self {
        Self {
            messages: value
                .messages
                .into_iter()
                .map(|(role, text)| (role, ImageOrText::Text(text)))
                .collect(),
            sampling: value.sampling,
            llguidance: value.llguidance,
            max_seq: value.max_seq,
            prefill: value.prefill,
            tmpl: value.tmpl,
        }
    }
}

pub struct RunnerWithRecommendedSampling<Inner> {
    pub inner: Inner,
    pub default_sampling: SimpleSamplingParams,
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

impl<'s, 'req, Inner> VisionLmRunner<'s, 'req> for RunnerWithRecommendedSampling<Inner>
where
    Inner: VisionLmRunner<'s, 'req>,
{
    fn stream_vlm_response<Tmpl>(
        &'s self,
        mut request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> impl Iterator<Item = Result<String, GenericRunnerError<Tmpl::Error>>>
    where
        Tmpl: ChatTemplate,
    {
        request.sampling = self.get_preprocessed_simple_sampling(request.sampling);
        self.inner.stream_vlm_response(request)
    }
}

impl<'s, 'req, Inner> TextLmRunner<'s, 'req> for RunnerWithRecommendedSampling<Inner>
where
    Inner: TextLmRunner<'s, 'req>,
{
    fn stream_lm_response<Tmpl>(
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

fn build_hf_api() -> Result<hf_hub::api::tokio::Api, hf_hub::api::tokio::ApiError> {
    let mut api = ApiBuilder::new()
        .with_progress(std::io::stdin().is_terminal())
        .with_token(std::env::var("HF_TOKEN").ok())
        .with_chunk_size(Some(2 << 28));
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        api = api.with_endpoint(endpoint);
    }
    if let Ok(cache) = std::env::var("HF_HOME") {
        api = api.with_cache_dir(
            PathBuf::from_str(&cache).expect("HF_HOME env var is not a valid path"),
        );
    }
    api.build()
}

impl<Tmpl> PrepareRun<Tmpl::Error> for Gemma3Stream<'_, ImageOrText<'_>, Gemma3VisionRunner, Tmpl>
where
    Tmpl: ChatTemplate,
{
    fn prepare(&mut self) -> Result<(), GenericRunnerError<Tmpl::Error>> {
        // Preprocess the message, flattening media
        let media_marker = mtmd::mtmd_default_marker();
        let messages = self
            .req
            .messages
            .iter()
            .fold(
                Vec::<(MessageRole, String)>::new(),
                |mut acc, (role, message)| {
                    let text = match message {
                        ImageOrText::Text(text) => text,
                        ImageOrText::Image(_) => media_marker,
                    };
                    if let Some(last) = acc.last()
                        && last.0 == *role
                    {
                        // merge adjacent
                        let (_, adj) = acc.remove(acc.len() - 1);
                        acc.push((role.clone(), format!("{0}\n{text}", adj)));
                        acc
                    } else {
                        acc.push((role.clone(), text.to_string()));
                        acc
                    }
                },
            )
            .into_iter()
            .collect::<Vec<_>>();
        log::debug!(target: "gemma", "preprocessed messages: {messages:?}");

        // apply custom template
        let formatted_prompt = self
            .req
            .tmpl
            .apply_template(self.model, &self.runner.chat_template, &messages)
            .map_err(GenericRunnerError::ApplyChatTemplate)?;

        // Aggregate images
        let bitmaps = self
            .req
            .messages
            .iter()
            .filter_map(|msg| match &msg.1 {
                ImageOrText::Image(image) => Some(image),
                _ => None,
            })
            .enumerate()
            .map(|(idx, im)| {
                MtmdBitmap::from_image_data(
                    im.width(),
                    im.height(),
                    im.to_rgb8().to_vec().as_slice(),
                )
                .expect(format!("image#{} has corrupted RGB data", idx).as_str())
            })
            .collect::<Vec<_>>();
        let bitmap_refs = bitmaps.iter().collect::<Vec<_>>();
        let chunks = self.runner.mtmd_ctx.tokenize(
            MtmdInputText {
                text: formatted_prompt,
                add_special: true,
                parse_special: true,
            },
            &bitmap_refs,
        )?;
        log::debug!(target: "gemma", "tokenization resulted in {} chunks", chunks.len());
        let n_past = chunks.eval_chunks(
            &self.runner.mtmd_ctx,
            self.ctx.as_ref().unwrap(),
            0,
            0,
            1,
            true,
        )?;

        // Generate preparation
        let mut preparation = Runtime {
            sampler: self.req.sampling.to_llama(),
            decoder: UTF_8.new_decoder(),
            batch: LlamaBatch::new(self.runner.ctx_size.get() as usize, 1),
            n_past,
            step: 0,
        };
        if let Some(llguidance) = &self.req.llguidance {
            let llg_sampler = llguidance.to_llama(&self.runner.model)?;
            preparation.sampler = LlamaSampler::chain_simple([llg_sampler, preparation.sampler]);
        }
        self.runtime = Some(preparation);

        Ok(())
    }
}

impl<S: AsRef<str>, Tmpl> PrepareRun<Tmpl::Error> for Gemma3Stream<'_, S, Gemma3TextRunner, Tmpl>
where
    Tmpl: ChatTemplate,
{
    fn prepare(&mut self) -> Result<(), GenericRunnerError<Tmpl::Error>> {
        // Preprocess the message
        let messages = self
            .req
            .messages
            .iter()
            .fold(
                Vec::<(MessageRole, String)>::new(),
                |mut acc, (role, message)| {
                    if let Some(last) = acc.last()
                        && last.0 == *role
                    {
                        // merge adjacent
                        let (_, adj) = acc.remove(acc.len() - 1);
                        acc.push((role.clone(), format!("{0}\n{1}", adj, message.as_ref())));
                        acc
                    } else {
                        acc.push((role.clone(), message.as_ref().to_string()));
                        acc
                    }
                },
            )
            .into_iter()
            .collect::<Vec<_>>();
        log::debug!(target: "gemma", "preprocessed messages: {messages:?}");

        // apply custom template
        let formatted_prompt = self
            .req
            .tmpl
            .apply_template(self.model, &self.runner.llama_template, &messages)
            .map_err(GenericRunnerError::ApplyChatTemplate)?;

        // Aggregate images
        let token_list = self.model.str_to_token(&formatted_prompt, AddBos::Always)?;
        let mut batch = LlamaBatch::new(self.runner.ctx_size.get() as usize, 1);
        let token_list_len = token_list.len();
        for (i, token) in token_list.into_iter().enumerate() {
            batch.add(token, i as i32, &[0], i == token_list_len - 1)?;
        }
        self.ctx.as_mut().unwrap().decode(&mut batch)?;

        // Generate preparation
        let mut preparation = Runtime {
            sampler: self.req.sampling.to_llama(),
            decoder: UTF_8.new_decoder(),
            batch,
            n_past: token_list_len as i32,
            step: 0,
        };
        if let Some(llguidance) = &self.req.llguidance {
            let llg_sampler = llguidance.to_llama(&self.runner.model)?;
            preparation.sampler = LlamaSampler::chain_simple([llg_sampler, preparation.sampler]);
        }
        self.runtime = Some(preparation);

        Ok(())
    }
}
