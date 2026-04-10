use std::{num::NonZeroU32, path::Path};

use crate::{error::CreateLlamaCppRunnerError, sample::SimpleSamplingParams};

pub trait TextRunnerFromFile
where
    Self: Sized,
{
    fn from_file(
        model_file: impl AsRef<Path>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError>;
}

pub trait VisionRunnerFromFiles
where
    Self: Sized,
{
    fn from_files(
        model_file: impl AsRef<Path>,
        multimodel_file: impl AsRef<Path>,
        ctx_size: NonZeroU32,
    ) -> Result<Self, CreateLlamaCppRunnerError>;
}

pub trait RecommendedSampling {
    fn recommend_sampling() -> SimpleSamplingParams;
}

