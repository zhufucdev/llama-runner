use crate::{GenericTextLmRequest, GenericVisionLmRequest, TextLmRunner, VisionLmRunner, error::GenericRunnerError, template::ChatTemplate};

pub trait TextLmRunnerExt<'s, 'req> {
    fn get_lm_response<Tmpl>(
        &'s self,
        request: GenericTextLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>>
    where
        Tmpl: ChatTemplate;
}

pub trait VisionLmRunnerExt<'s, 'req> {
    fn get_vlm_response<Tmpl>(
        &'s self,
        request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>>
    where
        Tmpl: ChatTemplate;
}

impl<'s, 'req, TextRunner> TextLmRunnerExt<'s, 'req> for TextRunner
where
    TextRunner: TextLmRunner<'s, 'req>,
{
    fn get_lm_response<Tmpl>(
        &'s self,
        request: GenericTextLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>>
    where
        Tmpl: ChatTemplate,
    {
        self.stream_lm_response(request)
            .collect::<Result<String, _>>()
    }
}

impl<'s, 'req, VisionRunner> VisionLmRunnerExt<'s, 'req> for VisionRunner
where
    VisionRunner: VisionLmRunner<'s, 'req>,
{
    fn get_vlm_response<Tmpl>(
        &'s self,
        request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>>
    where
        Tmpl: ChatTemplate,
    {
        self.stream_vlm_response(request)
            .collect::<Result<String, _>>()
    }
}

