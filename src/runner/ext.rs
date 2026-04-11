use crate::{
    GenericTextLmRequest, GenericVisionLmRequest, TextLmRunner, VisionLmRunner,
    error::GenericRunnerError, template::ChatTemplate,
};

pub trait TextLmRunnerExt<'s, 'req, Tmpl>
where
    Tmpl: ChatTemplate,
{
    fn get_lm_response(
        &'s self,
        request: GenericTextLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>>;
}

pub trait VisionLmRunnerExt<'s, 'req, Tmpl>
where
    Tmpl: ChatTemplate,
{
    fn get_vlm_response(
        &'s self,
        request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>>;
}

impl<'s, 'req, TextRunner, Tmpl> TextLmRunnerExt<'s, 'req, Tmpl> for TextRunner
where
    TextRunner: TextLmRunner<'s, 'req, Tmpl>,
    Tmpl: ChatTemplate,
{
    fn get_lm_response(
        &'s self,
        request: GenericTextLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>> {
        self.stream_lm_response(request)
            .collect::<Result<String, _>>()
    }
}

impl<'s, 'req, VisionRunner, Tmpl> VisionLmRunnerExt<'s, 'req, Tmpl> for VisionRunner
where
    VisionRunner: VisionLmRunner<'s, 'req, Tmpl>,
    Tmpl: ChatTemplate,
{
    fn get_vlm_response(
        &'s self,
        request: GenericVisionLmRequest<'req, Tmpl>,
    ) -> Result<String, GenericRunnerError<Tmpl::Error>> {
        self.stream_vlm_response(request)
            .collect::<Result<String, _>>()
    }
}
