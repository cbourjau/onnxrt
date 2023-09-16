use crate::opset::utils::*;

#[derive(Debug)]
enum CaseChange {
    LOWER,
    UPPER,
    NONE,
}

#[derive(Debug)]
struct StringNormalizer {
    case_change_action: CaseChange,
    is_case_sensitive: bool,
    stopwords: Vec<String>,
}

impl StringNormalizer {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let case_change_action =
            attrs.get_with_default("case_change_action", "NONE".to_string())?;

        let case_change_action = match case_change_action.as_str() {
            "NONE" => CaseChange::NONE,
            "UPPER" => CaseChange::UPPER,
            "LOWER" => CaseChange::LOWER,
            s => bail!(
                "StringNormalizer: Unexpected value for attribute 'case_change_action': '{}'",
                s
            ),
        };

        let is_case_sensitive = attrs.get_with_default("is_case_sensitive", 0i64)? == 1;
        let stopwords: Vec<String> = attrs.get_with_default("stopwords", vec![])?;

        let stopwords = if is_case_sensitive {
            stopwords
        } else {
            match case_change_action {
                CaseChange::NONE | CaseChange::LOWER => {
                    stopwords.into_iter().map(|s| s.to_lowercase()).collect()
                }
                CaseChange::UPPER => stopwords.into_iter().map(|s| s.to_uppercase()).collect(),
            }
        };

        Ok(Self {
            is_case_sensitive,
            case_change_action,
            stopwords,
        })
    }

    fn compute(&self, x: ArrayD<String>) -> Result<ArrayD<String>> {
        let ndim = x.ndim();
        let it = x.into_iter().filter_map(|s| {
            let s_for_match = if self.is_case_sensitive {
                s.clone()
            } else {
                match self.case_change_action {
                    CaseChange::NONE | CaseChange::LOWER => s.to_lowercase(),
                    CaseChange::UPPER => s.to_uppercase(),
                }
            };

            for sw in &self.stopwords {
                if s_for_match.find(sw.as_str()).is_some() {
                    return None;
                }
            }
            Some(match (&self.case_change_action, self.is_case_sensitive) {
                (CaseChange::NONE, _) => s,
                (CaseChange::LOWER, false) | (CaseChange::UPPER, false) => s_for_match,
                (CaseChange::LOWER, true) => s.to_lowercase(),
                (CaseChange::UPPER, true) => s.to_uppercase(),
            })
        });

        let out = Array1::from_iter(it);
        // Special case from standard if output would be empty...
        let out = if out.is_empty() {
            array!["".to_string()]
        } else {
            out
        }
        .into_dyn();
        if ndim == 1 {
            Ok(out)
        } else if ndim == 2 {
            Ok(out.insert_axis(Axis(0)))
        } else {
            bail!(
                "StringNormalizer: Expected input of rank 1 or 2 found '{}'",
                ndim
            )
        }
    }
}

pub fn setup_string_normalizer(ctx: OpContext) -> Result<ExecutionStep> {
    let state = StringNormalizer::from_attrs(&ctx.attrs)?;
    let x = ctx.subscribe_to_inputs().narrow()?;

    if let TaggedReceiver::String(x) = x {
        setup((x,), move |(x,)| {
            let x = x.to_owned();
            Ok::<ArcArrayD<_>, _>(state.compute(x)?.into_dyn().into())
        })
    } else {
        bail!("")
    }
}
