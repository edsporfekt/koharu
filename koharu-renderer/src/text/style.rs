use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextStyleKind {
    #[default]
    Regular,
    Italic,
    Bold,
    BoldItalic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyledSegment {
    pub text: String,
    pub kind: TextStyleKind,
}

static STYLE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\*{1,3})(.*?)(\1)").unwrap());

pub fn parse_styled_segments(text: &str) -> Vec<StyledSegment> {
    let mut segments = Vec::new();
    let mut last_end = 0;

    for cap in STYLE_RE.captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let marker = cap.get(1).unwrap().as_str();
        let content = cap.get(2).unwrap().as_str();

        if full_match.start() > last_end {
            segments.push(StyledSegment {
                text: text[last_end..full_match.start()].to_string(),
                kind: TextStyleKind::Regular,
            });
        }

        let kind = match marker.len() {
            3 => TextStyleKind::BoldItalic,
            2 => TextStyleKind::Bold,
            1 => TextStyleKind::Italic,
            _ => TextStyleKind::Regular,
        };

        segments.push(StyledSegment {
            text: content.to_string(),
            kind,
        });

        last_end = full_match.end();
    }

    if last_end < text.len() {
        segments.push(StyledSegment {
            text: text[last_end..].to_string(),
            kind: TextStyleKind::Regular,
        });
    }

    if segments.is_empty() && !text.is_empty() {
        segments.push(StyledSegment {
            text: text.to_string(),
            kind: TextStyleKind::Regular,
        });
    }

    segments
}
