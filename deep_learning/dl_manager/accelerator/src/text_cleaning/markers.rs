pub enum Marker {
    Log,
    Traceback,
    UnformattedLog,
    UnformattedTraceback
}

impl Marker {
    pub fn all_markers() -> Vec<Marker> {
        vec![]
    }

    pub fn string_marker(&self) -> String {
        match self {
            Self::Log => "LLLOG",
            Self::Traceback => "TTTRACEBACK",
            Self::UnformattedLog => "UNFORMATTEDLOGGINGOUTPUT",
            Self::UnformattedTraceback => "UNFORMATTEDTRACEBACK"
        }.into()
    }
}