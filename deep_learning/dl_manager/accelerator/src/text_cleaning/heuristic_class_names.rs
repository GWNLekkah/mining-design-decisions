use std::collections::HashSet;
use lazy_static::lazy_static;
use regex::{Captures, Regex};
use crate::text_cleaning::FormattingHandling;
use crate::text_cleaning::markers::Marker;
use crate::text_cleaning::regex_util::is_full_match;


pub fn determine_type(full_name: &str, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref LOWER_CC: Regex = Regex::new(r"[a-z][a-z\d]*([A-Z]\w*)+").unwrap();
        static ref UPPER_CC: Regex = Regex::new(r"([A-Z]\w*){2,}").unwrap();
    }
    if full_name.chars().all(|c| c.is_uppercase() || c.is_whitespace()) {
        return full_name.to_string();
    }
    match handling {
        FormattingHandling::Keep => full_name.to_string(),
        FormattingHandling::Remove => "".into(),
        FormattingHandling::Markers => {
            let name = full_name.split('.').last().expect("Expected an item");
            if is_full_match(&LOWER_CC, name) {
                Marker::MethodOrVariableName.string_marker()
            } else if is_full_match(&UPPER_CC, name) {
                Marker::ClassName.string_marker()
            } else {    // We omit package check because its slow
                Marker::MethodOrVariableName.string_marker()
            }
        }
    }
}


pub fn remove_class_names_heuristically(text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref PATTERN: Regex = Regex::new(r"(\w+\.)+\w+").unwrap();
    }
    PATTERN.replace_all(&text, |c: &Captures| replace_class_name(c, handling)).into()
}

fn replace_class_name(c: &Captures, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref EXTENSIONS: HashSet<&'static str> = HashSet::from([
            "yaml", "java", "xml", "json", "txt",
            "cfg", "yml", "py", "md", "info", "exe", "log",
            "h", "c", "zip", "class", "bat", "sh", "rar", "jar",
            "tbl", "dir", "dll", "so", "pdf", "out", "png",
            "diff", "php", "lib", "jsp", "asc"
        ]);
        static ref ABBREVIATIONS: HashSet<&'static str> = HashSet::from([
            "e.g.", "i.e.", "w.r.t", "i.e", "e.g", "w.r.t.", "p.s", "p.s.", "ph.d"
        ]);
        static ref VN_PATTERN: Regex = Regex::new(
            r"(\w+|v)?((\d+|x|X|y|Y|z|Z)\.)*(\d+|x|X|y|Y|z|Z)"
        ).unwrap();
        static ref SS_PATTERN: Regex = Regex::new(
            r"\d+\.\d+(M|G|K|KB|MB|GB|k|m|g|t|T|TB|B)"
        ).unwrap();
        static ref AWS_PATTERN: Regex = Regex::new(
            r"[a-z\d.]+\.(nano|micro|small|medium|large|xlarge|metal|(\d+xlarge))"
        ).unwrap();
        static ref VN2_PATTERN: Regex = Regex::new(
            r"v?(\d\.)+\d_?[a-z\d]+"
        ).unwrap();
        static ref FLOAT_PATTERN: Regex = Regex::new(
            r"\d+\.\d+f"
        ).unwrap();
    }
    let text = c.get(0).expect("Expected a match").as_str();
    let text_as_lower = text.to_lowercase();
    if handling == FormattingHandling::Keep {
        return text.to_string();
    }
    let parts = text.split('.').collect::<Vec<_>>();
    let last = parts.last().expect("Expected at least one element");
    if EXTENSIONS.contains(last) {
        if handling == FormattingHandling::Remove {
            "".into()
        } else {
            Marker::FilePath.string_marker()
        }
    } else if *last == "com" || *last == "edu" {
        if handling == FormattingHandling::Remove {
            "".into()
        } else {
            Marker::WebLink.string_marker()
        }
    } else if ABBREVIATIONS.contains(text_as_lower.as_str()) {
        text_as_lower.replace('.', "")
    } else if is_full_match(&VN_PATTERN, text) {
        if handling == FormattingHandling::Remove {
            "".into()
        } else {
            Marker::VersionNumber.string_marker()
        }
    } else if is_full_match(&SS_PATTERN, text) {
        if handling == FormattingHandling::Remove {
            "".into()
        } else {
            Marker::StorageSize.string_marker()
        }
    } else if parts.first().expect("Expected at least one element") == &"www" {
        if handling == FormattingHandling::Remove {
            "".into()
        } else {
            Marker::WebLink.string_marker()
        }
    } else if is_full_match(&AWS_PATTERN, text) {
        if handling == FormattingHandling::Remove {
            "".into()
        } else {
            Marker::CloudInstanceSpec.string_marker()
        }
    } else if is_full_match(&VN2_PATTERN, text) {
        if handling == FormattingHandling::Remove {
            "".into()
        } else {
            Marker::VersionNumber.string_marker()
        }
    } else if is_full_match(&FLOAT_PATTERN, text) {
        "".into()
    } else {
        determine_type(text_as_lower.as_str().trim(), handling)
    }
}


pub fn remove_file_paths_heuristically(mut text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref SIMPLE_FILE_PATTERN: Regex = Regex::new(
            r"[ \t\n](\./|/)[\w_\-]{2,}(\.[a-z]+)?"
        ).unwrap();
        static ref PATH_PATTERN: Regex = Regex::new(
            r"[ \t\n](\./|/)?([\w\-]/)+[\w\-]{2,}(\.[a-z]+)?"
        ).unwrap();
    }
    text = SIMPLE_FILE_PATTERN.replace_all(
        &text, |c: &Captures| replace_file_path(c, handling)
    ).into();
    text = PATH_PATTERN.replace_all(
        &text, |c: &Captures| replace_file_path(c, handling)
    ).into();
    text
}

fn replace_file_path(c: &Captures, handling: FormattingHandling) -> String {
    let text = c.get(0).expect("Expected a match").as_str();
    if text == "I/O" {
        return text.to_string();
    }
    let parts = text.trim().split('/').collect::<Vec<_>>();
    if (parts.len() == 1 && parts.first().unwrap() != &".") || parts.last().expect("Expected at least one item").chars().all(char::is_numeric) {
        return text.to_string();
    }
    match handling {
        FormattingHandling::Keep => text.to_string(),
        FormattingHandling::Remove => "".into(),
        FormattingHandling::Markers => {
            let first = *text.chars().take(1).collect::<Vec<_>>().first().expect("Expected at least one item");
            if first == ' ' || first == '\t' || first == '\n' {
                first.to_string() + Marker::FilePath.string_marker().as_str()
            } else {
                Marker::FilePath.string_marker()
            }
        }
    }
}


pub fn replace_class_names_no_path_heuristically(mut text: String, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref LOWER_CC: Regex = Regex::new(r"\s[a-z][a-z\d]*([A-Z]\w*)+").unwrap();
        static ref UPPER_CC: Regex = Regex::new(r"(\s|\(|\[)([A-Z]\w*){2,}").unwrap();
    }
    text = LOWER_CC.replace_all(&text, |c: &Captures| remove_lower_case(c, handling)).into();
    text = UPPER_CC.replace_all(&text, |c: &Captures| remove_upper_case(c, handling)).into();
    text
}

fn remove_lower_case(c: &Captures, handling: FormattingHandling) -> String {
    let text = c.get(0).expect("Expected match").as_str();
    let whitespace = &text[0..1];
    match handling {
        FormattingHandling::Keep => text.to_string(),
        FormattingHandling::Remove => whitespace.to_string(),
        FormattingHandling::Markers => whitespace.to_string() + Marker::SimpleMethodOrVariableName.string_marker().as_str()
    }
}

fn remove_upper_case(c: &Captures, handling: FormattingHandling) -> String {
    lazy_static! {
        static ref TECHNOLOGIES: HashSet<String> = read_ontology_file();
        static ref ABBREVIATIONS: HashSet<&'static str> = HashSet::from([
            "udt", "qos", "bbs", "ids", "cpus", "url",
            "id-ing", "udfs", "uda", "fcs", "pmcs", "rpc",
            "uri", "acls", "tpes", "dn", "ip", "jira", "id",
            "udas", "dcs", "mrv2", "gets", "lowercamelcase",
            "phd", "jars", "nns", "ugis", "ssd", "atsv2",
            "api", "rpcv9", "mofs", "slas", "ips", "urls",
            "dbs", "cpu", "echos", "rm", "acl", "jiras",
            "rpms", "uis", "cfses", "bb", "echo", "vms",
            "mvs", "dc", "rr", "jvm", "cfse", "sid", "am",
            "eofs", "io", "get", "pmc", "udf", "fsms", "rrs",
            "fc", "sla", "uris", "crc", "linkedin",
            "uppercamelcase", "idps", "idp", "ui", "ugi",
            "camelcase", "nms", "ad", "ads", "cfs", "uuids",
            "uuid", "cf", "rms", "nm", "sids", "udts", "db",
            "fsm", "rpcs", "crcs", "mv", "eof", "tpe", "ghz",
            "dns", "vm", "jvms", "nn", "nfsv3", "apis", "ios",
            "ssds", "jar", "mof", "rpm", "ams"
        ]);
    }
    let text = c.get(0).expect("Expected a match").as_str();
    let whitespace = &text[0..1];
    #[allow(clippy::if_same_then_else)]
    if text.chars().all(|c| c.is_uppercase() || c.is_whitespace()) {
        text.to_string()
    } else if TECHNOLOGIES.contains(text.to_lowercase().trim()) {
        //match handling {
        //    FormattingHandling::Keep => text.to_string(),
        //    FormattingHandling::Remove => "".into(),
        //    FormattingHandling::Markers => whitespace.to_string() + Marker::TechnologyName.string_marker().as_str()
        //}
        text.to_string()
    } else if ABBREVIATIONS.contains(text) || text.len() <= 6 {     // magic number
        text.to_string()
    } else {
        match handling {
            FormattingHandling::Keep => text.to_string(),
            FormattingHandling::Remove => "".into(),
            FormattingHandling::Markers => whitespace.to_string() + Marker::SimpleClassName.string_marker().as_str()
        }
    }
}

fn read_ontology_file() -> HashSet<String> {
    let raw = include_str!("../../../feature_generators/util/technologies.csv");
    let mut base: HashSet<String> = raw.lines().map(str::trim).map(str::to_lowercase).collect();
    let extra_technologies = [
        "CGroups", "JDiff", "CentOS5", "OpenJDK",
        "JUnit", "OAUTH", "OAUTH2", "MapReduce",
        "OpenMPI", "JMX4Perl", "CircleCI", "BZip2",
        "WinRT", "DistCP", "RxJava", "Jira", "HiveQL"
    ];
    for t in extra_technologies {
        base.insert(t.to_lowercase().to_string());
    }
    base
}
