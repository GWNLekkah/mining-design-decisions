##############################################################################
##############################################################################
# Imports
##############################################################################

import csv
import itertools
import json
import os
import re
import string

import contractions
import nltk

__all__ = [
    'reporter',
    'remove_formatting',
    'fix_punctuation'
]


class FormattingHandling:
    Remove = 0
    Markers = 1
    Keep = 2


format_settings: FormattingHandling = FormattingHandling.Markers


def set_formatting_handling(h: FormattingHandling):
    global format_settings
    format_settings = h


def get_formatting_settings() -> FormattingHandling:
    return format_settings


PATH_PREFIX = os.path.split(__file__)[0]


##############################################################################
##############################################################################
# Diagnostic Utility
##############################################################################


class Reporter:

    def __init__(self):
        self.__funcs = {}

    def wrap(self, func, name):
        if name not in self.__funcs:
            self.__funcs[name] = 0

        def wrapper(*args, **kwargs):
            self.__funcs[name] += 1
            return func(*args, **kwargs)

        return wrapper

    def inc(self, name):
        if name not in self.__funcs:
            self.__funcs[name] = 0
        self.__funcs[name] += 1

    def print_report(self):
        for func, count in sorted(self.__funcs.items()):
            print(f'{func}: {count}')


reporter = Reporter()

##############################################################################
##############################################################################
# Exposed Functions
##############################################################################


def fix_punctuation(text: str) -> str:
    text = contractions.fix(text)
    text = _remove_double_punctuation(text)
    # for sentence in nltk.sent_tokenize(text):
    #     if sentence.count('.') > 1:
    #         print('=' * 72)
    #         print(sentence)
    return text


def remove_formatting(text: str,
                      key: str,
                      formatting_handling: FormattingHandling) -> str:
    if formatting_handling == FormattingHandling.Remove:
        raise ValueError('Remove option no longer supported')
    elif formatting_handling == FormattingHandling.Markers:
        return _remove_formatting(text, key, FormattingHandling.Markers)
    else:
        text = _remove_formatting(text, key, FormattingHandling.Keep)
        return _remove_formatting(text, key, FormattingHandling.Markers)


def _remove_formatting(text: str,
                      key: str,
                      formatting_handling: FormattingHandling) -> str:
    set_formatting_handling(formatting_handling)

    # Do this first, because the removal of formatting
    # does not deal nicely with heuristic methods
    if formatting_handling != FormattingHandling.Keep:
        text = _remove_unformatted_lines(key, text)
    text = _remove_dates(text)
    text = _remove_ip_addresses(text)

    # Now, remove regular formatting
    text = re.sub(r'\[(?P<linkpart>(http|https|mailto|file).*?\|.*?)\]',
                  lambda match: _remove_link(match.group('linkpart')),
                  text)
    text = re.sub(r'\[(?P<linkpart>(#|\^|http|https|mailto|file).*?)\]',
                  lambda match: _remove_link(match.group('linkpart')),
                  text)
    text = re.sub(r'https?://\S+',
                  lambda match: _remove_link(match.group()),
                  text)
    # text = re.sub(r'\{code.*?\}(.|\s)*?\{code\}', 'CCCODEBLOCK', text)

    if formatting_handling == FormattingHandling.Keep:
        text = re.sub(r'\{code}', '', text)
        text = re.sub(r'\{code:\w+}', '', text)
        text = re.sub(r'\{noformat}', '', text)
    else:
        text = _remove_code_blocks(text,
                                   place_markers=formatting_handling == FormattingHandling.Markers)
        text = _remove_no_format_blocks(text,
                                        place_markers=formatting_handling == FormattingHandling.Markers)
    text = re.sub(r'\{\{(?P<code>.*?)\}\}',
                  reporter.wrap(_remove_inline_code, 'Inline Code'),
                  text)
    text = re.sub(r'\[\~[^\s]+\]',
                  reporter.wrap(lambda *_: 'USERPROFILELINK', 'User Link'),
                  text)
    text = re.sub(r'\!.*?\!',
                  reporter.wrap(lambda *_: 'IMAGEATTACHMENT', 'Image'),
                  text)
    text = re.sub(r'h[1-6]\.',
                  reporter.wrap(lambda *_: '', 'Header'),
                  text)
    text = re.sub(r'\|',
                  reporter.wrap(lambda *_: ' | ', 'Table Fragment'),
                  text)
    text = re.sub(r'bq\.',
                  reporter.wrap(lambda *_: '', 'Block Quote'),
                  text)
    text = re.sub(r'\{quote}', reporter.wrap(lambda _: ' ', 'Quote'), text)
    text = re.sub(r'\<.*?\>',
                  reporter.wrap(lambda *_: '', 'Misc'),
                  text)
    text = re.sub(r'\*(?P<content>.+?)\*',
                  reporter.wrap(lambda match: match.group('content'), 'Bold'),
                  text)
    text = re.sub(r'_(?P<content>.+?)_',
                  reporter.wrap(lambda match: match.group('content'), 'Italic'),
                  text)
    text = re.sub(r'\?\?(?P<content>.+?)\?\?',
                  reporter.wrap(lambda match: match.group('content'), 'Citation'),
                  text)
    #text = re.sub(r'-(?P<content>.+?)-',
    #              reporter.wrap(lambda match: match.group('content'), 'Strikethrough'),
    #              text)
    text = re.sub(r'\+(?P<content>.+?)\+',
                  reporter.wrap(lambda match: match.group('content'), 'Underline'),
                  text)
    text = re.sub(r'\^(?P<content>.+?)\^',
                  reporter.wrap(lambda match: match.group('content'), 'Superscript'),
                  text)
    text = re.sub(r'~(?P<content>.+?)~',
                  reporter.wrap(lambda match: match.group('content'), 'Subscript'),
                  text)
    text = re.sub(r'\{color:.+?}(?P<content>.*?)\{color}',
                  reporter.wrap(lambda match: match.group('content'), 'Color'),
                  text)

    # Remove empty lines caused up to this point
    text = _remove_empty_lines(text)

    # Remove lists
    text = _remove_lists_from_text(text)

    # At this point, we will try to remove any last left-over artifacts
    if formatting_handling != FormattingHandling.Keep:
        text = _remove_file_path_heuristic(text)
        text = _remove_class_names_heuristic(text)
        text = _remove_class_names_no_path(text)

    # TODO: improving this function/getting it to work, is probably a full Bachelor's project
    #text = _remove_unformatted_code(text)

    # Finally, clean up the lines once again
    text = _remove_empty_lines(text)
    if 'panel' in text:
        reporter.inc('panel')
    return text


##############################################################################
##############################################################################
# Punctuation Handling
##############################################################################


def _remove_double_punctuation(text: str) -> str:
    punctuation = '.,;:'
    for x in punctuation:
        for y in punctuation:
            pattern = re.compile(fr'{re.compile(x)}\s*{re.compile(y)}')
            while pattern.search(text) is not None:
                text = pattern.sub('', text)
    return text


##############################################################################
##############################################################################
# Link Removal
##############################################################################


_HTTP_VERSION_PATTERN = re.compile(r'http/\d.\d')


def _remove_link(link: str) -> str:
    if get_formatting_settings() == FormattingHandling.Keep:
        return link
    if get_formatting_settings() == FormattingHandling.Remove:
        return ''
    if link[0] == '^':
        return 'ATTACHMENT'
    if _HTTP_VERSION_PATTERN.fullmatch(link) is not None:
        return link
    if link.startswith('https://github.com'):
        return 'GITHUBLINK'
    if link.startswith('https://issues.apache.org/jira/browse/'):
        return 'ISSUELINK'
    return 'WEBLINK'


##############################################################################
##############################################################################
# Simple Removal Functions
##############################################################################


#######################################################################
# Date Removal

def _remove_dates(text: str) -> str:
    fmt_1 = r'\d\d\d\d[./]\d\d?[./]\d\d?'
    fmt_2 = r'\d\d?[./]\d\d?[./]\d\d\d\d'
    pattern = re.compile(f'({fmt_1})|({fmt_2})')
    return pattern.sub(_remove_date_ex, text)


def _remove_date_ex(match):
    reporter.inc('Date')
    return 'DATE'


#######################################################################
# IP Address Removal


def _remove_ip_addresses(text: str) -> str:
    pattern = re.compile(
        r'(\d{2,3}|xx)\.(\d{2,3}|xx)\.(\d{2,3}|xx)\.(\d{1,3}|xx)'
    )
    return pattern.sub(_remove_ip_ex, text)


def _remove_ip_ex(match):
    reporter.inc('IP Address')
    return ''


#######################################################################
# Empty Line Removal


def _remove_empty_lines(text: str) -> str:
    return '\n'.join(line
                     for line in text.splitlines()
                     if not _is_useless_line(line))


def _is_useless_line(line: str) -> bool:
    if not line.strip():
        return True
    if set(line.strip()) <= set(string.whitespace) | set(string.punctuation):
        return True
    return False


#######################################################################
# Inline Code Removal


def _remove_inline_code(match):
    if get_formatting_settings() == FormattingHandling.Keep:
        return match.group('code')
    if get_formatting_settings() == FormattingHandling.Remove:
        return ''
    code = match.group('code')
    if not re.fullmatch(r'[a-zA-Z\d_\-\.:#]+(\(.*\))?', code) or len(code) <= 1:
        return 'INLINECODESAMPLE'
    return _determine_type(code)


#######################################################################
# Heuristic File Path Removal


def _remove_file_path_heuristic(text: str):
    simple_file_pattern = re.compile(r'[ \t\n](\./|/)[\w_\-]{2,}(\.[a-z]+)?')
    path_pattern = re.compile(r'[ \t\n](\./|/)?([\w\-]/)+[\w\-]{2,}(\.[a-z]+)?')
    text = simple_file_pattern.sub(_replace_file_path, text)
    text = path_pattern.sub(_replace_file_path, text)
    return text


def _replace_file_path(match):
    # Check if we accidentally matched the 'I/O' string
    if match.group().upper().strip() == 'I/O':
        return match.group()
    parts = match.group().strip().split('/')
    if len(parts[0]) == 1 and parts[0] != '.':
        # I don't know what this is, but it is not a file path most likely
        return match.group()
    if parts[-1].isdigit():
        # Once again, no clue
        return match.group()
    # Heuristically, we pretty much have all file names at this point.
    reporter.inc('File Path (Heuristic)')
    if get_formatting_settings() == FormattingHandling.Keep:
        return match.group()
    if get_formatting_settings() == FormattingHandling.Remove:
        return ''
    if match.group()[0] in ' \t\n':
        return match.group()[0] + 'FilePath'
    return 'FILEPATH'


#######################################################################
# Heuristic Class Name Removal


package_path = os.path.join(PATH_PREFIX, 'packages')
PACKAGES = set()
for filename in os.listdir(package_path):
    path = os.path.join(package_path, filename)
    with open(path) as file:
        for line in file:
            PACKAGES.add(line.strip())


def _remove_class_names_heuristic(text: str):
    pattern = re.compile(r'(\w+\.)+\w+')
    text = pattern.sub(_replace_class_name, text)
    return text


def _replace_class_name(match):
    parts = match.group().split('.')
    extensions = {'yaml', 'java', 'xml', 'json', 'txt',
                  'cfg', 'yml', 'py', 'md', 'info', 'exe', 'log',
                  'h', 'c', 'zip', 'class', 'bat', 'sh', 'rar', 'jar',
                  'tbl', 'dir', 'dll', 'so', 'pdf', 'out', 'png',
                  'diff', 'php', 'lib', 'jsp', 'asc'}
    # Step 1: Filter out files
    if parts[-1].lower() in extensions:
        reporter.inc('File Path (Heuristic 4)')
        return 'FILEPATH'
    # Step 2: Filter out rogue website names
    if parts[-1].lower() in {'com', 'edu'}:
        reporter.inc('Link (Heuristic 2)')
        return 'WEBLINK'
    # Step 3: Filter out some common abbreviations
    if match.group().lower() in {'e.g.', 'i.e.', 'w.r.t', 'i.e', 'e.g', 'w.r.t.', 'p.s', 'p.s.', 'ph.d'}:
        return match.group().replace('.', '')
    # Step 4: Filter out version numbers. Version numbers may contain unknown/variable parts
    if re.fullmatch(r'(\w+|v)?((\d+|x|X|y|Y|z|Z)\.)*(\d+|x|X|y|Y|z|Z)', match.group()):
        reporter.inc('Version Number (1)')
        return 'VERSIONNUMBER'
    # Step 5: Filter out XGB/MB etc
    if re.fullmatch(r'\d+\.\d+(M|G|K|KB|MB|GB|k|m|g|t|T|TB|B)', match.group()) is not None:
        reporter.inc('Storage Size')
        return 'STORAGESIZE'
    # Step 6: More rogue web sites
    if parts[0] == 'www':
        reporter.inc('Link (Heuristic 3)')
        return 'WEBLINK'
    # Step 7: Amazone instance types
    # A full check would be rather expensive, so we assume that the
    # size suffix is sufficient for detection.
    sizes = ['nano', 'micro', 'small', 'medium', 'large', 'xlarge', 'metal']
    pattern = re.compile(
        fr'[a-z\d.]+\.({"|".join(sizes)}|(\d+xlarge))'
    )
    if pattern.fullmatch(match.group()) is not None:
        reporter.inc('Instance Type')
        return ''
    # Step 8: Additional version number check. No wildcards this time
    if re.fullmatch(r'v?(\d\.)+\d_?[a-z\d]+', match.group()):
        reporter.inc('Version Number (2)')
        return 'VERSIONNUMBER'
    # Step 9: Remove floating point numbers
    if re.fullmatch(r'\d+\.\d+f', match.group()) is not None:
        reporter.inc('Float')
        return ''
    #print(match.group())
    return _determine_type(match.group().lower().strip())


def _determine_type(fullname: str) -> str:
    if get_formatting_settings() == FormattingHandling.Keep:
        return fullname
    if get_formatting_settings() == FormattingHandling.Remove:
        return ''
    if '.' in fullname:
        name = fullname.split('.')[-1]
    else:
        name = fullname
    lower_camel_case = re.compile(r'[a-z][a-z\d]*([A-Z]\w*)+')
    upper_camel_case = re.compile(r'([A-Z]\w*){2,}')
    if lower_camel_case.fullmatch(name) is not None:
        reporter.inc('Method/Field Name')
        return 'METHODORVARIABLENAME'
    if upper_camel_case.fullmatch(name) is not None:
        reporter.inc('Class Name')
        return 'CLASSNAME'
    # Try to check for a package name
    for package in PACKAGES:
        if fullname in package:
            reporter.inc('Package')
            return 'PACKAGE'
    # Default
    reporter.inc('Method/Field Name')
    return 'METHODORVARIABLENAME'


#######################################################################
# Heuristic Class Name Removal (loose)


def _remove_class_names_no_path(text: str) -> str:
    lower_camel_case = re.compile(r'\s[a-z][a-z\d]*([A-Z]\w*)+')
    upper_camel_case = re.compile(r'\s([A-Z]\w*){2,}')
    text = lower_camel_case.sub(_remove_lower_cc, text)
    text = upper_camel_case.sub(_remove_upper_cc, text)
    return text


def _remove_lower_cc(match):
    whitespace = match.group()[0]
    reporter.inc('Field Name (No Package)')
    if get_formatting_settings() == FormattingHandling.Keep:
        return match.group()
    if get_formatting_settings() == FormattingHandling.Remove:
        return whitespace
    return whitespace + 'SIMPLEMETHODORVARIABLENAME'


onto_path = os.path.join(PATH_PREFIX, '../../datasets/ontologies/Technology Names.csv')
with open(onto_path) as file:
    TECHNOLOGIES = {line.strip().lower() for line in file}


def _remove_upper_cc(match):
    if match.group().isupper():
        return match.group()
    raw = match.group().lower().strip()
    lemmatized = nltk.stem.WordNetLemmatizer().lemmatize(raw)
    if lemmatized in TECHNOLOGIES or raw in TECHNOLOGIES:
        reporter.inc('Technology')
        if get_formatting_settings() == FormattingHandling.Keep:
            return match.group()
        return match.group()[0] + 'Technology Names'
    ignored_technologies = {
        'CGroups', 'JDiff', 'CentOS5', 'OpenJDK',
        'JUnit', 'OAUTH', 'OAUTH2', 'MapReduce',
        'OpenMPI', 'JMX4Perl', 'CircleCI', 'BZip2',
        'WinRT', 'DistCP', 'RxJava', 'Jira', 'HiveQL'
    }
    ignored_technologies = {x.lower() for x in ignored_technologies}
    if lemmatized in ignored_technologies or raw in ignored_technologies:
        return match.group()
    # Abbreviations to ignore. This list was collected based
    # on inspection of the dataset
    to_remove = {
        'IOs', 'RPCs', 'ACLs', 'APIs', 'AMs', 'DBs', 'VMs',
        'URLs', 'DNs', 'NNs', 'MVs', 'IPs', 'UIs', 'FCs',
        'FSMs', 'CRCs', 'URIs', 'IDs', 'RMs', 'DCs', 'CFs',
        'RRs', 'PhD', 'NMs', 'IDPs', 'CPUs', 'UDFs',
        'PMCs', 'SSDs', 'JARs', 'EOFs', 'UDAs', 'GETs',
        'JVMs', 'UDTs', 'UGIs', 'UUIDs', 'ADs', 'MOFs',
        'BBs', 'SIDs', 'SLAs', 'TPEs', 'CFSes', 'RPMs',
        'ECHOs', 'JIRAs'
    }
    to_remove |= {x[:-1] for x in to_remove if x[-1] == 's'}
    to_remove |= {'QoS', 'RPCv9', 'ATSv2', 'NFSv3', 'MRv2', 'GHz',
                  'ID-ing'}
    to_remove = {x.lower() for x in to_remove}
    if match.group().lower().strip() in to_remove:
        return match.group()    # Heuristically, this is not a class name
    misc_ignore = {'CamelCase', 'LinkedIn'}
    misc_ignore = {x.lower() for x in misc_ignore}
    if lemmatized in misc_ignore or raw in misc_ignore:
        return match.group()
    reporter.inc('Class Name (No Package)')
    if get_formatting_settings() == FormattingHandling.Keep:
        return match.group()
    if get_formatting_settings() == FormattingHandling.Remove:
        return ''
    return match.group()[0] + 'SIMPLECLASSNAME'


##############################################################################
##############################################################################
# List Removal
##############################################################################


def _remove_lists_from_text(text: str) -> str:
    parts = []
    capitalize_next = False
    for line in text.splitlines():
        capitalize = capitalize_next
        line, capitalize_next = _remove_list_item(line)
        if capitalize:
            line = line[0].upper() + line[1:]
        if line.strip():
            parts.append(line)
    return '\n'.join(parts)


def _remove_list_item(line: str) -> (str, bool):
    stripped_line = line.strip()
    index = 0
    try:
        while stripped_line[index] in '*-#':
            index += 1
    except IndexError:
        print(f'|{line}|')
    if index == 0:
        return _remove_list_item_heuristic(line)
    reporter.inc('List')
    modified = stripped_line[index:].capitalize()
    if not modified.endswith('.'):
        modified += '.'
    return modified, True


def _remove_list_item_heuristic(line: str) -> (str, bool):
    pattern = re.compile(r'(\d)+[.)](?P<payload>.+)')
    if (match := pattern.fullmatch(line)) is not None:
        reporter.inc('List (Heuristic)')
        modified = match.group('payload')
        if not modified.endswith('.'):
            modified += '.'
        return modified, True
    return line, False


##############################################################################
##############################################################################
# Traceback Removal
##############################################################################


def _remove_unformatted_lines(key: str, text: str):
    if get_formatting_settings() == FormattingHandling.Keep:
        return text
    lines = text.splitlines()
    trimmed = '\n'.join(_check_line(key, line) for line in lines)
    trimmed = re.sub(r'LLLOG(\s*LLLOG)*', ' UNFORMATTEDLOGGINGOUTPUT ', trimmed)
    trimmed = re.sub(r'TTTRACEBACK(\s*TTTRACEBACK)*', ' UNFORMATTEDTRACEBACK ', trimmed)
    if get_formatting_settings() == FormattingHandling.Remove:
        trimmed = trimmed.replace('UNFORMATTEDLOGGINGOUTPUT', '')
        trimmed = trimmed.replace('UNFORMATTEDTRACEBACK', '')
    return trimmed


def _check_line(key: str, line: str) -> str:
    if _test_is_log_line(line):
        reporter.inc('Log Line')
        return 'LLLOG'
    if _test_is_tb_line(line):
        reporter.inc('Traceback line')
        return 'TTTRACEBACK'
    return line


def _test_is_log_line(line: str) -> bool:
    regexes = [
        r'\s*\d\d/\d\d/\d\d \d\d:\d\d:\d\d (DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE) .*\s*',
        r'\s*\[(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE)\] .*\s*',
        r'\s*\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d (DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE) .*\s*',
        r'\s*(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE) .*? \d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d .*\s*',
        r'\s*[A-Z][a-z]{,2} \d\d?, \d\d\d\d \d\d?:\d\d?:\d\d? (AM|PM) .*?\s*',
        r'\s*(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|SEVERE): .*\s*',
        r'\s*ERROR - .*\s*',
        r'\s*INFO .*\s*',
    ]
    return any(
        re.fullmatch(pattern, line) is not None
        for pattern in regexes
    )


def _test_is_tb_line(line: str) -> bool:
    regexes = [
        r'\s*at [$#]?\w+([\.$#]\w+)*\(.*\)\s*',
        r'\s*Caused by: \w+(\.\w+)*: .*\s*',
        r'\s*(\w+\.)*\w+(Error|Exception)(: (\w+\.)*\w+(Error|Exception))*\s*'
    ]
    return any(
        re.fullmatch(pattern, line) is not None
        for pattern in regexes
    )


##############################################################################
##############################################################################
# Block Scanning and Removal
##############################################################################


# Removes code blocks from a text
def _remove_code_blocks(text: str, *, place_markers=True) -> str:
    # Step 1: find all text markers
    starts = list(re.finditer(r'\{code:.*?\}', text))
    generic = list(re.finditer(r'\{code\}', text))
    # Step 2: Filter out all equal objects
    pure_starts = []
    for match in starts:
        for item in generic:
            if match.group() == item.group() and match.start() == item.start():
                break
        else:
            pure_starts.append(match)
    # Step 3: Order all match objects
    markers = [(s, True) for s in pure_starts] + [(s, False) for s in generic]
    markers.sort(key=lambda x: x[0].start())
    # Step 4: Remove code blocks, or resolve ambiguity
    removals = []
    while len(markers) >= 2:
        (start, start_is_pure), (end, end_is_pure), *markers = markers
        if end_is_pure:
            # We have two starting tags; We ignore the second one
            markers.insert(0, (start, start_is_pure))
            continue
        removals.append((start.start(), end.end()))
    if markers:
        marker, is_pure = markers.pop()
        # assume this is an unmatched start; remove the entirety of the remaining string
        removals.append((marker.start(), len(text)))
    # Step 5: Remove parts from the string
    # print(f'Found {len(removals)} code blocks')
    for start, stop in reversed(removals):
        reporter.inc('Code Block')
        marker = _guess_marker(text[start:stop], 'STRUCTUREDCODEBLOCK')
        text = f'{text[:start]} {f"{marker} " if place_markers else ""}{text[stop + 1:]}'
    return text


def _remove_no_format_blocks(text: str, *, place_markers=True) -> str:
    matches = list(re.finditer(r'\{noformat\}', text))
    markers = []
    for i, match in enumerate(matches):
        if i % 2 == 0:
            markers.append(match.start())
        else:
            markers.append(match.end())
    # If the last block is not closed, remove all trailing content
    if len(markers) % 2 == 1:
        markers.append(len(text))
    # Create pairs of markers
    blocks = []
    for start, end in zip(markers[::2], markers[1::2]):
        blocks.append((start, end))
    # Remove code from input string
    # print(f'Found {len(blocks)} no-format blocks')
    for start, stop in reversed(blocks):
        reporter.inc('No-Format Block')
        marker = _guess_marker(text[start:stop], 'NOFORMATBLOCK')
        text = f'{text[:start]} {f"{marker} " if place_markers else ""}{text[stop:]}'
    return text


def _guess_marker(text, default):
    stripped = _remove_unformatted_lines('', text)
    log_key = 'UNFORMATTEDLOGGINGOUTPUT'
    ex_key = 'UNFORMATTEDTRACEBACK'
    if ex_key in stripped:
        return 'FORMATTEDTRACEBACK'
    if log_key in stripped:
        return 'FORMATTEDLOGGINGOUTPUT'
    return default 


##############################################################################
##############################################################################
# Heuristic Code Removal
##############################################################################


def _remove_unformatted_code(text: str) -> str:
    # In this function, we want to identify snippets of code located
    # inside the text without any additional formatting.
    #
    # We use the following heuristic method:
    # 1) And line containing ":", ";", "{", or "}", is marked as a
    #       candidate line of code.
    # 2) Blocks of consecutive candidate lines of code are
    #       identified and examined.
    #

    # Step 1: Identify candidate lines
    lines = text.splitlines()

    if not lines:
        return text

    # Step 2: Group candidate lines
    groups = []
    for key, grouper in itertools.groupby(lines, key=_is_candidate_code_line):
        groups.append((key, list(grouper)))

    # Step 3: Identify sufficiently large blocks of
    #           candidate lines
    new_lines = []
    for is_candidate, group in groups:
        if not is_candidate:
            new_lines.extend(group)
        elif _is_likely_json(group):
            new_lines.append('JSONORSCHEMA')
        elif not _has_strong_candidate(group):
            new_lines.extend(group)
        else:
            stream = _remove_weak_lines(group)
            for key, grouper in itertools.groupby(stream, lambda x: x[0]):
                if not key:
                    new_lines.extend(x[1] for x in grouper)
                else:
                    new_lines.append('UNFORMATTEDCODE')

    if lines != new_lines:
        print('=' * 72)
        print('=' * 72)
        print('=' * 72)
        print('\n'.join(lines))
        print('-' * 40)
        print('-' * 40)
        print('\n'.join(new_lines))

    return '\n'.join(new_lines)


def _is_candidate_code_line(test_line: str) -> bool:
    return any(symbol in test_line for symbol in [':', ';', '{', '}', '||', '&&'])


def _is_strong_candidate_line(g: str) -> bool:
    return g.strip().endswith((';', '{', '}')) or any(
        x in g for x in ['||', '&&']
    )


def _is_likely_json(group: str) -> bool:
    block = '\n'.join(group)
    try:
        json.loads(block)
    except json.JSONDecodeError:
        pass
    else:
        return True
    key_value_pattern = re.compile(r'"\w+":\s?.*?,')
    if len(key_value_pattern.findall(block)) > 1:
        return True


def _has_strong_candidate(group):
    return any(
        _is_strong_candidate_line(g)
        for g in group
    )


def _remove_weak_lines(group):
    without_prefix = []
    key = False
    for is_strong, text in _loop_with_pred(_is_strong_candidate_line, group):
        if not key and is_strong:
            key = True
        without_prefix.append((key, text))

    without_suffix = []
    key = False
    for is_strong, text in _loop_with_pred(_is_strong_candidate_line, reversed(group)):
        if not key and is_strong:
            key = True
        without_suffix.append((key, text))

    return reversed(without_suffix)


def _loop_with_pred(pred, stream):
    for x in stream:
        yield pred(x), x
