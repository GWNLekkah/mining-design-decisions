import collections
import json
import os.path

import issue_db_api

from ... import config
from ... import accelerator


def _check_one_none(a, b, name):
    if a is None and b is None:
        raise ValueError(f'For the {name} file, either the ident or the path must be given.')
    if a is not None and b is not None:
        raise ValueError(f'For the {name} file, not both the ident and the path can be given')

def replace_technologies(issues: list[list[str]],
                         keys: list[str],
                         project_names_ident: str | None,
                         project_name_lookup_ident: str | None,
                         this_project_replacement: str,
                         other_project_replacement: str,
                         conf: config.Config, *,
                         project_names_file=None,
                         name_lookup_file=None) -> list[list[str]]:
    _check_one_none(project_names_ident, project_names_file, 'project_names')
    _check_one_none(project_name_lookup_ident, name_lookup_file, 'name_lookup')
    num_threads = conf.get('system.resources.threads')
    if project_name_lookup_ident:
        name_lookup_file = load_db_file(project_name_lookup_ident, conf)
    if name_lookup_file:
        if not this_project_replacement:
            raise ValueError('this-technology-replacement must be given')
        with open(name_lookup_file) as file:
            project_name_lookup = json.load(file)
        issues = replace_this_system(
            keys,
            issues,
            project_name_lookup,
            this_project_replacement,
            num_threads
        )
    if project_names_ident:
        project_names_file = load_db_file(project_names_ident, conf)
    if project_names_file:
        if not other_project_replacement:
            raise ValueError('other-technology-replacement must be given')
        with open(project_names_file) as file:
            project_names = json.load(file)
        issues = replace_other_systems(
            project_names,
            issues,
            other_project_replacement,
            num_threads
        )
    return issues


def replace_this_system(keys: list[str],
                        issues: list[list[str]],
                        lookup: dict[str, list[str]],
                        replacement: str,
                        num_threads: int) -> list[list[str]]:
    # result = []
    # for key, issue in zip(keys, issues):
    #     new_issue = []
    #     for sent in issue:
    #         if key in lookup:
    #             for repl in lookup[key]:
    #                 sent = sent.replace(repl, replacement)
    #         new_issue.append(sent)
    #     result.append(new_issue)
    # return result
    issues_by_project = collections.defaultdict(list)
    for (i, key), issue in zip(enumerate(keys), issues):
        project = key.split('-')[0]
        issues_by_project[project].append((i, issue))
    result = []
    for project, issues in issues_by_project.items():
        if project in lookup:
            indices, documents = zip(*issues)
            documents = accelerator.bulk_replace_parallel_string(
                list(documents),
                lookup[project],
                replacement,
                num_threads
            )
            result.extend(zip(indices, documents))
        else:
            result.extend(issues)
    result.sort()
    return [pair[1] for pair in result]


def replace_other_systems(projects: list[str],
                          issues: list[list[str]],
                          replacement: str,
                          num_threads: int) -> list[list[str]]:
    # result = []
    # for issue in issues:
    #     new_issue = []
    #     for sent in issue:
    #         for p in projects:
    #             sent = sent.replace(p, replacement)
    #         new_issue.append(sent)
    #     result.append(new_issue)
    # return result
    return accelerator.bulk_replace_parallel_string(
        issues,
        projects,
        replacement,
        num_threads
    )

# def replace_technologies(issues: list[list[str]],
#                          keys: list[str],
#                          project_names_ident: str,
#                          project_name_lookup_ident: str,
#                          this_project_replacement: list[str],
#                          other_project_replacement: list[str],
#                          conf: config.Config) -> list[list[str]]:
#     num_threads = conf.get('system.resources.num-threads')
#     if project_name_lookup_ident:
#         project_name_lookup = load_db_file(project_name_lookup_ident, conf)
#         if not this_project_replacement:
#             raise ValueError('this-technology-replacement must be given')
#         issues = replace_this_system(
#             keys,
#             issues,
#             project_name_lookup,
#             this_project_replacement,
#             num_threads
#         )
#     if project_names_ident:
#         project_names = load_db_file(project_names_ident, conf)
#         if not other_project_replacement:
#             raise ValueError('other-technology-replacement must be given')
#         issues = replace_other_systems(
#             project_names,
#             issues,
#             other_project_replacement,
#             num_threads
#         )
#     return issues
#
#
# def replace_this_system(keys: list[str],
#                         issues: list[list[str]],
#                         lookup: dict[str, list[list[str]]],
#                         replacement: list[str],
#                         num_threads: int) -> list[list[str]]:
#     issues_by_project = collections.defaultdict(list)
#     for (i, key), issue in zip(enumerate(keys), issues):
#         project = key.split('-')[0]
#         issues_by_project[project].append((i, issue))
#     result = []
#     for project, issues in issues_by_project.items():
#         if project in lookup:
#             indices, documents = zip(*issues)
#             documents = accelerator.bulk_replace_parallel(
#                 list(documents),
#                 lookup[project],
#                 replacement,
#                 num_threads
#             )
#             result.extend(zip(indices, documents))
#         else:
#             result.extend(issues)
#     result.sort()
#     return [pair[1] for pair in result]
#
#
# def replace_other_systems(project_names: list[list[str]],
#                           issues: list[list[str]],
#                           replacement: list[str],
#                           num_threads: int) -> list[list[str]]:
#     return accelerator.bulk_replace_parallel(issues,
#                                              project_names,
#                                              replacement,
#                                              num_threads)


def get_filename(ident: str, conf: config.Config):
    path = os.path.join(
        conf.get('system.os.scratch-directory'),
        ident
    )
    return path


def load_db_file(ident: str, conf: config.Config):
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    file = db.get_file_by_id(ident)
    path = get_filename(ident, conf)
    file.download(path)
    # with open(path) as f:
    #     return json.load(f)
    return path
