import collections
import json
import os.path

import issue_db_api

from ... import config
from ... import accelerator


def replace_technologies(issues: list[list[str]],
                         keys: list[str],
                         project_names_ident: str,
                         project_name_lookup_ident: str,
                         this_project_replacement: list[str],
                         other_project_replacement: list[str],
                         conf: config.Config) -> list[list[str]]:
    num_threads = conf.get('system.resources.num-threads')
    if project_name_lookup_ident:
        project_name_lookup = load_db_file(project_name_lookup_ident, conf)
        if not this_project_replacement:
            raise ValueError('this-technology-replacement must be given')
        issues = replace_this_system(
            keys,
            issues,
            project_name_lookup,
            this_project_replacement,
            num_threads
        )
    if project_names_ident:
        project_names = load_db_file(project_names_ident, conf)
        if not other_project_replacement:
            raise ValueError('other-technology-replacement must be given')
        issues = replace_other_systems(
            project_names,
            issues,
            other_project_replacement,
            num_threads
        )
    return issues


def replace_this_system(keys: list[str],
                        issues: list[list[str]],
                        lookup: dict[str, list[list[str]]],
                        replacement: list[str],
                        num_threads: int) -> list[list[str]]:
    issues_by_project = collections.defaultdict(list)
    for (i, key), issue in zip(enumerate(keys), issues):
        project = key.split('-')[0]
        issues_by_project[project].append((i, issue))
    result = []
    for project, issues in issues_by_project.items():
        if project in lookup:
            indices, documents = zip(*issues)
            documents = accelerator.bulk_replace_parallel(
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


def replace_other_systems(project_names: list[list[str]],
                          issues: list[list[str]],
                          replacement: list[str],
                          num_threads: int) -> list[list[str]]:
    return accelerator.bulk_replace_parallel(issues,
                                             project_names,
                                             replacement,
                                             num_threads)


def load_db_file(ident: str, conf: config.Config):
    db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
    file = db.get_file_by_id(ident)
    path = os.path.join(
        conf.get('system.os.scratch-directory'),
        ident
    )
    file.download(path)
    with open(path) as f:
        return json.load(f)
