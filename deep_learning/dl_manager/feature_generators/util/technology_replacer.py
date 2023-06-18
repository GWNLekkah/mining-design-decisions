import json
import os.path

import issue_db_api

from ... import config


def replace_technologies(issues: list[list[str]],
                         keys: list[str],
                         project_names_ident: set[str],
                         project_name_lookup_ident: dict[str, str],
                         this_project_replacement: list[str],
                         other_project_replacement: str,
                         conf: config.Config) -> list[list[str]]:
    if project_names_ident:
        project_names = load_db_file(project_names_ident, conf)
    else:
        project_names = set()
    if project_name_lookup_ident:
        project_name_lookup = load_db_file(project_name_lookup_ident, conf)
    else:
        project_name_lookup = {}
    if project_names and not other_project_replacement:
        raise ValueError('other-technology-replacement must be given')
    if project_name_lookup and not this_project_replacement:
        raise ValueError('this-technology-replacement must be given')
    return replace_other_systems(
        project_names,
        replace_this_system(
            keys,
            issues,
            project_name_lookup,
            this_project_replacement
        ),
        other_project_replacement
    )

def replace_this_system(keys: list[str],
                        issues: list[list[str]],
                        lookup: dict[str, str],
                        replacement: list[str]) -> list[list[str]]:
    result = []
    for key, issue in zip(keys, issues):
        project_key = key.split('-')[0]
        if project_key not in lookup:
            result.append(issue)
        else:
            new = []
            project_name = lookup[project_key]
            for word in issue:
                if word == project_name:
                    new.extend(replacement)
                else:
                    new.append(word)
            result.append(new)
    return result


def replace_other_systems(project_names: set[str],
                          issues: list[list[str]],
                          replacement: str) -> list[list[str]]:
    result = []
    for issue in issues:
        issue = [word if word not in project_names else replacement
                 for word in issue]
        result.append(issue)
    return result


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
