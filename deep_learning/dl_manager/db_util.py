import json

from issue_db_api import Query


def json_to_query(j) -> Query:
    if isinstance(j, str):
        return string_to_query(j)
    return object_to_query(j)


def string_to_query(s: str) -> Query:
    return object_to_query(json.loads(s))


def object_to_query(o: object) -> Query:
    match o:
        case {'$and': [*children]}:
            return Query().land(*(object_to_query(c) for c in children))
        case {'$or': [*children]}:
            return Query().lor(*(object_to_query(c) for c in children))
        case {'tags': {'$eq': tag}}:
            return Query().tag(tag)
        case {'tags': {'$ne': tag}}:
            return Query().not_tag(tag)
        case _ as x:
            raise ValueError(f'Could not convert object to query: {x}')
