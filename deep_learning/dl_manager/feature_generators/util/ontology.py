import dataclasses
import json
import os.path
import string

import nltk

POS_CONVERSION = {
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


_loc = os.path.split(__file__)[0]
with open(os.path.join(_loc, 'part_of_speech.json')) as _file:
    _part_of_speech_table = json.load(_file)


@dataclasses.dataclass(frozen=True, slots=True)
class OntologyClass:
    class_name: str
    part_of_speech: str | None
    content: set[str]


class OntologyTable:

    def __init__(self, *ontologies: OntologyClass):
        self.__lookup = {
            (word, cls.part_of_speech): cls.class_name
            for cls in ontologies
            for word in cls.content
        }
        self.__fallback = {
            word: cls.class_name
            for cls in ontologies
            for word in cls.content
        }
        self.__classes = {
            cls.class_name: cls
            for cls in ontologies
        }

    def get_ontology_class(self, word, pos) -> str:
        try:
            return self.__lookup[(word, pos)]
        except KeyError:
            try:
                return self.__fallback[word]
            except KeyError:
                return word

    def get_class_by_name(self, name: str) -> OntologyClass:
        return self.__classes[name]

    @property
    def classes(self) -> set[str]:
        return set(self.__classes)


def load_ontology(filename: str) -> OntologyTable:
    classes = []
    with open(filename) as file:
        for spec in json.load(file):
            cls = OntologyClass(
                class_name=spec['name'],
                part_of_speech=spec.get('part_of_speech', None),
                content=spec['content']
            )
            classes.append(cls)
    return OntologyTable(*classes)


def _simplify_tag(tag: str) -> str:
    tag = _part_of_speech_table['translations'][tag]
    while tag in _part_of_speech_table['simplifications']:
        tag = _part_of_speech_table['simplifications'][tag]
    return tag


def apply_ontologies_to_sentence(words: list[tuple[str, str]],
                                 ontology: OntologyTable) -> list[str]:
    #words = nltk.pos_tag(words)
    words = [(nltk.stem.WordNetLemmatizer().lemmatize(word, pos=POS_CONVERSION.get(pos, 'n')), pos) for word, pos in words]
    result = []
    for word, tag in words:
        if tag in string.punctuation or tag in ['``']:
            continue
        if tag == "''":
            tag = 'EMPTY'
        tag = _simplify_tag(tag)
        result.append(
            (ontology.get_ontology_class(word, tag), tag)
        )
    return result   # Result preserves tagging
