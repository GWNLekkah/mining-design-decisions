import abc
import dataclasses
import pathlib
import typing

import nltk
from nltk.tag import _get_tagger as get_tagger_internal
from nltk.tag import _pos_tag as pos_tag_internal

from ..config import conf
from ..database import DatabaseAPI
from ..feature_generators.util.text_cleaner import FormattingHandling
from ..feature_generators.util.text_cleaner import clean_issue_text
from ..logger import get_logger

log = get_logger('Embedding Generator')


@dataclasses.dataclass
class EmbeddingGeneratorParam:
    description: str
    data_type: str
    minimum: int | None = None
    maximum: int | None = None
    options: typing.Any | None = None


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


class AbstractEmbeddingGenerator(abc.ABC):

    def __init__(self, **params):
        self.params = params

    def make_embedding(self, query: str, path: pathlib.Path, formatting_handling: str):
        db: DatabaseAPI = conf.get('system.storage.database-api')
        issues = db.select_issues(query)
        data = db.get_issue_data(issues, ['summary', 'description'])
        handling = FormattingHandling.from_string(formatting_handling)
        stopwords = nltk.corpus.stopwords.words('english')
        use_lemmatization = self.params.get('use-lemmatization', 'False') == 'True'
        use_stemming = self.params.get('use-stemming', 'False') == 'True'
        use_pos = self.params.get('use-pos', 'False') == 'True'
        if use_stemming and use_lemmatization:
            raise ValueError('Cannot use both stemming and lemmatization')
        if not (use_stemming or use_lemmatization):
            log.warning('Not using stemming or lemmatization')
        stemmer = None
        lemmatizer = None
        if use_stemming:
            stemmer = nltk.stem.PorterStemmer()
        if use_lemmatization:
            lemmatizer = nltk.stem.WordNetLemmatizer()
        tagger = get_tagger_internal(lang='eng')
        documents = []
        for issue in data:
            summary = clean_issue_text(issue['summary'], handling)
            description = clean_issue_text(issue['description'], handling)
            text = summary + description
            for sentence in (sent.lower() for sent in text):
                words = nltk.word_tokenize(sentence)
                words = pos_tag_internal(words, None, tagger, 'eng')
                words = [(word, tag) for word, tag in words if word not in stopwords]
                if use_lemmatization:
                    words = [
                        (lemmatizer.lemmatize(word, pos=POS_CONVERSION.get(tag, 'n')), tag)
                        for word, tag in words
                    ]
                if use_stemming:
                    words = [(stemmer.stem(word), tag) for word, tag in words]
                if use_pos:
                    words = [f'{word}_{POS_CONVERSION.get(tag, tag)}' for word, tag in words]
                else:
                    words = [word for word, _ in words]
                documents.append(words)
        self.generate_embedding(documents, path)

    @abc.abstractmethod
    def generate_embedding(self, issues: list[str], path: pathlib.Path):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_params() -> dict[str, EmbeddingGeneratorParam]:
        return {
            'use-stemming': EmbeddingGeneratorParam(
                description='stem the words in the text',
                data_type='bool'
            ),
            'use-lemmatization': EmbeddingGeneratorParam(
                description='Use lemmatization on words in the text',
                data_type='bool'
            ),
            'use-pos': EmbeddingGeneratorParam(
                'Enhance words in the text with part of speech information',
                data_type='bool'
            )
        }
