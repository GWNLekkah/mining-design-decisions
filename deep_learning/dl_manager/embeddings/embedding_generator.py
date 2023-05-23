import abc
import itertools
import os.path
import pathlib

import nltk

import issue_db_api

from .. import accelerator
from ..config import Config, BoolArgument, Argument, ArgumentConsumer, EnumArgument, StringArgument
from ..feature_generators.util.text_cleaner import FormattingHandling
from ..feature_generators.util.text_cleaner import clean_issue_text
from ..feature_generators.util.ontology import load_ontology, apply_ontologies_to_sentence
from ..logger import get_logger

log = get_logger('Embedding Generator')


TEMP_EMBEDDING_PATH = pathlib.Path('embedding_binary.bin')


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


class AbstractEmbeddingGenerator(abc.ABC, ArgumentConsumer):

    def __init__(self, **params):
        self.params = params

    def make_embedding(self,
                       query: issue_db_api.Query,
                       conf: Config):
        # Loading issues from database
        # db: DatabaseAPI = conf.get('system.storage.database-api')
        # issues = db.select_issues(query)
        # data = db.get_issue_data(issues, ['summary', 'description'])
        db: issue_db_api.IssueRepository = conf.get('system.storage.database-api')
        issues = db.search(query, attributes=['summary', 'description'])
        log.info(f'Training embedding on {len(issues)} issues (query: {query})')

        # Setting up NLP stuff
        formatting_handling = self.params['formatting-handling']
        handling = FormattingHandling.from_string(formatting_handling)
        stopwords = nltk.corpus.stopwords.words('english')
        use_lemmatization = self.params['use-lemmatization']
        use_stemming = self.params['use-stemming']
        use_pos = self.params['use-pos']
        use_ontologies = self.params['use-ontologies']
        ontology_id = self.params['ontology-id']
        if use_stemming and use_lemmatization:
            raise ValueError('Cannot use both stemming and lemmatization')
        if not (use_stemming or use_lemmatization):
            log.warning('Not using stemming or lemmatization')
        if use_ontologies and not ontology_id:
            raise ValueError('ontology-id must be given is use-ontologies is true')
        if use_ontologies:
            ontology_file = db.get_file_by_id(ontology_id)
            ontology_filename = f'{conf.get("system.storage.file-prefix")}_ontologies.json'
            ontology_file.download(ontology_filename)
        else:
            ontology_filename = None 
        stemmer = None
        lemmatizer = None
        if use_stemming:
            stemmer = nltk.stem.PorterStemmer()
        if use_lemmatization:
            lemmatizer = nltk.stem.WordNetLemmatizer()
        weights, tagdict, classes = nltk.load(
            'taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle'
        )
        tagger = accelerator.Tagger(weights, classes, tagdict)

        # Bulk processing
        summaries = [issue.summary for issue in issues]
        descriptions = [issue.description for issue in issues]
        summaries = accelerator.bulk_clean_text_parallel(
            summaries, handling.as_string(), conf.get('system.resources.threads')
        )
        summaries = [clean_issue_text(summary) for summary in summaries]
        descriptions = accelerator.bulk_clean_text_parallel(
            descriptions, handling.as_string(), conf.get('system.resources.threads')
        )
        descriptions = [clean_issue_text(description) for description in descriptions]
        texts = [
            [
                nltk.word_tokenize(sent.lower())
                for sent in itertools.chain(summary, description)
            ]
            for summary, description in zip(summaries, descriptions)
        ]
        texts = tagger.bulk_tag_parallel(texts, conf.get('system.resources.threads'))

        # Per-issue processing
        documents = []
        if use_ontologies:
            ontology_table = load_ontology(ontology_filename)
        else:
            ontology_table = None
        for issue in texts:
            document = []
            for words in issue:
                words = [(word, tag) for word, tag in words if word not in stopwords]
                if use_ontologies:
                    assert ontology_table is not None 
                    words = apply_ontologies_to_sentence(words, ontology_table)
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
                document.extend(words)
            documents.append(document)

        # Embedding generation
        embedding_path = os.path.join(
            conf.get('system.os.scratch-directory'),
            TEMP_EMBEDDING_PATH
        )
        self.generate_embedding(documents, pathlib.Path(embedding_path))

        # Upload binary file
        embedding_id = conf.get('generate-embedding-internal.embedding-id')
        embedding = db.get_embedding_by_id(embedding_id)
        embedding.upload_binary(embedding_path)


    @abc.abstractmethod
    def generate_embedding(self, issues: list[list[str]], path: pathlib.Path):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_arguments() -> dict[str, Argument]:
        return {
            'use-stemming': BoolArgument(
                name='use-stemming',
                description='stem the words in the text',
                default=False
            ),
            'use-lemmatization': BoolArgument(
                name='use-lemmatization',
                description='Use lemmatization on words in the text',
                default=True
            ),
            'use-pos': BoolArgument(
                name='use-pos',
                description='Enhance words in the text with part of speech information',
                default=False,
            ),
            'formatting-handling': EnumArgument(
                name='formatting-handling',
                description='How to handle formatting in issues.',
                options=['markers', 'keep', 'remove']
            ),
            'use-ontologies': BoolArgument(
                name='use-ontologies',
                description='If True, apply ontology classes to the input text.',
                default=False
            ),
            'ontology-id': StringArgument(
                name='ontology-id',
                description='ID to a file containing ontology classes.',
                default=''
            )
        }
