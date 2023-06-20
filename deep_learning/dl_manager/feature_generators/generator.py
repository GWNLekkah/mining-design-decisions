##############################################################################
##############################################################################
# Imports
##############################################################################

from __future__ import annotations

import abc
import csv
import enum
import hashlib
import itertools
import json
import os.path
import random
import string

import nltk

import issue_db_api

from ..config import (
    Argument,
    BoolArgument,
    IntArgument,
    StringArgument,
    EnumArgument,
    ArgumentConsumer,
)
from .util.text_cleaner import FormattingHandling, clean_issue_text
from .. import accelerator
from ..model_io import InputEncoding, classification8_lookup
from ..custom_kfold import stratified_trim
from .util import ontology
from .util.technology_replacer import replace_technologies, get_filename as get_technology_file_filename
from ..config import Config
from ..logger import get_logger, timer
from ..data_manager import Dataset


log = get_logger("Base Feature Generator")

from ..data_manager_bootstrap import get_raw_text_file_name

csv.field_size_limit(100000000)

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

ATTRIBUTE_CONSTANTS = {
    "n_attachments": "n_attachments",
    "n_comments": "n_comments",
    "len_comments": "len_comments",
    "n_components": "n_components",
    "len_description": "len_description",
    "n_issuelinks": "n_issuelinks",
    "n_labels": "n_labels",
    "parent": "parent",
    "n_subtasks": "n_subtasks",
    "len_summary": "len_summary",
    "n_votes": "n_votes",
    "n_watches": "n_watches",
    "issuetype": "issuetype",
    "labels": "labels",
    "priority": "priority",
    "resolution": "resolution",
    "status": "status",
}

##############################################################################
##############################################################################
# Auxiliary Classes
##############################################################################


class FeatureEncoding(enum.Enum):
    Numerical = enum.auto()  # No metadata
    Categorical = enum.auto()  # No metadata
    Mixed = enum.auto()  # Metadata = Indices of categorical features
    Bert = enum.auto()  # No metadata

    def as_string(self):
        match self:
            case self.Numerical:
                return "numerical"
            case self.Categorical:
                return "categorical"
            case self.Mixed:
                return "mixed"
            case self.Bert:
                return "bert"

    def from_string(self, x: str):
        match x:
            case "numerical":
                return self.Numerical
            case "categorical":
                return self.Categorical
            case "mixed":
                return self.Mixed
            case "bert":
                return self.Bert
            case _:
                raise ValueError(f"Invalid feature encoding: {x}")


def _escape(x):
    for ws in string.whitespace:
        x = x.replace(ws, "_")
    x = x.replace(".", "dot")
    for illegal in "/<>:\"/\\|?*'":
        x = x.replace(illegal, "")
    return x


class _NullDict(dict):
    def __missing__(self, key):
        return key


##############################################################################
##############################################################################
# Main Class
##############################################################################


class AbstractFeatureGenerator(abc.ABC, ArgumentConsumer):
    def __init__(
        self,
        conf: Config,
        /,
        *,
        pretrained_generator_settings: dict | None = None,
        **params,
    ):
        self.__params = params
        self.__pretrained = pretrained_generator_settings
        self.__colors = None
        self.__keys = None
        self.conf = conf
        self.__ontology_classes = None
        self.__apply_ontologies = False
        self._project_names_file = ''
        self._name_lookup_file = ''
        if self.__pretrained is not None:
            if self.__params:
                raise ValueError(
                    "Feature generator does not take params when pretrained settings are given"
                )
            # Populate params for default pre-processing,
            # which does not require any trained settings.
            for name in AbstractFeatureGenerator.get_arguments():
                if name in self.__pretrained:
                    self.__params[name] = self.__pretrained[name]
            aux = self.conf.get("system.storage.auxiliary-map")
            if "ontology-classes" in self.__pretrained:
                # self.conf.set('run.ontology-classes', aux[self.__pretrained['ontology-classes']])
                self.__ontology_classes = aux[self.__pretrained["ontology-classes"]]
                self.__apply_ontologies = self.__pretrained["use-ontology-classes"]
            if x := self.__pretrained['$project-names-file']:
                self._project_names_file = aux[x]
            if x := self.__pretrained['$project-lookup-file']:
                self._name_lookup_file = aux[x]
        else:
            self.__ontology_classes = (
                f'{conf.get("system.storage.file-prefix")}_ontologies.json'
            )
            self.__have_ontology_classes = False
            if ident := conf.get("run.ontology-classes"):
                repo: issue_db_api.IssueRepository = conf.get(
                    "system.storage.database-api"
                )
                ontology_file = repo.get_file_by_id(ident)
                ontology_file.download(self.__ontology_classes)
                self.__apply_ontologies = conf.get("run.apply-ontology-classes")
                self.__have_ontology_classes = True

    def require_ontology_classes(self):
        if not self.__have_ontology_classes:
            raise ValueError("Need ontology classes")
        return self.__ontology_classes

    @property
    def params(self) -> dict[str, str]:
        return self.__params

    @property
    def pretrained(self) -> dict | None:
        return self.__pretrained

    @property
    def colors(self) -> list[int]:
        if self.__colors is None:
            raise RuntimeError("No colors yet")
        return self.__colors

    @property
    def issue_keys(self) -> list[str]:
        if self.__keys is None:
            raise RuntimeError("No keys yet")
        return self.__keys

    def save_pretrained(
        self, pretrained_settings: dict, auxiliary_files: list[str] = None
    ):
        if auxiliary_files is None:
            auxiliary_files = []
        log.info(f"Saving {self.__class__.__name__} feature encoding")
        settings = "_".join(f"{key}-{value}" for key, value in self.__params.items())
        filename = f"{self.__class__.__name__}__{settings}"
        prefix = self.conf.get("system.storage.file-prefix")
        filename = f"{prefix}_{hashlib.sha512(filename.encode()).hexdigest()}.json"
        filename = os.path.join(self.conf.get("system.os.scratch-directory"), filename)
        for name in AbstractFeatureGenerator.get_arguments():
            if name in self.__params:
                pretrained_settings[name] = self.__params[name]
        if self.__have_ontology_classes:
            pretrained_settings["ontology-classes"] = self.__ontology_classes
            self.conf.get("system.storage.auxiliary").append(self.__ontology_classes)
        pretrained_settings["use-ontology-classes"] = self.conf.get(
            "run.apply-ontology-classes"
        )
        if x := self.params['replace-other-technologies-list']:
            pretrained_settings['$project-names-file'] = get_technology_file_filename(x, self.conf)
            self.conf.get('system.storage.auxiliary').append(
                get_technology_file_filename(x, self.conf)
            )
        if x := self.params['replace-this-technology-mapping']:
            pretrained_settings['$project-lookup-file'] = get_technology_file_filename(x, self.conf)
            self.conf.get('system.storage.auxiliary').append(
                get_technology_file_filename(x, self.conf)
            )
        self.conf.get("system.storage.generators").append(filename)
        self.conf.get("system.storage.auxiliary").extend(auxiliary_files)
        with open(filename, "w") as file:
            json.dump(
                {
                    "settings": pretrained_settings,
                    "generator": self.__class__.__name__,
                },
                file,
            )

    @staticmethod
    @abc.abstractmethod
    def input_encoding_type() -> InputEncoding:
        """Type of input encoding generated by this generator."""

    @abc.abstractmethod
    def generate_vectors(
        self, tokenized_issues: list[list[str]], metadata, args: dict[str, str]
    ):
        # TODO: implement this method
        # TODO: this method should take in data, and generate
        # TODO: the corresponding feature vectors
        pass

    @staticmethod
    @abc.abstractmethod
    def feature_encoding() -> FeatureEncoding:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_arguments() -> dict[str, Argument]:
        return {
            "max-len": IntArgument(
                name="max-len",
                description="words limit of the issue text. Set to -1 to disable.",
                minimum=-1,
                default=-1,
            ),
            "disable-lowercase": BoolArgument(
                name="disable-lowercase",
                description="transform words to lowercase",
                default=False,
            ),
            "disable-stopwords": BoolArgument(
                name="disable-stopwords",
                description="remove stopwords from text",
                default=False,
            ),
            "use-stemming": BoolArgument(
                name="use-stemming",
                description="stem the words in the text",
                default=False,
            ),
            "use-lemmatization": BoolArgument(
                name="use-lemmatization",
                description="Use lemmatization on words in the text",
                default=False,
            ),
            "use-pos": BoolArgument(
                name="use-pos",
                description="Enhance words in the text with part of speech information",
                default=False,
            ),
            "class-limit": IntArgument(
                name="class-limit",
                description="limit the amount of items per class. Set to -1 to disable",
                default=-1,
                minimum=-1,
            ),
            "metadata-attributes": StringArgument(
                name="metadata-attributes",
                description="Comma-separated list of metadata attributes to fetch for use in feature generation",
                default="",
            ),
            "formatting-handling": EnumArgument(
                name="formatting-handling",
                description="How to handle formatting",
                options=["markers", "remove", "keep"],
                default="markers",
            ),
            'replace-this-technology-mapping': StringArgument(
                name='replace-this-technology-mapping',
                description='If given, should be a file mapping project keys to project names. '
                            'Project names in text will be replacement with `this-technology-replacement`.',
                default=''
            ),
            'this-technology-replacement': StringArgument(
                name='this-technology-replacement',
                description='See description of `replace-this-technology-mapping`',
                default=''
            ),
            'replace-other-technologies-list': StringArgument(
                name='replace-other-technologies-list',
                description='If given, should be a file containing a list of project names. '
                            'Project names will be replaced with `other-technology-replacement`',
                default=''
            ),
            'other-technology-replacement': StringArgument(
                name='other-technology-replacement',
                description='See description of `replace-other-technology-list`.',
                default=''
            ),
        }

    def load_data_from_db(self, query: issue_db_api.Query, metadata_attributes):
        api: issue_db_api.IssueRepository = self.conf.get("system.storage.database-api")
        issues = api.search(
            query,
            attributes=metadata_attributes + ["key", "summary", "description"],
            load_labels=True,
        )
        issues.sort(key=lambda i: i.identifier)
        labels = {
            "detection": [],
            "classification3": [],
            "classification3simplified": [],
            "classification8": [],
            "issue_keys": [issue.key for issue in issues],
            "issue_ids": [issue.identifier for issue in issues],
        }
        classification_indices = {
            "Existence": [],
            "Property": [],
            "Executive": [],
            "Non-Architectural": [],
        }
        if self.pretrained is None:
            raw_labels = [issue.manual_label for issue in issues]
            for index, raw in enumerate(raw_labels):
                self.update_labels(
                    labels,
                    classification_indices,
                    index,
                    raw.existence,
                    raw.executive,
                    raw.property,
                )
        texts = []
        for issue in issues:
            # summary = x if (x := issue.pop('summary')) is not None else ''
            # description = x if (x := issue.pop('description')) is not None else ''
            # labels['issue_keys'].append(issue.pop('key'))
            texts.append([issue.summary, issue.description])
        metadata = []
        for issue in issues:
            metadata.append(
                {attr: getattr(issue, attr) for attr in metadata_attributes}
            )
        return texts, metadata, labels, classification_indices

    def update_labels(
        self,
        labels,
        classification_indices,
        current_index,
        is_existence,
        is_executive,
        is_property,
    ):
        if self.__colors is None:
            self.__colors = []
        if is_executive:  # Executive
            labels["classification3simplified"].append((0, 1, 0, 0))
            classification_indices["Executive"].append(current_index)
            self.__colors.append(0)
        elif is_property:  # Property
            labels["classification3simplified"].append((0, 0, 1, 0))
            classification_indices["Property"].append(current_index)
            self.__colors.append(1)
        elif is_existence:  # Existence
            labels["classification3simplified"].append((1, 0, 0, 0))
            classification_indices["Existence"].append(current_index)
            self.__colors.append(2)
        else:  # Non-architectural
            labels["classification3simplified"].append((0, 0, 0, 1))
            classification_indices["Non-Architectural"].append(current_index)
            self.__colors.append(3)

        if is_executive or is_property or is_existence:
            labels["detection"].append(True)
        else:
            labels["detection"].append(False)

        key = (is_existence, is_executive, is_property)
        labels["classification8"].append(classification8_lookup[key])
        labels["classification3"].append(key)

    def generate_features(self, query: issue_db_api.Query, output_mode: str):
        """Generate features from the data in the given source file,
        and store the results in the given target file.
        """
        metadata_attributes = [
            attr for attr in self.__params["metadata-attributes"].split(",") if attr
        ]
        for attr in metadata_attributes:
            if attr not in ATTRIBUTE_CONSTANTS:
                raise ValueError(f"Unknown metadata attribute: {attr}")

        texts, metadata, labels, classification_indices = self.load_data_from_db(
            query, metadata_attributes
        )

        limit = self.params["class-limit"]
        if limit != -1 and self.pretrained is None:  # Only execute if not pretrained
            stratified_indices = []
            for issue_type in classification_indices.keys():
                project_labels = [
                    label
                    for index, label in enumerate(
                        [label.split("-")[0] for label in labels["issue_keys"]]
                    )
                    if index in classification_indices[issue_type]
                ]
                trimmed_indices = stratified_trim(limit, project_labels)
                stratified_indices.extend(
                    [classification_indices[issue_type][idx] for idx in trimmed_indices]
                )
            texts = [
                text for idx, text in enumerate(texts) if idx in stratified_indices
            ]
            for key in labels.keys():
                labels[key] = [
                    label
                    for idx, label in enumerate(labels[key])
                    if idx in stratified_indices
                ]

        # The replace_technologies function
        # already performs the substitutions conditionally,
        # so it can be in the main code path here.
        if self.pretrained is None:
            texts = replace_technologies(
                issues=texts,
                keys=labels['issue_keys'],
                project_names_ident=self.params['replace-other-technologies-list'],
                project_name_lookup_ident=self.params['replace-this-technology-mapping'],
                this_project_replacement=self.params['this-technology-replacement'],
                other_project_replacement=self.params['other-technology-replacement'],
                conf=self.conf
            )
        else:
            # Cry
            texts = replace_technologies(
                issues=texts,
                keys=labels['issue_keys'],
                project_names_ident=None,
                project_name_lookup_ident=None,
                this_project_replacement=self.params['this-technology-replacement'],
                other_project_replacement=self.params['other-technology-replacement'],
                conf=self.conf,
                project_names_file=self._project_names_file,
                name_lookup_file=self._name_lookup_file
            )

        if self.input_encoding_type() == InputEncoding.Text:
            tokenized_issues = [[". ".join(text)] for text in texts]
        else:
            # with cProfile.Profile() as p:
            #    tokenized_issues = self.preprocess(texts)
            # p.dump_stats('profile.txt')
            tokenized_issues = self.preprocess(texts, labels['issue_keys'])

        log.info("Generating feature vectors")
        with timer("Feature Generation"):
            output = self.generate_vectors(tokenized_issues, metadata, self.__params)
        output["labels"] = labels  # labels is empty when pretrained

        output["original"] = tokenized_issues
        if (
            "original" in output and not self.pretrained
        ):  # Only dump original text when not pre-trained.
            with open(get_raw_text_file_name(self.conf), "w") as file:
                mapping = {
                    key: text
                    for key, text in zip(labels["issue_ids"], output["original"])
                }
                json.dump(mapping, file)
            del output["original"]
        elif "original" in output:
            del output["original"]

        return Dataset(
            features=output["features"],
            labels=output["labels"][output_mode.lower()],
            shape=output["feature_shape"],
            embedding_weights=output.get("weights", None),
            vocab_size=output.get("vocab_size", None),
            weight_vector_length=output.get("word_vector_length", None),
            binary_labels=output["labels"]["detection"],
            issue_keys=output["labels"]["issue_keys"],
            ids=output["labels"]["issue_ids"],
        )

    def apply_technology_substitutions(self):
        pass

    def preprocess(self, issues, issue_keys):
        log.info("Preprocessing Features")
        with timer("Feature Preprocessing"):
            if self.__apply_ontologies:
                ontology_table = ontology.load_ontology(self.__ontology_classes)
            else:
                ontology_table = None

            stopwords = nltk.corpus.stopwords.words("english")
            use_stemming = self.__params["use-stemming"]
            use_lemmatization = self.__params["use-lemmatization"]
            use_pos = self.__params["use-pos"]
            stemmer = nltk.stem.PorterStemmer()
            lemmatizer = nltk.stem.WordNetLemmatizer()
            use_lowercase = not self.__params["disable-lowercase"]
            use_ontologies = self.__apply_ontologies
            handling_string = self.__params["formatting-handling"]
            handling = FormattingHandling.from_string(handling_string)
            weights, tagdict, classes = nltk.load(
                "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
            )
            tagger = accelerator.Tagger(weights, classes, tagdict)

            summaries, descriptions = (list(x) for x in zip(*issues))
            summaries = accelerator.bulk_clean_text_parallel(
                summaries,
                handling.as_string(),
                self.conf.get("system.resources.threads"),
            )
            summaries = [clean_issue_text(summary) for summary in summaries]
            descriptions = accelerator.bulk_clean_text_parallel(
                descriptions,
                handling.as_string(),
                self.conf.get("system.resources.threads"),
            )
            descriptions = [
                clean_issue_text(description) for description in descriptions
            ]
            texts = [
                [
                    nltk.word_tokenize(sent.lower() if use_lowercase else sent)
                    for sent in itertools.chain(summary, description)
                ]
                for summary, description in zip(summaries, descriptions)
            ]
            # old version which operated on tokens. No clue why I did that
            # texts = replace_technologies(
            #     keys=issue_keys,
            #     issues=texts,
            #     project_names_ident=self.params['replace-other-technologies-list'],
            #     project_name_lookup_ident=self.params['replace-this-technology-mapping'],
            #     this_project_replacement=self.params['this-technology-replacement'].split(),
            #     other_project_replacement=self.params['other-technology-replacement'].split(),
            #     conf=self.conf
            # )
            tagged = tagger.bulk_tag_parallel(
                texts, self.conf.get("system.resources.threads")
            )
            tokenized_issues = []
            for issue in tagged:
                all_words = []

                # Tokenize
                for words in issue:
                    # Apply ontology simplification. Must be done before stemming/lemmatization
                    if use_ontologies:
                        # assert ontology_table is not None, 'Missing --ontology-classes'
                        words = ontology.apply_ontologies_to_sentence(
                            words, ontology_table
                        )

                    # Remove stopwords
                    if not self.__params['disable-stopwords']:
                        words = [(word, tag) for word, tag in words if word not in stopwords]

                    if use_stemming and use_lemmatization:
                        raise ValueError("Cannot use both stemming and lemmatization")

                    if use_stemming:
                        words = [(stemmer.stem(word), tag) for word, tag in words]

                    if use_lemmatization:
                        words = [
                            (
                                lemmatizer.lemmatize(
                                    word, pos=POS_CONVERSION.get(tag, "n")
                                ),
                                tag,
                            )
                            for word, tag in words
                        ]

                    if use_pos:
                        words = [
                            f"{word}_{POS_CONVERSION.get(tag, tag)}"
                            for word, tag in words
                        ]
                    else:
                        words = [word for word, _ in words]

                    # At this point, we forget about sentence order
                    all_words.extend(words)

                # Limit issue length
                if (m := self.__params["max-len"]) > 0:
                    if len(all_words) > m:
                        all_words = all_words[0:m]

                tokenized_issues.append(all_words)

        log.info("Finished preprocessing")
        return tokenized_issues
