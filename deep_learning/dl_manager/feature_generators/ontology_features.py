import collections

from . import ParameterSpec
from .generator import AbstractFeatureGenerator
from ..classifiers import InputEncoding

from ..config import conf

from .util.ontology import load_ontology, OntologyTable


class OntologyFeatures(AbstractFeatureGenerator):

    MARKERS = (
        'attachment',
        'githublink',
        'issuelink',
        'weblink',
        'inlinecodesample',
        'filepath',
        'versionnumber',
        'storagesize',
        'methodorvariablename',
        'classname',
        'package',
        'simplemethodorvariablename',
        'simpleclassname',
        'unformattedloggingoutput',
        'unformattedtraceback',
        'structuredcodeblock',
        'noformatblock',
        'jsonschema',
        'unformattedcode',
    )

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    def generate_vectors(self,
                         tokenized_issues: list[list[str]],
                         metadata,
                         args: dict[str, str]):
        # This pre-processing does not require learning
        # anything.
        # If this is changed, be sure to implement
        # functionality for dealing with the .pretrained
        # attribute.
        ontology_path = conf.get('make-features.ontology-classes')
        if ontology_path == '':
            raise ValueError('--ontology-classes parameter must be given')
        table = load_ontology(ontology_path)
        order = tuple(table.classes)
        features = [
            self._make_feature(issue, table, order)
            for issue in tokenized_issues
        ]

        if self.pretrained is None:
            self.save_pretrained({})

        return {
            'features': features,
            'feature_shape': len(order)
        }

    def _make_feature(self,
                      issue: list[str],
                      table: OntologyTable,
                      order: tuple[str]) -> list[int]:
        counts = collections.defaultdict(int)
        for word in issue:
            if word in table.classes or word in self.MARKERS:
                counts[word] += 1
            else:
                cls = table.get_ontology_class(word, '')
                if cls != word:
                    counts[cls] += 1
        return [counts[x] for x in order]
        #return [len(issue)] + [counts[x] for x in order]

    @staticmethod
    def get_parameters() -> dict[str, ParameterSpec]:
        return super(OntologyFeatures, OntologyFeatures).get_parameters()
