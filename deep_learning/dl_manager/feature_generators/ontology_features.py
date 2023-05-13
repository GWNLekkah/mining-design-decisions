import collections

from ..config import Argument
from .generator import AbstractFeatureGenerator, FeatureEncoding
from ..model_io import InputEncoding


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
        if self.pretrained is None:
            ontology_path = self.conf.get('run.ontology-classes')
            if ontology_path == '':
                raise ValueError('--ontology-classes parameter must be given')
        else:
            aux_map = self.conf.get('system.storage.auxiliary_map')
            ontology_path = aux_map[self.pretrained['ontologies']]
        table = load_ontology(ontology_path)
        order = tuple(table.classes)
        features = [
            self._make_feature(issue, table, order)
            for issue in tokenized_issues
        ]

        if self.pretrained is None:
            self.save_pretrained(
                {
                    'ontologies': ontology_path
                },
                [
                   ontology_path
                ]
            )

        return {
            'features': features,
            'feature_shape': len(order),
            'feature_encoding': {
                'encoding': self.feature_encoding(),
                'metadata': []
            }
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
    def feature_encoding() -> FeatureEncoding:
        return FeatureEncoding.Numerical

    @staticmethod
    def get_arguments() -> dict[str, Argument]:
        return super(OntologyFeatures, OntologyFeatures).get_arguments()
