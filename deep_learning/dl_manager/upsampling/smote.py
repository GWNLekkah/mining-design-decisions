import imblearn

from ..config import EnumArgument
from .base import AbstractUpSampler
from ..feature_generators import generators, FeatureEncoding
from ..data_utilities import Features2Vector


class SmoteUpSampler(AbstractUpSampler):

    def _check_feature_encoding(self):
        encodings = [
            generators[i].feature_encoding()
            for i in self.conf.get('run.input-mode')
        ]
        if any(e != FeatureEncoding.Numerical for e in encodings):
            raise ValueError('Can only apply SMOTE when using purely numerical features')

    @staticmethod
    def _get_smote(x: str):
        match x:
            case 'default':
                return imblearn.over_sampling.SMOTE()
            case 'kmeans':
                return imblearn.over_sampling.KMeansSMOTE()
            case 'svm':
                return imblearn.over_sampling.SVMSMOTE()
            case 'adasyn':
                return imblearn.over_sampling.ADASYN()
            case 'borderline':
                return imblearn.over_sampling.BorderlineSMOTE()
            case _:
                raise ValueError(f'Unknown SMOTE variant: {x}')

    def upsample(self, indices, targets, labels, keys, *features):
        self._check_feature_encoding()
        smote = self._get_smote(self.hyper_params['smote'])
        transformer = Features2Vector(self.conf.get('run.input-mode'), features)
        transformed = transformer.forward_transform(features)
        sampler = imblearn.combine.SMOTETomek(targets, smote=smote)
        new_transformed, new_labels = sampler.fit_resample(transformed, labels)
        new_features = transformer.backward_transform(new_transformed)
        return new_labels, self.synthetic_keys(len(new_labels)), new_features

    def upsample_class(self, indices, target, labels, keys, *features):
        raise NotImplementedError('upsample_class not used for smote upsampling')

    @staticmethod
    def get_arguments():
        return {
            'smote': EnumArgument(
                name='smote',
                description='Variant of the SMOTE algorithm to use',
                options=['default', 'kmeans', 'svm', 'borderline', 'adasyn'],
                default='default',
            )
        }
