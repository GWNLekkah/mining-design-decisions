import imblearn

from .base import AbstractUpSampler, UpsamplerHyperParam
from ..config import conf
from ..data_utilities import Features2Vector


class SmoteUpSampler(AbstractUpSampler):

    @staticmethod
    def _get_smote(x: str):
        match x:
            case 'default':
                pass
            case 'kmeans':
                pass
            case 'svm':
                pass
            case 'adasyn':
                pass
            case 'borderline':
                pass
            case _:
                raise ValueError(f'Unknown SMOTE variant: {x}')

    def upsample(self, indices, targets, labels, keys, *features):
        transformer = Features2Vector(conf.get('run.input-mode'), features)
        transformed = transformer.forward_transform(features)
        sampler = imblearn.combine.SMOTETomek(targets)
        new_transformed, new_labels = sampler.fit_resample(transformed, labels)
        new_features = transformer.backward_transform(new_transformed)
        return new_labels, self.synthetic_keys(len(new_labels)), new_features

    def upsample_class(self, indices, target, labels, keys, *features):
        raise NotImplementedError('upsample_class not used for smote upsampling')

    @staticmethod
    def get_hyper_params():
        return {
            'smote': UpsamplerHyperParam(
                description='Variant of the SMOTE algorithm to use',
                allowed_values=['default', 'kmeans', 'svm', 'borderline', 'adasyn'],
                default='default',
                data_type='str'
            )
        }
