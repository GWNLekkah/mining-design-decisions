import imblearn

from .base import AbstractUpSampler, UpsamplerHyperParam


class SmoteUpSampler(AbstractUpSampler):

    def upsample(self, indices, targets, labels, keys, *features):
        raise NotImplementedError('SMOTE upsampler currently not implemented')
        sampler = imblearn.combine.SMOTETomek(targets)

    def upsample_class(self, indices, target, labels, keys, *features):
        raise NotImplementedError('upsample_class not used for smote upsampling')

    @staticmethod
    def get_hyper_params():
        return {
            'smote': UpsamplerHyperParam(
                description='Variant of the SMOTE algorithm to use',
                allowed_values=['default', 'kmeans', 'svm'],
                default='default',
                data_type='str'
            )
        }
