import os

from .config import Config

##############################################################################
##############################################################################
# Raw text
##############################################################################


def get_raw_text_file_name(conf: Config) -> str:
    return os.path.join(conf.get('system.os.scratch-directory'),
                        f'{conf.get("system.storage.file_prefix")}_raw_words')
