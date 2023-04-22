import os

from .config import Config

##############################################################################
##############################################################################
# Raw text
##############################################################################


def get_raw_text_file_name(conf: Config) -> str:
    if conf.get('system.os.peregrine'):
        path = os.path.join(conf.get('system.os.data-directory'),
                            f'{conf.get("system.storage.file_prefix")}_raw_words')
    else:
        path = f'{conf.get("system.storage.file_prefix")}_raw_words'
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f'{conf.get("system.storage.file_prefix")}_raw_words.json')
    return filename
