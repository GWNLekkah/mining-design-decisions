import json
import os

from .config import conf

##############################################################################
##############################################################################
# Raw text
##############################################################################


def get_raw_text_file_name() -> str:
    if conf.get('system.peregrine'):
        path = os.path.join(conf.get('system.peregrine.data'),
                            f'{conf.get("system.storage.file_prefix")}_raw_words')
    else:
        path = f'{conf.get("system.storage.file_prefix")}_raw_words'
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f'{conf.get("system.storage.file_prefix")}_raw_words.json')
    return filename
