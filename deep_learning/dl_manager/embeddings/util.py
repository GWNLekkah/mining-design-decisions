import os.path
import shutil

import issue_db_api
from .. import config

def load_embedding(ident, repo: issue_db_api.IssueRepository, conf: config.Config):
    embedding = repo.get_embedding_by_id(ident)
    temp_file = os.path.join(conf.get('system.os.scratch-directory'), 'embedding-file.zip')
    target_dir =  os.path.join(conf.get('system.os.scratch-directory'), ident)
    if os.path.exists(temp_file):
        os.remove(temp_file)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    embedding.download_binary(temp_file)
    shutil.unpack_archive(temp_file, target_dir)
    return os.path.join(target_dir, 'embedding_binary.bin')      # Actual embedding file
