import issue_db_api
import requests


def update_embedding(repo: issue_db_api.IssueRepository, embedding_id, *, all_data):
    embedding = repo.get_embedding_by_id(embedding_id)
    config = embedding.config.copy()
    if all_data:
        config['training-data-query'] = {
            '$or': [
                {'tags': {'$eq': 'Apache-TAJO'}},
                {'tags': {'$eq': 'Apache-YARN'}},
                {'tags': {'$eq': 'Apache-HDFS'}},
                {'tags': {'$eq': 'Apache-HADOOP'}},
                {'tags': {'$eq': 'Apache-MAPREDUCE'}},
                {'tags': {'$eq': 'Apache-CASSANDRA'}},
                {'tags': {'$eq': 'Apache-SOLR'}},
                {'tags': {'$eq': 'Apache-JSPWIKI'}},
                {'tags': {'$eq': 'Apache-TOMEE'}},
                {'tags': {'$eq': 'Apache-BROOKLYN'}},
                {'tags': {'$eq': 'Apache-CLOUDSTACK'}},
            ]
        }
    else:
        config['training-data-query'] = {'tags': {'$eq': 'has-label'}}
    match embedding.name:
        case 'Internship Dictionary':
            name = 'Dictionary Data Science + Web (min-count = 0)'
        case 'Internship TFIDF':
            name = 'IDF Data Science + Web (min-count = 0)'
        case 'Internship Doc2Vec (25)':
            name = 'Doc2Vec (25) Data Science + Web'
        case 'Internship Doc2Vec (100)':
            name = 'Doc2Vec (100) Data Science + Web'
        case 'Internship Word2Vec (10)':
            name = 'Word2Vec (10) Data Science + Web'
        case 'Internship Word2Vec (25)':
            name = 'Word2Vec (25) Data Science + Web'
        case 'Internship Word2Vec (300)':
            name = 'Word2Vec (300) Data Science + Web'
        case _ as x:
            raise ValueError(x)
    print('New Embedding:', name)
    #return ''
    return repo.create_embedding(name, config)

def compute_updated_config(repo: issue_db_api.IssueRepository,
                           model: issue_db_api.Model,
                           new_embeddings):
    config = model.config.copy()
    # Update query
    config['training-data-query'] = {'tags': {'$eq': 'has-label'}}
    # Update batch size
    config['batch-size'] = 10_000
    # Update embedding if necessary
    print('>', model.name)
    for key, value in config['params'].items():
        if 'dictionary-id' in value:
            if 'BOWNormalized' in key or 'BOWFrequency' in key:
                embedding_id = value['dictionary-id']
                if embedding_id not in new_embeddings:
                    new_embeddings[embedding_id] = update_embedding(repo, embedding_id, all_data=False)
                config['params'][key]['dictionary-id'] = new_embeddings[embedding_id]
            elif 'TfidfGenerator' in key:
                embedding_id = value['dictionary-id']
                if embedding_id not in new_embeddings:
                    new_embeddings[embedding_id] = update_embedding(repo, embedding_id, all_data=False)
                config['params'][key]['dictionary-id'] = new_embeddings[embedding_id]
            else:
                raise ValueError(key)
        elif 'embedding-id' in value:
            if 'Word2Vec' in key or 'Word2Vec' in key:
                embedding_id = value['embedding-id']
                if embedding_id not in new_embeddings:
                    new_embeddings[embedding_id] = update_embedding(repo, embedding_id, all_data=True)
                config['params'][key]['embedding-id'] = new_embeddings[embedding_id]
            elif 'Doc2Vec' in key:
                embedding_id = value['embedding-id']
                if embedding_id not in new_embeddings:
                    new_embeddings[embedding_id] = update_embedding(repo, embedding_id, all_data=True)
                config['params'][key]['embedding-id'] = new_embeddings[embedding_id]
            else:
                raise ValueError(key)
        else:
            raise ValueError(value)
    return config


def main():
    repo = issue_db_api.IssueRepository(
        'https://issues-db.nl:8000',
        credentials=('jesse', 'GevlektePiemelodus')
    )
    identifiers = [
        '646b86563613911624e8c5f9',     # BOWFreq Detection
        '646b86873613911624e8c5fa',     # BOWNorm Detection
        '646b86c03613911624e8c5fb',     # TFIDF Detection
        '646b86fa3613911624e8c5fc',     # Doc2Vec Detection
        '646b91d73613911624e8c5fe',     # BOWFreq Classification
        '646b92273613911624e8c5ff',     # BOWnorm Classification
        '646b92773613911624e8c600',     # Doc2Vec Classification
        '646b93163613911624e8c601',     # TFIDF Classification
        '646b9d0a3613911624e8c603',     # CNN Classification
        '646b9ecb3613911624e8c604',     # CNN Detection
        '646b93d93613911624e8c602',     # RNN Classification
        '646b87993613911624e8c5fd',     #  RNN Detection
        '646c80f03613911624e8cdaf',     # combination ensemble,
        '646c80f03613911624e8cdb0',     # voting ensemble
        '646c80f03613911624e8cdb1',     # stacking ensemble
    ]
    # new_embeddings = {}
    # new_ids = []
    # for ident in identifiers:
    #     old_model = repo.get_model_by_id(ident)
    #     new_config = compute_updated_config(repo, old_model, new_embeddings)
    #     print(f'New model: ', old_model.name + ' - New Dataset')
    #     new_ids.append(
    #        repo.add_model(old_model.name + ' - New Dataset', new_config)
    #     )
    # print(new_ids)
    for ident in identifiers:
        model = repo.get_model_by_id(ident)
        response = requests.post(
            'https://192.168.2.255:9011/metrics',
            json={
                'auth': {
                    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6Implc3NlIiwiZXhwIjoxNjg0OTM0Mzg5fQ.gAWAzSHaHWJ8i-knKlAE7BuBrnKAFjal0b8r_sMKexU'
                },
                'config': {
                    'database-url': 'https://issues-db.nl:8000',
                    'model-id': ident,
                    'version-id': model.test_runs[-1].run_id,
                    'epoch': 'stopping-point',
                    'metrics': [
                        {
                            'dataset': 'testing',
                            'metric': 'f1_score',
                            'variant': 'macro'
                        },
                        {
                            'dataset': 'testing',
                            'metric': 'precision',
                            'variant': 'macro'
                        },
                        {
                            'dataset': 'testing',
                            'metric': 'recall',
                            'variant': 'macro'
                        }
                    ]
                }
            }
        )
        response.raise_for_status()
        metrics = response.json()
        prec = metrics['testing']['precision[macro]'][-1]
        recall = metrics['testing']['recall[macro]'][-1]
        f1 = metrics['testing']['f1_score[macro]'][-1]
        print(model.name, ' -- precision / recall / F1 --', f'{prec:.4f} / {recall:.4f} / {f1:.4f}')

if __name__ == '__main__':
    main()