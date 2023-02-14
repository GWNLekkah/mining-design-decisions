python __main__.py run FullyConnectedModel FullyConnectedModel ^
    -i OntologyFeatures Doc2Vec ^
    -o Detection ^
    --store-model ^
    --target-model-path model-dump ^
    --force-regenerate-data ^
    --ontology-classes dl_manager/feature_generators/util/ontologies.json ^
    -e 3 ^
    -f ../datasets/issuedata/EBSE_issues_formatting-markers.json ^
    -p Doc2Vec.vector-length=10
