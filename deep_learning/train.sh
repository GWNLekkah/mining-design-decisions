python3.10 __main__.py run LinearConv1Model \
  -o Detection \
  -i Word2Vec1D \
  -f ../datasets/issuedata/EBSE_issues_formatting-markers.json \
  --use-early-stopping \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 0.001 \
  --split 0.1 \
  -e 1000 \
  --analyze-keywords \
  --force-regenerate-data \
  -p \
    max-len=400 \
    pretrained-binary=True \
    pretrained-file=pretrained_word2vec_bigdataAll_5_10.bin \
    vector-length=10 \
    use-lemmatization=True \
    min-count=5 \
  -hp \
    number-of-convolutions=3 \
    kernel-1-size=1 \
    kernel-2-size=2 \
    kernel-3-size=3 \
    optimizer=sgd_0.25 \
    fully-connected-layer-size=0 \
    filters=64 \
    loss=hinge
