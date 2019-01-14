Clone repozitorija z naslova https://github.com/stanfordnlp/coqa-baselines

# Fasttext

Uporaba (po nastavitvi stanford repozitorija)


## Obvezne verzije paketov
```
pip install torch==0.4.0
pip install torchtext==0.2.1
pip install gensim==3.6.0
pip install pycorenlp==0.3.0
```

## Prenosi
Fasttext word embedding:
```bash
  wget -P wordvecs https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
```

Zaradi preprostosti uporabe datoteki zgolj preimenujemo:
glove.42B.300d.txt -> glove.42B.300d.txt.old
wiki.en.vec -> glove.42B.300d.txt 

Po potrebi prilagodimo parametre (primerno za 6GB graficno kartico).
Potrebujemo tudi 50GB RAM (ali RAM + SWAP).

1.)
rc/main.py ---> set:  batch_size = 25,  max_epochs = 30


2.)
v rc\word_model.py
nadomestimo
```python
for line in input_file.readlines():
	splitLine = line.split(' ')
```
z
```python
  for line in input_file.readlines()[1:]:
        splitLine = line.rstrip().split(' ')
```

pri predprocesiranju uporabimo dodaten flag: -type word2vec
    
## Zazenemo skripto 

```bash
  java -mx4g -cp lib/stanford-corenlp-3.9.1.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## Fasttext pipeline model
### Predprocesiranje
```bash
  python scripts/gen_pipeline_data.py --data_file data/coqa-train-v1.0.json --output_file1 data/coqa.train.pipeline.json --output_file2 data/seq2seq-train-pipeline
  python scripts/gen_pipeline_data.py --data_file data/coqa-dev-v1.0.json --output_file1 data/coqa.dev.pipeline.json --output_file2 data/seq2seq-dev-pipeline
  python seq2seq/preprocess.py -train_src data/seq2seq-train-pipeline-src.txt -train_tgt data/seq2seq-train-pipeline-tgt.txt -valid_src data/seq2seq-dev-pipeline-src.txt -valid_tgt data/seq2seq-dev-pipeline-tgt.txt -save_data data/seq2seq-pipeline -lower -dynamic_dict -src_seq_length 10000
  PYTHONPATH=seq2seq python seq2seq/tools/embeddings_to_torch.py -emb_file_enc wordvecs/glove.42B.300d.txt -emb_file_dec wordvecs/glove.42B.300d.txt -dict_file data/seq2seq-pipeline.vocab.pt -output_file data/seq2seq-pipeline.embed -type word2vec
```

### Ucenje

```bash
  python rc/main.py --trainset data/coqa.train.pipeline.json --devset data/coqa.dev.pipeline.json --n_history 2 --dir pipeline_models --embed_file wordvecs/glove.42B.300d.txt --predict_raw_text n
  python seq2seq/train.py -data data/seq2seq-pipeline -save_model pipeline_models/seq2seq_copy -copy_attn -reuse_copy_attn -word_vec_size 300 -pre_word_vecs_enc data/seq2seq-pipeline.embed.enc.pt -pre_word_vecs_dec data/seq2seq-pipeline.embed.dec.pt -epochs 30 -gpuid 0 -seed 123
```

### Testiranje
```bash
  python rc/main.py --testset data/coqa.dev.pipeline.json --n_history 2 --pretrained pipeline_models
  python scripts/gen_pipeline_for_seq2seq.py --data_file data/coqa.dev.pipeline.json --output_file pipeline_models/pipeline-seq2seq-src.txt --pred_file pipeline_models/predictions.json
  python seq2seq/translate.py -model pipeline_models/seq2seq_copy_acc_84.77_ppl_2.21_e30.pt -src pipeline_models/pipeline-seq2seq-src.txt -output pipeline_models/pred.txt -replace_unk -verbose -gpu 0
  python scripts/gen_seq2seq_output.py --data_file data/coqa-dev-v1.0.json --pred_file pipeline_models/pred.txt --output_file pipeline_models/pipeline.prediction.json
```

Cas predprocesiranja in ucenja na GTX 1060 6 GB ~ 20h

## Rezultati

All the results are based on `n_history = 2`:

| Model  | Dev F1 | Dev EM |
| ------------- | ------------- | ------------- |
| seq2seq | 20.9 | 17.7 |
| seq2seq_copy  | 45.2  | 38.0 |
| DrQA | 55.6 | 46.2 |
| pipeline | 65.0 | 54.9 |
| ------------- | ------------- | ------------- |
| pipeline | 49.49 | 38.69 |

