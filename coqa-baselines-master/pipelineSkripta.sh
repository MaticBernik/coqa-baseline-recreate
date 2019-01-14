#!/bin/bash


####### START NLP SERVER #########
#java -mx4g -cp lib/stanford-corenlp-3.9.1.jar edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000


####### PREPROCESSING #######
#python3 scripts/gen_pipeline_data.py --data_file data/coqa-train-v1.0.json --output_file1 data/coqa.train.pipeline.json --output_file2 data/seq2seq-train-pipeline
#python3 scripts/gen_pipeline_data.py --data_file data/coqa-dev-v1.0.json --output_file1 data/coqa.dev.pipeline.json --output_file2 data/seq2seq-dev-pipeline
#python3 seq2seq/preprocess.py -train_src data/seq2seq-train-pipeline-src.txt -train_tgt data/seq2seq-train-pipeline-tgt.txt -valid_src data/seq2seq-dev-pipeline-src.txt -valid_tgt data/seq2seq-dev-pipeline-tgt.txt -save_data data/seq2seq-pipeline -lower -dynamic_dict -src_seq_length 10000
#PYTHONPATH=seq2seq python3 seq2seq/tools/embeddings_to_torch.py -emb_file_enc wordvecs/glove.42B.300d.txt -emb_file_dec wordvecs/glove.42B.300d.txt -dict_file data/seq2seq-pipeline.vocab.pt -output_file data/seq2seq-pipeline.embed

####### TRAINING #######
MAX_EPOCHS_N=5
BATCH_SIZE_N=12

python3 rc/main.py --trainset data/coqa.train.pipeline.json --devset data/coqa.dev.pipeline.json --n_history 2 --batch_size $BATCH_SIZE_N --max_epochs $MAX_EPOCHS_N --dir pipeline_models --embed_file wordvecs/glove.840B.300d.txt --predict_raw_text n > Izpis_korak1_training.txt
python3 seq2seq/train.py -data data/seq2seq-pipeline -save_model pipeline_models/seq2seq_copy -copy_attn -reuse_copy_attn -word_vec_size 300 -pre_word_vecs_enc data/seq2seq-pipeline.embed.enc.pt -pre_word_vecs_dec data/seq2seq-pipeline.embed.dec.pt -train_steps $MAX_EPOCHS_N --batch_size $BATCH_SIZE_N -world_size 1 -gpu_ranks 0 -seed 123 > Izpis_korak2_training.txt #-gpuid 0 -epochs 50

####### TESTING ########
#python3 rc/main.py --testset data/coqa.dev.pipeline.json --n_history 2 --pretrained pipeline_models
#python3 scripts/gen_pipeline_for_seq2seq.py --data_file data/coqa.dev.pipeline.json --output_file pipeline_models/pipeline-seq2seq-src.txt --pred_file pipeline_models/predictions.json
#python3 seq2seq/translate.py -model pipeline_models/seq2seq_copy_acc_85.00_ppl_2.18_e16.pt -src pipeline_models/pipeline-seq2seq-src.txt -output pipeline_models/pred.txt -replace_unk -verbose -gpu 0
#python3 scripts/gen_seq2seq_output.py --data_file data/coqa-dev-v1.0.json --pred_file pipeline_models/pred.txt --output_file pipeline_models/pipeline.prediction.json
