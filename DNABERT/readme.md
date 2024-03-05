# DNABERT Prediction
This folder contains 2 sub-folders dhs and dhs2 with the DNABERT6 fine tuned pre-trained model using kmers=6. However, we must first clone the DNABERT repo and then add these 2 folders in the cloned DNABERT/examples folder. All data generated for DNABERT training will be saved in these folders.

To use DNABERT we must <b>clone the DNABERT repo</b> and <b><u>set up the environment</u></b> using instructions from the [official DNABERT site](https://github.com/jerryji1993/DNABERT), one level above the current working directory. Hence you will have a folder DNABERT one level above your current working folder, i.e. <b>../DNABERT/</b>

For Problem1, we must next create the subfolder ```/dhs/ft/6/``` in the folder in the folder ```../DNABERT/examples/```. We will next save our datasets for training and testing in the new folder ```../DNABERT/examples/dhs/ft/6/```

For Problem2, we must next create the subfolder ```/dhs2/ft/6/``` in the folder in the folder ```../DNABERT/examples/```. We will next save our datasets for training and testing in the new folder ```../DNABERT/examples/dhs2/ft/6/```

These are already speciified in the notebooks.

If the prepared data are saved in the specified DNABERT folder, then we can run the scripts (from the folder <b>examples</b> in DNABERT, to do the prediction and find motifs if there are any using the following bash:


## Fine-tuning the DNABERT model using our dataset - if we have time.
This takes at least 24 hrs on a zbook with GPU and 8 cores (when tried with the sample data). So I did not do this step due to lack of time. I just used the fined-tuned pretrained DNABERT6 model.
```
export KMER=6
export MODEL_PATH=./dhs/ft/$KMER/dna_model # a copy saved in this folder
export DATA_PATH=dhs/ft/$KMER
export OUTPUT_PATH=./dhs/ft/$KMER/new_model #must create the folder new_model

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 100 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8

```

## Running prediction with the downloaded fine-tuned pretrained DNABERT6 model
```
#For first problem, just uncomment this and comment the lines for Problem 2
#export KMER=6
#export MODEL_PATH=./dhs/ft/$KMER/dna_model
#export DATA_PATH=dhs/ft/$KMER
#export PREDICTION_PATH=./dhs/result/$KMER

# For second problem
export KMER=6
export MODEL_PATH=./dhs2/ft/$KMER/dna_model
export DATA_PATH=dhs2/ft/$KMER
export PREDICTION_PATH=./dhs2/result/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 75 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 48
```

