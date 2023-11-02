# nlp_bert_-sentiment_analysis

The goal of this homework is to get familiar with how to fine-tune language models for a specific task and understand the challenges involved in it. More specifically, we will first fine-tune a BERT-base model for sentiment analysis using the IMDB dataset.


Next, we will look at one of the assumptions commonly made in supervised learning — we often assume i.i.d. (independent and identically distributed) test distribution i.e. the test data is drawn from the same distribution as the training data (when we create random splits). But this assumption may not always hold in practice e.g. there could be certain features specific to that dataset which won’t work for other examples of the same task. 
 
The main objective in this homework will be creating transformations of the dataset as out-of-distribution (OOD) data, and evaluate your fine-tuned model on this transformed dataset. The aim will be to construct transformations that are “reasonable” (e.g. something we can actually expect at test time) but not very trivial.


Dataset:  https://huggingface.co/datasets/imdb. 


```
ssh wh2405@gw.hpc.nyu.edu
ssh wh2405@greene.hpc.nyu.edu
ssh burst
srun --account=ds_ga_1011-2023fa --partition=n1s8-v100-1 --gres=gpu --time=1:00:00 --pty /bin/bash
cd /scratch/wh2405
singularity exec --bind /scratch --nv --overlay /scratch/wh2405/overlay-25GB-500K.ext3:rw /scratch/wh2405/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash
conda activate 2590-hw3
pip install -r requirements.txt

```

Before proceeding with training on the whole dataset, test your training loop on a small subset of the data to verify its correctness. 
```
#Train on a small subset
Use the following command: python3 main.py --train --eval --debug_train.
```

```
#Train on the whole training set 
Use the following command: python3 main.py --train --eval
```

PartII: 

Design transformations of the evaluation dataset which will serve as out-of-distribution (OOD) evaluation for models. These transformations should be designed so that in most cases, the new transformed example has the same label as the original example — a human would assign the same label to the original and transformed example. e.g. For an original movie review “Titanic is the best movie I have ever seen.”, a transformation which maintains the label “Titanic is the best film I have ever seen.”.


Referencing: https://arxiv.org/pdf/1901.11196.pdf

To ensure that your transformation is working as expected and to see a few examples of the transformed text, use the following command: 

```
#Debug transformation
python3 main.py --eval_transformed --debug_transformation
```

#Evaluation: To assess the performance of the trained model on your transformed test set
```
python3 main.py --eval_transformed
```

Train a model using this augmented data by executing the following command

```
#Train and Evaluation 
python3 main.py --train_augmented --eval_transformed
```


Evaluate the performance of the above-trained model. 

```
#Evaluation on Original Test Data
python3 main.py --eval --model_dir out_augmented

#Evaluation on Transformed Test Data
python3 main.py --eval_transformed --model_dir out_augmented
```
