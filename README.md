# nlp_bert_-sentiment_analysis


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

