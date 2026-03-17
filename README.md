# CRTypist

## Installing

Build the environment

```
conda env create -f env.yml
conda activate deeptyping
```

Update the requirements

```
conda env update -f env.yml
```

Install PyTorch with CUDA

```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Play with the model

Let *CRTypist* type "hello world" on different keyboards

```
python main.py --demo --phrase "hello world"
```

## Training Workflow

### 1. Pre-training for the internal environment

#### Vision

Train the foveal vision encoder using autoencoder architecture.

```
python main.py --train --vision-encoder --no-cuda --epoch 5
```

Train the peripheral vision encoder using autoencoder architecture.

```
python main.py --train --peripheral-encoder --batch-size 1 --no-cuda --epoch 50
```

Train the vision agent (or continue training)

```
python main.py --train --vision-agent --total-timesteps 500000
python main.py --train --continue-training --vision-agent --total-timesteps 500000
```

Check Tensorboard

```
tensorboard --logdir runs/vision_agent
```

Evaluate the vision agent.

```
python main.py --evaluate --vision-agent --no-cuda
```

#### Finger

Train the finger agent.

```
python main.py --train --finger-agent --total-timesteps 1000000
```

Check Tensorboard

```
tensorboard --logdir runs/finger_agent
```

Continue training

```
python main.py --train --continue-training --finger-agent --total-timesteps 1000000
```

Evaluate the finger agent.

```
python main.py --evaluate --finger-agent
```

#### Working Memory

Train the working memory clf.

```
python main.py --train --memory --epoch 50 --batch-size 64 --no-cuda
```

Check Tensorboard

```
tensorboard --logdir runs/working_memory
```

Evaluate the working memory clf.

```
python main.py --evaluate --memory
```

### 2. Policy optimization

```
python main.py --train --supervisor-agent --total-timesteps 20000000
python main.py --train --continue-training --supervisor-agent --total-timesteps 100000
```

### 3. Parameter Fitting

```
python -W ignore optimization.py
```

### Test

Performance metrics

```
python main.py --evaluate --supervisor-agent
```

Test the model with random sentences

```
python main.py --demo --gboard
```