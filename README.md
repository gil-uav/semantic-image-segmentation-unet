# Semantic Image Segmentation Unet
**Members** : <a href="https://github.com/vegovs">Vegard Bergsvik Øvstegård</a>

**Supervisors** : <a href="https://www.mn.uio.no/ifi/personer/vit/jimtoer/">Jim Tørresen</a>

## Description

This repository aims to implement and produce trained networks in semantic image segmentation for
[orthopohots](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/orthophoto).
Current network structure is [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

## Dependencies
* [Python](https://www.python.org/) (version 3.8)
* [Pip](https://virtualenv.pypa.io/en/latest/)
* [conda](https://docs.conda.io/en/latest/)
* [Cuda](https://developer.nvidia.com/cuda-10.2-download-archive) version 10.2

## Installation

```console
git clone https://github.com/gil-uav/semantic-image-segmentation.git
```

#### Conda
```console
conda create --name seg --file spec-file.txt
conda activate seg
pip install kornia
```

## Usage

### Training
The application fetches some configurations and parameters from a .env file if it exists.
Run `python train.py --help` to see all other arguments. The package is using [pytorch-lighting](https://github.com/PyTorchLightning/pytorch-lightning) and inherits all its arguments.

The data is expected to be structured like this:
```
data/
    images/
    masks/
```
The path do data us set using --dp argument.

#### Console example
This example stores the checkpoints and logs under the default_root_dir, uses all available GPUs and
fetches training data from --dp.

```console
python train.py --default_root_dir=/shared/use/this/ --gpus=-1 --dp=/data/is/here/
```

#### .env example
Only these arguments are fetched from .env, the rest must be passed through the CLI.
```
# Model config
N_CHANNELS=3
N_CLASSES=1
BILINEAR=True

# Hyperparameters
EPOCHS=300 # Epochs
BATCH_SIZE=4 # Batch size
LRN_RATE=0.001 # Learning rate
VAL_PERC=15 # Validation percent
TEST_PERC=15 # Testing percent
IMG_SIZE=512  # Image size
VAL_INT_PER=1 # Validation interval percentage
ACC_GRAD=4 # Accumulated gradients, number = K.
GRAD_CLIP=1.0 # Clip gradients with norm above give value
EARLY_STOP=10 # Early stopping patience(Epochs)

# Other
PROD=False # Turn on or off debugging APIs
DIR_DATA="data/" # Where dataset is stored
DIR_ROOT_DIR="/shared/use/this/" # Where logs and checkpoint will be stored
WORKERS=4 # Number of workers for data- and validation loading
DISCORD_WH=httpsomethingwebhoowawnserisalways42
```

### Performance tips:
* Try with different number of workers, but more than 0. A good starting point
is `workers = cores * (threads per core)`.

## Features
### ML:
* Auto learning rate tuner
* Distributed data parallel training
* Early stopping
* ADAM optimizer
* Gradient clipping
* ReduceLROnPlateau learning rate scheduler
* Logging to Tensorboard
* Metrics:
    * Loss
    * F1
    * Precision
    * Recall
    * Visualise images
* Gradient Accumulation(NB! Might conflict with batch-norm!)

### Ease of use:
* Add hyper-parameters and arguments from console
* Load hyper-parameters from .env
* Training finished notification to Discord

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT Licence](https://github.com/gil-uav/semantic-image-segmentation/blob/master/LICENSE)
