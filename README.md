# LittleLM

LittleLM is a self-made project for building a Language Model (LM). This project includes data preparation, model training, and evaluation scripts.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Appreciation](#appreciation)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/minhaj1403/LittleLM.git
    cd LittleLM
    ```

2. Create a Conda environment with Python 3.12.1 and activate it:
    ```sh
    conda create -n littlelm python=3.12.1 -y
    conda activate littlelm
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Configuration

The configuration for the model and training parameters is done through command-line arguments. You can see all available options by running:
```sh
python main.py --help
```

## Appreciation

This development is largely inspired from Elliot Arledge's workthrough and special shout-out to this guy: https://x.com/elliotarledge
