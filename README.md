# Bachelor Thesis

## Requirements

MATLab (version R2025b) - This is for bridging our python code with the RS analysis written in MATLAB and located in the re_analysis subrepo. This is a fork from @attribution

Python 3.12. At the time of writting this, any python version later than 3.12 had missing wheels for packeges requiered by pytorch, and thus 3.12 was used.

## Setup

Start by cloning a repo, fetching the the subrepos, and creating a venv with python 3.12. 
This can be done with the following commads:

windowns:
```powershell
git clone https://github.com/Gullbra/ThesisWorkBachelor.git
cd ThesisWorkBachelor
git submodule update --init --recursive
./install.ps1
```

Linux/mac:
```bash
git clone https://github.com/Gullbra/ThesisWorkBachelor
cd ThesisWorkBachelor
git submodule update --init --recursive
chmod +x install.sh
./install.sh
```

This creates a local python environment with python 3.12, and tries to install all required packages into it.



## Command-Line Usage

The project is controlled through `main.py` using two subcommands:

* `stego` – Generate stego images from the cover image dataset.
* `analysis` – Run steganalysis on a previously generated stego dataset.

### General Syntax

```bash
python main.py [--dataset-path PATH] <command> [options]
```

### Global Arguments

| Argument                    | Description                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `-d`, `--dataset-path PATH` | Path to the dataset root directory. Defaults to the project's configured dataset path. |

---

### Generate Stego Images

Create stego images using one of the available embedding methods.

```bash
python main.py stego <stego_method> [--threshold VALUE]
```

#### Arguments

| Argument            | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| `stego_method`      | Embedding algorithm to use. One of: `sequential`, `random`, `matching`, `adaptive`. |
| `-t`, `--threshold` | Embedding threshold between `0.0` and `1.0`. Default: `0.25`.                       |

#### Examples

Generate images using sequential LSB embedding:

```bash
python main.py stego sequential
```

Generate images using adaptive embedding with a threshold of 0.40:

```bash
python main.py stego adaptive --threshold 0.40
```

Generate images from a custom dataset location:

```bash
python main.py --dataset-path ./images/BOSSbase stego random
```

#### Output Structure

Generated images are stored in method-specific directories:

```text
train/
├── cover/
├── stego_sequential/
├── stego_random/
├── stego_matching/
└── stego_adaptive/

test/
├── cover/
└── stego_sequential/

val/
├── cover/
└── stego_sequential/
```

---

### Run Analysis

Run a steganalysis method against a previously generated stego dataset.

```bash
python main.py analysis <analysis_method> <stego_method>
```

#### Arguments

| Argument          | Description                                                                             |
| ----------------- | --------------------------------------------------------------------------------------- |
| `analysis_method` | Analysis method to use. Currently: `cnn`, `rs`.                                         |
| `stego_method`    | Which stego dataset to analyze. One of: `sequential`, `random`, `matching`, `adaptive`. |

#### Examples

Run CNN analysis on images generated using sequential embedding:

```bash
python main.py analysis cnn sequential
```

Run RS analysis on images generated using adaptive embedding:

```bash
python main.py analysis rs adaptive
```

### Available Steganography Methods

| Method       | Description                                         |
| ------------ | --------------------------------------------------- |
| `sequential` | Sequential LSB embedding.                           |
| `random`     | Pseudo-randomized LSB embedding.                    |
| `matching`   | LSB matching embedding.                             |
| `adaptive`   | Edge-adaptive embedding using Sobel edge detection. |

### Available Analysis Methods

| Method | Description                     |
| ------ | ------------------------------- |
| `cnn`  | CNN-based steganalysis (SRNet). |
| `rs`   | RS steganalysis.                |

