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
git submodule update --init --recursive
./install.ps1
```

Linux/mac:
```bash
git clone https://github.com/Gullbra/ThesisWorkBachelor
git submodule update --init --recursive
chmod +x install.sh
./install.sh
```

This creates a local python environment with python 3.12, and tries to install all required packages into it.

