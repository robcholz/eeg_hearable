# Setup

Run the following commands in the project directory.

## Init

```shell
#!/usr/bin/env bash
set -e

# Install Miniconda
mkdir -p "$HOME/miniconda3"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O "$HOME/miniconda3/miniconda.sh"
bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
rm "$HOME/miniconda3/miniconda.sh"
```

## Run in your terminal

```shell
source "$HOME/miniconda3/bin/activate"
conda init --all
```

## Codex

```shell
# Install NVM + Node.js (LTS)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
nvm install --lts

# Install OpenAI Codex CLI
npm install -g @openai/codex
```

## Conda env Setup

```shell
conda env create -f environment.yml
```