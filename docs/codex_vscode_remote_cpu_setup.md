# VS Code Remote + Codex CPU Server Setup

Generated: 2026-08-11 KST  
Project: `/home/ted1204/MS_datamaker`

This note captures the current server-side VS Code/Codex setup so the same workflow can be reproduced from a notebook using VS Code Remote SSH and Codex.

## Current Remote Environment

### Remote Host

```text
Host nickname in this session: master
Project root: /home/ted1204/MS_datamaker
Likely SSH target used by VS Code: ted1204@147.46.137.69
```

The exact SSH alias on the notebook can be chosen freely. The important part is that VS Code opens this remote folder:

```text
/home/ted1204/MS_datamaker
```

### VS Code Server

```text
VS Code version: 1.98.2
VS Code commit: ddc367ed5c8936efe395cffeec279b04ffd7db78
Architecture: x64
Remote code binary: /home/ted1204/.vscode-server/cli/servers/Stable-ddc367ed5c8936efe395cffeec279b04ffd7db78/server/bin/remote-cli/code
```

Installed VS Code extension on the remote server:

```text
openai.chatgpt@26.803.61601
```

Codex binary currently comes from the VS Code ChatGPT/Codex extension:

```text
/home/ted1204/.vscode-server/extensions/openai.chatgpt-26.727.40816-linux-x64/bin/linux-x86_64/codex
```

Codex CLI version observed:

```text
codex-cli 0.146.0-alpha.9.2
```

### Python / Conda

Base Python observed:

```text
Python 3.9.7
```

Available conda environments:

```text
base                         /home/anaconda3
CNN_Transformer_Datamaker    /home/ted1204/.conda/envs/CNN_Transformer_Datamaker
MS_datamaker                 /home/ted1204/.conda/envs/MS_datamaker
[hsenv]                      /home/ted1204/.conda/envs/[hsenv]
codex                        /home/ted1204/.conda/envs/codex
hsenv                        /home/ted1204/.conda/envs/hsenv
```

CPU environment file in this repo:

```yaml
name: ms_datamaker
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - matplotlib
  - scikit-learn
  - pyyaml
  - tqdm
  - pip
```

Path:

```text
/home/ted1204/MS_datamaker/env/environment_cpu.yml
```

## Codex Config Snapshot

Current non-secret Codex config at `/home/ted1204/.codex/config.toml`:

```toml
[projects."/home/ted1204"]
trust_level = "trusted"

[projects."/home/ted1204/CNN_Transformer_Datamaker"]
trust_level = "trusted"

[projects."/home/ted1204/MS_datamaker"]
trust_level = "trusted"

[projects."/home/ted1204/Diffusion_datamaker"]
trust_level = "trusted"

[plugins."github@openai-curated"]
enabled = true

[tui.model_availability_nux]
"gpt-5.5" = 4
```

Do **not** copy these files into another machine manually:

```text
/home/ted1204/.codex/auth.json
/home/ted1204/.codex/cache/*
/home/ted1204/.codex/attachments/*
```

`auth.json` is credential material. On a new notebook or remote host, run Codex login again instead of copying it.

## Notebook Setup Steps

### 1. Install VS Code Locally

Install the same or newer VS Code on the notebook. Current remote server was created by:

```text
VS Code 1.98.2
```

Exact match is not strictly required, but using the latest stable VS Code is usually fine.

### 2. Install Local VS Code Extensions

Install these on the notebook VS Code:

```text
Remote - SSH
ChatGPT / Codex extension: openai.chatgpt
```

The remote currently has:

```text
openai.chatgpt@26.803.61601
```

If VS Code prompts to install the extension on the remote server, allow it.

### 3. Add SSH Config On Notebook

Example `~/.ssh/config` on the notebook:

```sshconfig
Host ms-cpu
    HostName 147.46.137.69
    User ted1204
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

Then connect from VS Code:

```text
Remote-SSH: Connect to Host... -> ms-cpu
Open Folder -> /home/ted1204/MS_datamaker
```

### 4. Verify Remote Shell

Inside the VS Code remote terminal:

```bash
whoami
hostname
pwd
code --version
codex --version
conda env list
```

Expected project folder:

```text
/home/ted1204/MS_datamaker
```

### 5. Activate Project Environment

Recommended working environment:

```bash
conda activate MS_datamaker
export PYTHONPATH=/home/ted1204/MS_datamaker/src:${PYTHONPATH:-}
```

If the environment needs to be recreated on another CPU server:

```bash
cd /home/ted1204/MS_datamaker
conda env create -f env/environment_cpu.yml
conda activate ms_datamaker
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

Note: the existing server environment is named `MS_datamaker`, while the YAML creates `ms_datamaker`. That naming difference is cosmetic but worth remembering.

### 6. Login To Codex On The Remote

Run this from the remote terminal, not the notebook local terminal:

```bash
codex
```

Then complete login with the same ChatGPT/OpenAI account. This creates fresh credentials under the remote user's `~/.codex`.

Do not copy `~/.codex/auth.json` between machines.

### 7. Trust Project In Codex

The current project is trusted in Codex:

```text
/home/ted1204/MS_datamaker -> trusted
```

If Codex asks whether to trust the folder, choose trust for this project.

## Current Project Paths To Open

Main repo:

```text
/home/ted1204/MS_datamaker
```

Important subdirectories:

```text
src/                         Python source modules
scripts/                     CPU / Slurm scripts
scripts/cpu/sbatch/final/    Final dataset batch scripts
env/                         Conda environment files
outputs/                     Segment plans and intermediate outputs
output_npy/                  Generated NPY datasets and figures
MS_datamaker_PH/             Pohang source processing
MS_datamaker_UTAH_2019/      Utah 2019 source processing
MS_datamaker_UTAH_2023/      Utah 2023 source processing
```

## Useful Commands

Check dataset job status:

```bash
squeue -u ted1204
```

Follow a Slurm log:

```bash
tail -f logs/<job_log>.out
```

Run project Python modules:

```bash
cd /home/ted1204/MS_datamaker
conda activate MS_datamaker
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
python -m datamaker.final.check_raw_sampling_rates --help
```

Check current data size:

```bash
du -sh output_npy/final/*
df -h /home
```

## Safety Rules For Remote Codex Work

Use these instructions when starting a Codex session from the notebook:

```text
Work only inside /home/ted1204/MS_datamaker unless I explicitly say otherwise.
Do not delete original raw datasets.
Do not run sudo.
Before launching long Slurm jobs, show me the exact command.
For destructive commands such as rm -rf, ask first.
Keep generated datasets under output_npy/final.
Prefer reading metadata before modifying split files.
```

## Minimal Reconnection Workflow

From the notebook:

1. Open VS Code.
2. Connect via Remote SSH to `ms-cpu`.
3. Open `/home/ted1204/MS_datamaker`.
4. Open terminal.
5. Run:

```bash
conda activate MS_datamaker
export PYTHONPATH=/home/ted1204/MS_datamaker/src:${PYTHONPATH:-}
codex
```

Then use Codex normally from the VS Code/Codex panel or terminal session.

## Notes

- VS Code Remote SSH means code and data stay on the CPU server.
- The notebook acts mainly as a UI client.
- Codex commands execute on the remote server when launched inside the remote VS Code session.
- Credentials should be created by login, not copied.
- If the ChatGPT/Codex extension is missing on the remote, install/sync the `openai.chatgpt` extension from VS Code.
