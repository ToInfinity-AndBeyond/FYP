# Imperial DoC GPU Cluster Quick Start

This repo already uses `torch.cuda.is_available()`, so once the job lands on a GPU node and the
correct Python environment is active, the training scripts will move onto CUDA automatically.

## Access route

As of **December 17, 2025**, direct external SSH access to `gpucluster2` and `gpucluster3` is
disabled. Use one of these routes instead:

- On campus: connect from a DoC lab or office machine, or campus WiFi.
- Off campus: use Imperial Unified Access or the approved remote-access path.
- SSH via `shell1` to `shell5` with your public key set up.

From this workspace, the hosts resolve correctly:

- `gpucluster2.doc.ic.ac.uk`
- `gpucluster3.doc.ic.ac.uk`
- `shell1.doc.ic.ac.uk`

The remaining blocker is authentication, not DNS or host reachability.

## Recommended setup

Prepare your Python environment on a lab PC or another allowed machine, not on the GPU head node:

```bash
mkdir -p /vol/bitbucket/${USER}
python3 -m virtualenv /vol/bitbucket/${USER}/myvenv
source /vol/bitbucket/${USER}/myvenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r /homes/${USER}/FYP/Atrial-Fibrillation-ML-main/requirements.txt
```

## SSH examples

If you are using the shell jump hosts, an `~/.ssh/config` entry like this is convenient:

```sshconfig
Host ic-shell
  HostName shell1.doc.ic.ac.uk
  User mc1920

Host ic-gpu-head
  HostName gpucluster2.doc.ic.ac.uk
  User mc1920
  ProxyJump ic-shell
```

Then connect with:

```bash
ssh ic-gpu-head
```

If you are on campus or using the approved remote-access path, you can also connect directly:

```bash
ssh gpucluster2.doc.ic.ac.uk
```

## Sanity check job

The repo includes a lightweight Slurm job that only checks Python, PyTorch, and CUDA visibility.
It does **not** require the training dataset anymore.

```bash
cd /homes/${USER}/FYP/Atrial-Fibrillation-ML-main
PROJECT_DIR=/homes/${USER}/FYP/Atrial-Fibrillation-ML-main \
VENV_PATH=/vol/bitbucket/${USER}/myvenv \
sbatch run_ppg_gpu.slurm
```

Useful follow-up commands:

```bash
squeue --me
less slurm-ppg_gpu_sanity-<jobid>.out
```

Expected output includes lines like:

- `torch_version=...`
- `cuda_available=True`
- `cuda_device_count=1`
- `cuda_device_name=...`

## Interactive GPU session

For debugging or VS Code attach:

```bash
salloc --partition=a30 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --no-shell
squeue --me
ssh ${USER}@<allocated-node>.doc.ic.ac.uk
```

If you want a shell immediately instead of reconnecting later:

```bash
salloc --partition=a30 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00
```

## Training jobs

PPG hybrid training:

```bash
cd /homes/${USER}/FYP/Atrial-Fibrillation-ML-main
PROJECT_DIR=/homes/${USER}/FYP/Atrial-Fibrillation-ML-main \
VENV_PATH=/vol/bitbucket/${USER}/myvenv \
sbatch run_ppg_hybrid_train.slurm
```

Physiology-aware distillation training:

```bash
cd /homes/${USER}/FYP/Atrial-Fibrillation-ML-main
PROJECT_DIR=/homes/${USER}/FYP/Atrial-Fibrillation-ML-main \
VENV_PATH=/vol/bitbucket/${USER}/myvenv \
sbatch run_physio_distill_train.slurm
```

Both scripts now:

- request GPUs with the cluster-guide-compatible form `#SBATCH --gres=gpu:1`
- allow `PROJECT_DIR`, `VENV_PATH`, `SEGMENTS_PATH`, `SUMMARY_PATH`, and `OUTPUT_DIR` overrides
- write Slurm output to `slurm-<job-name>-<jobid>.out` in the submission directory

You can still override the GPU partition at submission time if needed:

```bash
sbatch --partition=t4 run_ppg_gpu.slurm
sbatch --partition=a16 run_ppg_hybrid_train.slurm
```

## Notes

- The main repo code does not need a separate GPU switch; it already selects CUDA when available.
- If a job stays in `PD`, check `squeue --me --start` for the estimated start time.
- If you hit `Permission denied`, the usual fix is to use the approved access route and make sure your
  SSH key or Imperial auth path is configured for `shell1-5` or the campus-access method you are using.
