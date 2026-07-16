# Running TopoBrain on CERN — step by step

**I (Claude) cannot log into CERN for you** — the Kerberos + 2FA login is interactive
and only you can complete it. This is the exact sequence to run yourself. Paste the
output of each step back and I'll tell you what's next.

There are two files here:
- `topo.sub` — the **submit file**: the form you hand the scheduler ("run this, with a GPU, this long").
- `run_job.sh` — the **payload**: what actually runs on the GPU node (resume training to 40k).

Both have placeholders marked `TBD` / `<-- ASK YOUR FRIEND` that we fill in together
once you tell me your CERN layout.

---

## Stage 1 — prove the interactive GPU works (do this FIRST, ~10 min)

You never debug a pipeline through a batch queue. Confirm the basics interactively.

```bash
ssh lxplus-gpu                 # T4 GPU node. enter CERN password + 2FA.
nvidia-smi                     # is a GPU attached? which one?
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
echo $HOME ; fs listquota      # how much AFS space (usually only ~10 GB -- too small for our data)
```

**Paste that output back.** It tells me the GPU, the torch version, and whether we
need EOS for the data (we almost certainly do).

## Stage 2 — get code + data onto CERN

- **Code:** `git clone` the repo into your EOS work dir (I'll give the exact path once
  Stage 1 shows me where you have space).
- **Data:** the normalised `.nii` files currently live on Kaggle. We copy them to CERN.
  ~1.4 GB — options are `scp` from your laptop or a direct download. We decide after Stage 1.

## Stage 3 — one tiny interactive test run

Before batch, run **20 steps** interactively to prove the environment and data paths work:

```bash
python scripts/train_cascaded.py --pairs-csv <data>/pairs_cern.csv \
    --resume <ckpt> --n-iters 20 --batch-size 8 --out-dir /tmp/test
```

If that prints loss numbers without crashing, the pipeline is sound and we scale up.

## Stage 4 — submit the real run to batch

```bash
cd cern
mkdir -p logs
condor_submit topo.sub         # hands the job to the scheduler
condor_q                       # shows your job: Idle -> Running -> (gone = done)
```

Watch it:
```bash
condor_q                       # status
tail -f logs/topo.*.out        # live log once it starts running
```

When `condor_q` shows nothing, the job finished — the outputs are in the `OUT` dir
set inside `run_job.sh`.

---

## What still needs answers from your friend (2 questions)

1. **A100 request line** — "what line in an HTCondor submit file requests an A100 here?"
2. **GPU container image** — "what apptainer/container image should I use for a
   PyTorch GPU job?" (the `IMAGE=` line in `run_job.sh`).

Everything else I can fill in from your Stage 1 output.
