# Fine-tuned LLM reranker — running on Kaggle (2× T4)

Trains Qwen3-4B-Instruct-2507 (4-bit QLoRA) to pick the coders' answer from the engine's top-10 candidates.
Produces honest out-of-fold scores (5 folds) + a final adapter. Locally they become ranker features.

## Files to upload
- `kaggle/data/llm_prompts.jsonl` — the training prompts (≈9.6k terms × 10 candidates). **Proprietary — keep the
  Kaggle dataset PRIVATE.**
- `kaggle/llm_rerank_kaggle.py` — the job.

## Steps
1. kaggle.com → **Datasets → New Dataset** → upload both files → title e.g. `hilabs-llm` → visibility **Private** → Create.
2. **Code → New Notebook**. In the right panel:
   - **Accelerator: GPU T4 x2**
   - **Internet: On** (needed to install packages and download the model)
   - **Add Input** → your private dataset `hilabs-llm`
3. First cell — install:
   ```
   !pip install -q -U transformers peft bitsandbytes accelerate
   ```
4. Second cell — smoke test (≈10 min; prints the time per step so we can estimate the full run):
   ```
   !python /kaggle/input/hilabs-llm/llm_rerank_kaggle.py --data /kaggle/input/hilabs-llm/llm_prompts.jsonl --units 0 --limit 200
   ```
   Send me the last ~15 lines it prints.
5. Full run (≈5–7 h). Use **Save Version → Save & Run All (Commit)** so it keeps running in the background and the
   outputs are saved even if you close the browser. Put this as the notebook's main cell:
   ```
   !python /kaggle/input/hilabs-llm/llm_rerank_kaggle.py --data /kaggle/input/hilabs-llm/llm_prompts.jsonl --plan auto
   !tail -5 /kaggle/working/llm_out/worker_gpu0.log /kaggle/working/llm_out/worker_gpu1.log
   ```
6. When the version finishes: open it → **Output** → download `llm_out/` (the five `llm_scores_f*.csv` files and
   `adapter_final.zip`) and put them in `hilabs/kaggle/llm_out/` on the laptop.

Progress lines look like `[fold0] step 120/406 loss 1.21 | 35.2 min elapsed, ~84.0 min left`.
If a session is cut, re-running skips finished folds (each fold's CSV is written as soon as it is done).

## Optional second LLM on Google Colab (runs in parallel, adds model diversity)
colab.research.google.com → New notebook → Runtime → Change runtime type → **T4 GPU**. Upload the two files
(left sidebar → Files → upload), then:
```
!pip install -q -U transformers peft bitsandbytes accelerate
!python llm_rerank_kaggle.py --data llm_prompts.jsonl --out /content/llm_out_phi --model unsloth/Phi-3.5-mini-instruct-bnb-4bit --units 0 --limit 200
```
If the smoke test works:
```
!python llm_rerank_kaggle.py --data llm_prompts.jsonl --out /content/llm_out_phi --model unsloth/Phi-3.5-mini-instruct-bnb-4bit --units 0,1,2,3,4,final
```
(~8–10 h on one T4; keep the tab open — free Colab may disconnect; finished folds are kept in /content/llm_out_phi,
download them periodically.) Put the results in `hilabs/kaggle/llm_out_phi/`.
