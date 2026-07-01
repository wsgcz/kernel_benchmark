# Run layout: `p*` problem directories

Under each run folder (for example `kb1/` or `kb2/`), problems are named
**`p<id>`** (for example `p01`, `p80`, `p100`).

All paths below are relative to the repository root `lumen-sosp26-ae/`.

A typical problem directory looks like this:

```text
p80/
├── meta.json          ← summary for this problem
├── error.txt          ← sometimes present
└── round0/            ← one attempt; you may also see round1/, round2/, …
    ├── meta.json
    ├── input_model.py
    ├── output_model_new.py
    ├── eval_config.json
    ├── prompt.txt
    └── error.txt      ← sometimes present
```

`roundN` holds the artifacts for that attempt; several rounds mean multiple
tries for the same `p*`.
