# Manifest contract

Offline scoring consumes JSONL with one row per candidate waveform.

Required fields:

```json
{
  "candidate_id": "ikala_song001_clean",
  "source_id": "ikala_song001",
  "split": "test",
  "vocal_path": "/absolute/path/vocal.wav",
  "accompaniment_path": "/absolute/path/accompaniment.wav",
  "variant": "clean"
}
```

Recommended provenance:

```json
{
  "dataset": "iKala",
  "pair_id": "ikala_song001_offset_160",
  "pair_side": "A",
  "pair_label": 1,
  "perturbation_family": "constant_offset",
  "target_dimension": "beat",
  "pair_kind": "directional_preference",
  "severity": 0.160,
  "parent_candidate_id": null,
  "audio_sha256": "...",
  "transform_config_sha256": "..."
}
```

Rules:

- paths are absolute inside one run receipt but are never committed;
- `source_id` is the split/bootstrap cluster;
- all rows from one source remain in one split;
- `pair_label` is A-preferred `+1`, ambiguous `0`, or B-preferred `-1`;
- `target_dimension` identifies beat, coverage, guardrail, or nuisance scope;
- primary accuracy is computed only when the reward and target dimension match;
- `pair_kind` distinguishes directional preferences from invariance tests;
- scorer output preserves unknown metadata fields;
- reward scores are never stored in ground-truth fields;
- separated Muse rows include separator ID and checkpoint hash.
