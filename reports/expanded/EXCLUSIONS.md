# Expanded experiment: inclusion and exclusion audit

All 80 development participants supplied usable trials. Of 3,600 candidate cues in 240 recordings, 3,417 were accepted and 183 rejected. Every development person appears in exactly one validation fold, with no overlap between that fold's training and validation people.

The intended test group was S081–S109 (29 people). Four supplied no usable trials under the unchanged preprocessing policy:

| Participant | Reason |
|---|---|
| S088 | All three imagery runs have 128 Hz sampling; the fixed pipeline requires 160 Hz. |
| S092 | All three imagery runs have 128 Hz sampling; the fixed pipeline requires 160 Hz. |
| S100 | All three imagery runs have 128 Hz sampling; the fixed pipeline requires 160 Hz. |
| S109 | All 45 cues failed the 500 µV post-filter peak-to-peak threshold at C4; one also exceeded it at F6. This flags large amplitude, not a diagnosed artifact type. |

Among the 78 test recordings that passed metadata validation, there were 1,168 candidate cues: 1,084 accepted and 84 rejected. These counts do not include annotations from the nine unsupported-rate recordings. Twenty-five participants contributed to the test balanced-accuracy mean.

No resampling or special channel repair was added after seeing results. Generalization claims therefore apply only to participants/recordings supported by the fixed pipeline, not all 109 people without qualification. The recording and event-level JSON audits are the source of these counts.
