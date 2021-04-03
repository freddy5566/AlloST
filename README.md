## Official Implementation of AlloST: Low-resource Speech Translation without Source Transcription

### Dependency:
  1. [ESPnet](https://github.com/espnet/espnet): please follow the instruction to install ESPnet
  2. [Allosaurus](https://github.com/xinjli/allosaurus): `pip install allosaurus`

### Results

| Method | fisher-dev | fisher-dev2 | fisher-test |
| ----- | ----------- | ----------- | ----------- |
| Conformer | 20.84 | 21.49 | 21.06 |
| Conformer w/ MTL | 31.21 | **32.59** | 31.47 |
| Phoneme Segment | 21.00 | N/A | 19.8 |
| Phoneme Concat. | **34.50** | N/A | 33.00 |
| Word Embedding | 23.45| N/A | 21.72 |
| Encoder Fusion | **29.52** | 30.01 | 29.01|
| Decoder Fusion | 26.32 | 27.14 | 26.1 |
| Encoder Fusion + Decoder Fusion w/o BPE| 26.52 | 26.92 | 26.12 |
| Encoder Fusion + Decoder Fusion w/ BPE 1k | 28.95 | 30.18 | 29.49 |
| Encoder Fusion + Decoder Fusion w/ BPE 10k | 29.26 | 29.90 | 29.37 |
| Encoder Fusion + Decoder Fusion w/ BPE 16k | 28.89 | 30.18 | 29.73 |
| Encoder Fusion + Decoder Fusion w/ BPE 32k | 29.19 | **31.05** | **30.30** |
| Encoder Fusion + Decoder Fusion w/ BPE 48k | 29.37 | 30.26 | 29.70 |
| Encoder Fusion + Decoder Fusion w/ BPE 64k | 29.44 | 30.73 | 29.12 |