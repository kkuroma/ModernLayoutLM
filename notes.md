# Notes

This file contains facts and observations we made while building the project, sorted from newest to oldest.

## Contents

1. [ModernLayoutLM with FA2 reads whole documents at the speed of chunked LayoutLMv1](#modernlayoutlm-with-fa2-reads-whole-documents-at-the-speed-of-chunked-layoutlmv1)
2. [RealKIE fine-tuning performs better at 10 epochs than at 15](#realkie-fine-tuning-performs-better-at-10-epochs-than-at-15)
3. [2048-token uptraining does not improve RealKIE fine-tuning](#2048-token-uptraining-does-not-improve-realkie-fine-tuning)
4. [Zeroed boxes leave the layout checkpoint below plain ModernBERT on every benchmark](#zeroed-boxes-leave-the-layout-checkpoint-below-plain-modernbert-on-every-benchmark)
5. [Pretraining only at 512 tokens erodes the use of context past 2048](#pretraining-only-at-512-tokens-erodes-the-use-of-context-past-2048)
6. [600 steps at 2048 tokens restore the use of context up to 4096](#600-steps-at-2048-tokens-restore-the-use-of-context-up-to-4096)
7. [Kleister-Charity's stacked page boxes break the pretrained model](#kleister-charitys-stacked-page-boxes-break-the-pretrained-model)
8. [Kleister-Charity's spans are recovered by string matching, not annotated](#kleister-charitys-spans-are-recovered-by-string-matching-not-annotated)
9. [11M pages lift FUNSD from 0.7153 to 0.7565 at the same rate](#11m-pages-lift-funsd-from-07153-to-07565-at-the-same-rate)
10. [3e-4 stays the best FUNSD rate on the 11M-page checkpoint](#3e-4-stays-the-best-funsd-rate-on-the-11m-page-checkpoint)
11. [Per-model rate sweeps move FUNSD less than its seed noise](#per-model-rate-sweeps-move-funsd-less-than-its-seed-noise)
12. [FUNSD fine-tuning is not deterministic at a fixed seed on this GPU](#funsd-fine-tuning-is-not-deterministic-at-a-fixed-seed-on-this-gpu)
13. [The fast pretraining stack doubles throughput to 104 seq/s](#the-fast-pretraining-stack-doubles-throughput-to-104-seqs)
14. [The fast stack trains the same model as the old one](#the-fast-stack-trains-the-same-model-as-the-old-one)
15. [Pretraining is power-bound at the 270W cap, so TF32 buys nothing](#pretraining-is-power-bound-at-the-270w-cap-so-tf32-buys-nothing)
16. [Our entity F1 matches LayoutLM v1 and v3 scoring exactly](#our-entity-f1-matches-layoutlm-v1-and-v3-scoring-exactly)
17. [SROIE's published metric is field-level Hmean, not entity F1](#sroies-published-metric-is-field-level-hmean-not-entity-f1)
18. [SROIE's official keys disagree with its tags on a fifth of addresses](#sroies-official-keys-disagree-with-its-tags-on-a-fifth-of-addresses)
19. [Scoring overflow chunks separately penalizes LayoutLM's 512 window](#scoring-overflow-chunks-separately-penalizes-layoutlms-512-window)
20. [Plain ModernBERT beats LayoutLM without boxes on text alone](#plain-modernbert-beats-layoutlm-without-boxes-on-text-alone)
21. [The nielsr/funsd-layoutlmv3 boxes leak every entity boundary](#the-nielsrfunsd-layoutlmv3-boxes-leak-every-entity-boundary)
22. [Word-level FUNSD reproduces LayoutLM's paper score](#word-level-funsd-reproduces-layoutlms-paper-score)
23. [ModernBERT fine-tunes best at 3e-4, not LayoutLM's 5e-5](#modernbert-fine-tunes-best-at-3e-4-not-layoutlms-5e-5)
24. [The best fine-tuning rate differs by variant](#the-best-fine-tuning-rate-differs-by-variant)
25. [bf16 master weights cost about 20 FUNSD F1 while pretraining loss looks fine](#bf16-master-weights-cost-about-20-funsd-f1-while-pretraining-loss-looks-fine)
26. [flash-attn needs bf16 inputs, not bf16 weights](#flash-attn-needs-bf16-inputs-not-bf16-weights)

## ModernLayoutLM with FA2 reads whole documents at the speed of chunked LayoutLMv1

- Summary: With FA2, ModernLayoutLM ran one document of 1024 to 4096 tokens within 13% of LayoutLMv1's time over 512-token windows, and needed half its memory at 8192 tokens.
- Date: 2026-10-05
- Evidence: we timed one document per forward pass in bf16 on the RTX 3090, LayoutLMv1 in windows overlapping by 128 with eager attention and ModernLayoutLM whole with SDPA and with FA2, as the median of three rounds of 30 passes, as follows:

| tokens | LayoutLMv1 windows | LayoutLMv1 ms | SDPA ms | FA2 ms | LayoutLMv1 GiB | SDPA GiB | FA2 GiB |
|---|---|---|---|---|---|---|---|
| 512 | 1 | 5.8 | 7.4 | 7.4 | 0.24 | 0.30 | 0.30 |
| 1024 | 3 | 14.3 | 13.9 | 12.5 | 0.31 | 0.31 | 0.31 |
| 2048 | 6 | 27.3 | 30.7 | 24.8 | 0.41 | 0.35 | 0.33 |
| 4096 | 11 | 48.6 | 76.9 | 52.2 | 0.57 | 0.54 | 0.37 |
| 8192 | 22 | 95.1 | 215.7 | 113.8 | 0.94 | 1.30 | 0.46 |

## RealKIE fine-tuning performs better at 10 epochs than at 15

- Summary: Fine-tuning on RealKIE for 15 epochs instead of 10 lowered mean validation F1 for both ModernLayoutLM and LayoutLMv1.
- Date: 2026-10-05
- Evidence: we compared validation F1 at seed 42 between fine-tuning runs of 10 and 15 epochs as follows:

| task | ModernLayoutLM, 10 | ModernLayoutLM, 15 | LayoutLMv1, 10 | LayoutLMv1, 15 |
|---|---|---|---|---|
| FCC | 0.8761 | 0.8609 | 0.8613 | 0.8309 |
| NDA | 0.8586 | 0.8517 | 0.8539 | 0.8571 |
| Charities | 0.8043 | 0.8082 | 0.8071 | 0.8140 |
| mean | 0.8464 | 0.8403 | 0.8408 | 0.8340 |

## 2048-token uptraining does not improve RealKIE fine-tuning

- Summary: Continuing pretraining on whole documents at 2048 tokens for 28,000 steps left ModernLayoutLM's mean RealKIE F1 slightly lower than before.
- Date: 2026-10-05
- Evidence: we compared validation F1 at seed 42 and 10 epochs between the 11M-page checkpoint and the same checkpoint after 917M more tokens of whole IDL documents in 2048-token windows, as follows:

| task | before | after |
|---|---|---|
| FCC | 0.8761 | 0.8784 |
| NDA | 0.8586 | 0.8542 |
| Charities | 0.8043 | 0.7955 |
| mean | 0.8464 | 0.8427 |

## Zeroed boxes leave the layout checkpoint below plain ModernBERT on every benchmark

- Summary: ModernLayoutLM fine-tuned with every box at the origin scored below ModernBERT fine-tuned on text alone on all six benchmarks.
- Date: 2026-10-05
- Evidence: we compared the two as follows, as three-seed means on FUNSD, CORD and SROIE and at seed 42 on RealKIE:

| benchmark | ModernLayoutLM (no-bbox) | ModernBERT |
|---|---|---|
| FUNSD | 0.3000 | 0.5884 |
| CORD | 0.9312 | 0.9424 |
| SROIE | 0.8411 | 0.8955 |
| RealKIE FCC | 0.7178 | 0.8607 |
| RealKIE NDA | 0.5735 | 0.8571 |
| RealKIE Charities | 0.3740 | 0.7725 |

## Pretraining only at 512 tokens erodes the use of context past 2048

- Summary: After 11M pages at 512 tokens, masked-token loss rose when the visible window grew from 2048 to 4096 tokens, while ModernBERT's kept falling.
- Date: 2026-10-04
- Evidence: we masked the same 15% of tokens in 64 whole IDL documents of 4096 tokens, from a shard no run had read, and compared the loss by visible window as follows:

| model | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| ModernLayoutLM, 11M pages | 0.8361 | 0.7582 | 0.7159 | 0.7354 |
| ModernBERT, text only | 2.2621 | 1.9838 | 1.7708 | 1.6453 |

The same rise from 2048 to 4096 appeared at every saved checkpoint of the 11M-page run.

## 600 steps at 2048 tokens restore the use of context up to 4096

- Summary: 600 steps of continued pretraining on whole documents at 2048 tokens brought masked-token loss at 4096 to 0.6542, against 0.6853 for the same tokens in 512-token windows.
- Date: 2026-10-04
- Evidence: on the same 64 documents and masks, we compared the 11M-page checkpoint before and after 600 steps of 32,768 tokens as follows:

| model | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| 11M pages | 0.8361 | 0.7582 | 0.7159 | 0.7354 |
| plus 600 steps at 512 | 0.8285 | 0.7524 | 0.7004 | 0.6853 |
| plus 600 steps at 2048 | 0.8326 | 0.7525 | 0.6872 | 0.6542 |

## Kleister-Charity's stacked page boxes break the pretrained model

- Summary: Kleister-Charity's processed data put up to 24 pages on one 0-1000 grid, which made 96% of boxes 0 or 1 unit tall and pushed masked-token loss to about 5.6.
- Date: 2026-10-04
- Evidence: we compared the 11M-page checkpoint's masked-token loss on 64 Kleister-Charity documents with the stacked boxes and with each page on its own grid, same tokens, as follows:

| boxes | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| stacked pages | 5.5823 | 5.6526 | 5.7675 | 5.9004 |
| one grid per page | 1.0371 | 0.9445 | 0.8950 | 0.8829 |

## Kleister-Charity's spans are recovered by string matching, not annotated

- Summary: Kleister-Charity labels each document with field=value strings and no positions, so its token tags can only come from searching the OCR text for each value.
- Date: 2026-10-04
- Evidence: our adapter recovered every span by locating the value's string in the document text, while RealKIE, which replaced it, annotates each field as a character span over its own OCR.

## 11M pages lift FUNSD from 0.7153 to 0.7565 at the same rate

- Summary: Continuing layout pretraining from 2.19M to about 11M pages raised ModernLayoutLM's FUNSD F1 by 0.041 with the fine-tuning setup unchanged.
- Date: 2026-10-02
- Evidence: we compared FUNSD test F1 over seeds 42 to 44 at a 3e-4 rate as follows, beside what LayoutLM's paper reports at similar page counts:

| pages | ModernLayoutLM | LayoutLM paper |
|---|---|---|
| 2.19M, 2M for the paper | 0.7153 ± 0.0035 | 0.6957 |
| 11M | 0.7565 ± 0.0037 | 0.7636 |

## 3e-4 stays the best FUNSD rate on the 11M-page checkpoint

- Summary: On the 11M-page checkpoint, neither 1e-4 nor 5e-4 beat the 3e-4 fine-tuning rate on FUNSD.
- Date: 2026-10-02
- Evidence: we compared FUNSD test F1 over seeds 42 to 44 at three rates as follows:

| rate | seed 42 | seed 43 | seed 44 | mean ± std |
|---|---|---|---|---|
| 1e-4 | 0.7631 | 0.7452 | 0.7518 | 0.7534 ± 0.0090 |
| 3e-4 | 0.7499 | 0.7745 | 0.7475 | 0.7573 ± 0.0150 |
| 5e-4 | 0.7261 | 0.7446 | 0.7626 | 0.7444 ± 0.0182 |

## Per-model rate sweeps move FUNSD less than its seed noise

- Summary: Mean FUNSD F1 varies less across fine-tuning rates than single runs vary across seeds at any one rate, so per-model rate sweeps were dropped.
- Date: 2026-10-02
- Evidence: on the 11M-page checkpoint, mean F1 across 1e-4, 3e-4 and 5e-4 spans 0.0129, while the three seeds at a single rate span 0.0179 to 0.0365.

## FUNSD fine-tuning is not deterministic at a fixed seed on this GPU

- Summary: Two FUNSD fine-tuning runs with the same seed, data and setup on the RTX 3090 finished 0.0164 F1 apart.
- Date: 2026-10-02
- Evidence: seed 43 at 3e-4 on the 11M-page checkpoint scored 0.7745 in the rate sweep and 0.7581 in the run that went into the table.

## The fast pretraining stack doubles throughput to 104 seq/s

- Summary: Stacking torch.compile, an MLM loss on masked tokens only, fused AdamW, micro-batch 32 and flex attention raised pretraining throughput from 54.7 to 104.2 sequences per second.
- Date: 2026-09-30
- Evidence: we measured sequences per second over 20 optimizer steps of 64 sequences of 512 tokens, adding one change at a time, as follows:

| setup | seq/s |
|---|---|
| bf16 autocast, flash-attn 2, foreach AdamW, micro-batch 16 | 54.7 |
| plus torch.compile | 74.8 |
| plus MLM loss on masked tokens only | 90.6 |
| plus fused AdamW | 92.0 |
| plus micro-batch 32 | 101.1 |
| plus flex attention in place of flash-attn | 104.2 |

## The fast stack trains the same model as the old one

- Summary: Over 20 identical optimizer steps, the fast stack's weight updates differed from the old stack's by only a little more than two runs of the old stack differ from each other.
- Date: 2026-09-30
- Evidence: we trained 20 steps on the same batches and masks, twice with the old stack and once with the fast one, and compared each against the first old run as follows:

| run | cosine of weight updates | relative L2 difference | mean loss |
|---|---|---|---|
| old stack, first run | reference | reference | 1.0172 |
| old stack, second run | 0.9739 | 0.2285 | 1.0173 |
| fast stack | 0.9605 | 0.2809 | 1.0170 |

## Pretraining is power-bound at the 270W cap, so TF32 buys nothing

- Summary: The RTX 3090 sat at its 270W software power cap throughout pretraining, so enabling TF32 matmuls changed throughput by -1.8%.
- Date: 2026-09-30
- Evidence: board power was 269W median in every configuration measured, the baseline spent 100% of samples at the cap at 99% GPU utilization, and SM clocks ran at 1530 to 1658 MHz against a 2205 MHz maximum.

## Our entity F1 matches LayoutLM v1 and v3 scoring exactly

- Summary: Our entity F1 gives the same number as the scoring code of LayoutLM v1 and LayoutLMv3 on every case tested.
- Date: 2026-08-05
- Evidence: we scored 20 random label sets plus all-empty, partly-empty and perfect cases with both implementations, and every case agreed to 0.00e+00, as both wrap the same seqeval calls, mask -100 and score first subwords only.

## SROIE's published metric is field-level Hmean, not entity F1

- Summary: ICDAR 2019 task 3 scores SROIE on four exact strings per receipt, company, date, address and total, micro-averaged as Hmean, which differs from entity F1 over tagged spans.
- Date: 2026-08-05
- Evidence: we scored the same SROIE test predictions both ways as follows:

| model | entity F1 | field-level Hmean |
|---|---|---|
| ModernLayoutLM, 2.19M pages | 0.8243 | 0.9004 |
| LayoutLMv1 | 0.8362 | 0.9089 |

## SROIE's official keys disagree with its tags on a fifth of addresses

- Summary: The official SROIE key files disagree with the word-level tags on about 21% of addresses, through words absent from the OCR text, so ground truth is decoded from the tags instead.
- Date: 2026-08-05
- Evidence: we compared each field decoded from the tags with the official keys and found these exact-match rates: date 97.7%, company 95.9%, total 94.2%, address 78.9%.

## Scoring overflow chunks separately penalizes LayoutLM's 512 window

- Summary: Scoring each overflow chunk on its own penalized LayoutLMv1, whose 512-token window cuts receipts into more chunks, and stitching the chunks back reversed the SROIE ranking.
- Date: 2026-08-05
- Evidence: we compared SROIE test entity F1 with chunks of stride 128 scored separately and stitched back into one receipt, as follows:

| model | chunks scored separately | chunks stitched |
|---|---|---|
| ModernLayoutLM, 1024-token window | 0.8390 | 0.8243 |
| LayoutLMv1, 512-token window | 0.8222 | 0.8362 |

## Plain ModernBERT beats LayoutLM without boxes on text alone

- Summary: Without boxes, plain ModernBERT reads FUNSD better than LayoutLMv1 does.
- Date: 2026-08-05
- Evidence: we fine-tuned both on word-level FUNSD with the layout input removed and compared test F1: ModernBERT 0.5585, LayoutLMv1 0.4814.

## The `nielsr/funsd-layoutlmv3` boxes leak every entity boundary

- Summary: The `nielsr/funsd-layoutlmv3` mirror ships one box per text segment, and every repeated box covers exactly one entity span, which tells the model where each entity starts and ends.
- Date: 2026-08-04
- Evidence: on the test split the mirror has 3.84 words per box with 90.4% of words sharing a box, and all 1437 repeated boxes coincide with one entity span each, while the official release has 1.00 words per box over the identical text and 1998 entities.

## Word-level FUNSD reproduces LayoutLM's paper score

- Summary: Rebuilt with one box per word, FUNSD gives LayoutLMv1 0.7798 F1 against the 0.7866 its paper reports.
- Date: 2026-08-04
- Evidence: we compared test F1 on the segment-level mirror and on word-level boxes as follows, with ModernLayoutLM at 2.19M pages:

| model | segment-level boxes | word-level boxes |
|---|---|---|
| LayoutLMv1 | 0.8616 | 0.7798 |
| ModernLayoutLM | 0.7610 | 0.6674 |
| ModernLayoutLM (no-pretrain) | 0.5138 | 0.4278 |
| ModernLayoutLM (no-bbox) | 0.3121 | 0.2581 |

## ModernBERT fine-tunes best at 3e-4, not LayoutLM's 5e-5

- Summary: Raising the fine-tuning rate from the 5e-5 of LayoutLM's BERT recipe to 3e-4 added 0.051 F1 on FUNSD and also helped CORD and SROIE.
- Date: 2026-08-04
- Evidence: we swept the backbone rate for ModernLayoutLM on word-level FUNSD at 2.19M pages as follows:

| rate | 3e-5 | 5e-5 | 1e-4 | 2e-4 | 3e-4 | 5e-4 | 8e-4 |
|---|---|---|---|---|---|---|---|
| FUNSD F1 | 0.5990 | 0.6674 | 0.6743 | 0.6891 | 0.7188 | 0.7206 | 0.6939 |

Moving from 5e-5 to 3e-4 also took CORD from 0.9418 to 0.9535 and SROIE from 0.8372 to 0.8390.

## The best fine-tuning rate differs by variant

- Summary: On FUNSD, ModernLayoutLM peaked at 5e-4 while the no-bbox variant peaked at 1e-4, so one shared rate favors whichever variant it suits.
- Date: 2026-08-04
- Evidence: in the same rate sweep the no-bbox variant fell to 0.2077 F1 at 5e-4, which is also why LayoutLMv1 keeps the 5e-5 of its own paper.

## bf16 master weights cost about 20 FUNSD F1 while pretraining loss looks fine

- Summary: Pretraining with the weights stored in bf16 instead of fp32 cost 17 to 21 FUNSD F1 points, while pretraining eval loss and the saved config, which still read float32, gave no sign of it.
- Date: 2026-08-03
- Evidence: we ran 13 short pretraining runs of 1100 steps on 25k pages per corpus, each changing one setting, and compared FUNSD F1 on the segment-level boxes of the time as follows:

| master weights | runs | FUNSD F1 |
|---|---|---|
| fp32 | 5 | 0.5148 to 0.5443 |
| none, no pretraining | 1 | 0.4976 |
| bf16 | 7 | 0.2879 to 0.3269 |

After the same steps, the bf16 weights had moved about 8x less from the backbone, a median absolute change of 9.5e-5 against 7.5e-4.

## flash-attn needs bf16 inputs, not bf16 weights

- Summary: flash-attn only requires bf16 activations, which bf16 autocast already produces from fp32 weights, so loading the weights in bf16 for it was unnecessary.
- Date: 2026-08-03
- Evidence: the bf16 weights came from passing `torch_dtype=torch.bfloat16` to `from_pretrained` whenever flash-attn was on, and with fp32 weights under autocast flash-attn ran forward and backward with finite gradients and gave the best FUNSD F1 of the 13 runs, 0.5443.
