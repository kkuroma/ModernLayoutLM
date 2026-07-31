# Results

## cord

| variant | f1 | precision | recall | token_accuracy | loss | what it is |
|---|---|---|---|---|---|---|
| layout | 0.9454 | 0.9393 | 0.9516 | 0.9597 | 0.2582 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.9661 | 0.9625 | 0.9696 | 0.9721 | 0.2016 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.9362 | 0.9282 | 0.9442 | 0.9543 | 0.2622 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.9164 | 0.9029 | 0.9303 | 0.9478 | 0.2484 | ours, boxes withheld at fine-tune time |

- plain ModernBERT: not run
- does not beat layoutlm-base-uncased by -0.0207 f1
- beats the same model without the layout-MLM stage by +0.0092 f1

## funsd

| variant | f1 | precision | recall | token_accuracy | loss | what it is |
|---|---|---|---|---|---|---|
| layout | 0.6933 | 0.6575 | 0.7332 | 0.6605 | 2.1278 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.8616 | 0.8504 | 0.8731 | 0.7907 | 1.1751 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.5138 | 0.4634 | 0.5766 | 0.6319 | 2.3650 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.2935 | 0.2801 | 0.3083 | 0.4927 | 1.3725 | ours, boxes withheld at fine-tune time |
| plain | 0.5084 | 0.4594 | 0.5691 | 0.6274 | 1.9006 | plain ModernBERT: no layout-MLM, no boxes |

- beats plain ModernBERT by +0.1849 f1
- does not beat layoutlm-base-uncased by -0.1683 f1
- beats the same model without the layout-MLM stage by +0.1795 f1

## rvlcdip

| variant | accuracy | macro_f1 | loss | what it is |
|---|---|---|---|---|
| layout | 0.8422 | 0.8379 | 0.5756 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.8852 | 0.8750 | 0.4452 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.8315 | 0.8252 | 0.6390 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.7348 | 0.7275 | 0.8548 | ours, boxes withheld at fine-tune time |

- plain ModernBERT: not run
- does not beat layoutlm-base-uncased by -0.0430 accuracy
- beats the same model without the layout-MLM stage by +0.0107 accuracy

## sroie

| variant | f1 | precision | recall | token_accuracy | loss | what it is |
|---|---|---|---|---|---|---|
| layout | 0.8046 | 0.8268 | 0.7836 | 0.9836 | 0.1370 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.8222 | 0.8162 | 0.8284 | 0.9865 | 0.0957 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.7923 | 0.8175 | 0.7687 | 0.9867 | 0.1063 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.7586 | 0.7795 | 0.7388 | 0.9833 | 0.1019 | ours, boxes withheld at fine-tune time |

- plain ModernBERT: not run
- does not beat layoutlm-base-uncased by -0.0176 f1
- beats the same model without the layout-MLM stage by +0.0123 f1

