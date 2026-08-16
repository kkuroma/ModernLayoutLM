# Results

## cord

| variant | f1 | precision | recall | token_accuracy | loss | what it is |
|---|---|---|---|---|---|---|
| layout | 0.9418 | 0.9346 | 0.9491 | 0.9570 | 0.2405 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.9661 | 0.9625 | 0.9696 | 0.9721 | 0.2016 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.9362 | 0.9282 | 0.9442 | 0.9543 | 0.2622 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.9290 | 0.9189 | 0.9393 | 0.9492 | 0.2422 | ours, boxes withheld at fine-tune time |

- plain ModernBERT: not run
- does not beat layoutlm-base-uncased by -0.0243 f1
- beats the same model without the layout-MLM stage by +0.0056 f1

## funsd

| variant | f1 | precision | recall | token_accuracy | loss | what it is |
|---|---|---|---|---|---|---|
| layout | 0.7610 | 0.7400 | 0.7833 | 0.6901 | 1.8226 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.8616 | 0.8504 | 0.8731 | 0.7907 | 1.1751 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.5138 | 0.4634 | 0.5766 | 0.6319 | 2.3650 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.3121 | 0.2776 | 0.3564 | 0.5080 | 1.2590 | ours, boxes withheld at fine-tune time |
| plain | 0.5084 | 0.4594 | 0.5691 | 0.6274 | 1.9006 | plain ModernBERT: no layout-MLM, no boxes |

- beats plain ModernBERT by +0.2526 f1
- does not beat layoutlm-base-uncased by -0.1006 f1
- beats the same model without the layout-MLM stage by +0.2472 f1

## rvlcdip

| variant | accuracy | macro_f1 | loss | what it is |
|---|---|---|---|---|
| layout | 0.8582 | 0.8537 | 0.5319 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.8852 | 0.8750 | 0.4452 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.8315 | 0.8252 | 0.6390 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.7454 | 0.7377 | 0.8151 | ours, boxes withheld at fine-tune time |

- plain ModernBERT: not run
- does not beat layoutlm-base-uncased by -0.0270 accuracy
- beats the same model without the layout-MLM stage by +0.0267 accuracy

## sroie

| variant | f1 | precision | recall | token_accuracy | loss | what it is |
|---|---|---|---|---|---|---|
| layout | 0.8372 | 0.8710 | 0.8060 | 0.9867 | 0.1055 | ModernLayoutLM: layout-MLM then fine-tune |
| layoutlm | 0.8222 | 0.8162 | 0.8284 | 0.9865 | 0.0957 | microsoft/layoutlm-base-uncased, identical data |
| skipmlm | 0.7923 | 0.8175 | 0.7687 | 0.9867 | 0.1063 | ours, fine-tuned straight off ModernBERT |
| nobbox | 0.7954 | 0.8240 | 0.7687 | 0.9859 | 0.0868 | ours, boxes withheld at fine-tune time |

- plain ModernBERT: not run
- beats layoutlm-base-uncased by +0.0150 f1
- beats the same model without the layout-MLM stage by +0.0449 f1

