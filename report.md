# EEG Transformer Classification Report

## Dataset Description

- Number of training subjects: 18 (S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17, S18)
- Number of test subjects: 3 (S19, S20, S21)
- Total training trials: 80615
- Total test trials: 13280

## Training Results

- Final training accuracy: 0.7919
- Final validation accuracy: 0.7800

## Test Results

- Test accuracy: 0.7139

### Confusion Matrix (Table)

| True \ Pred | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 8644 | 231 | 7 | 118 |
| 2 | 1238 | 148 | 11 | 43 |
| 3 | 412 | 45 | 4 | 19 |
| 4 | 1557 | 102 | 17 | 684 |

### Visualizations

![Loss Curve](plots/loss.png)
![Training Accuracy](plots/train_accuracy.png)
![Validation Accuracy](plots/val_accuracy.png)
![Confusion Matrix](plots/confusion_matrix.png)
