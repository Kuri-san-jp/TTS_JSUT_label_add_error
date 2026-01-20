# JSUT BASIC5000 Accent Error Generator

A Python tool for generating intentional accent errors in Japanese speech synthesis (TTS) datasets. This tool modifies accent nucleus positions in the JSUT BASIC5000 dataset while respecting Japanese phonological constraints.

## Overview

This tool downloads the ground truth accent labels from [sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label) and generates modified versions with intentionally shifted accent nucleus (`]`) positions. The generated data can be used for:

- Training accent error detection models
- Evaluating TTS systems' sensitivity to accent errors
- Research on Japanese prosody and accent

## Features

- **Automatic data download**: Fetches ground truth from the official JSUT label repository
- **Phonologically valid errors**: Follows Kubozono's accent rules for Japanese
- **Comprehensive validation**: 11 validation rules ensure data quality
- **Visual diff output**: Generates color-coded HTML diff reports

## Japanese Accent Notation

The tool uses the following notation system:

| Symbol | Meaning |
|--------|---------|
| `^` | Utterance start |
| `$` | Utterance end |
| `#` | Bunsetsu (phrase) boundary |
| `_` | Pause |
| `[` | Phrase-initial pitch rise (after 1st mora) |
| `]` | Accent nucleus (pitch fall point) |

## Phonological Constraints (Kubozono's Rules)

The accent nucleus (`]`) cannot be placed on:

1. **Special morae**: `N` (moraic nasal), `cl` (geminate), `q` (glottal stop)
2. **Long vowel second halves**: The second part of long vowels (e.g., `o` in `oo`)
3. **Positions before `[`**: The nucleus must come after the phrase-initial rise
4. **Final phoneme**: At least one phoneme must follow the nucleus

## Validation Rules

The tool enforces 11 validation rules:

| Rule | Description |
|------|-------------|
| 1 | Maximum one nucleus per accent phrase |
| 2 | Nucleus only after valid vowels |
| 3-4 | Original `[` positions must be preserved |
| 7 | Phrases with original `]` must have `]` in output |
| 8 | All phrases need a nucleus (with exceptions) |
| 9 | `[` must precede `]` in a phrase |
| 10 | `]` requires `[` (except for 1-2 mora phrases) |
| 11 | `]` cannot immediately precede `#` |

## Usage

```bash
python generate_error_dataset_with_diff.py
```

### Output Files

- `basic5000_accent_error.yaml`: Generated accent error data (5000 utterances)
- `basic5000_diff.html`: Color-coded HTML diff report

### Parameters

Edit the script to modify:

- `error_rate`: Probability of modifying each accent phrase (default: 0.8)
- `random_seed`: Seed for reproducibility (default: 42)

## Requirements

- Python 3.7+
- PyYAML
- requests

```bash
pip install pyyaml requests
```

## File Structure

```
.
├── README.md
├── generate_error_dataset_with_diff.py  # Main script
├── basic5000_accent_error.yaml          # Generated error data
└── basic5000_diff.html                  # Visual diff report
```

## Example

**Original**:
```
^-w-a-t-a-sh-i-w-a-#-[-k-yo-o-t-o-]-n-i-#-i-k-i-m-a-]-sh-i-t-a-$
```

**With accent error** (nucleus shifted):
```
^-w-a-t-a-sh-i-w-a-#-[-k-yo-]-o-t-o-n-i-#-i-]-k-i-m-a-sh-i-t-a-$
```

## Data Source

Ground truth labels are from:
- Repository: [sarulab-speech/jsut-label](https://github.com/sarulab-speech/jsut-label)
- File: `e2e_symbol/phoneme.yaml`

## License

This tool is provided for research purposes. Please refer to the original JSUT corpus license for data usage terms.

## References

- JSUT corpus: https://github.com/sarulab-speech/jsut-label
- Kubozono, H. (1999). *Mora and Syllable*. In N. Tsujimura (Ed.), The Handbook of Japanese Linguistics.

## Acknowledgments

- [sarulab-speech](https://github.com/sarulab-speech) for the JSUT label annotations
