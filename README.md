# LLM-anonymization

This repository provides utilities for anonymizing, pseudonymizing, and simplifying Dutch text using various NLP techniques. The main functionalities are implemented in the `anonymization_utils.py` file, with an example usage provided in the `example.ipynb` notebook.

## Features

- **Anonymization**: Replace sensitive entities in the text with generic placeholders.
- **Pseudonymization**: Replace sensitive entities with unique identifiers.
- **Text Simplification**: Simplify Dutch text using a pre-trained model.
- **SARI Score Calculation**: Evaluate the quality of text simplification.
- **Entity Extraction**: Extract named entities from the text.
- **Entity Comparison**: Compare entities between original and simplified texts.
- **Score Plotting**: Visualize the performance metrics.

## Usage

### Anonymization

To anonymize text, use the `anonymize_text` function:

```python
from anonymization_utils import anonymize_text

text = "Your text here"
anonymized_text, analyzer_results = anonymize_text(text)
print(anonymized_text)
```

### Pseudonymization

To pseudonymize text, use the `pseudonymize_text` function:

```python
from anonymization_utils import pseudonymize_text

pseudonymized_text = pseudonymize_text(text, analyzer_results)
print(pseudonymized_text)
```

### Text Simplification

To simplify text, use the `simplify_text` function:

```python
from anonymization_utils import simplify_text

simplified_text = simplify_text(anonymized_text)
print(simplified_text)
```

### SARI Score Calculation

To calculate the SARI score, use the `calculate_sari_score` function:

```python
from anonymization_utils import calculate_sari_score

sari_score = calculate_sari_score(text, simplified_text)
print(sari_score)
```

### Entity Extraction

To extract entities from text, use the `extract_entities` function:

```python
from anonymization_utils import extract_entities

entities = extract_entities(text)
print(entities)
```

### Entity Comparison

To compare entities between original and simplified texts, use the `compare_entities` function:

```python
from anonymization_utils import compare_entities

precision, recall, f1, true_positives, false_positives, false_negatives = compare_entities(original_entities, simplified_entities)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
```

### Score Plotting

To plot the performance metrics, use the `plot_scores` function:

```python
from anonymization_utils import plot_scores

anonymized_scores = [precision_anonymized, recall_anonymized, f1_anonymized, sari_score_anonymized]
pseudonymized_scores = [precision_pseudonymized, recall_pseudonymized, f1_pseudonymized, sari_score_pseudonymized]

plot_scores(anonymized_scores, pseudonymized_scores)
```

## Evaluation Metrics

To ensure the effectiveness and accuracy of our text processing techniques, we use several evaluation metrics, particularly in the context of text simplification and anonymization/pseudonymization:

- **Precision**: Indicates the accuracy of the entities identified by the model. High precision means that the model correctly identifies and anonymizes sensitive information without affecting irrelevant data.
- **Recall**: Reflects the model's ability to find all relevant instances in the text. High recall ensures that most, if not all, sensitive information is identified and anonymized.
- **F1 Score**: Balances precision and recall, providing a single metric that considers both the accuracy and completeness of the anonymization process. A high F1 score indicates a well-performing model that effectively anonymizes sensitive information while minimizing errors.
- **SARI Score**: Evaluates the quality of text simplification by comparing the simplified text to the original and reference texts. It helps assess how well the simplified text retains its meaning and readability after anonymization.

These metrics complement each other by providing a comprehensive evaluation of the model's performance in both simplifying and anonymizing text. Precision and recall focus on the accuracy and completeness of anonymization, while the F1 score offers a balanced view. The SARI score ensures that the simplified text remains clear and understandable, even after anonymization.

## Example

For a complete example, refer to the `example.ipynb` notebook in this repository. It demonstrates how to use the utilities step-by-step with sample text.

## Sources

- [Presidio Pseudonymization](https://microsoft.github.io/presidio/samples/python/pseudonomyzation/)
- [spaCy Documentation](https://spacy.io)
- [UL2-large-Dutch-Simplification-MAI-2023 Model](https://huggingface.co/BramVanroy/ul2-large-dutch-simplification-mai-2023)
- [Optimizing Statistical Machine Translation for Text Simplification](https://www.aclweb.org/anthology/Q16-1029)
- [Presidio Pseudonymization](https://microsoft.github.io/presidio/samples/python/pseudonomyzation/)
- [spaCy Models for Dutch](https://spacy.io/models/nl)
- [Document-Level Text Simplification: Dataset, Criteria and Baseline](https://arxiv.org/abs/2110.05071)
- [Project Jupyter](https://jupyter.org/)
- [Matplotlib](https://matplotlib.org/)
- [A Review of Research-based Automatic Text Simplification Tools](https://aclanthology.org/2023.ranlp-1.36.pdf)
- Pseudonymization of patient identifiers for translational research
- Analysis of Data Anonymization Techniques
- Privacy-and utility-preserving NLP with anonymized data: A case study of pseudonymization
- A review of anonymization for healthcare data
