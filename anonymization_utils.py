import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType
from transformers import pipeline, AutoTokenizer
from nltk.tokenize import sent_tokenize
from evaluate import load
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from typing import Dict

# Custom Spacy NLP engine that uses a pre-loaded Spacy model
class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"nl": loaded_spacy_model}

# Custom anonymizer operator that replaces entities with a format <ENTITY_TYPE_INDEX>
class InstanceCounterAnonymizer(Operator):
    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: Dict = None) -> str:
        entity_type: str = params["entity_type"]
        entity_mapping: Dict[Dict:str] = params["entity_mapping"]
        entity_mapping_for_type = entity_mapping.get(entity_type)
        if not entity_mapping_for_type:
            new_text = self.REPLACING_FORMAT.format(entity_type=entity_type, index=0)
            entity_mapping[entity_type] = {}
        else:
            if text in entity_mapping_for_type:
                return entity_mapping_for_type[text]
            previous_index = self._get_last_index(entity_mapping_for_type)
            new_text = self.REPLACING_FORMAT.format(entity_type=entity_type, index=previous_index + 1)
        entity_mapping[entity_type][text] = new_text
        return new_text

    @staticmethod
    def _get_last_index(entity_mapping_for_type: Dict) -> int:
        def get_index(value: str) -> int:
            return int(value.split("_")[-1][:-1])
        indices = [get_index(v) for v in entity_mapping_for_type.values()]
        return max(indices)

    def validate(self, params: Dict = None) -> None:
        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return "entity_counter"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize

# Function to anonymize text using Presidio
def anonymize_text(text):
    nlp = spacy.load("nl_core_news_lg")
    loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model=nlp)
    analyzer = AnalyzerEngine(nlp_engine=loaded_nlp_engine)
    analyzer_results = analyzer.analyze(text=text, language="nl")
    anonymizer_engine = AnonymizerEngine()
    anonymized_result = anonymizer_engine.anonymize(text, analyzer_results)
    return anonymized_result.text, analyzer_results

# Function to pseudonymize text using custom InstanceCounterAnonymizer
def pseudonymize_text(text, analyzer_results):
    anonymizer_engine = AnonymizerEngine()
    anonymizer_engine.add_anonymizer(InstanceCounterAnonymizer)
    entity_mapping = dict()
    pseudonymized_result = anonymizer_engine.anonymize(
        text,
        analyzer_results,
        {"DEFAULT": OperatorConfig("entity_counter", {"entity_mapping": entity_mapping})}
    )
    return pseudonymized_result.text

# Function to simplify text using a pre-trained model
def simplify_text(text):
    model_name = "BramVanroy/ul2-large-dutch-simplification-mai-2023"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    simplifier = pipeline(model=model_name, tokenizer=tokenizer)
    sentences = sent_tokenize(text)
    simplified_sentences = [simplifier(sentence)[0]['generated_text'] for sentence in sentences]
    return ' '.join(simplified_sentences)

# Function to calculate SARI score for text simplification
def calculate_sari_score(text, simplified_text):
    source_sentences = sent_tokenize(text)
    simplified_sentences = sent_tokenize(simplified_text)

    sari_score = calculate_sari_avg(source_sentences, simplified_sentences)
    
    return sari_score

# Helper function to calculate average SARI score
def calculate_sari_avg(source_sentences, simplified_sentences):
    sari = load("sari")
    scores = [
        sari.compute(sources=[src], predictions=[pred], references=[[src]])['sari']
        for src, pred in zip(source_sentences, simplified_sentences)
        if pred.strip()
    ]
    return sum(scores) / len(scores) if scores else 0

# Function to extract entities from text using Spacy
def extract_entities(text):
    nlp = spacy.load("nl_core_news_lg")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to compare entities between original and simplified text
def compare_entities(original_entities, simplified_entities):
    original_set = set(original_entities)
    simplified_set = set(simplified_entities)
    true_positives = original_set & simplified_set
    false_positives = simplified_set - original_set
    false_negatives = original_set - simplified_set
    y_true = [1 if ent in original_set else 0 for ent in original_set | simplified_set]
    y_pred = [1 if ent in simplified_set else 0 for ent in original_set | simplified_set]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1, true_positives, false_positives, false_negatives

# Function to plot comparison scores
def plot_scores(anonymized_scores, pseudonymized_scores):
    labels = ['Precision', 'Recall', 'F1 Score', 'SARI']
    x = range(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, anonymized_scores, width=0.4, label='Anonymized', align='center')
    ax.bar(x, pseudonymized_scores, width=0.4, label='Pseudonymized', align='edge')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of NER Metrics and SARI Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, v in enumerate(anonymized_scores):
        ax.text(i - 0.2, v + 0.01, f"{v:.2f}", color='blue', fontweight='bold')
    for i, v in enumerate(pseudonymized_scores):
        ax.text(i + 0.2, v + 0.01, f"{v:.2f}", color='orange', fontweight='bold')

    plt.show()

# Main function to run the anonymization, pseudonymization, simplification, and evaluation
def main():
    text = '''
    Geachte klantenservice van Coolblue,

    Mijn naam is Bram Ekelschot en met deze brief wil ik mijn ernstige ongenoegen uiten over zowel de levering als de klantenservice van Coolblue. Op 5 januari 2025 heb ik via uw website, www.coolblue.nl, een bestelling geplaatst voor een Samsung Galaxy S23 Ultra (512GB, Phantom Black). Het ordernummer van deze bestelling is CB2025001234. Volgens de orderbevestiging zou het pakket op 7 januari 2025 bezorgd worden op mijn huisadres in Amsterdam. Helaas is de levering met maar liefst vijf dagen vertraagd, waardoor ik het toestel pas op 12 januari 2025 heb ontvangen.

    Tot mijn grote teleurstelling bleek bij het uitpakken dat de Samsung Galaxy S23 Ultra beschadigd was. De behuizing had zichtbare krassen en de camera vertoonde een defect, waardoor het toestel niet naar behoren functioneerde. Direct op 13 januari 2025 heb ik telefonisch contact opgenomen met uw klantenservice via het telefoonnummer 010-798 8999. Een medewerker, genaamd Mark van den Berg, verzekerde mij dat er binnen drie werkdagen een passende oplossing zou komen, zoals een vervangend toestel of een terugbetaling van het aankoopbedrag van €1.399,-. Helaas heb ik tot op heden niets van u vernomen.

    Na meerdere vergeefse pogingen om telefonisch en per e-mail contact met Coolblue op te nemen, zie ik mij genoodzaakt deze klacht schriftelijk in te dienen. Ik verzoek u dan ook dringend om binnen vijf werkdagen met een oplossing te komen. Dit kan in de vorm van een vervangend toestel, een volledige terugbetaling van het aankoopbedrag of een alternatieve compensatie. Daarnaast verwacht ik een vergoeding voor het ongemak dat ik heb ondervonden door de vertraging en de gebrekkige klantenservice.

    Indien er geen tijdige en bevredigende oplossing komt, zal ik mij genoodzaakt zien verdere stappen te ondernemen. Ik overweeg een klacht in te dienen bij de Consumentenbond en, indien nodig, een formele melding te maken bij de Autoriteit Consument & Markt (ACM). Daarnaast behoud ik mij het recht voor om juridische stappen te ondernemen via een geschillencommissie of een gerechtelijke procedure.

    Ik verzoek u vriendelijk om mij per omgaande te informeren over de voortgang van mijn klacht. U kunt mij schriftelijk bereiken op mijn woonadres in Amsterdam of per e-mail via bram.ekelschot@gmail.com. Bij deze brief voeg ik een kopie van mijn bestelbevestiging, de factuur en foto’s van het beschadigde product toe ter ondersteuning van mijn klacht.

    Ik vertrouw erop dat Coolblue zijn verantwoordelijkheid zal nemen en op korte termijn met een passende oplossing zal komen. Ik zie uw reactie dan ook met belangstelling tegemoet.

    Met vriendelijke groet,

    Bram Ekelschot
    '''

    # Anonymize the text
    anonymized_text, analyzer_results = anonymize_text(text)
    # Pseudonymize the text
    pseudonymized_text = pseudonymize_text(text, analyzer_results)
    # Simplify the anonymized text
    simplified_anonymized_text = simplify_text(anonymized_text)
    # Simplify the pseudonymized text
    simplified_pseudonymized_text = simplify_text(pseudonymized_text)

    # Calculate SARI scores for anonymized and pseudonymized texts
    sari_score_anonymized = calculate_sari_score(text, simplified_anonymized_text)
    sari_score_pseudonymized = calculate_sari_score(text, simplified_pseudonymized_text)

    # Extract entities from original, simplified anonymized, and simplified pseudonymized texts
    original_entities = extract_entities(text)
    simplified_anonymized_entities = extract_entities(simplified_anonymized_text)
    simplified_pseudonymized_entities = extract_entities(simplified_pseudonymized_text)

    # Compare entities and calculate precision, recall, and F1 scores
    precision_anonymized, recall_anonymized, f1_anonymized, _, _, _ = compare_entities(original_entities, simplified_anonymized_entities)
    precision_pseudonymized, recall_pseudonymized, f1_pseudonymized, _, _, _ = compare_entities(original_entities, simplified_pseudonymized_entities)

    # Prepare scores for plotting
    anonymized_scores = [precision_anonymized, recall_anonymized, f1_anonymized, sari_score_anonymized]
    pseudonymized_scores = [precision_pseudonymized, recall_pseudonymized, f1_pseudonymized, sari_score_pseudonymized]

    # Plot the comparison scores
    plot_scores(anonymized_scores, pseudonymized_scores)

if __name__ == "__main__":
    main()