# Hate Speech Detection - Transformer-basiertes Modell für Deutsch und Schweizerdeutsch

Ein umfassendes Projekt zur Erkennung von Hate Speech in deutschen und schweizerdeutschen Texten mit Hilfe von Transformer-Modellen.

## 📋 Inhaltsverzeichnis

- [Überblick](#überblick)
- [Projektstruktur](#projektstruktur)
- [Daten](#daten)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Modelle](#modelle)
- [Ergebnisse](#ergebnisse)
- [Deployment](#deployment)

## 🎯 Überblick

Dieses Projekt implementiert ein Transformer-basiertes Modell zur Klassifikation von Hate Speech in deutschen und schweizerdeutschen Texten. Das Modell basiert auf `deepset/gbert-base` und wurde auf einem speziell aufbereiteten Datensatz trainiert.

### Hauptfunktionen

- **Transformer-basierte Klassifikation**: Verwendet `gbert-base` für die Hate Speech Erkennung
- **Umfassende Evaluation**: Confusion Matrix, ROC Curve, Precision-Recall Curve
- **Bias-Screening**: Analyse der Modell-Performance für verschiedene demografische Gruppen
- **Active Learning**: Export von unsicheren Vorhersagen für manuelle Annotation
- **Explainability**: SHAP-basierte Erklärungen für Modellvorhersagen
- **Deployment**: ONNX-Export für Produktionseinsatz

## 📁 Projektstruktur

```
KETE/
├── data/                          # Datensätze
│   ├── de_hf_112024.csv          # Original-Datensatz
│   ├── de_hf_112024_train.csv    # Trainingsdatensatz (80%)
│   ├── de_hf_112024_val.csv      # Validierungsdatensatz (5%)
│   ├── de_hf_112024_test.csv     # Testdatensatz (20%)
│   └── vulgaer.txt               # Vulgärwörter-Liste
│
├── notebook/                      # Jupyter Notebooks
│   ├── hateSpeeech_splitData.ipynb    # Datenaufbereitung und Split
│   ├── hateSpeech_trainmodel.ipynb    # Modelltraining
│   ├── hateSpeech_validate.ipynb      # Validierung und Evaluation
│   └── results_*/                # Trainingsergebnisse
│       ├── final_model/          # Finales trainiertes Modell
│       ├── checkpoint-*/        # Training-Checkpoints
│       └── validation_results/  # Validierungsergebnisse
│
├── lambda-hatespeech/            # AWS Lambda Deployment
│   ├── index.ts                  # Lambda Handler
│   └── model/                    # ONNX-Modell für Lambda
│
├── modelsagemaker/               # SageMaker Deployment
│   ├── code/                     # Inference-Code
│   └── model.ckpt                # SageMaker-Modell
│
└── dokumentation/                # Dokumentationsbilder
```

## 📊 Daten

### Datensatz: `de_hf_112024.csv`

Der Hauptdatensatz enthält deutsche und schweizerdeutsche Texte mit Labels für Hate Speech.

**Spalten:**
- `text`: Der zu klassifizierende Text
- `labels`: Binäres Label (0 = Non-Hate, 1 = Hate Speech)

**Aufteilung:**
- **Training**: 80% (ca. 39.000 Samples)
- **Validation**: 5% (ca. 2.500 Samples)
- **Test**: 20% (ca. 9.800 Samples)

Die Aufteilung erfolgt stratifiziert, um die Label-Verteilung in allen Sets zu erhalten.

### Datenaufbereitung

1. **Filterung**: Nur Einträge mit Labels 0 oder 1 werden behalten
2. **Bereinigung**: Entfernung von NaN-Werten
3. **Split**: Stratifizierte Aufteilung in Train/Val/Test

## 📓 Notebooks

### 1. `hateSpeeech_splitData.ipynb`

**Zweck**: Datenaufbereitung und Aufteilung in Train/Val/Test-Sets

**Funktionen:**
- Lädt den Original-Datensatz `de_hf_112024.csv`
- Filtert nach Labels 0 und 1
- Führt stratifizierten Split durch (5% Val, Rest 80/20 Train/Test)
- Speichert die aufgeteilten Datensätze als CSV

**Ausgabe:**
- `data/de_hf_112024_train.csv`
- `data/de_hf_112024_val.csv`
- `data/de_hf_112024_test.csv`

### 2. `hateSpeech_trainmodel.ipynb`

**Zweck**: Training des Transformer-Modells

**Funktionen:**
- **Setup**: Installation von Paketen, Initialisierung
- **Daten laden**: Lädt Train- und Test-Datensätze
- **Tokenisierung**: Verwendet `gbert-base` Tokenizer
- **Modell-Training**: 
  - Base Model: `deepset/gbert-base`
  - Training mit Early Stopping
  - Mixed Precision Training (CUDA)
  - Evaluation auf Testdaten
- **ONNX-Export**: Exportiert Modell für Produktion
- **Bias-Screening**: Analysiert Performance für verschiedene Gruppen
- **Active Learning**: Exportiert unsichere Vorhersagen
- **SHAP Explainability**: Erklärt Modellvorhersagen

**Ausgabe:**
- `results_*/final_model/`: Finales trainiertes Modell
- `results_*/model.onnx`: ONNX-Export
- `results_*/active_learning_*.csv`: Active Learning Exports
- `results_*/model_card_*.json`: Modell-Metadaten

### 3. `hateSpeech_validate.ipynb`

**Zweck**: Umfassende Validierung und Evaluation des trainierten Modells

**Funktionen:**
- Lädt das trainierte Modell
- Erstellt Vorhersagen auf Validierungsdatensatz
- **Metriken**: Accuracy, Precision, Recall, F1, ROC AUC
- **Visualisierungen**:
  - Confusion Matrix (2 Varianten)
  - ROC Curve
  - Precision-Recall Curve
  - Metriken-Bar Charts
  - Wahrscheinlichkeitsverteilungen
  - Fehleranalyse
- **Fehleranalyse**: Identifiziert False Positives/Negatives
- **Export**: Speichert alle Ergebnisse als CSV

**Ausgabe:**
- `validation_results/validation_summary.csv`: Zusammenfassung der Metriken
- `validation_results/validation_predictions.csv`: Alle Vorhersagen
- `validation_results/false_positives.csv`: Falsch Positive Beispiele
- `validation_results/false_negatives.csv`: Falsch Negative Beispiele

## 🚀 Installation

### Voraussetzungen

- Python 3.8+
- Jupyter Notebook
- CUDA-fähige GPU (empfohlen, aber nicht erforderlich)

### Pakete installieren

```bash
pip install transformers datasets accelerate torch shap scikit-learn pandas matplotlib seaborn tf-keras
```

Für ONNX-Export:
```bash
pip install optimum[onnxruntime] onnxruntime onnxscript
```

## 💻 Verwendung

### 1. Datenaufbereitung

```bash
# Öffne das Notebook
jupyter notebook notebook/hateSpeeech_splitData.ipynb

# Führe alle Zellen aus
# Dies erstellt die Train/Val/Test-Splits
```

### 2. Modelltraining

```bash
# Öffne das Training-Notebook
jupyter notebook notebook/hateSpeech_trainmodel.ipynb

# Führe alle Zellen der Reihe nach aus
# Das Training kann mehrere Stunden dauern (abhängig von GPU)
```

**Wichtige Parameter:**
- `BASE_MODEL_NAME`: `"deepset/gbert-base"`
- `MAX_LEN`: 512 (maximale Sequenzlänge)
- `BATCH_SIZE`: 16 (kann je nach GPU angepasst werden)
- `LEARNING_RATE`: 2e-5
- `NUM_EPOCHS`: 5

### 3. Validierung

```bash
# Öffne das Validierungs-Notebook
jupyter notebook notebook/hateSpeech_validate.ipynb

# Führe alle Zellen aus
# Dies erstellt alle Visualisierungen und Metriken
```

## 🤖 Modelle

### Base Model: `deepset/gbert-base`

- **Architektur**: BERT-basiert, speziell für Deutsch trainiert
- **Parameter**: ~110 Millionen
- **Tokenisierung**: SentencePiece
- **Maximale Sequenzlänge**: 512 Tokens

### Training

- **Optimizer**: AdamW
- **Learning Rate**: 2e-5 mit Linear Decay
- **Warmup Steps**: 500
- **Early Stopping**: Basierend auf Validation Loss
- **Mixed Precision**: FP16 (wenn CUDA verfügbar)

### Modellformate

1. **PyTorch** (`model.safetensors`): Für weitere Training/Feintuning
2. **ONNX** (`model.onnx`): Für Produktionseinsatz
3. **HuggingFace** (`final_model/`): Standard HuggingFace Format

## 📈 Ergebnisse

### Metriken (Beispiel)

Die genauen Metriken hängen vom trainierten Modell ab. Typische Werte:

- **Accuracy**: ~92-95%
- **F1-Score (Weighted)**: ~0.85-0.90
- **ROC AUC**: ~0.90-0.95
- **Precision (Hate Speech)**: ~0.70-0.80
- **Recall (Hate Speech)**: ~0.60-0.70

### Visualisierungen

Das Validierungsnotebook erstellt folgende Visualisierungen:

1. **Confusion Matrix**: Zeigt Klassifikationsfehler
2. **ROC Curve**: Zeigt Trade-off zwischen True/False Positive Rate
3. **Precision-Recall Curve**: Zeigt Trade-off zwischen Precision und Recall
4. **Metriken-Vergleich**: Bar Charts für alle Metriken
5. **Wahrscheinlichkeitsverteilung**: Histogramm und Boxplot
6. **Fehleranalyse**: Analyse von False Positives/Negatives

## 🚢 Deployment

### AWS Lambda

Das Modell kann als AWS Lambda Function deployed werden:

```bash
cd lambda-hatespeech
npm install
# Modell muss in model/ Verzeichnis vorhanden sein
zip -r ../lambda-hatespeech.zip .
```

### SageMaker

Das Modell kann auch auf AWS SageMaker deployed werden:

```bash
# Modell muss als model.tar.gz gepackt sein
# Siehe modelsagemaker/ Verzeichnis für Inference-Code
```

### Lokale Verwendung

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Lade Modell
model_path = "notebook/results_*/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Vorhersage
text = "Ihr Text hier"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    confidence = predictions[0][predicted_class].item()

print(f"Klasse: {predicted_class}, Konfidenz: {confidence:.4f}")
```

## 🔍 Bias-Screening

Das Modell analysiert die Performance für verschiedene Gruppen:

- **Ethnisch/Migration**: Flüchtling, Migrant, Ausländer, etc.
- **Geschlecht/LGBTQ**: Frau, Mann, LGBTQ, etc.
- **Religion**: Christ, Muslim, Jude, etc.

Die Ergebnisse werden im Training-Notebook ausgegeben.

## 📚 Active Learning

Das Modell exportiert automatisch:

1. **Unsichere Vorhersagen**: 300 Samples mit höchster Unsicherheit
2. **Fehlerhafte Vorhersagen**: Alle falsch klassifizierten Samples

Diese können für manuelle Annotation und weiteres Training verwendet werden.

## 🛠️ Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'tf_keras'`

**Lösung**: Installiere `tf-keras`:
```bash
pip install tf-keras
```

### Problem: `optimum` nicht gefunden beim ONNX-Export

**Lösung**: Installiere optimum:
```bash
pip install optimum[onnxruntime]
```

### Problem: CUDA Out of Memory

**Lösung**: Reduziere `BATCH_SIZE` im Training-Notebook

### Problem: Modellpfad nicht gefunden

**Lösung**: Passe den `MODEL_PATH` im Validierungsnotebook an

## 📝 Lizenz

Dieses Projekt wurde im Rahmen eines Master-Studiums erstellt.

## 👥 Autoren

- Erstellt für KETE (Key Technology)

## 📞 Kontakt

Bei Fragen oder Problemen bitte ein Issue erstellen.

---

**Hinweis**: Dieses Modell ist für Forschungs- und Bildungszwecke gedacht. Bei Verwendung in Produktion sollten zusätzliche Tests und Validierungen durchgeführt werden.

