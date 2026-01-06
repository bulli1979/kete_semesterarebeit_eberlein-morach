# Hatespeech Detection Demo

Eine Electron-Anwendung zur Erkennung von Hatespeech in Kommentaren.

## Features

- **Kommentaranalyse**: Automatische Erkennung von Hatespeech in eingegebenen Kommentaren
- **Kommentarverwaltung**: Übersichtliche Tabelle mit allen Kommentaren
- **Status-Verwaltung**: Kommentare können gesperrt oder veröffentlicht werden
- **Echtzeit-Score**: Anzeige der Vorhersage-Wahrscheinlichkeiten des Modells

## Installation

1. Installiere die Abhängigkeiten:
```bash
npm install
```

2. Stelle sicher, dass die Model-Dateien im `lambda-hatespeech/model/` Ordner vorhanden sind:
   - `model_single.onnx`
   - `tokenizer.json`
   - `tokenizer_config.json`

## Entwicklung

Starte die Anwendung im Entwicklungsmodus:

```bash
npm run dev
```

Dies startet:
- Vite Dev Server für den Renderer-Prozess (React)
- TypeScript Compiler im Watch-Modus für Main- und Preload-Prozesse
- Electron-Anwendung

## Build

Erstelle eine Produktionsversion:

```bash
npm run build
```

Starte die gebaute Anwendung:

```bash
npm start
```

## Verwendung

1. **Kommentar eingeben**: Geben Sie einen Kommentar in das linke Textfeld ein
2. **Senden**: Klicken Sie auf "Kommentar senden" oder drücken Sie Strg+Enter
3. **Ergebnis ansehen**: Der Kommentar wird analysiert und in der Tabelle angezeigt
4. **Status ändern**: Verwenden Sie die Aktion-Buttons, um Kommentare zu sperren oder zu veröffentlichen
5. **Hatespeech bestätigen**: Bei erkanntem Hatespeech können Sie es bestätigen oder nicht bestätigen (Funktion noch nicht implementiert)

## Technologie-Stack

- **Electron**: Desktop-Anwendung
- **React**: UI-Framework
- **TypeScript**: Typsichere Programmierung
- **ONNX Runtime**: Machine Learning Inference
- **Hugging Face Tokenizers**: Text-Tokenisierung

