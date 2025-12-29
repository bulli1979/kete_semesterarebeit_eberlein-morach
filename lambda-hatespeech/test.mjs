import ort from "onnxruntime-node";
import fs from "fs";
import { Tokenizer } from "@huggingface/tokenizers";

const MODEL_PATH = "./model/model_single.onnx";
const TOKENIZER_JSON_PATH = "./model/tokenizer.json";
const TOKENIZER_CONFIG_PATH = "./model/tokenizer_config.json";

async function main() {
  console.log("🔹 Lade ONNX Modell ...");
  const session = await ort.InferenceSession.create(MODEL_PATH);

  // Prüfe die erwartete Eingabegröße
  console.log("🔹 Input Names:", session.inputNames);
  console.log("🔹 Input Metadata:", session.inputMetadata);

  // Versuche die Shape aus den Metadaten zu extrahieren
  let maxLength = 12; // Fallback
  if (session.inputNames.length > 0) {
    const firstInputName = session.inputNames[0];
    const metadata = session.inputMetadata[firstInputName];
    if (metadata && metadata.dims && metadata.dims.length > 1) {
      maxLength = metadata.dims[1];
    }
  }
  console.log("🔹 Erwartete Sequenzlänge:", maxLength);

  console.log("🔹 Initialisiere Tokenizer ...");

  // Lade beide JSON-Dateien als Objekte (nicht als Strings)
  const tokenizerJson = JSON.parse(
    fs.readFileSync(TOKENIZER_JSON_PATH, "utf8")
  );
  const tokenizerConfig = JSON.parse(
    fs.readFileSync(TOKENIZER_CONFIG_PATH, "utf8")
  );

  // Erstelle Tokenizer mit der korrekten API für Version 0.0.4
  const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);

  const text = "Neger sind doof.";
  const encoded = tokenizer.encode(text);

  const padId = 0; // PAD Token ID (aus tokenizer.json)

  // Kürze oder padde die Sequenz auf die erwartete Länge
  let ids = encoded.ids;
  let attention_mask = encoded.attention_mask;

  if (ids.length > maxLength) {
    // Kürze auf maxLength
    ids = ids.slice(0, maxLength);
    attention_mask = attention_mask.slice(0, maxLength);
  } else if (ids.length < maxLength) {
    // Padde auf maxLength
    while (ids.length < maxLength) {
      ids.push(padId);
      attention_mask.push(0);
    }
  }

  const inputIds = new BigInt64Array(ids.map(BigInt));
  const attentionMask = new BigInt64Array(attention_mask.map(BigInt));

  const feeds = {
    input_ids: new ort.Tensor("int64", inputIds, [1, maxLength]),
    attention_mask: new ort.Tensor("int64", attentionMask, [1, maxLength]),
  };

  console.log("🔹 Führe Inferenz aus ...");
  const results = await session.run(feeds);

  console.log("✅ Ergebnis (logits):", results.logits.data);
  const logits = Array.from(results.logits.data);

  // Softmax berechnen
  function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
  }
  const probs = softmax(logits);

  // Labels definieren
  const labels = ["non_hate", "hate"];

  // Wahrscheinlichkeiten für jedes Label ausgeben
  console.log(`🔹 Label 0 (${labels[0]}): ${(probs[0] * 100).toFixed(2)} %`);
  console.log(`🔹 Label 1 (${labels[1]}): ${(probs[1] * 100).toFixed(2)} %`);

  // Gewinnerlabel bestimmen
  const maxIndex = probs.indexOf(Math.max(...probs));
  console.log(`✅ Vorhersage: ${labels[maxIndex]}`);
}

main().catch(console.error);
