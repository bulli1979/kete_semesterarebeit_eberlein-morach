import * as ort from "onnxruntime-node";
import * as fs from "fs";
import * as path from "path";
import { Tokenizer } from "@huggingface/tokenizers";

// Pfade relativ zum Projekt-Root (nicht zum dist-Ordner)
const getModelPath = (filename: string) => {
  // __dirname ist dist/main/ in der kompilierten Version
  // Gehe drei Ebenen hoch zum Projekt-Root (KETE)
  // dist/main/ -> dist/ -> demo/ -> KETE/
  const projectRoot = path.resolve(__dirname, "../../..");
  const modelPath = path.join(projectRoot, "lambda-hatespeech/model", filename);
  console.log(`__dirname: ${__dirname}`);
  console.log(`Projekt-Root: ${projectRoot}`);
  console.log(`Suche Modell-Datei: ${modelPath}`);
  console.log(`Datei existiert: ${fs.existsSync(modelPath)}`);
  return modelPath;
};

interface PredictionResult {
  label: string;
  probability: number;
  probabilities: {
    non_hate: number;
    hate: number;
  };
}

let session: ort.InferenceSession | null = null;
let tokenizer: Tokenizer | null = null;
let maxLength: number = 12; // Standardwert, wird aus Modell-Metadaten gelesen

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

export async function loadModel(): Promise<void> {
  if (!session || !tokenizer) {
    console.log("🔄 Lade Modell (Cold Start) ...");

    // Prüfe ob Dateien existieren
    const modelPath = getModelPath("model_single.onnx");
    const tokenizerPath = getModelPath("tokenizer.json");

    if (!fs.existsSync(modelPath)) {
      throw new Error(`Modell-Datei nicht gefunden: ${modelPath}`);
    }
    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer-Datei nicht gefunden: ${tokenizerPath}`);
    }

    // Lade ONNX Modell
    session = await ort.InferenceSession.create(modelPath);
    console.log("✅ ONNX Modell geladen");

    // Bestimme maxLength aus Metadaten
    if (session.inputNames.length > 0) {
      const firstInputName = session.inputNames[0];
      console.log(`🔍 Input-Name: ${firstInputName}`);
      const metadata =
        session.inputMetadata[
          firstInputName as keyof typeof session.inputMetadata
        ];
      console.log(`🔍 Metadaten:`, JSON.stringify(metadata, null, 2));
      if (metadata && typeof metadata === "object" && "dims" in metadata) {
        const valueMetadata = metadata as { dims: readonly number[] };
        console.log(`🔍 Dimensions:`, valueMetadata.dims);
        if (valueMetadata.dims && valueMetadata.dims.length > 1) {
          maxLength = valueMetadata.dims[1] as number;
          console.log(`✅ Sequenzlänge aus Metadaten gelesen: ${maxLength}`);
        } else {
          console.warn(
            `⚠️ Konnte Sequenzlänge nicht aus Metadaten lesen, verwende Standard: ${maxLength}`
          );
        }
      } else {
        console.warn(
          `⚠️ Metadaten nicht gefunden, verwende Standard: ${maxLength}`
        );
      }
    } else {
      console.warn(
        `⚠️ Keine Input-Namen gefunden, verwende Standard: ${maxLength}`
      );
    }
    console.log(`🔹 Finale erwartete Sequenzlänge: ${maxLength}`);

    // Lade Tokenizer
    const tokenizerJson = JSON.parse(fs.readFileSync(tokenizerPath, "utf8"));
    const tokenizerConfigPath = getModelPath("tokenizer_config.json");
    const tokenizerConfig = JSON.parse(
      fs.readFileSync(tokenizerConfigPath, "utf8")
    );
    tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    console.log("✅ Tokenizer geladen");
  }
}

export async function predict(text: string): Promise<PredictionResult> {
  if (!session || !tokenizer) {
    throw new Error("Modell nicht geladen");
  }

  // Tokenisiere Text
  const encoded = tokenizer.encode(text);
  const padId = 0;

  // Kürze oder padde die Sequenz auf die erwartete Länge
  let ids = encoded.ids;
  let attention_mask = encoded.attention_mask;

  if (ids.length > maxLength) {
    ids = ids.slice(0, maxLength);
    attention_mask = attention_mask.slice(0, maxLength);
  } else if (ids.length < maxLength) {
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

  // Führe Inferenz aus
  const results = await session.run(feeds);
  const logits = Array.from(results.logits.data as Float32Array);
  const probs = softmax(logits);

  const labels = ["non_hate", "hate"];
  const maxIndex = probs.indexOf(Math.max(...probs));

  return {
    label: labels[maxIndex],
    probability: probs[maxIndex],
    probabilities: {
      non_hate: probs[0],
      hate: probs[1],
    },
  };
}
