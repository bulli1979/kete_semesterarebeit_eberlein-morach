import * as onyxRuntime from "onnxruntime-node";
import * as fs from "fs";
import { Tokenizer } from "@huggingface/tokenizers";

const MODEL_PATH = "./model/model_single.onnx";
const TOKENIZER_JSON_PATH = "./model/tokenizer.json";
const TOKENIZER_CONFIG_PATH = "./model/tokenizer_config.json";

interface LambdaEvent {
  body?: string;
  [key: string]: any;
}

interface LambdaResponse {
  statusCode: number;
  headers: { [key: string]: string };
  body: string;
}

interface PredictionResult {
  input: string;
  prediction: {
    label: string;
    probability: number;
    probabilities: {
      non_hate: number;
      hate: number;
    };
  };
}

let session: onyxRuntime.InferenceSession | null = null;
let tokenizer: Tokenizer | null = null;
let maxLength: number = 12;

async function loadModel(): Promise<void> {
  if (!session || !tokenizer) {
    console.log("🔄 Lade Modell (Cold Start) ...");

    // Lade ONNX Modell
    session = await onyxRuntime.InferenceSession.create(MODEL_PATH);
    console.log("✅ ONNX Modell geladen");

    // Bestimme maxLength aus Metadaten
    if (session.inputNames.length > 0) {
      const firstInputName = session.inputNames[0];
      const metadata =
        session.inputMetadata[
          firstInputName as keyof typeof session.inputMetadata
        ];
      if (metadata && typeof metadata === "object" && "dims" in metadata) {
        const valueMetadata = metadata as { dims: readonly number[] };
        if (valueMetadata.dims && valueMetadata.dims.length > 1) {
          maxLength = valueMetadata.dims[1] as number;
        }
      }
    }
    console.log(`🔹 Erwartete Sequenzlänge: ${maxLength}`);

    // Lade Tokenizer
    const tokenizerJson = JSON.parse(
      fs.readFileSync(TOKENIZER_JSON_PATH, "utf8")
    );
    const tokenizerConfig = JSON.parse(
      fs.readFileSync(TOKENIZER_CONFIG_PATH, "utf8")
    );
    tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    console.log("✅ Tokenizer geladen");
  }
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

async function predict(text: string): Promise<PredictionResult["prediction"]> {
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
    input_ids: new onyxRuntime.Tensor("int64", inputIds, [1, maxLength]),
    attention_mask: new onyxRuntime.Tensor("int64", attentionMask, [
      1,
      maxLength,
    ]),
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

export const handler = async (event: LambdaEvent): Promise<LambdaResponse> => {
  try {
    // Lade Modell beim ersten Aufruf
    await loadModel();

    // Parse Request Body
    const body = event.body ? JSON.parse(event.body) : {};
    const text = body.text || "";

    if (!text) {
      return {
        statusCode: 400,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Kein Text übergeben" }),
      };
    }

    // Führe Vorhersage aus
    const prediction = await predict(text);

    const result: PredictionResult = {
      input: text,
      prediction,
    };

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(result),
    };
  } catch (err: any) {
    console.error("Fehler:", err);
    return {
      statusCode: 500,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: err.message || "Interner Serverfehler" }),
    };
  }
};

// Lokaler Test (nur wenn direkt ausgeführt)
if (process.argv[1] && process.argv[1].includes("index")) {
  (async () => {
    const event: LambdaEvent = {
      body: JSON.stringify({ text: "Schwule sind super." }),
    };
    const res = await handler(event);
    console.log("\nLokaler Test:\n", res.body);
  })().catch(console.error);
}
