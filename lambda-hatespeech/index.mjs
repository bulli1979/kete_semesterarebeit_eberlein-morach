import { pipeline } from "@huggingface/transformers";

let classifier;

async function loadModel() {
  if (!classifier) {
    console.log("🔄 Lade Modell (Cold Start) ...");
    classifier = await pipeline("text-classification", "deepset/gbert-base");
    console.log("✅ Modell geladen");
  }
  return classifier;
}

export const handler = async (event) => {
  try {
    const body = event.body ? JSON.parse(event.body) : {};
    const text = body.text || "Kein Text übergeben";

    const model = await loadModel();
    const result = await model(text);

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: text, prediction: result })
    };
  } catch (err) {
    console.error("Fehler:", err);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: err.message })
    };
  }
};

// Lokaler Test
if (process.argv[1].includes("index.mjs")) {
  const event = { body: JSON.stringify({ text: "Ich hasse alle Menschen dieser Gruppe." }) };
  const res = await handler(event);
  console.log("\nLokaler Test:\n", res.body);
}
