import express from "express";
import cors from "cors";
import "dotenv/config";
import OpenAI from "openai";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";

const execFileAsync = promisify(execFile);

const app = express();

app.use(cors());
app.use(express.json());

const selora = new OpenAI({
  apiKey: process.env.SELORA_API_KEY,
  baseURL: "https://api.selora.lol/v1",
});

const pythonPath = path.resolve(
  __dirname,
  "../../ml/.venv/bin/python"
);

const predictorPath = path.resolve(
  __dirname,
  "../../ml/predict.py"
);

app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    service: "digital-twin-backend",
  });
});

app.get("/api/test-ai", async (_req, res) => {
  try {
    const response = await selora.chat.completions.create({
      model: "claude-fable-5.1",
      messages: [
        {
          role: "user",
          content:
            "Explain digital twins in healthcare in one sentence.",
        },
      ],
      max_tokens: 100,
    });

    res.json({
      response: response.choices[0]?.message?.content,
      usage: response.usage,
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      error: "AI request failed",
    });
  }
});

app.get("/api/patients/:patientId/twin", async (req, res) => {
  const { patientId } = req.params;
  const { at } = req.query;

  if (!/^P\d{4}$/.test(patientId)) {
    return res.status(400).json({
      error: "Invalid patient ID format.",
    });
  }

  try {
    const { stdout, stderr } = await execFileAsync(
      pythonPath,
      [
        predictorPath,
        "--patient-id",
        patientId,
        ...(typeof at === "string"
          ? ["--timestamp", at]
          : []),
      ],
      {
        cwd: path.resolve(__dirname, "../.."),
        maxBuffer: 1024 * 1024,
      }
    );

    if (stderr) {
      console.warn("Python stderr:", stderr);
    }

    const prediction = JSON.parse(stdout);

    if (prediction.error) {
      return res.status(404).json(prediction);
    }

    res.json({
      twin: {
        patientId: prediction.patient_id,
        timestamp: prediction.timestamp,

        currentState: prediction.current_state,

        prediction: {
          event: "glucose_spike",
          horizonMinutes:
            prediction.prediction_horizon_minutes,
          probability:
            prediction.glucose_spike_probability,
          probabilityPercent:
            prediction.glucose_spike_probability_percent,
          predicted:
            Boolean(prediction.predicted_spike),
          riskLevel: prediction.risk_level,
        },
      },
    });
  } catch (error) {
    console.error("Twin prediction error:", error);

    res.status(500).json({
      error: "Unable to generate digital twin state.",
    });
  }
});

const PORT = 3000;

app.listen(PORT, () => {
  console.log(
    `Server running on http://localhost:${PORT}`
  );
});