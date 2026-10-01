import express from "express";
import cors from "cors";
import "dotenv/config";
import OpenAI from "openai";

const app = express();

app.use(cors());
app.use(express.json());

const selora = new OpenAI({
  apiKey: process.env.SELORA_API_KEY,
  baseURL: "https://api.selora.lol/v1",
});

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
          content: "Explain digital twins in healthcare in one sentence.",
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

const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});