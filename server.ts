import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// Initialize GoogleGenAI
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.warn("[Warning] GEMINI_API_KEY is not set. AI interactions will fail securely.");
}

const ai = new GoogleGenAI({
  apiKey: apiKey || "",
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

// Model definitions
const CORE_MODEL = "gemini-3.5-flash";
const FALLBACK_MODEL = "gemini-3.1-flash-lite";

// Robust helper to perform AI queries with high-availability fallbacks
async function generateContentWithFallback(params: {
  contents: any;
  config?: any;
}) {
  try {
    return await ai.models.generateContent({
      model: CORE_MODEL,
      contents: params.contents,
      config: params.config,
    });
  } catch (error: any) {
    const errorStr = String(error.message || "").toLowerCase();
    const isTemporarySpike = 
      errorStr.includes("503") || 
      errorStr.includes("demand") || 
      errorStr.includes("temporary") || 
      errorStr.includes("unavailable") ||
      errorStr.includes("exhausted") ||
      errorStr.includes("overloaded");
    
    if (isTemporarySpike) {
      console.warn(`[Gemini Safe Guard] ${CORE_MODEL} is currently highly loaded. Re-routing request to ${FALLBACK_MODEL}...`);
      try {
        return await ai.models.generateContent({
          model: FALLBACK_MODEL,
          contents: params.contents,
          config: params.config,
        });
      } catch (fallbackError: any) {
        console.error(`[Gemini Safe Guard Error] Fallback query to ${FALLBACK_MODEL} struggled:`, fallbackError);
        throw fallbackError;
      }
    }
    throw error;
  }
}

// -------------------------------------------------------------
// SECURE WORKSPACE API ENDPOINTS
// -------------------------------------------------------------

// Post general conversation with system behavior (Tutors) with attachment support
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, systemInstruction } = req.body;

    if (!messages || !Array.isArray(messages)) {
      res.status(400).json({ error: "Invalid or empty messages payload" });
      return;
    }

    // Map history to official parts structure, including potential inline multimedia files
    const contents = messages.map((m: any) => {
      const parts: any[] = [{ text: m.content || "" }];
      
      if (m.attachments && Array.isArray(m.attachments)) {
        m.attachments.forEach((attach: any) => {
          let cleanData = attach.base64Data || "";
          if (cleanData.includes(";base64,")) {
            cleanData = cleanData.split(";base64,").pop() || "";
          }
          parts.push({
            inlineData: {
              mimeType: attach.mimeType || "application/octet-stream",
              data: cleanData,
            }
          });
        });
      }
      
      return {
        role: m.role === "user" ? "user" : "model",
        parts,
      };
    });

    const response = await generateContentWithFallback({
      contents,
      config: {
        systemInstruction: systemInstruction || "You are an expert student advisor helping a student with interactive tasks.",
        temperature: 0.7,
      },
    });

    res.json({ text: response.text });
  } catch (error: any) {
    console.error("Error in AI dialogue execution:", error);
    res.status(500).json({ error: error.message || "An error occurred with the AI model" });
  }
});

// Flashcard Generator Endpoint
app.post("/api/generate-flashcards", async (req, res) => {
  try {
    const { topic, customNotes } = req.body;

    if (!topic && !customNotes) {
      res.status(400).json({ error: "Topic or custom notes is required" });
      return;
    }

    const inputData = customNotes 
      ? `Notes provided: ${customNotes}` 
      : `Topic requested: ${topic}`;

    const prompt = `Generate exactly 5 memory-boosting flashcards for a student. ${inputData}. 
Provide the front (a targeted concise question or core concept) and the back (the detailed yet crystal-clear explanation or answer), along with a short 1-word or 2-word subject category (e.g. Physics, History).`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction: "You are a professional retrieval-practice designer. Keep flashcard cards highly actionable and scientifically accurate.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          description: "List of produced flashcards",
          items: {
            type: Type.OBJECT,
            properties: {
              front: { type: Type.STRING, description: "The term, question or concept prompt" },
              back: { type: Type.STRING, description: "The definition, answer or core explanation" },
              category: { type: Type.STRING, description: "1-2 word classification, e.g. chemistry" }
            },
            required: ["front", "back", "category"]
          }
        }
      }
    });

    const parsed = JSON.parse(response.text || "[]");
    res.json({ flashcards: parsed });
  } catch (error: any) {
    console.error("Error generating flashcards:", error);
    res.status(500).json({ error: error.message || "Failed to generate dynamic study cards" });
  }
});

// Interactive Diagnostic Quiz Generator
app.post("/api/generate-quiz", async (req, res) => {
  try {
    const { concept, level } = req.body;

    if (!concept) {
      res.status(400).json({ error: "A quiz subject concept is required" });
      return;
    }

    const difficulty = level || "Intermediate";
    const prompt = `Design a comprehensive diagnostic quiz comprising exactly 3 diverse multiple-choice questions on: "${concept}". Difficulty level: ${difficulty}.
For each question, draft 4 distinctive plausible choices option lines. Specifiy the zero-based index of the single correct answer, and compose a short pedagogical explanation resolving why that option is correct and why the others are distractors.`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction: "You are an adaptive educational evaluator. Ensure distractors are robust and diagnostic. Do not reference layout in text.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          description: "Syllabic set of multiple choice questions",
          items: {
            type: Type.OBJECT,
            properties: {
              question: { type: Type.STRING, description: "The quiz question content" },
              options: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: "Exactly four choices"
              },
              answerIndex: { type: Type.INTEGER, description: "Correct option index from 0 to 3" },
              explanation: { type: Type.STRING, description: "Pedagogical summary of the core concept and correct option." }
            },
            required: ["question", "options", "answerIndex", "explanation"]
          }
        }
      }
    });

    const parsed = JSON.parse(response.text || "[]");
    res.json({ quiz: parsed });
  } catch (error: any) {
    console.error("Error creating quiz:", error);
    res.status(500).json({ error: error.message || "Failed to construct the requested quiz." });
  }
});

// AI Goal Roadmap Checklist breakdown
app.post("/api/breakdown-tasks", async (req, res) => {
  try {
    const { goal } = req.body;

    if (!goal) {
      res.status(400).json({ error: "Goal content is required" });
      return;
    }

    const prompt = `I want to cover or master the study goal: "${goal}". Break this down into exactly 4-5 incremental, bite-sized chronological task items that I can complete sequentially to capture the understanding fully. Keep each task descriptive and short (10-15 words).`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction: "You are a methodical curriculum planner. Break objectives down into logical, chronological study actions.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: { type: Type.STRING }
        }
      }
    });

    const parsed = JSON.parse(response.text || "[]");
    res.json({ tasks: parsed });
  } catch (error: any) {
    console.error("Error breaking down roadmap:", error);
    res.status(500).json({ error: error.message || "Failed to decompose educational goal." });
  }
});

// Explain Simplifier & Smart Summarizer
app.post("/api/explain-concept", async (req, res) => {
  try {
    const { text, mode } = req.body; // modes: "eli5" (Explain like I'm 5), "detailed", "formulas", "summary"

    if (!text) {
      res.status(400).json({ error: "Text to explain is required" });
      return;
    }

    let instruction = "";
    if (mode === "eli5") {
      instruction = "Explain this complex concept completely but using extremely simple analogies suitable for a 5-year old or an absolute beginner. Use warm, highly readable explanations, and simple visual metaphors.";
    } else if (mode === "formulas") {
      instruction = "Analyze the text and extract or deduce all core mathematical formulas, key chemical equations, or scientific laws. Present each with a brief 1-line definition. If no equations exist, present core rules or logical definitions.";
    } else if (mode === "detailed") {
      instruction = "Provide a comprehensive academic breakdown, detailing historical context, structural applications, nuances, and edge cases.";
    } else {
      instruction = "Provide a high-quality summary. Detail the primary thesis, list exactly 3-4 bulleted takeaways, and a concise conclusion.";
    }

    const prompt = `Deconstruct and explain the following material:\n\n${text}`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction: `You are an exceptional science communicator and private academic coach. ${instruction}`,
        temperature: 0.6,
      }
    });

    res.json({ explanation: response.text });
  } catch (error: any) {
    console.error("Error explaining concepts:", error);
    res.status(500).json({ error: error.message || "Failed to simplify content." });
  }
});

// -------------------------------------------------------------
// VITE OR STATIC RUNTIME MIDDLEWARE
// -------------------------------------------------------------
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[DeskMate Server] Running on http://localhost:${PORT}`);
  });
}

startServer();
