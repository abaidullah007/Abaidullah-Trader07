import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const PORT = 3000;

// Helper to sanitize Gemini response JSON
function parseGeminiJson(rawText: string) {
  let cleaned = rawText.trim();
  if (cleaned.startsWith('```json')) {
    cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
  } else if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
  }
  try {
    return JSON.parse(cleaned);
  } catch {
    const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
    throw new Error('Failed to parse JSON response from AI model.');
  }
}

const SYSTEM_PROMPT = `
# ABAIDULLAH TRADER ANALYSIS — MASTER SYSTEM PROMPT

You are “Abaidullah Trader Analysis”, an advanced scalping chart-analysis AI assistant.
Your primary job is to analyze TradingView chart screenshots uploaded by the user and identify the highest-quality potential scalping setup with Entry, Stop Loss, Take Profit and a confidence score.

1. IMAGE INPUT
The user provides TradingView chart screenshots. Analyze ALL uploaded screenshots together.
Preferred timeframe hierarchy: 15M → 5M → 3M → 1M.
Never pretend to see information that is not visible. If important information is missing, clearly state what is missing.

2. PRIMARY OBJECTIVE
Your objective is NOT to generate many trades. Find ONE BEST SCALPING SETUP, if a high-quality setup exists.
Prioritize:
1. Market structure
2. Liquidity (sweeps, stop hunts, equal highs/lows)
3. Trend/bias
4. Support & resistance
5. Supply & demand / Order blocks / Breaker blocks
6. Fair Value Gaps (FVG) / Imbalances
7. BOS — Break of Structure
8. CHoCH/MSS — Change of Character / Market Structure Shift
9. Candlestick confirmation & momentum
10. VWAP/EMA/RSI/MACD or other indicators if visible
11. Risk-to-reward (prefer min ~1:2)
12. Multi-timeframe confluence

3. MARKET ANALYSIS
Classify market condition as one of:
"Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish", "Ranging", "Choppy / Unclear"

4. PERFECT ENTRY RULE
Never provide an entry simply because market is moving.
Prefer: Liquidity sweep → Structure shift → Displacement → Retest → Entry.
If price is too far from ideal entry, say "WAIT FOR RETEST". Do not chase price.
If no high-quality setup exists, return "NO TRADE".

5. BUY/SELL DECISION
Choose strictly ONE: "BUY", "SELL", or "NO TRADE".

6. CONFIDENCE SCORE (0-100%)
Evaluate based on confluence points (HTF bias, structure, liquidity, entry zone, PA, FVG/OB, momentum, R/R, MTF alignment).
90-100%: Exceptional confluence
80-89%: Strong setup
70-79%: Good setup
60-69%: Moderate setup (prefer caution)
Below 60%: Prefer NO TRADE

7. EXECUTION DIRECTIVE
Must be one of: "ENTER NOW", "WAIT FOR RETEST", "WAIT FOR CONFIRMATION", "NO TRADE"

8. RESPONSE FORMAT REQUIREMENTS
You must return a strict JSON object that strictly conforms to the JSON schema provided.
The rawReportMarkdown field inside the JSON MUST contain the exact full formatted text matching this layout:

## ABAIDULLAH TRADER ANALYSIS

**Market:** [asset]
**Timeframes Analyzed:** [list]
**Market Bias:** [bias]

**Final Signal:**
🟢 **BUY** (or 🔴 **SELL** or ⚪ **NO TRADE**)

### TRADE SETUP
**Entry:** [price/zone]
**Stop Loss:** [price]
**Take Profit 1:** [price]
**Take Profit 2:** [price]
**Risk/Reward:** [ratio]
**Confidence:** [XX]%

### WHY THIS TRADE?
1. [Reason 1]
2. [Reason 2]
3. [Reason 3]
4. [Reason 4]

### MARKET STRUCTURE
* Trend: [details]
* BOS/CHoCH: [details]
* Liquidity: [details]
* Support/Resistance: [details]
* Order Block: [details]
* FVG: [details]
* Momentum: [details]

### INVALIDATION
[Explanation of price action that invalidates setup]

### EXECUTION
[ENTER NOW / WAIT FOR RETEST / WAIT FOR CONFIRMATION / NO TRADE]

### RISK WARNING
This is technical chart analysis, not a guaranteed prediction. The confidence score represents setup quality based on uploaded charts and does not guarantee profit.
`;

async function startServer() {
  const app = express();

  // Increase JSON payload limit to handle multi-image base64 uploads
  app.use(express.json({ limit: '50mb' }));

  // Health check endpoint
  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
  });

  // API route for chart analysis
  app.post('/api/analyze-chart', async (req, res) => {
    try {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        return res.status(500).json({
          error: 'GEMINI_API_KEY environment variable is missing. Please configure it in Settings > Secrets.',
        });
      }

      const { images, assetName, accountBalance, riskPercentage, userNotes } = req.body;

      if (!images || !Array.isArray(images) || images.length === 0) {
        return res.status(400).json({ error: 'At least one chart screenshot is required for analysis.' });
      }

      const ai = new GoogleGenAI({ apiKey });

      // Build Gemini contents array with image parts + text prompt
      const parts: any[] = [];

      images.forEach((img: { data: string; mimeType?: string; timeframe?: string; name?: string }, index: number) => {
        let cleanBase64 = img.data;
        if (cleanBase64.includes('base64,')) {
          cleanBase64 = cleanBase64.split('base64,')[1];
        }

        // Ensure supported image MIME type for Gemini Vision
        let validMime = img.mimeType || 'image/png';
        if (validMime.includes('svg')) {
          validMime = 'image/png';
        }

        parts.push({
          inlineData: {
            mimeType: validMime,
            data: cleanBase64,
          },
        });

        parts.push({
          text: `[Uploaded Screenshot #${index + 1}: Timeframe = ${img.timeframe || 'Unspecified'}, File = ${img.name || 'chart.png'}]`,
        });
      });

      let extraContextPrompt = `You have been provided with ${images.length} TradingView chart screenshot(s).`;
      if (assetName) extraContextPrompt += `\nSpecified Asset/Pair: ${assetName}`;
      if (accountBalance) extraContextPrompt += `\nTrader Account Balance: $${accountBalance}`;
      if (riskPercentage) extraContextPrompt += `\nTrader Max Risk Per Trade: ${riskPercentage}%`;
      if (userNotes) extraContextPrompt += `\nTrader Additional Notes / Session Context: "${userNotes}"`;

      extraContextPrompt += `\nPerform a comprehensive multi-timeframe scalping analysis using Smart Money Concepts (SMC) and price action following the Abaidullah Trader Analysis master rules.`;

      parts.push({ text: extraContextPrompt });

      let responseText: string | null = null;
      let lastError: any = null;

      // Candidate models in order of availability and throughput
      const candidateModels = [
        'gemini-3.8-flash',
        'gemini-3.7-flash',
        'gemini-3.6-flash',
        'gemini-3.1-flash-lite',
        'gemini-3.7-pro',
        'gemini-3.6-pro',
      ];

      for (const modelName of candidateModels) {
        try {
          const response = await ai.models.generateContent({
            model: modelName,
            contents: { parts },
            config: {
              systemInstruction: SYSTEM_PROMPT,
              temperature: 0.2, // Low temperature for high precision & consistency
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  market: { type: Type.STRING, description: 'Asset or currency pair identified from chart or user input (e.g. BTC/USDT, EUR/USD)' },
                  timeframesAnalyzed: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: 'List of timeframes identified across screenshots (e.g. ["15M", "5M", "1M"])',
                  },
                  marketBias: {
                    type: Type.STRING,
                    description: 'Market classification: "Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish", "Ranging", or "Choppy / Unclear"',
                  },
                  finalSignal: {
                    type: Type.STRING,
                    description: 'Final trade recommendation: strictly "BUY", "SELL", or "NO TRADE"',
                  },
                  tradeSetup: {
                    type: Type.OBJECT,
                    properties: {
                      entry: { type: Type.STRING, description: 'Entry price level or zone (e.g. 96,480 - 96,520 Order Block Retest)' },
                      entryPriceNum: { type: Type.NUMBER, description: 'Numeric representation of entry price if discernible, for calculations' },
                      stopLoss: { type: Type.STRING, description: 'Stop loss level and invalidation reason (e.g. 96,180 below recent swing low)' },
                      stopLossNum: { type: Type.NUMBER, description: 'Numeric representation of stop loss price' },
                      takeProfit1: { type: Type.STRING, description: 'First Take Profit target' },
                      takeProfit1Num: { type: Type.NUMBER, description: 'Numeric representation of TP1 price' },
                      takeProfit2: { type: Type.STRING, description: 'Second Take Profit target' },
                      takeProfit2Num: { type: Type.NUMBER, description: 'Numeric representation of TP2 price' },
                      riskRewardRatio: { type: Type.STRING, description: 'Calculated Risk to Reward ratio (e.g. 1:2.4)' },
                      confidenceScore: { type: Type.NUMBER, description: 'Confidence score percentage from 0 to 100' },
                    },
                    required: ['entry', 'stopLoss', 'takeProfit1', 'takeProfit2', 'riskRewardRatio', 'confidenceScore'],
                  },
                  execution: {
                    type: Type.STRING,
                    description: 'Execution directive: "ENTER NOW", "WAIT FOR RETEST", "WAIT FOR CONFIRMATION", or "NO TRADE"',
                  },
                  reasonsForTrade: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: 'Key core justifications for the trade setup under WHY THIS TRADE?',
                  },
                  marketStructure: {
                    type: Type.OBJECT,
                    properties: {
                      trend: { type: Type.STRING },
                      bosChoch: { type: Type.STRING },
                      liquidity: { type: Type.STRING },
                      supportResistance: { type: Type.STRING },
                      orderBlock: { type: Type.STRING },
                      fvg: { type: Type.STRING },
                      momentum: { type: Type.STRING },
                    },
                    required: ['trend', 'bosChoch', 'liquidity', 'supportResistance', 'orderBlock', 'fvg', 'momentum'],
                  },
                  confluenceScores: {
                    type: Type.OBJECT,
                    properties: {
                      higherTimeframeBias: { type: Type.NUMBER, description: 'Score out of 15' },
                      marketStructure: { type: Type.NUMBER, description: 'Score out of 15' },
                      liquidity: { type: Type.NUMBER, description: 'Score out of 15' },
                      entryZoneQuality: { type: Type.NUMBER, description: 'Score out of 15' },
                      priceActionConfirmation: { type: Type.NUMBER, description: 'Score out of 10' },
                      fvgOrderBlock: { type: Type.NUMBER, description: 'Score out of 10' },
                      indicatorMomentum: { type: Type.NUMBER, description: 'Score out of 5' },
                      riskReward: { type: Type.NUMBER, description: 'Score out of 10' },
                      multiTimeframeAlignment: { type: Type.NUMBER, description: 'Score out of 5' },
                    },
                  },
                  invalidation: { type: Type.STRING, description: 'Exact price action or structural invalidation condition' },
                  riskWarning: { type: Type.STRING, description: 'Standard educational risk warning and disclaimer' },
                  rawReportMarkdown: { type: Type.STRING, description: 'Full formatted Markdown text report matching Abaidullah Trader Analysis template' },
                },
                required: [
                  'market',
                  'timeframesAnalyzed',
                  'marketBias',
                  'finalSignal',
                  'tradeSetup',
                  'execution',
                  'reasonsForTrade',
                  'marketStructure',
                  'invalidation',
                  'riskWarning',
                  'rawReportMarkdown',
                ],
              },
            },
          });

          if (response.text) {
            responseText = response.text;
            console.log(`Successfully generated chart analysis using model: ${modelName}`);
            break;
          }
        } catch (err: any) {
          lastError = err;
          const errStr = err?.message || JSON.stringify(err);
          // If 503 high demand or temporary capacity spike, immediately try next candidate model
          if (errStr.includes('503') || errStr.includes('UNAVAILABLE') || errStr.includes('high demand')) {
            console.log(`Model ${modelName} busy (503), switching to next fallback model...`);
          } else {
            console.warn(`Model ${modelName} invocation failed:`, err?.message || err);
          }
        }
      }

      if (!responseText) {
        const errorMsg = lastError?.message || (typeof lastError === 'string' ? lastError : 'Unknown error');
        if (errorMsg.includes('503') || errorMsg.includes('UNAVAILABLE') || errorMsg.includes('high demand')) {
          return res.status(503).json({
            error: 'AI analysis engine is currently experiencing high demand. Please try again in a few moments.',
          });
        }
        return res.status(500).json({
          error: errorMsg || 'Failed to receive analysis response from AI.',
        });
      }

      const parsedData = parseGeminiJson(responseText);

      // Position sizing calculation if account balance & risk % supplied
      let positionSizing = undefined;
      if (accountBalance && riskPercentage) {
        const riskAmountDollars = (accountBalance * riskPercentage) / 100;
        positionSizing = {
          accountBalance,
          riskPercentage,
          riskAmountDollars,
          pipRiskValue: parsedData.tradeSetup.entryPriceNum && parsedData.tradeSetup.stopLossNum
            ? Math.abs(parsedData.tradeSetup.entryPriceNum - parsedData.tradeSetup.stopLossNum).toFixed(4)
            : 'N/A',
          lotOrContractAdvice: `Risk amount set to $${riskAmountDollars.toFixed(2)} (${riskPercentage}% of $${accountBalance.toLocaleString()}). Calculate standard lot/contract size based on exchange pip value and SL distance.`,
        };
      }

      return res.json({
        success: true,
        analysis: {
          id: `analysis-${Date.now()}`,
          timestamp: Date.now(),
          ...parsedData,
          positionSizing,
        },
      });
    } catch (err: any) {
      console.error('Error in /api/analyze-chart:', err);
      return res.status(500).json({
        error: err?.message || 'Failed to analyze chart screenshots. Please try again.',
      });
    }
  });

  // Catch-all 404 for unmatched /api/* routes to prevent Vite from returning index.html
  app.all('/api/*', (req, res) => {
    res.status(404).json({ error: `API endpoint ${req.method} ${req.path} not found.` });
  });

  // Global API Error handler for payload limits or JSON errors
  app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    if (req.path.startsWith('/api') || req.url.startsWith('/api')) {
      console.error('API Middleware Error:', err);
      return res.status(err.status || 500).json({
        error: err.type === 'entity.too.large' 
          ? 'Uploaded chart images exceed size limit. Please upload compressed or smaller screenshots.' 
          : (err.message || 'Server API error'),
      });
    }
    next(err);
  });

  // Vite middleware setup
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
        watch: process.env.DISABLE_HMR === 'true' ? null : {},
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to initialize server:', err);
});
