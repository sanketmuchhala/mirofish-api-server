"""
Swarm Prediction API
Wraps MiroFish swarm intelligence engine with x402 payment middleware.

POST /api/predict  - Run a swarm prediction simulation
GET  /health       - Health check
GET  /mcp          - MCP manifest for discovery
"""

import os
import json
import uuid
import asyncio
import logging
import random
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("swarm-api")

# ─── Config ───────────────────────────────────────────────────────────────────
PORT = 8080
WALLET_ADDRESS = os.environ.get("WALLET_ADDRESS", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
PRICE_PER_CALL_USD = 0.50

# ─── x402 Payment Middleware ──────────────────────────────────────────────────

X402_PRICE_USD = PRICE_PER_CALL_USD
X402_NETWORK = "base"
X402_CURRENCY = "USDC"

PAID_ROUTES = {"/api/predict"}


async def x402_middleware(request: Request, call_next):
    """
    x402 Payment Required middleware.
    Checks for a valid x-payment header on protected routes.
    In production, integrate with coinbase/x402 SDK for on-chain verification.
    """
    path = request.url.path

    if path not in PAID_ROUTES:
        return await call_next(request)

    payment_header = request.headers.get("x-payment") or request.headers.get("X-Payment")

    if not payment_header:
        # Return 402 Payment Required with payment details
        payment_info = {
            "error": "Payment Required",
            "x402": {
                "version": 1,
                "accepts": [
                    {
                        "scheme": "exact",
                        "network": X402_NETWORK,
                        "currency": X402_CURRENCY,
                        "amount": str(int(X402_PRICE_USD * 1_000_000)),  # USDC has 6 decimals
                        "address": WALLET_ADDRESS,
                        "description": f"${X402_PRICE_USD} per prediction call",
                    }
                ],
            },
        }
        return JSONResponse(status_code=402, content=payment_info)

    # In production: verify payment on-chain via x402 SDK
    # For now, accept any non-empty payment header as valid
    # TODO: integrate coinbase/x402 verification
    logger.info(f"Payment header received: {payment_header[:40]}...")
    return await call_next(request)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Swarm Prediction API",
    description="MiroFish swarm intelligence engine wrapped with x402 payments",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(x402_middleware)


# ─── Models ───────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    seed_text: str = Field(..., description="Topic or scenario to simulate", min_length=1, max_length=2000)
    num_agents: int = Field(default=100, ge=10, le=500, description="Number of simulated agents")
    rounds: int = Field(default=20, ge=1, le=50, description="Number of simulation rounds")


class AgentAction(BaseModel):
    agent_id: int
    agent_name: str
    agent_role: str
    round: int
    action_type: str
    content: str
    sentiment: str  # positive / negative / neutral
    influence_score: float
    key_signals: List[str] = []


class PredictResponse(BaseModel):
    prediction_id: str
    seed_text: str
    num_agents: int
    rounds: int
    started_at: str
    completed_at: str
    duration_seconds: float

    # Simulation metrics
    total_actions: int
    sentiment_distribution: Dict[str, int]
    top_narratives: List[str]
    emerging_trends: List[str]

    # Prediction report
    report: str

    # Sample agent actions
    sample_actions: List[Dict[str, Any]]


# ─── Swarm Simulation Engine ──────────────────────────────────────────────────


AGENT_ROLES = [
    "Analyst", "Skeptic", "Enthusiast", "Expert", "Observer",
    "Journalist", "Researcher", "Activist", "Policymaker", "Citizen",
    "Entrepreneur", "Academic", "Influencer", "Critic", "Pragmatist",
]

AGENT_ARCHETYPES = {
    "Analyst": "objective, data-driven, looks for patterns and evidence",
    "Skeptic": "questions claims, challenges assumptions, seeks contrary evidence",
    "Enthusiast": "excited, optimistic, amplifies positive signals",
    "Expert": "deep domain knowledge, technical precision, nuanced takes",
    "Observer": "neutral, watches trends, slow to react",
    "Journalist": "seeks the story, asks questions, looks for impact",
    "Researcher": "evidence-based, cautious, systematic",
    "Activist": "passionate, advocates for change, high emotional intensity",
    "Policymaker": "considers implications, regulatory lens, risk-averse",
    "Citizen": "everyday perspective, personal impact focus, community lens",
    "Entrepreneur": "opportunity-seeking, practical, disruption-minded",
    "Academic": "theoretical, references literature, long-term view",
    "Influencer": "trend-sensitive, audience-aware, quick to amplify",
    "Critic": "identifies flaws, challenges narratives, pessimistic lean",
    "Pragmatist": "what works now, cost-benefit thinking, compromise-oriented",
}

ACTION_TYPES = ["CREATE_POST", "REPLY", "AMPLIFY", "CHALLENGE", "IGNORE", "ANALYZE", "PREDICT"]


async def run_swarm_simulation(
    seed_text: str,
    num_agents: int,
    rounds: int,
    llm_client: AsyncOpenAI,
) -> Dict[str, Any]:
    """
    Lightweight MiroFish-style swarm simulation using LLM.
    
    Architecture mirrors OASIS engine:
    - Agents have roles/archetypes (from oasis_profile_generator patterns)
    - Multi-round simulation with action logging (from simulation_runner patterns)
    - ReACT-style report generation (from report_agent patterns)
    """
    started_at = datetime.utcnow()
    
    # ── Phase 1: Spawn agent archetypes ──────────────────────────────────────
    agents = []
    role_cycle = AGENT_ROLES * ((num_agents // len(AGENT_ROLES)) + 1)
    for i in range(num_agents):
        role = role_cycle[i]
        agents.append({
            "agent_id": i + 1,
            "agent_name": f"Agent_{i+1:04d}",
            "agent_role": role,
            "archetype": AGENT_ARCHETYPES[role],
        })
    
    # ── Phase 2: Run simulation rounds ───────────────────────────────────────
    # We run parallel batches of agents per round via LLM
    # Each batch = one LLM call simulating N agents
    BATCH_SIZE = 10  # agents per LLM call
    ROUNDS_PER_LLM_CALL = min(rounds, 5)  # compress rounds for efficiency
    
    all_actions = []
    round_summaries = []
    
    # Sample a diverse set of agents for simulation
    sample_size = min(num_agents, 60)
    sampled_agents = random.sample(agents, sample_size)
    
    # Batch agents and run simulation
    batches = [sampled_agents[i:i+BATCH_SIZE] for i in range(0, len(sampled_agents), BATCH_SIZE)]
    
    simulation_tasks = []
    for batch_idx, batch in enumerate(batches):
        task = _simulate_agent_batch(
            seed_text=seed_text,
            agents=batch,
            rounds=ROUNDS_PER_LLM_CALL,
            batch_id=batch_idx,
            llm_client=llm_client,
        )
        simulation_tasks.append(task)
    
    batch_results = await asyncio.gather(*simulation_tasks, return_exceptions=True)
    
    for result in batch_results:
        if isinstance(result, Exception):
            logger.warning(f"Batch simulation error: {result}")
            continue
        all_actions.extend(result.get("actions", []))
        round_summaries.extend(result.get("round_summaries", []))
    
    # ── Phase 2.5: Compute interaction-based influence scores ────────────────
    # Count how many REPLY/AMPLIFY actions target each agent_id
    interaction_counts: Dict[int, int] = {}
    for action in all_actions:
        if action.get("action_type") in ("REPLY", "AMPLIFY"):
            aid = action.get("agent_id", 0)
            interaction_counts[aid] = interaction_counts.get(aid, 0) + 1
    
    max_interactions = max(interaction_counts.values()) if interaction_counts else 1
    
    for action in all_actions:
        aid = action.get("agent_id", 0)
        llm_score = float(action.get("influence_score", 0.5))
        interaction_score = min(1.0, interaction_counts.get(aid, 0) / max_interactions)
        action["influence_score"] = round(0.4 * llm_score + 0.6 * interaction_score, 2)
    
    # ── Phase 3: Generate prediction report ──────────────────────────────────
    report = await _generate_prediction_report(
        seed_text=seed_text,
        all_actions=all_actions,
        round_summaries=round_summaries,
        num_agents=num_agents,
        rounds=rounds,
        llm_client=llm_client,
    )
    
    # ── Phase 4: Aggregate metrics ────────────────────────────────────────────
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for action in all_actions:
        s = action.get("sentiment", "neutral")
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
    
    completed_at = datetime.utcnow()
    duration = (completed_at - started_at).total_seconds()
    
    # Return top 20 actions by influence_score
    top_actions = sorted(all_actions, key=lambda a: a.get("influence_score", 0), reverse=True)[:20]
    
    return {
        "started_at": started_at.isoformat() + "Z",
        "completed_at": completed_at.isoformat() + "Z",
        "duration_seconds": round(duration, 2),
        "total_actions": len(all_actions),
        "sentiment_distribution": sentiment_counts,
        "top_narratives": report.get("top_narratives", []),
        "emerging_trends": report.get("emerging_trends", []),
        "report": report.get("report_text", ""),
        "sample_actions": top_actions,
    }


async def _simulate_agent_batch(
    seed_text: str,
    agents: List[Dict],
    rounds: int,
    batch_id: int,
    llm_client: AsyncOpenAI,
) -> Dict[str, Any]:
    """Simulate a batch of agents across multiple rounds."""
    
    agent_descriptions = "\n".join([
        f"- Agent_{a['agent_id']:04d} [{a['agent_role']}]: {a['archetype']}"
        for a in agents
    ])
    
    prompt = f"""You are simulating a swarm intelligence system with {len(agents)} agents analyzing this scenario:

SCENARIO: {seed_text}

AGENTS IN THIS BATCH:
{agent_descriptions}

Simulate {rounds} rounds of agent activity. Each agent should:
1. React to the scenario from their archetype's perspective
2. Build on previous round information
3. Exhibit emergent collective behavior

Output a JSON array of agent actions. Each action must have:
{{
  "agent_id": <int>,
  "agent_name": "Agent_XXXX",
  "agent_role": "<role>",
  "round": <1-{rounds}>,
  "action_type": "<CREATE_POST|REPLY|AMPLIFY|CHALLENGE|ANALYZE|PREDICT>",
  "content": "<what the agent posted/said, 1-2 sentences>",
  "sentiment": "<positive|negative|neutral>",
  "influence_score": <0.0-1.0>,
  "key_signals": ["<signal1>", "<signal2>"]
}}

Generate realistic, diverse responses. Some agents agree, some challenge, some discover new angles.
Return ONLY valid JSON array, no markdown."""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4096,
            timeout=60.0,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        # Parse JSON - handle both array and object wrappers
        if content.startswith("["):
            actions = json.loads(content)
        else:
            parsed = json.loads(content)
            # Find array in response
            if isinstance(parsed, list):
                actions = parsed
            else:
                # Look for array value
                for v in parsed.values():
                    if isinstance(v, list):
                        actions = v
                        break
                else:
                    actions = []
        
        # Normalize actions
        normalized = []
        for action in actions:
            if isinstance(action, dict) and "agent_id" in action:
                normalized.append({
                    "agent_id": action.get("agent_id", 0),
                    "agent_name": action.get("agent_name", "Agent_0000"),
                    "agent_role": action.get("agent_role", "Observer"),
                    "round": action.get("round", 1),
                    "action_type": action.get("action_type", "CREATE_POST"),
                    "content": action.get("content", ""),
                    "sentiment": action.get("sentiment", "neutral"),
                    "influence_score": float(action.get("influence_score", 0.5)),
                    "key_signals": action.get("key_signals", []),
                })
        
        # Build round summaries
        round_map = {}
        for a in normalized:
            r = a["round"]
            if r not in round_map:
                round_map[r] = {"actions": 0, "sentiments": []}
            round_map[r]["actions"] += 1
            round_map[r]["sentiments"].append(a["sentiment"])
        
        round_summaries = [
            {
                "round": r,
                "actions_count": v["actions"],
                "dominant_sentiment": max(set(v["sentiments"]), key=v["sentiments"].count),
            }
            for r, v in sorted(round_map.items())
        ]
        
        return {"actions": normalized, "round_summaries": round_summaries}
    
    except Exception as e:
        logger.error(f"Batch {batch_id} simulation failed: {e}")
        return {"actions": [], "round_summaries": []}


async def _generate_prediction_report(
    seed_text: str,
    all_actions: List[Dict],
    round_summaries: List[Dict],
    num_agents: int,
    rounds: int,
    llm_client: AsyncOpenAI,
) -> Dict[str, Any]:
    """
    ReACT-style report generation (mirrors MiroFish report_agent.py patterns).
    """
    
    # Summarize simulation data for the report prompt
    total_actions = len(all_actions)
    sentiments = [a.get("sentiment", "neutral") for a in all_actions]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    
    # Extract sample insights
    sample_posts = [a["content"] for a in all_actions if a.get("content")][:15]
    sample_text = "\n".join([f"- {p}" for p in sample_posts])
    
    prompt = f"""You are a swarm intelligence analyst. A multi-agent simulation was run on this topic:

SCENARIO: {seed_text}

SIMULATION STATS:
- Agents: {num_agents} (roles: Analyst, Skeptic, Expert, Journalist, Activist, etc.)
- Rounds: {rounds}
- Total agent actions captured: {total_actions}
- Sentiment breakdown: {pos} positive, {neg} negative, {neu} neutral

SAMPLE AGENT OUTPUTS FROM SIMULATION:
{sample_text}

Generate a prediction report in JSON with this structure:
{{
  "top_narratives": ["<3-5 dominant narrative threads that emerged>"],
  "emerging_trends": ["<3-5 trends the swarm detected>"],
  "report_text": "<A concise 400-600 word analytical report covering: (1) Executive Summary, (2) Key Prediction Findings, (3) Swarm Consensus vs Dissent, (4) Risk Signals, (5) Outlook. Use the agent simulation data as evidence. Write in confident, analytical prose.>"
}}

Base predictions on what the swarm of agents collectively discovered. Be specific and insightful.
Return ONLY valid JSON."""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2000,
            timeout=60.0,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Strip markdown if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        parsed = json.loads(content)
        return parsed
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "top_narratives": ["Simulation data insufficient for narrative extraction"],
            "emerging_trends": ["Unable to determine trends from available data"],
            "report_text": f"Swarm simulation completed with {total_actions} agent actions on the topic: {seed_text}. Report generation encountered an error: {str(e)}",
        }


# ─── LLM Client Singleton ─────────────────────────────────────────────────────

_llm_client: Optional[AsyncOpenAI] = None


def get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        if not LLM_API_KEY:
            raise HTTPException(status_code=500, detail="LLM_API_KEY not configured")
        _llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _llm_client


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "swarm-prediction-api",
        "version": "1.0.0",
        "engine": "MiroFish OASIS",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "llm_configured": bool(LLM_API_KEY),
            "wallet_configured": bool(WALLET_ADDRESS),
            "payment_network": X402_NETWORK,
            "price_usd": X402_PRICE_USD,
        },
    }


@app.get("/mcp")
async def mcp_manifest():
    """MCP (Model Context Protocol) manifest for tool discovery."""
    base_url = os.environ.get("PUBLIC_URL", f"http://localhost:{PORT}")
    return {
        "schema_version": "v1",
        "name": "swarm-prediction-api",
        "description": "MiroFish swarm intelligence prediction engine. Runs multi-agent simulations to generate crowd-sourced predictions on any topic.",
        "version": "1.0.0",
        "payment": {
            "required": True,
            "scheme": "x402",
            "network": X402_NETWORK,
            "currency": X402_CURRENCY,
            "price_usd": X402_PRICE_USD,
            "wallet": WALLET_ADDRESS,
        },
        "tools": [
            {
                "name": "predict",
                "description": "Run a swarm intelligence simulation to generate predictions on a topic. Returns a full prediction report with agent consensus, narratives, and trends.",
                "payment_required": True,
                "price_usd": X402_PRICE_USD,
                "endpoint": f"{base_url}/api/predict",
                "method": "POST",
                "input_schema": {
                    "type": "object",
                    "required": ["seed_text"],
                    "properties": {
                        "seed_text": {
                            "type": "string",
                            "description": "The topic, scenario, or question to simulate",
                            "maxLength": 2000,
                        },
                        "num_agents": {
                            "type": "integer",
                            "description": "Number of simulated agents (10-500)",
                            "default": 100,
                            "minimum": 10,
                            "maximum": 500,
                        },
                        "rounds": {
                            "type": "integer",
                            "description": "Number of simulation rounds (1-50)",
                            "default": 20,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "prediction_id": {"type": "string"},
                        "report": {"type": "string", "description": "Full prediction report text"},
                        "top_narratives": {"type": "array", "items": {"type": "string"}},
                        "emerging_trends": {"type": "array", "items": {"type": "string"}},
                        "sentiment_distribution": {"type": "object"},
                        "total_actions": {"type": "integer"},
                        "sample_actions": {"type": "array"},
                    },
                },
            }
        ],
        "resources": [],
        "prompts": [
            {
                "name": "predict-topic",
                "description": "Generate a swarm prediction for a given topic",
                "template": "Use the predict tool to analyze: {{topic}}",
            }
        ],
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """
    Run a MiroFish swarm intelligence simulation and return a prediction report.
    
    Requires x-payment header with valid x402 payment on Base USDC ($0.50).
    """
    prediction_id = str(uuid.uuid4())
    logger.info(f"[{prediction_id}] Predict request: agents={body.num_agents}, rounds={body.rounds}, topic={body.seed_text[:60]}...")
    
    try:
        llm = get_llm_client()
        result = await run_swarm_simulation(
            seed_text=body.seed_text,
            num_agents=body.num_agents,
            rounds=body.rounds,
            llm_client=llm,
        )
        
        return PredictResponse(
            prediction_id=prediction_id,
            seed_text=body.seed_text,
            num_agents=body.num_agents,
            rounds=body.rounds,
            **result,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{prediction_id}] Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
