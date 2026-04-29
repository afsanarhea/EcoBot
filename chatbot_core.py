# chatbot_core.py

import os
import json
from typing import Dict, Any


class EcobotInterface:
    """
    A simple interface for handling chatbot queries.
    You can plug in:
        - LLM (OpenAI, HF, etc.)
        - Vector database (FAISS, Chroma, Pinecone)
        - Computer vision findings
    For now, answers are generated from a small built-in knowledge base.
    """

    def __init__(self):
        # Example small knowledge base
        self.knowledge_base = {
            "aloe vera": [
                "Aloe Vera often dies from overwatering. Their roots rot easily if water stays too long.",
                "Ensure the soil fully dries between watering.",
                "Use well-draining cactus/mix soil."
            ],
            "snake plant": [
                "Snake plants die from prolonged fungal infections, too much watering, or lack of light.",
                "Water only when soil is fully dry.",
                "Provide indirect bright light."
            ],
            "greenhouse maintenance": [
                "Ensure stable temperature, proper humidity, and airflow.",
                "Regularly check for pests and mold.",
                "Do not overwater plants."
            ]
        }

        print("[EcoBot] Loaded successfully.")

    # ------------------------------------------------------------------
    # 🔹 Simple retrieval function (replace with vector search later)
    # ------------------------------------------------------------------
    def _retrieve_best_match(self, user_input: str) -> str:
        """Return the knowledge entry most closely related to the query."""
        user_input = user_input.lower()
        for key, values in self.knowledge_base.items():
            if key in user_input:
                return "\n".join(values)

        # Default fallback answer
        return (
            "I couldn't find specific plant information related to your question, "
            "but maintaining proper sunlight, watering cycles, and soil quality is essential for greenhouse plant health."
        )

    # ------------------------------------------------------------------
    # 🔹 Public chatbot API
    # ------------------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        """
        Takes user question → returns dict:
        {
            "result": "...bot answer..."
        }
        """

        print(f"[EcoBot] Received query: {question}")

        answer = self._retrieve_best_match(question)

        return {
            "result": answer
        }