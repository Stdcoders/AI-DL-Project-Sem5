# tools/improvement_tools.py
import traceback
from langchain_core.tools import tool
from typing import Dict

# Import your DeepSeekLLM wrapper (the file you posted)
# Make sure DeepSeekLLM class is importable from its module (e.g., deepseek_llm.py)
from deepseek_llm import DeepSeekLLM

@tool
def improve_analysis(user_question: str, original_answer: str, critique: str) -> Dict:
    """
    Uses DeepSeekLLM to improve an analysis answer based on critic feedback.
    Returns a dict with key 'improved_answer'.
    """

    prompt = f"""
You are an expert data analyst and editor. Improve the given answer using the critique.
User question:
{user_question}

Original answer:
{original_answer}

Critique (fix these issues):
{critique}

INSTRUCTIONS:
- Produce a corrected, concise answer that directly answers the user's question.
- If calculations are needed, provide numeric results or explain how to compute them.
- Keep it factual and do not mention internal critique/process.
- If you cannot improve, return the original answer verbatim.

Provide only the improved answer (no extra commentary).
"""

    # Initialize DeepSeek LLM (adjust model_name and temperature as needed)
    try:
        # Use a reasonably creative but conservative temperature
        llm = DeepSeekLLM(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            temperature=0.2,
            max_tokens=1024,
            use_inference_client=True,
            provider="nscale"
        )

        if not llm.is_available():
            # If model is not available, attempt a single request anyway
            print("⚠️ DeepSeekLLM reported unavailable. Attempting request as fallback...")

        improved = llm.generate(prompt)
        if improved is None:
            raise RuntimeError("DeepSeekLLM returned no text.")

        improved_text = improved.strip()
        # basic safety: if result too short, fall back
        if len(improved_text) < 10:
            print("⚠️ Improved text too short, falling back to original answer.")
            return {"improved_answer": original_answer}

        return {"improved_answer": improved_text}

    except Exception as e:
        print("❌ improve_analysis: DeepSeekLLM failed:", e)
        traceback.print_exc()
        # Fallback: return the original answer unchanged
        return {"improved_answer": original_answer}
