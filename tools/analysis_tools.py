import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# LangChain import for creating the tool
from langchain_core.tools import tool

# --- Import all required specialized modules ---
try:
    from nodes.llm_question_classifier import LLMQuestionClassifier, QuestionIntent
    from nodes.hybrid_calculation_engine import HybridCalculationEngine
    from gemma_llm import GemmaLLM
    from groq_llm import GroqLLM
except ImportError:
    print("FATAL ERROR: Could not import required modules. Please ensure all tool files are in the correct path.")
    # Provide minimal fallbacks for basic script loading
    class LLMQuestionClassifier:
        def analyze_question_intent(self, q, df_context): return type('obj', (object,), {'intent': 'descriptive', 'confidence': 0, 'reasoning': 'Fallback'})()
    class HybridCalculationEngine:
        def analyze_with_calculations(self, q, df): return "Hybrid Engine not available."
    class GemmaLLM:
        def __init__(self, **kwargs): pass
        def is_available(self): return False
        def __call__(self, prompt): return "Gemma LLM not available."
    class QuestionIntent:
        STATISTICAL, COMPARATIVE, ANALYTICAL, DESCRIPTIVE = "statistical", "comparative", "analytical", "descriptive"

# Optional Plotly imports for visualization
try:
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    px, pio = None, None
    print("Warning: Plotly not found. Visualization will be disabled.")

# ================== THE LANGCHAIN TOOL ==================
@tool
def analyze_user_question(question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Answers a user's question by classifying intent with an LLM and routing to specialized engines
    for calculation, visualization, and descriptive answers.
    """
    print(f"🔬 Tool 'analyze_user_question' starting for question: '{question}'")
    try:
        engine = UnifiedAnalysisEngine()
        result = engine.analyze_question(question, df, dataset_name)
        print("✅ Tool 'analyze_user_question' finished successfully.")
        return result
    except Exception as e:
        print(f"❌ Error in 'analyze_user_question' tool: {e}")
        return {"error": str(e), "answer": f"An error occurred during analysis: {e}"}

# ========== INTERNAL CLASSES AND LOGIC ==========

class InsightAgent:
    """
    A specialized agent using LLMs for two key roles:
    1.  Generating qualitative, descriptive answers (with Gemma).
    2.  Generating dynamic, intelligent data visualizations (with Groq/Gemma).
    """
    def __init__(self):
        self.desc_llm = GemmaLLM(model_name="google/gemma-2-9b-it", temperature=0.2, max_tokens=1500)
        self.viz_llm = GroqLLM(model_name="openai/gpt-oss-120b", temperature=0.0, max_tokens=300)

    def generate_descriptive_answer(self, question: str, df: pd.DataFrame, dataset_name: str) -> str:
        # (This method remains the same as the previous version)
        if not self.desc_llm.is_available(): return "Insight LLM is not available."
        context = (f"Dataset: '{dataset_name}'\nShape: {df.shape}\nColumns: {df.columns.tolist()}\nHead:\n{df.head(3).to_markdown()}")
        prompt = (f"Context:\n{context}\n\nUser Question: \"{question}\"\n\nAnswer the question based only on the context provided.")
        return self.desc_llm(prompt)

    def create_visualization(self, df: pd.DataFrame, question: str):
        """
        Dynamically generates a Plotly visualization using an LLM to write the code.
        """
        if not px or not self.viz_llm.is_available():
            print("⚠️ Visualization skipped: Plotly or Viz LLM not available.")
            return None, None

        prompt = self._create_viz_prompt(df, question)
        print("⚡ Generating visualization code with LLM...")
        viz_code = self.viz_llm(prompt)

        if not viz_code or 'px.' not in viz_code:
            print("❌ LLM failed to generate valid visualization code.")
            return None, None

        cleaned_code = self._clean_generated_code(viz_code)
        print(f"✅ Generated Plotly Code: `{cleaned_code}`")

        try:
            # Safely execute the generated code
            fig = eval(cleaned_code, {"df": df, "px": px})
            if fig:
                return fig.to_html(full_html=False, include_plotlyjs='cdn'), pio.to_json(fig)
        except Exception as e:
            print(f"❌ Visualization execution failed: {e}\n   Code: {cleaned_code}")
        return None, None

    def _create_viz_prompt(self, df: pd.DataFrame, question: str) -> str:
        """Creates a detailed prompt for the LLM to generate Plotly Express code."""
        col_details = "\n".join([f"- '{col}' (type: {dtype})" for col, dtype in df.dtypes.items()])
        return f"""
        Given a pandas DataFrame named 'df', write a single, executable line of Python code using the Plotly Express library (aliased as `px`) to best visualize the answer to the user's question.

        USER QUESTION: "{question}"

        DATAFRAME COLUMNS:
        {col_details}

        AVAILABLE PLOT TYPES:
        - `px.histogram`: For distributions of a single numeric variable.
        - `px.bar`: For comparing a numeric value across categories.
        - `px.scatter`: For relationships between two numeric variables.
        - `px.box`: For visualizing the distribution and outliers of a numeric variable across categories.
        - `px.pie`: For showing proportions of a categorical variable (use sparingly).
        - `px.line`: For showing trends over time (if a date/time column is identified).

        RULES:
        1.  Choose the MOST APPROPRIATE plot type from the list above based on the question and data types.
        2.  Your output must be a single line of Python code starting with `fig = px.`
        3.  Use the correct column names from the list.
        4.  Do not include any explanation, comments, or markdown formatting like ```python.
        5.  For bar charts, if the user asks for an "average" or "total", use the DataFrame's aggregation capabilities within the plot call if appropriate (e.g., by passing a pre-grouped DataFrame). Often, it's better to just plot the raw values if the question is general.

        EXAMPLE:
        User Question: "Show me the distribution of charges"
        Code: fig = px.histogram(df, x='charges', title='Distribution of Charges')

        User Question: "What are the average charges per region?"
        Code: fig = px.bar(df.groupby('region')['charges'].mean().reset_index(), x='region', y='charges', title='Average Charges per Region')

        CODE:
        """.strip()

    def _clean_generated_code(self, code: str) -> str:
        """Removes markdown and 'fig =' prefix."""
        code = code.strip().replace("`python", "").replace("`", "")
        if code.startswith("fig = "):
            code = code[6:]
        return code.strip()

class UnifiedAnalysisEngine:
    """The orchestrator that uses the LLM classifier to route tasks."""
    def __init__(self):
        self.classifier = LLMQuestionClassifier()
        self.hybrid_engine = HybridCalculationEngine()
        self.insight_agent = InsightAgent()

    def analyze_question(self, question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        df_context = f"Dataset has columns: {df.columns.tolist()}"
        classification = self.classifier.analyze_question_intent(question, df_context)

        print(f"🎯 LLM Classification: {classification.intent.value.upper()} (Confidence: {classification.confidence:.1%})")
        print(f"   Reasoning: {classification.reasoning}")

        intent = classification.intent
        if intent in [QuestionIntent.STATISTICAL, QuestionIntent.COMPARATIVE, QuestionIntent.ANALYTICAL]:
            return self._handle_statistical_question(question, df, classification)
        else:
            return self._handle_descriptive_question(question, df, dataset_name, classification)

    def _handle_statistical_question(self, question: str, df: pd.DataFrame, classification) -> Dict[str, Any]:
        print(f"⚡ Using Hybrid Calculation Engine for: {classification.intent.value}")
        calculation_result = self.hybrid_engine.analyze_with_calculations(question, df)
        viz_html, viz_json = self.insight_agent.create_visualization(df, question)
        return {
            'question': question, 'answer': calculation_result, 'visualization_html': viz_html,
            'visualization_json': viz_json, 'method': f'LLM-Classified ({classification.intent.value}) -> Hybrid Engine'
        }

    def _handle_descriptive_question(self, question: str, df: pd.DataFrame, dataset_name: str, classification) -> Dict[str, Any]:
        """Handles non-analytical questions by routing them intelligently."""
        print("📋 Routing Descriptive Question...")
        q_lower = question.lower()
        if 'columns' in q_lower or 'variables' in q_lower:
            answer = f"The dataset '{dataset_name}' contains {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
            method = 'descriptive_summary'
        elif 'shape' in q_lower or 'size' in q_lower:
            answer = f"The dataset has {df.shape[0]:,} rows and {df.shape[1]} columns."
            method = 'descriptive_summary'
        else:
            print("🧠 Rerouting to InsightAgent for a qualitative answer...")
            answer = self.insight_agent.generate_descriptive_answer(question, df, dataset_name)
            method = f'LLM-Classified ({classification.intent.value}) -> Insight Agent'
        return {
            'question': question, 'answer': answer, 'visualization_html': None,
            'visualization_json': None, 'method': method
        }