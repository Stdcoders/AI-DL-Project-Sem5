#!/usr/bin/env python3
"""
Ultra-Fast Triple-Model Hybrid Calculation Engine
Uses Groq GPT-OSS-120B for ultra-fast mathematical reasoning (PRIMARY)
Uses DeepSeek-R1-Distill-Qwen-7B for mathematical reasoning (BACKUP)
Uses Gemma-2-9B-IT for contextual explanations and insights
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from deepseek_llm import DeepSeekLLM
from gemma_llm import GemmaLLM
from groq_llm import GroqLLM
import json

class HybridCalculationEngine:
    """
    Advanced ultra-fast triple-model hybrid system:
    - Groq GPT-OSS-120B: PRIMARY for ultra-fast mathematical reasoning (lightning speed)
    - DeepSeek-R1-Distill-Qwen-7B: BACKUP for mathematical reasoning when Groq unavailable
    - Gemma-2-9B-IT: Contextual explanations, insights, interpretations
    """
    
    def __init__(self):
        # Initialize Groq GPT-OSS-120B as PRIMARY calculation engine
        try:
            self.groq = GroqLLM(
                model_name="openai/gpt-oss-120b",
                temperature=0.0,   # Zero temperature for precise calculations
                max_tokens=2500,   # Higher for detailed mathematical work
                timeout=30         # Reasonable timeout
            )
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")
            self.groq = None
        
        # Initialize DeepSeek-R1 as BACKUP calculation engine
        self.deepseek = DeepSeekLLM(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            temperature=0.1,   # Low for precise reasoning
            max_tokens=2000,   # Higher for detailed mathematical work
            provider="nscale"   # Use nscale provider
        )
        
        # Initialize Gemma for explanations and context
        self.gemma = GemmaLLM(
            model_name="google/gemma-2-9b-it",
            temperature=0.25,  # Slightly higher for creative explanations
            max_tokens=2000    # Higher for detailed explanations
        )
        
        print(f"🔥 Initializing Ultra-Fast Triple-Model Hybrid Calculation Engine:")
        print(f"   ⚡ Groq GPT-OSS-120B (PRIMARY): {'✅ Available' if self.groq and self.groq.is_available() else '❌ Unavailable'}")
        print(f"   🦀 DeepSeek-R1-7B (BACKUP): {'✅ Available' if self.deepseek.is_available() else '❌ Unavailable'}")
        print(f"   🧠 Gemma-2-9B-IT (CONTEXT): {'✅ Available' if self.gemma.is_available() else '❌ Unavailable'}")
    
    def analyze_with_calculations(self, question: str, data: pd.DataFrame) -> str:
        """
        Primary method: Uses triple-model hybrid approach for comprehensive analysis.
        OpenAI (PRIMARY) handles complex calculations, DeepSeek (BACKUP) for math, Gemma for explanations.
        """
        try:
            print(f"🔥 [HybridEngine] Starting ultra-fast triple-model analysis...")
            
            # Step 1: Create rich data context
            data_context = self._create_comprehensive_data_context(data)
            
            # Step 2: Use Groq (PRIMARY) for lightning-fast mathematical analysis
            calculation_result = self._groq_mathematical_analysis(question, data_context)
            
            # Step 3: If Groq failed, fallback to DeepSeek
            if not calculation_result:
                print(f"🔄 [HybridEngine] Groq unavailable, using DeepSeek backup...")
                calculation_result = self._deepseek_mathematical_analysis(question, data_context)
            
            # Step 4: Use Gemma for contextual explanation  
            if calculation_result:
                explanation_result = self._gemma_contextual_explanation(
                    question, data_context, calculation_result
                )
                
                if explanation_result:
                    # Combine both results with cost tracking
                    return self._combine_triple_model_results(
                        calculation_result, explanation_result, question
                    )
            
            # Step 5: Fallback strategies
            return self._hybrid_fallback_analysis(question, data)
            
        except Exception as e:
            print(f"⚠️ [HybridEngine] Error in analysis: {e}")
            return self._basic_fallback(question, data)
    
    def _groq_mathematical_analysis(self, question: str, data_context: str) -> Optional[str]:
        """
        Use Groq GPT-OSS-120B for ultra-fast mathematical reasoning and calculations.
        Completely dataset-agnostic - works with any type of data.
        """
        if not self.groq or not self.groq.is_available():
            print(f"⚠️ [Groq] GPT-OSS-120B unavailable, skipping primary mathematical analysis")
            return None
        
        try:
            print(f"⚡ [Groq] Performing lightning-fast mathematical analysis...")
            
            # Create universal calculation prompt that works with ANY dataset
            calculation_prompt = f"""You are GPT-OSS-120B, a mathematical calculation expert. Perform precise statistical analysis using the provided dataset.

**CRITICAL REQUIREMENTS:**
- Perform ACTUAL calculations using the real data provided
- DO NOT generate code or programming examples
- Use the specific data values, column names, and statistics shown in the context
- Provide exact numerical results with proper units/labels
- Be mathematically precise and show your methodology

QUESTION: {question}

DATA CONTEXT:
{data_context}

**ANALYSIS APPROACH:**
1. **EXAMINE THE DATA**: Study the column types, values, and statistics provided
2. **IDENTIFY RELEVANT VARIABLES**: Determine which columns/metrics answer the question
3. **PERFORM CALCULATIONS**: Use the actual data values to compute results
4. **COMPARE & ANALYZE**: Find patterns, differences, relationships as needed
5. **PROVIDE SPECIFIC ANSWERS**: Give exact numbers with proper context

**YOUR MATHEMATICAL ANALYSIS MUST INCLUDE:**
- Specific numerical results based on the actual data
- Clear identification of which data elements you used
- Brief explanation of your calculation methodology
- Confidence level in your results

**FOCUS ON THE ACTUAL QUESTION AND DATASET PROVIDED**

MATHEMATICAL ANALYSIS:"""
            
            response = self.groq.calculate(calculation_prompt)
            
            if response and len(response.strip()) > 50:
                print(f"✅ [Groq] Generated ultra-fast mathematical analysis: {len(response)} chars")
                # Display performance tracking
                if hasattr(self.groq, 'get_usage_summary'):
                    usage_summary = self.groq.get_usage_summary()
                    # Show speed metrics
                    speed_line = [line for line in usage_summary.split('\n') if 'speed' in line.lower()]
                    if speed_line:
                        print(f"⚡ [Groq] {speed_line[0].strip()}")
                return response
            else:
                print(f"⚠️ [Groq] Mathematical analysis insufficient")
                return None
                
        except Exception as e:
            print(f"⚠️ [Groq] Mathematical analysis error: {e}")
            return None
    
    def _deepseek_mathematical_analysis(self, question: str, data_context: str) -> Optional[str]:
        """
        Use DeepSeek R1 for mathematical reasoning and calculations.
        Completely dataset-agnostic - works with any type of data.
        """
        if not self.deepseek.is_available():
            print(f"⚠️ [HybridEngine] DeepSeek unavailable, skipping mathematical analysis")
            return None
        
        try:
            print(f"🧮 [DeepSeek] Performing mathematical analysis...")
            
            # Create universal calculation prompt for any dataset
            calculation_prompt = f"""You are DeepSeek-R1, a mathematical calculation expert. Analyze the provided dataset to answer the question with precise calculations.

**REQUIREMENTS:**
- Perform ACTUAL calculations using the real data provided
- DO NOT generate code or suggest programming approaches
- Use the specific data values, statistics, and samples shown in the context
- Provide exact numerical results with appropriate units
- Show your mathematical reasoning

QUESTION: {question}

DATA CONTEXT:
{data_context}

**CALCULATION APPROACH:**
1. **DATA EXAMINATION**: Study the provided columns, values, and statistics
2. **VARIABLE IDENTIFICATION**: Determine which data elements answer the question
3. **MATHEMATICAL OPERATIONS**: Perform relevant calculations (sums, averages, comparisons, etc.)
4. **RESULT DERIVATION**: Compute specific answers using the actual data
5. **VERIFICATION**: Ensure results make sense given the data context

**YOUR ANALYSIS MUST PROVIDE:**
- Specific numerical results based on the actual dataset
- Clear identification of which data you analyzed
- Methodology explaining your calculations
- Confidence assessment of your results

**FOCUS ON THE SPECIFIC QUESTION AND PROVIDED DATA**

MATHEMATICAL SOLUTION:"""
            
            response = self.deepseek.calculate(calculation_prompt)
            
            if response and len(response.strip()) > 50:
                print(f"✅ [DeepSeek] Generated mathematical analysis: {len(response)} chars")
                return response
            else:
                print(f"⚠️ [DeepSeek] Mathematical analysis insufficient")
                return None
                
        except Exception as e:
            print(f"⚠️ [DeepSeek] Mathematical analysis error: {e}")
            return None
    
    def _gemma_contextual_explanation(self, question: str, data_context: str, 
                                    calculation_result: str) -> Optional[str]:
        """
        Use Gemma-2-9B-IT for contextual explanations and insights.
        """
        if not self.gemma.is_available():
            print(f"⚠️ [HybridEngine] Gemma unavailable, skipping explanation")
            return None
        
        try:
            print(f"🧠 [Gemma] Generating contextual explanation...")
            
            # Create explanation-focused prompt for Gemma
            explanation_prompt = f"""You are an expert data analyst providing contextual explanations and insights.

ORIGINAL QUESTION: {question}

MATHEMATICAL CALCULATIONS (from DeepSeek):
{calculation_result}

DATA CONTEXT:
{data_context}

EXPLANATION TASK:
1. Interpret the mathematical results in business/practical context
2. Explain what the numbers mean in real-world terms
3. Provide insights about trends and patterns
4. Discuss implications and significance
5. Suggest actionable insights or next steps
6. Connect results to broader domain knowledge

Focus on:
- Clear, accessible explanations
- Business/practical implications  
- Contextual meaning of the calculations
- Actionable insights
- Professional interpretation

Do NOT repeat the calculations - focus on explaining their meaning and significance.

EXPLANATION:"""
            
            response = self.gemma.generate(explanation_prompt)
            
            if response and len(response.strip()) > 100:
                print(f"✅ [Gemma] Generated contextual explanation: {len(response)} chars")
                return response
            else:
                print(f"⚠️ [Gemma] Explanation insufficient")
                return None
                
        except Exception as e:
            print(f"⚠️ [Gemma] Explanation error: {e}")
            return None
    
    def _combine_triple_model_results(self, calculation: str, explanation: str, 
                                    question: str) -> str:
        """
        Combine OpenAI/DeepSeek calculations with Gemma explanations into a cohesive response.
        """
        # Determine which model was used for calculations
        calc_model = "Groq GPT-OSS-120B" if self.groq and self.groq.is_available() else "DeepSeek R1"
        calc_emoji = "⚡" if "Groq" in calc_model else "🦀"
        
        combined_response = f"""# 🔥 Ultra-Fast Triple-Model Hybrid Analysis

## {calc_emoji} **Mathematical Analysis** ({calc_model})

{calculation}

---

## 🧠 **Contextual Explanation** (Gemma-2-9B-IT)

{explanation}

---

## 📊 **Analysis Summary**
This premium hybrid analysis combines advanced mathematical reasoning from {calc_model} with rich contextual insights from Gemma-2-9B-IT to provide both precise calculations and meaningful business interpretations of your question: "{question}"
"""
        
        # Add performance tracking if Groq was used
        if self.groq and "Groq" in calc_model:
            try:
                performance_info = self.groq.get_usage_summary()
                combined_response += f"\n\n{performance_info}"
            except:
                pass
        
        return combined_response
    
    def _combine_calculation_and_explanation(self, calculation: str, explanation: str, 
                                           question: str) -> str:
        """
        Legacy method - now redirects to triple-model combination.
        """
        return self._combine_triple_model_results(calculation, explanation, question)
    
    def _hybrid_fallback_analysis(self, question: str, data: pd.DataFrame) -> str:
        """
        Fallback analysis when one or both models fail.
        """
        print(f"🔄 [HybridEngine] Using fallback analysis...")
        
        # Try Gemma-only analysis if DeepSeek failed
        if self.gemma.is_available():
            try:
                data_context = self._create_comprehensive_data_context(data)
                gemma_prompt = f"""Analyze this data question comprehensively:

QUESTION: {question}

DATA CONTEXT:
{data_context}

Provide both mathematical analysis and contextual explanation:
1. Identify relevant calculations needed
2. Estimate or compute numerical results
3. Explain the meaning and implications
4. Provide actionable insights

ANALYSIS:"""
                
                response = self.gemma.generate(gemma_prompt)
                if response and len(response.strip()) > 100:
                    return f"## 🧠 **Analysis** (Gemma-2-9B-IT)\n\n{response}\n\n*Note: Using single-model fallback analysis*"
            
            except Exception as e:
                print(f"⚠️ [HybridEngine] Gemma fallback error: {e}")
        
        # Final computational fallback
        return self._basic_computational_analysis(question, data)
    
    def _basic_fallback(self, question: str, data: pd.DataFrame) -> str:
        """Basic fallback when both models are unavailable."""
        return self._basic_computational_analysis(question, data)
    
    
    def _create_comprehensive_data_context(self, data: pd.DataFrame) -> str:
        """Create universal data context for any dataset type."""
        try:
            context_parts = []
            
            # Basic dataset overview
            context_parts.append(f"DATASET OVERVIEW:")
            context_parts.append(f"- Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")
            context_parts.append(f"- Columns: {list(data.columns)}")
            
            # Analyze each column type and provide key statistics
            context_parts.append(f"\n📊 COLUMN ANALYSIS:")
            
            for col in data.columns[:10]:  # Limit to first 10 columns to avoid token overflow
                try:
                    col_data = data[col].dropna()
                    if len(col_data) == 0:
                        context_parts.append(f"• {col}: No data (all null)")
                        continue
                    
                    # Try to convert to numeric
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    numeric_ratio = len(numeric_data) / len(col_data)
                    
                    if numeric_ratio > 0.8:  # Mostly numeric
                        stats = numeric_data.describe()
                        context_parts.append(f"• {col}: NUMERIC")
                        context_parts.append(f"  Statistics: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
                        context_parts.append(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                        context_parts.append(f"  Sample values: {numeric_data.head(5).tolist()}")
                    else:  # Categorical/text
                        unique_count = col_data.nunique()
                        context_parts.append(f"• {col}: CATEGORICAL ({unique_count} unique values)")
                        
                        if unique_count <= 20:  # Show value counts for low cardinality
                            top_values = col_data.value_counts().head(5)
                            context_parts.append(f"  Top values: {dict(top_values)}")
                        else:
                            sample_values = col_data.head(5).tolist()
                            context_parts.append(f"  Sample values: {sample_values}")
                            
                except Exception as e:
                    context_parts.append(f"• {col}: Analysis error - {str(e)[:100]}")
            
            # Add data samples for reference
            context_parts.append(f"\n📋 SAMPLE DATA (first 3 rows):")
            sample_df = data.head(3)
            for idx, row in sample_df.iterrows():
                row_dict = row.to_dict()
                context_parts.append(f"Row {idx + 1}: {row_dict}")
            
            # Add basic relationships if multiple numeric columns exist
            numeric_cols = [col for col in data.columns if pd.to_numeric(data[col], errors='coerce').count() > len(data) * 0.8]
            if len(numeric_cols) >= 2:
                context_parts.append(f"\n🔗 NUMERIC COLUMNS FOR ANALYSIS: {numeric_cols[:5]}")
                
                # Simple correlation info
                try:
                    corr_df = data[numeric_cols[:3]].apply(pd.to_numeric, errors='coerce').corr()
                    context_parts.append(f"Basic correlations available between: {', '.join(numeric_cols[:3])}")
                except:
                    pass
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return f"Dataset: {data.shape[0]} rows × {data.shape[1]} columns\nError creating context: {str(e)}"
    
    def _basic_computational_analysis(self, question: str, data: pd.DataFrame) -> str:
        """Basic computational fallback analysis for any dataset."""
        try:
            # Get basic statistics for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            analysis_parts = []
            analysis_parts.append(f"## 📊 **Basic Analysis**")
            analysis_parts.append(f"Dataset: {data.shape[0]} rows × {data.shape[1]} columns")
            
            if numeric_cols:
                analysis_parts.append(f"\n**Numeric Columns Summary:**")
                for col in numeric_cols[:3]:
                    stats = data[col].describe()
                    analysis_parts.append(f"• {col}: mean={stats['mean']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
            
            if categorical_cols:
                analysis_parts.append(f"\n**Categorical Columns:**")
                for col in categorical_cols[:3]:
                    unique_count = data[col].nunique()
                    analysis_parts.append(f"• {col}: {unique_count} unique values")
                    
                    if unique_count <= 10:
                        top_val = data[col].mode().iloc[0] if not data[col].mode().empty else "N/A"
                        analysis_parts.append(f"  Most frequent: {top_val}")
            
            analysis_parts.append(f"\n**Question Analysis:**")
            analysis_parts.append(f"Your question: '{question}'")
            analysis_parts.append(f"Based on the available data, this appears to be a {data.shape[1]}-column dataset with both numeric and categorical information.")
            analysis_parts.append(f"For more detailed analysis, please ensure the calculation engines are properly connected.")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Basic analysis failed: {str(e)}"
    
