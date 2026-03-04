"""
Context Intelligence Engine — Research-backed techniques to make small LLMs smarter.

Combines 4 layers to dramatically improve response quality:
1. RAG Retrieval — pull relevant chunks from uploaded documents (Lewis et al., 2020)
2. Recursive Context — decompose complex queries into sub-questions (MIT RLMs, 2025)
3. Self-Refine — auto-critique and improve answers (Madaan et al., ICLR 2024)
4. Adaptive Prompting — Chain-of-Thought and task-specific templates (Wei et al., 2022)

All processing happens locally. No data leaves the user's machine.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Generator

from . import config

logger = logging.getLogger(__name__)


# ─── Prompt Templates ────────────────────────────────────────────────────────

PROMPTS = {
    # RAG grounding: model only uses provided context
    "rag_system": (
        "You are a helpful AI assistant. Answer the user's question using ONLY "
        "the context provided below. If the context doesn't contain enough "
        "information, say so honestly. Cite which source you used.\n\n"
        "--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
    ),

    # Chain-of-Thought: forces step-by-step reasoning
    "cot_prefix": (
        "Think step by step. First identify what's being asked, "
        "then reason through each part before giving your final answer.\n\n"
    ),

    # Self-Refine: critique pass
    "refine_critique": (
        "You are a critical reviewer. Evaluate the following answer for:\n"
        "1. Accuracy — are there factual errors?\n"
        "2. Completeness — does it answer the full question?\n"
        "3. Clarity — is it well-organized and easy to understand?\n\n"
        "Question: {question}\n\n"
        "Answer to evaluate:\n{answer}\n\n"
        "List specific issues (or say 'NONE' if the answer is good):"
    ),

    # Self-Refine: improvement pass
    "refine_improve": (
        "Improve the following answer based on the feedback provided. "
        "Keep what's good, fix what's wrong, and fill in any gaps.\n\n"
        "Original question: {question}\n\n"
        "Original answer:\n{answer}\n\n"
        "Feedback:\n{critique}\n\n"
        "Improved answer:"
    ),

    # Recursive decomposition: break complex queries
    "decompose": (
        "Break this question into 2-3 simpler sub-questions that can be "
        "answered independently. Output ONLY the sub-questions, one per line, "
        "prefixed with 'Q:'\n\n"
        "Complex question: {question}\n\n"
        "Sub-questions:"
    ),

    # Code questions
    "code_system": (
        "You are a coding assistant. Provide clear, working code with "
        "brief explanations. Use proper formatting with code blocks."
    ),
}


# ─── Query Complexity Detection ──────────────────────────────────────────────

# Patterns that suggest a complex, multi-part query
_COMPLEX_PATTERNS = [
    r'\b(compare|contrast|difference between|similarities)\b',
    r'\b(analyze|evaluate|assess|critique)\b',
    r'\b(explain how|explain why|what are the|how does .+ relate)\b',
    r'\b(step by step|in detail|thoroughly|comprehensive)\b',
    r'\b(pros and cons|advantages and disadvantages)\b',
    r'\band\b.*\band\b',  # multiple "and"s suggest multi-part
    r'\?.*\?',            # multiple questions
]

_SIMPLE_PATTERNS = [
    r'^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure)\b',
    r'^(what is|who is|when was|where is)\b.{0,30}$',  # short factual
]

_CODE_PATTERNS = [
    r'\b(code|function|class|module|script|program|bug|error|debug)\b',
    r'\b(python|javascript|java|rust|go|sql|html|css|react|node)\b',
    r'\b(implement|refactor|optimize|fix|write)\b.*\b(code|function|method)\b',
]


def classify_query(message: str) -> str:
    """
    Classify a user query as 'simple', 'complex', or 'code'.
    Used to decide which processing layers to activate.
    """
    lower = message.lower().strip()

    # Check simple first
    for pattern in _SIMPLE_PATTERNS:
        if re.search(pattern, lower):
            return "simple"

    # Check code
    for pattern in _CODE_PATTERNS:
        if re.search(pattern, lower):
            return "code"

    # Check complex
    complexity_score = 0
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, lower):
            complexity_score += 1

    # Long queries are more likely complex
    if len(message.split()) > 25:
        complexity_score += 1

    if complexity_score >= 2:
        return "complex"

    return "standard"


# ─── Context Intelligence Engine ─────────────────────────────────────────────

class ContextEngine:
    """
    Orchestrates 4 research-backed layers to improve LLM response quality.

    Usage:
        engine = ContextEngine(llm_engine, knowledge_base)
        # For streaming:
        for event in engine.process_stream(message, conversation, settings):
            yield event  # {"type": "status"|"token"|"done", ...}

        # For non-streaming:
        result = engine.process(message, conversation, settings)
    """

    def __init__(self, llm_engine, knowledge_base=None):
        self.llm = llm_engine
        self.kb = knowledge_base

    # ── Layer 1: RAG Retrieval ────────────────────────────────────────────

    def retrieve_context(self, message: str, n_results: int = None) -> str:
        """
        Query the knowledge base for relevant document chunks.
        Returns formatted context string, or empty string if KB unavailable.
        """
        if not self.kb:
            return ""

        try:
            context = self.kb.query(message, n_results=n_results)
            if context:
                logger.info(f"RAG: Retrieved context ({len(context)} chars)")
            return context
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""

    # ── Layer 2: Recursive Context Decomposition ─────────────────────────

    def recursive_retrieve(self, message: str, initial_context: str = "") -> str:
        """
        For complex queries, decompose into sub-questions and retrieve
        context for each independently. Merges results.

        Inspired by MIT CSAIL Recursive Language Models (RLMs, 2025).
        """
        if not self.llm.is_loaded or not self.kb:
            return initial_context

        try:
            # Ask the model to decompose the question
            decompose_prompt = PROMPTS["decompose"].format(question=message)
            sub_questions_raw = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You break questions into sub-questions."},
                    {"role": "user", "content": decompose_prompt},
                ],
                max_tokens=256,
                temperature=0.3,
                stream=False,
            )

            # Parse sub-questions
            sub_questions = []
            for line in sub_questions_raw.strip().split("\n"):
                line = line.strip()
                if line.startswith("Q:"):
                    sub_questions.append(line[2:].strip())
                elif line and len(line) > 10:  # fallback: any substantial line
                    sub_questions.append(line.lstrip("0123456789.-) "))

            if not sub_questions:
                logger.info("Recursive: No sub-questions extracted, using initial context")
                return initial_context

            logger.info(f"Recursive: Decomposed into {len(sub_questions)} sub-questions")

            # Retrieve context for each sub-question
            all_contexts = set()  # deduplicate
            if initial_context:
                all_contexts.add(initial_context)

            for sq in sub_questions[:3]:  # max 3 sub-questions
                ctx = self.retrieve_context(sq, n_results=2)
                if ctx:
                    all_contexts.add(ctx)

            merged = "\n\n---\n\n".join(all_contexts)
            logger.info(f"Recursive: Merged context ({len(merged)} chars)")
            return merged

        except Exception as e:
            logger.warning(f"Recursive decomposition failed: {e}")
            return initial_context

    # ── Layer 3: Self-Refine Loop ────────────────────────────────────────

    def self_refine(self, question: str, initial_answer: str, depth: int = 1) -> str:
        """
        Iteratively critique and improve an answer.

        Based on Self-Refine (Madaan et al., ICLR 2024):
        1. Generate → Critique → Refine → (repeat)

        Args:
            question: Original user question
            initial_answer: First-pass answer to improve
            depth: Number of refinement iterations (1 or 2)

        Returns:
            Refined answer string
        """
        if not self.llm.is_loaded:
            return initial_answer

        answer = initial_answer

        for i in range(min(depth, 2)):  # cap at 2 iterations
            try:
                # Step 1: Critique
                critique_prompt = PROMPTS["refine_critique"].format(
                    question=question, answer=answer
                )
                critique = self.llm.chat(
                    messages=[
                        {"role": "system", "content": "You are a critical reviewer."},
                        {"role": "user", "content": critique_prompt},
                    ],
                    max_tokens=512,
                    temperature=0.3,
                    stream=False,
                )

                # If no issues found, stop refining
                if "NONE" in critique.upper() or "no issues" in critique.lower():
                    logger.info(f"Self-Refine pass {i+1}: No issues found, stopping")
                    break

                logger.info(f"Self-Refine pass {i+1}: Found issues, improving...")

                # Step 2: Improve
                improve_prompt = PROMPTS["refine_improve"].format(
                    question=question, answer=answer, critique=critique
                )
                improved = self.llm.chat(
                    messages=[
                        {"role": "system", "content": "You improve answers based on feedback."},
                        {"role": "user", "content": improve_prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.5,
                    stream=False,
                )

                if improved and len(improved.strip()) > len(answer.strip()) * 0.3:
                    answer = improved.strip()
                else:
                    logger.info(f"Self-Refine pass {i+1}: Improvement too short, keeping original")
                    break

            except Exception as e:
                logger.warning(f"Self-Refine pass {i+1} failed: {e}")
                break

        return answer

    # ── Layer 4: Adaptive Prompt Engineering ──────────────────────────────

    def build_system_prompt(
        self,
        message: str,
        rag_context: str = "",
        use_cot: bool = False,
        base_prompt: str = "",
    ) -> str:
        """
        Build an optimized system prompt based on query type and available context.

        Combines:
        - Base system prompt
        - RAG context grounding
        - Chain-of-Thought instructions
        - Task-specific formatting
        """
        query_type = classify_query(message)
        parts = []

        # Base prompt
        if rag_context:
            parts.append(PROMPTS["rag_system"].format(context=rag_context))
        elif query_type == "code":
            parts.append(PROMPTS["code_system"])
        elif base_prompt:
            parts.append(base_prompt)
        else:
            parts.append(
                "You are a helpful, knowledgeable AI assistant. "
                "Answer questions clearly and thoroughly."
            )

        # Chain-of-Thought for complex queries
        if use_cot or query_type == "complex":
            parts.append(PROMPTS["cot_prefix"])

        return "\n\n".join(parts)

    # ── Main Processing Pipeline ─────────────────────────────────────────

    def process(
        self,
        message: str,
        conversation=None,
        use_rag: bool = False,
        refine_depth: int = 0,
        use_cot: bool = False,
        base_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Full pipeline: RAG → Recursive → Generate → Self-Refine.
        Returns the final answer string (non-streaming).
        """
        # Layer 1 & 2: Knowledge retrieval
        rag_context = ""
        if use_rag:
            rag_context = self.retrieve_context(message)
            query_type = classify_query(message)
            if query_type == "complex" and rag_context:
                rag_context = self.recursive_retrieve(message, rag_context)

        # Layer 4: Build optimized system prompt
        system_prompt = self.build_system_prompt(
            message, rag_context=rag_context, use_cot=use_cot, base_prompt=base_prompt
        )

        # Generate initial response
        if conversation:
            # Use full conversation history with the enhanced system prompt
            messages = conversation.get_context_messages()
            # Replace the system prompt with our enhanced one
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]
        response = self.llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        # Layer 3: Self-Refine
        if refine_depth > 0 and response:
            response = self.self_refine(message, response, depth=refine_depth)

        return response

    def process_stream(
        self,
        message: str,
        conversation=None,
        use_rag: bool = False,
        refine_depth: int = 0,
        use_cot: bool = False,
        base_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming pipeline: yields status updates and tokens.

        Yields events:
            {"type": "status", "message": "Searching documents..."}
            {"type": "context", "sources": [...]}
            {"type": "token", "content": "..."}
            {"type": "refine_start"}
            {"type": "refine_token", "content": "..."}
            {"type": "done", "refined": bool}
        """
        # Layer 1 & 2: Knowledge retrieval
        rag_context = ""
        sources = []

        if use_rag:
            yield {"type": "status", "message": "🔍 Searching your documents..."}
            rag_context = self.retrieve_context(message)

            if rag_context:
                # Extract source names
                for line in rag_context.split("\n"):
                    if line.startswith("[Source:"):
                        src = line.split("]")[0].replace("[Source: ", "")
                        if src not in sources:
                            sources.append(src)

                yield {"type": "context", "sources": sources}

                # Recursive decomposition for complex queries
                query_type = classify_query(message)
                if query_type == "complex":
                    yield {"type": "status", "message": "🧠 Breaking down your question..."}
                    rag_context = self.recursive_retrieve(message, rag_context)

        # Layer 4: Build system prompt
        auto_cot = use_cot or classify_query(message) == "complex"
        if auto_cot and not use_cot:
            yield {"type": "status", "message": "💭 Activating deeper reasoning..."}

        system_prompt = self.build_system_prompt(
            message, rag_context=rag_context, use_cot=use_cot, base_prompt=base_prompt
        )

        # Generate initial response (streaming)
        if conversation:
            # Use full conversation history with the enhanced system prompt
            messages = conversation.get_context_messages()
            # Replace the system prompt with our enhanced one
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]

        full_response = ""
        try:
            for token in self.llm.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                full_response += token
                yield {"type": "token", "content": token}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        # Layer 3: Self-Refine (if enabled)
        if refine_depth > 0 and full_response.strip():
            yield {"type": "refine_start"}
            yield {"type": "status", "message": f"✨ Thinking deeper (pass 1/{refine_depth})..."}

            refined = self.self_refine(message, full_response, depth=refine_depth)

            if refined != full_response:
                yield {"type": "refine_token", "content": refined}
                yield {"type": "done", "refined": True}
            else:
                yield {"type": "done", "refined": False}
        else:
            yield {"type": "done", "refined": False}
