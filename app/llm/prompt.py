from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_prompt(mode: str = "default"):
    try:
        logger.info(f"Loading prompt with mode: {mode}")

        if mode == "interview":
            template = """
You are an ML expert preparing someone for interviews.

Answer using ONLY the context below.

Format:
- Use bullet points
- Be concise
- Include key concepts and definitions

If you don't know, say "I don't know".

Context:
{context}

Question:
{question}
"""

        elif mode == "beginner":
            template = """
You are a teacher explaining machine learning concepts to a beginner.

Answer using ONLY the context below.

Format:
- Use simple language
- Use bullet points
- Avoid technical jargon

If you don't know, say "I don't know".

Context:
{context}

Question:
{question}
"""

        else:
            template = """
You are an ML expert.

Answer the question using ONLY the context below.

Format:
- Use bullet points
- Keep each point short and clear
- Avoid long paragraphs

If you don't know, say "I don't know".

Context:
{context}

Question:
{question}
"""

        return ChatPromptTemplate.from_template(template)

    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise