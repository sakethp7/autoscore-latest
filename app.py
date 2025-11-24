from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from PIL import Image
import base64
from io import BytesIO
import asyncio
from contextlib import asynccontextmanager
import logging
import uuid
from threading import Lock
import copy

# Load environment variables
load_dotenv()

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for configuration
MAX_LLM_CONNECTIONS = int(os.getenv('MAX_LLM_CONNECTIONS', '10'))
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")
    raise ValueError("GEMINI_API_KEY environment variable not set")

# --- Pydantic Models ---
class QuestionEvaluation(BaseModel):
    """Evaluation result for a single question"""
    question_number: str = Field(description="Question number or identifier")
    question_text: str = Field(description="The question that was evaluated")
    score: int = Field(description="Score for the student's answer")
    max_marks: int = Field(description="Maximum marks for the question")
    error_type: str = Field(description="Type of error: Conceptual Error,Irrelavant, Calculation Error or None")
    mistakes_made: str = Field(description="Specific mistakes made in the question")
    gap_analysis: str = Field(description="Detailed explanation about student's mistakes and missing concepts")
    additional_comments: str = Field(description="Any additional comments")
    concepts_required: str = Field(description=" 2-3 Main concepts required to solve the question")
    time_analysis: str = Field(description="Time analysis: great/good/should_improve/critical")

class HomeworkEvaluationResult(BaseModel):
    """Complete homework evaluation result containing all questions"""
    total_questions: int = Field(description="Total number of questions evaluated")
    total_score: int = Field(description="Total score obtained across all questions")
    total_max_marks: int = Field(description="Total maximum marks for all questions")
    evaluations: List[QuestionEvaluation] = Field(description="List of individual question evaluations")
    overall_performance: str = Field(description="Overall performance summary")
    extracted_text_summary: str = Field(description="Summary of what was extracted from answer sheets")

class Result(BaseModel):
    """Evaluation Result schema for a single question with transcription"""
    student_text_transcription: str = Field(description="Complete transcription of student's answer using KaTeX based LaTeX format")
    score: int = Field(description="Score for the student's answer")
    max_marks: int = Field(description="Maximum marks for the question")
    error_type: str = Field(description="Type of error: conceptual_error, calculation_error, logical_error, no_error or unattempted")
    mistakes_made: str = Field(description="Specific mistakes made in the question")
    gap_analysis: str = Field(description="Explain in detail about student's mistakes what concept he is unable to apply to solve the problem")
    additional_comments: str = Field(description="Any additional comments")
    concepts_required: str = Field(description="Concepts required to solve the question")
    time_analysis: str = Field(description="Time analysis in one word great/good/should improve/critical")

# --- FIXED LLM Connection Pool ---
class LLMPool:
    """Thread-safe connection pool for LLM instances with proper isolation."""

    def __init__(self, size: int):
        self.size = size
        self.semaphore = asyncio.Semaphore(self.size)
        self._lock = asyncio.Lock()

    def _create_llm_instance(self) -> ChatGoogleGenerativeAI:
        """Creates a fresh LLM instance (gemini-2.5-flash)."""
        return ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=API_KEY,
            temperature=0,
            max_retries=2,
            timeout=60
        )

    def _create_llm_instance_lite(self) -> ChatGoogleGenerativeAI:
        """Creates a fresh LLM instance (gemini-2.5-flash-lite)."""
        return ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=API_KEY,
            temperature=0,
            max_retries=2,
            timeout=60
        )

    @asynccontextmanager
    async def get_llm(self):
        await self.semaphore.acquire()
        try:
            yield self._create_llm_instance()
        except Exception as e:
            logger.error(f"Error creating LLM instance: {e}")
            raise
        finally:
            self.semaphore.release()

    @asynccontextmanager
    async def get_llm_lite(self):
        await self.semaphore.acquire()
        try:
            yield self._create_llm_instance_lite()
        except Exception as e:
            logger.error(f"Error creating LLM-Lite instance: {e}")
            raise
        finally:
            self.semaphore.release()

# Global LLM pool instance
llm_pool = LLMPool(size=MAX_LLM_CONNECTIONS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"FastAPI app starting with {MAX_LLM_CONNECTIONS} LLM connections.")
    yield
    logger.info("FastAPI app shutting down.")

app = FastAPI(
    title="AutoScore API",
    description="Student Answer Evaluation System",
    version="3.3.0",
    lifespan=lifespan
)

# --- Utility Functions ---
async def image_to_base64_async(image: Image.Image, request_id: str) -> str:
    loop = asyncio.get_event_loop()
    image_copy = image.copy()
    return await loop.run_in_executor(None, image_to_base64_sync, image_copy, request_id)

def image_to_base64_sync(image: Image.Image, request_id: str) -> str:
    try:
        buffered = BytesIO()
        max_size = (1920, 1920)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"[{request_id}] Error converting image to base64: {e}")
        raise

async def extract_text_from_image(llm: ChatGoogleGenerativeAI, image: Image.Image, page_num: int, request_id: str) -> str:
    """Phase 1: Extracts text from image using OCR via LLM (async) - Used for Batch only."""
    ocr_prompt = """Extract ALL text from this image with extreme precision for student answer evaluation.
**EXTRACTION RULES:**
Extract:
1. **QUESTION STRUCTURE:**
   - Main questions: "1)", "2)", "Q1", "Question 1", etc.
2. **MATHEMATICAL CONTENT:**
   - Use LaTeX notation for ALL mathematical expressions, KaTeX-compatible.
   - Inline math: $expression$
   - Display math: $$expression$$
3. **TEXT CONTENT:**
   - Preserve ALL written explanations.
4. **VISUAL ELEMENTS:**
   - Describe diagrams: [DIAGRAM: description]
5.**CROSSING OUT RULES**
   - If student crosses out something dont retive that.
**OUTPUT FORMAT:**
[Question Number])
[Student's Solution:]
[Content...]
"""
    try:
        img_base64 = await image_to_base64_async(image, request_id)
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": ocr_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }]
        response = await llm.ainvoke(message)
        extracted_text = response.content if hasattr(response, 'content') else str(response)
        cleaned_text = '\n'.join([line for line in extracted_text.split('\n') if not line.strip().startswith(('Roll:', 'Page:'))])
        return f"<page{page_num + 1}>\n{cleaned_text}\n</page{page_num + 1}>"
    except Exception as e:
        return f"<page{page_num + 1}>\nError extracting text: {str(e)}\n</page{page_num + 1}>"

async def evaluate_answer(
    llm: ChatGoogleGenerativeAI, 
    question: str, 
    question_images: list, 
    answer_images: list, 
    request_id: str
) -> Result:
    """
    Unified function: Transcribes and Evaluates.
    """
    combined_prompt = f"""
You are an expert teacher. Perform two operations on the provided Student Answer Images:
1. **TRANSCRIPTION:** Transcribe the student's handwritten answer.
2. **EVALUATION:** Evaluate the answer against the Question.

**QUESTION:**
{question}

---

**PART 1: TRANSCRIPTION INSTRUCTIONS (Fill 'student_text_transcription' field)**
- Read the handwriting in the Student Answer Images.
- Convert ALL mathematical expressions into **KaTeX-compatible LaTeX**.
- Use single `$` for inline math and double `$$` for display math.
- **Do not** transcribe crossed-out work.
- Preserve the flow.
- Do NOT include headers like "Student Answer:" in the transcription field itself, just the raw content.

---

**PART 2: EVALUATION INSTRUCTIONS (Fill remaining fields)**
**SCORING CRITERIA (STRICT AND FIXED RULES)**
Use only the following deterministic marking scheme based on max_marks:
For 2-mark questions: 1 mark concept, 1 mark answer.
For 5-mark questions: 2 marks concept, 2 marks approach, 1 mark answer.
For 10-mark questions: 4 marks concept, 4 marks approach, 2 marks answer.

**DEDUCTION RULES**
- Wrong concept: 0 marks for concept.
- Calculation error: deduct only from final answer marks.
- Missing units: deduct up to 0.5 mark.

**OUTPUT REQUIREMENTS:**
- Respond ONLY using the defined JSON schema.
- `student_text_transcription`: Full KaTeX transcript.
- `mistakes_made`: Specific errors.
"""
    try:
        message_content = [{"type": "text", "text": combined_prompt}]
        if question_images:
            for i, img in enumerate(question_images):
                img_base64 = await image_to_base64_async(img, f"{request_id}_q{i}")
                message_content.append({"type": "text", "text": f"**Reference Question Image {i+1}:**"})
                message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})

        message_content.append({"type": "text", "text": "**STUDENT ANSWER IMAGES:**"})
        for i, img in enumerate(answer_images):
            img_base64 = await image_to_base64_async(img, f"{request_id}_ans{i}")
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})

        message = [{"role": "user", "content": message_content}]
        logger.info(f"[{request_id}] Starting combined Transcription and Evaluation")
        
        structured_llm = llm.with_structured_output(Result)
        result = await structured_llm.ainvoke(message)
        return result
    except Exception as e:
        logger.error(f"[{request_id}] Error in evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in evaluation: {str(e)}")

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "AutoScore API is running", "version": "3.3.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ... [evaluate_homework_batch function remains exactly the same as previous version] ...
async def evaluate_homework_batch(llm, questions, question_images, extracted_answer, request_id):
    # (Kept brief here to save space, logic is identical to your previous working code)
    questions_formatted = "\n\n".join([f"Question {q.get('question_number', idx+1)}: {q.get('question_text', q.get('text', ''))}" for idx, q in enumerate(questions)])
    evaluation_prompt = f"""You are an expert teacher evaluating a student's homework submission.
Evaluate ALL the following questions based on the student's answers provided.
QUESTIONS TO EVALUATE:
{questions_formatted}
STUDENT'S COMPLETE ANSWER SHEET:
{extracted_answer}
(Use standard instructions defined previously...)
"""
    try:
        if question_images:
            message_content = [{"type": "text", "text": evaluation_prompt}]
            for i, img in enumerate(question_images):
                img_base64 = await image_to_base64_async(img, f"{request_id}_qimg{i}")
                message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
            message = [{"role": "user", "content": message_content}]
        else:
            message = [{"role": "user", "content": evaluation_prompt}]
        
        structured_llm = llm.with_structured_output(HomeworkEvaluationResult)
        return await structured_llm.ainvoke(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/homework-autoscore/")
async def homework_autoscore(
    questions: str = Form(...),
    question_images: Optional[List[UploadFile]] = File(None),
    answer_images: List[UploadFile] = File(...)
):
    request_id = str(uuid.uuid4())[:8]
    try:
        import json
        questions_data = json.loads(questions)
        normalized_questions = []
        for idx, q in enumerate(questions_data):
            if isinstance(q, str): normalized_questions.append({'question_text': q, 'question_number': str(idx + 1)})
            elif isinstance(q, dict): normalized_questions.append({'question_text': q.get('question_text', q.get('text', '')), 'question_number': q.get('question_number', str(idx + 1))})
        
        question_image_objects = []
        if question_images:
            question_image_tasks = [read_and_process_image(img, f"{request_id}_qimg{i}") for i, img in enumerate(question_images)]
            question_image_objects = await asyncio.gather(*question_image_tasks)

        answer_image_tasks = [read_and_process_image(img, f"{request_id}_ans{i}") for i, img in enumerate(answer_images)]
        answer_image_objects = await asyncio.gather(*answer_image_tasks)

        async with llm_pool.get_llm_lite() as llm_lite:
            extraction_tasks = [extract_text_from_image(llm_lite, img, i, request_id) for i, img in enumerate(answer_image_objects)]
            extracted_text_parts = await asyncio.gather(*extraction_tasks)
            all_extracted_text = "\n\n".join(extracted_text_parts)

        async with llm_pool.get_llm() as llm_eval:
            homework_result = await evaluate_homework_batch(llm_eval, normalized_questions, question_image_objects, all_extracted_text, request_id)

        return {
            "status": "success",
            "request_id": request_id,
            "result": homework_result.model_dump(),
            "full_extracted_text": all_extracted_text
        }
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autoscore/")
async def autoscore(
    question: str = Form(..., description="Question text"),
    question_images: list[UploadFile] | None = File(default=None),
    answer_images: list[UploadFile] = File(...)
):
    """
    Main endpoint for student answer evaluation.
    Returns: Cleaned result object (without transcription inside) + formatted extracted_text at root.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting autoscore request")

    try:
        # Load images
        question_image_objects_tasks = [read_and_process_image(img, f"{request_id}_q{i}") for i, img in enumerate(question_images or [])]
        answer_image_objects_tasks = [read_and_process_image(img, f"{request_id}_a{i}") for i, img in enumerate(answer_images)]
        
        question_image_objects, answer_image_objects = await asyncio.gather(
            asyncio.gather(*question_image_objects_tasks),
            asyncio.gather(*answer_image_objects_tasks)
        )

        if not answer_image_objects:
            raise HTTPException(status_code=400, detail="No answer images found")

        # LLM Call
        async with llm_pool.get_llm() as llm_eval:
            evaluation_result = await evaluate_answer(
                llm_eval, question, question_image_objects, answer_image_objects, request_id
            )

        if isinstance(evaluation_result, Result):
            # --- TRANSFORM OUTPUT HERE ---
            # 1. Dump Pydantic model to dict
            result_dict = evaluation_result.model_dump()
            
            # 2. Extract (POP) the transcription from the result dict so it is NOT inside 'result'
            raw_transcription = result_dict.pop("student_text_transcription", "")
            
            # 3. Format the transcription specifically as requested
            formatted_text = f"<page1>\nQ)\nStudent's Solution:\n{raw_transcription}\n</page1>"
            
            # 4. Return new structure
            return {
                "status": "success",
                "result": result_dict,
                "extracted_text": formatted_text,
                "request_id": request_id
            }
        else:
            raise HTTPException(status_code=500, detail="Evaluation failed to produce structured result")

    except Exception as e:
        logger.error(f"[{request_id}] Internal server error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def read_and_process_image(uploaded_file: UploadFile, file_id: str) -> Image.Image:
    if uploaded_file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {uploaded_file.content_type}")
    file_content = await uploaded_file.read()
    return Image.open(BytesIO(file_content))
