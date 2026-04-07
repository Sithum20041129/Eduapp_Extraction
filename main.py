from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from model_loader import ModelLoader
from token_logger import token_logger
from vertexai.generative_models import Part
from PIL import Image
import io
import os
import base64

# Subject-specific extraction prompts for better accuracy
SUBJECT_PROMPTS = {
    "Mathematics": """Extract all text and mathematical expressions from this image.
- Use LaTeX for formulas: \\frac{3}{4}, x^2, \\int, \\sum
- Preserve equation structure exactly as shown
- Include all questions, working, and answers
- Common patterns: algebraic equations, calculus, geometry, statistics""",
    
    "Physics": """Extract all text, formulas, and diagrams from this image.
- Use LaTeX for physics notation: F=ma, v=\\frac{d}{t}, E=mc^2
- Note any graphs, force diagrams, or circuit diagrams
- Units are critical: include all units (m/s, kg, N, J)
- Include all questions and numerical values""",
    
    "Chemistry": """Extract all text and chemical content from this image.
- Chemical formulas: H₂O, CH₃COOH (use subscripts/superscripts)
- Reactions: use → for reaction arrows
- Include molecular structures, equations, and lab procedure steps
- Note any diagrams or tables""",
    
    "Biology": """Extract all text and diagrams from this image.
- Preserve labeled diagrams, tables, and classifications
- Use italics for Latin/scientific names
- Include process flows, cycles, and hierarchies
- Extract all questions and descriptive content"""
}

def get_extraction_prompt(subject_name: Optional[str] = None, lesson_name: Optional[str] = None, doc_type: str = "question"):
    """
    Get context-primed extraction prompt
    
    Args:
        subject_name: Subject name (e.g., "Mathematics")
        lesson_name: Lesson name (e.g., "Trigonometry", "Calculus")
        doc_type: Type of document - "question", "modelanswer", or "handwritten"
    """
    context_prefix = ""
    
    # Build context information
    if subject_name or lesson_name:
        context_parts = []
        if subject_name:
            context_parts.append(f"Subject: {subject_name}")
        if lesson_name:
            context_parts.append(f"Lesson: {lesson_name}")
        context_prefix = f"Context: {', '.join(context_parts)}\n\n"
    
    # Get base subject-specific prompt
    base_prompt = ""
    if subject_name and subject_name in SUBJECT_PROMPTS:
        base_prompt = SUBJECT_PROMPTS[subject_name]
    else:
        base_prompt = """Extract absolutely all text, tables, and mathematical formulas from this image.
Output mathematical formulas in LaTeX format. Do not add any conversational text, just the extracted content."""
    
    # Add handwritten-specific instructions
    if doc_type == "handwritten":
        handwritten_instructions = """
**CRITICAL - Handwritten Student Answer:**
- Transcribe EXACTLY what is written
- **DO NOT correct mathematical errors or logical mistakes**
- If handwriting is unclear, infer the most likely characters based on the context above
- Preserve the student's original work and intent, including any mistakes
- This is for grading purposes - we need the student's actual answer, not a corrected version"""
        return context_prefix + handwritten_instructions + "\n\n" + base_prompt
    
    # For questions and model answers, just use context + base prompt
    return context_prefix + base_prompt



app = FastAPI(title="EduApp AI Service")

# Initialize model loader
model_loader = ModelLoader.get_instance()




class ExtractionResponse(BaseModel):
    extracted_text: str




class BatchExtractionItem(BaseModel):
    id: str
    extracted_text: str

class BatchExtractionResponse(BaseModel):
    results: List[BatchExtractionItem]
    total_processed: int

@app.on_event("startup")
async def startup_event():
    # Preload model on startup to avoid delay on first request
    # model_loader.load_model() # Commented out for dev speed, uncomment for prod
    pass

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "EduApp AI Service"}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_text(
    file: UploadFile = File(...), 
    subject: Optional[str] = Form(default=None),
    lesson: Optional[str] = Form(default=None),
    docType: Optional[str] = Form(default="question")
):
    try:
        # DEBUG: Log received parameters with VISIBLE formatting
        print("=" * 60)
        print("[EXTRACTION DEBUG] === RECEIVED PARAMS ===")
        print(f"  subject: '{subject}' (type: {type(subject).__name__})")
        print(f"  lesson:  '{lesson}' (type: {type(lesson).__name__})")
        print(f"  docType: '{docType}' (type: {type(docType).__name__})")
        print("=" * 60)
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Use context-primed prompt for better extraction
        prompt = get_extraction_prompt(subject, lesson, docType if docType else "question")
        
        # DEBUG: Log the FULL generated prompt
        print("[EXTRACTION DEBUG] === FULL PROMPT ===")
        print(prompt)
        print("=" * 60)
        
        extracted, usage = model_loader.predict(image, prompt)
        
        # Log token usage
        token_logger.log_usage(
            operation="extract",
            input_tokens=usage['prompt_token_count'],
            output_tokens=usage['candidates_token_count'],
            doc_type=docType or "question",
            subject=subject,
            lesson=lesson,
            image_included=usage['image_included'],
            batch_size=1
        )
        
        return ExtractionResponse(extracted_text=extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-batch", response_model=BatchExtractionResponse)
async def extract_text_batch(
    files: List[UploadFile] = File(...),
    ids: str = Form(...),  # Comma-separated IDs matching each file
    contexts: Optional[str] = Form(None),  # JSON mapping id -> {subject, lesson, docType}
    subject: Optional[str] = Form(None),  # Fallback if contexts not provided
    lesson: Optional[str] = Form(None),  # Fallback if contexts not provided
    docType: Optional[str] = Form(default="question")  # Fallback if contexts not provided
):
    """
    Batch extract text from multiple images in a single request with per-image context.
    
    - files: List of image files to process
    - ids: Comma-separated list of IDs (e.g., "q1,q2,q3") matching each file
    - contexts: Optional JSON string mapping each ID to its context:
        Example: '{"q1": {"subject": "Mathematics", "lesson": "Trigonometry", "docType": "question"},
                   "q2": {"subject": "Physics", "lesson": "Mechanics", "docType": "question"}}'
    - subject, lesson, docType: Fallback values if contexts not provided (all images use same context)
    
    This reduces API overhead by processing multiple images together.
    """
    try:
        import json
        
        id_list = [id.strip() for id in ids.split(",")]
        
        if len(files) != len(id_list):
            raise HTTPException(
                status_code=400, 
                detail=f"Mismatch: {len(files)} files but {len(id_list)} IDs provided"
            )
        
        if len(files) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 images per batch request"
            )
        
        # Parse contexts if provided
        context_map = {}
        if contexts:
            try:
                context_map = json.loads(contexts)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in contexts parameter: {str(e)}"
                )
        
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for file, item_id in zip(files, id_list):
            try:
                # Get context for this specific image
                if item_id in context_map:
                    ctx = context_map[item_id]
                    img_subject = ctx.get("subject", subject)
                    img_lesson = ctx.get("lesson", lesson)
                    img_docType = ctx.get("docType", docType if docType else "question")
                else:
                    # Use fallback values
                    img_subject = subject
                    img_lesson = lesson
                    img_docType = docType if docType else "question"
                
                # Generate prompt for this specific image
                prompt = get_extraction_prompt(img_subject, img_lesson, img_docType)
                
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                extracted, usage = model_loader.predict(image, prompt)
                
                # Aggregate token usage
                total_input_tokens += usage['prompt_token_count']
                total_output_tokens += usage['candidates_token_count']
                
                results.append(BatchExtractionItem(
                    id=item_id,
                    extracted_text=extracted
                ))
            except Exception as e:
                # Add error result but continue processing other images
                results.append(BatchExtractionItem(
                    id=item_id,
                    extracted_text=f"[ERROR: {str(e)}]"
                ))
        
        # Log aggregated batch usage
        token_logger.log_usage(
            operation="extract_batch",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            doc_type=docType or "question",
            subject=subject,  # Use fallback subject for logging
            lesson=lesson,    # Use fallback lesson for logging
            image_included=True,
            batch_size=len(results)
        )
        
        return BatchExtractionResponse(
            results=results,
            total_processed=len(results)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ParsedQuestion(BaseModel):
    questionNumber: int
    text: str
    type: str  # "MCQ" or "ESSAY"
    marks: Optional[int] = None
    options: Optional[List[dict]] = None  # [{"text": "...", "isCorrect": true/false}]
    modelAnswer: Optional[str] = None
    startPage: Optional[int] = None  # 1-indexed page number where question starts
    endPage: Optional[int] = None    # 1-indexed page number where question ends

class PaperExtractionResponse(BaseModel):
    questions: List[ParsedQuestion]
    totalQuestions: int
    paperTitle: Optional[str] = None
    questionImages: Optional[List[Optional[str]]] = None  # base64 PNG - one per question (null if failed)


@app.post("/extract-paper", response_model=PaperExtractionResponse)
async def extract_paper(
    questionPaper: UploadFile = File(...),
    answerPaper: Optional[UploadFile] = File(None),
    subject: Optional[str] = Form(None),
    lesson: Optional[str] = Form(None),
    paperType: Optional[str] = Form("MIXED"),
    defaultMarks: Optional[int] = Form(1)
):
    """
    Extract structured questions from a PDF question paper.
    Optionally accepts an answer paper PDF to pair model answers with questions.
    
    Returns a JSON list of parsed questions with text, type, marks, options, and answers.
    """
    try:
        import json as json_module

        print("=" * 60)
        print("[PAPER EXTRACTION] === RECEIVED PARAMS ===")
        print(f"  subject: '{subject}'")
        print(f"  lesson: '{lesson}'")
        print(f"  paperType: '{paperType}'")
        print(f"  defaultMarks: {defaultMarks}")
        print(f"  questionPaper: {questionPaper.filename} ({questionPaper.content_type})")
        print(f"  answerPaper: {answerPaper.filename if answerPaper else 'None'}")
        print("=" * 60)

        # Read PDF bytes
        question_bytes = await questionPaper.read()
        
        # Determine mime type
        q_mime = questionPaper.content_type or "application/pdf"
        if not q_mime.startswith("application/pdf") and not q_mime.startswith("image/"):
            raise HTTPException(status_code=400, detail="Question paper must be a PDF or image file")

        # Convert PDF pages to PIL images, collect text blocks and vector drawings
        page_pil_images = []    # PIL Images (RGB), one per page
        page_text_data  = []    # Text blocks per page (for question boundary + compaction)
        page_drawing_data = []  # Vector drawings per page (for diagram detection)
        if q_mime.startswith("application/pdf"):
            try:
                import fitz  # PyMuPDF
                pdf_doc = fitz.open(stream=question_bytes, filetype="pdf")
                scale = 200 / 72  # 200 DPI
                mat = fitz.Matrix(scale, scale)
                print(f"[PAPER EXTRACTION] Converting {len(pdf_doc)} PDF pages to PIL images...")
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    # Force RGB colorspace — avoids issues with CMYK/grayscale PDFs
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    # Direct PIL conversion via raw samples — no intermediate PNG encode/decode
                    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_pil_images.append(pil_img)
                    # Text blocks: (x0,y0,x1,y1,text,block_no,block_type) in PDF pts
                    # Words: (x0,y0,x1,y1,word,block_no,line_no,word_no) in PDF pts
                    page_text_data.append({
                        "blocks": page.get_text("blocks"),
                        "words": page.get_text("words"),
                        "page_width": page.rect.width,  # PDF pts, used for left-margin ratio
                        "scale": scale,
                    })
                    # Vector drawings (coordinate axes, shapes, dashed lines, etc.)
                    page_drawing_data.append(page.get_drawings())
                pdf_doc.close()
                print(f"[PAPER EXTRACTION] Converted {len(page_pil_images)} pages")
            except ImportError:
                print("[PAPER EXTRACTION] PyMuPDF not installed, skipping page image conversion")
            except Exception as img_err:
                print(f"[PAPER EXTRACTION] PDF→PIL conversion failed: {img_err}")
                import traceback; traceback.print_exc()

        # Build the parts list for Gemini
        parts = []
        question_part = Part.from_data(data=question_bytes, mime_type=q_mime)
        parts.append(question_part)

        answer_part = None
        if answerPaper:
            answer_bytes = await answerPaper.read()
            a_mime = answerPaper.content_type or "application/pdf"
            answer_part = Part.from_data(data=answer_bytes, mime_type=a_mime)
            parts.append(answer_part)

        # Build context string
        context = ""
        if subject:
            context += f"Subject: {subject}\n"
        if lesson:
            context += f"Lesson/Topic: {lesson}\n"

        # Build the extraction prompt
        type_instruction = ""
        if paperType == "MCQ":
            type_instruction = "All questions in this paper are Multiple Choice Questions (MCQ). Each question MUST have options."
        elif paperType == "ESSAY":
            type_instruction = "All questions in this paper are Essay/Short Answer type. Do NOT create options for any question."
        else:
            type_instruction = "This paper may contain a mix of MCQ and Essay questions. Identify the type of each question based on whether it has options/choices listed."

        answer_instruction = ""
        if answerPaper:
            answer_instruction = """
A second document (answer paper) is also provided. Match each answer to its corresponding question by question number.
For MCQ questions, mark the correct option as isCorrect: true based on the answer paper.
For Essay questions, include the model answer text in the "modelAnswer" field."""
        
        prompt = f"""{context}
{type_instruction}
{answer_instruction}

Analyze the provided question paper document(s) and extract ALL questions into a structured JSON format.

IMPORTANT RULES:
1. Extract EVERY question from the paper, do not skip any
2. Preserve the exact question text as written in the paper
3. For MCQ questions, extract ALL options exactly as written
4. Identify marks for each question if shown (look for patterns like "(5 marks)", "[2]", etc.)
5. If marks are not specified, use {defaultMarks} as the default
6. Preserve mathematical expressions, formulas, and special notation
7. If a question has sub-parts (a, b, c...), combine them into one question text preserving the structure
8. Return ONLY valid JSON, no markdown code blocks, no explanation text

Return the following JSON structure:
{{
    "paperTitle": "Title of the paper if visible, otherwise null",
    "questions": [
        {{
            "questionNumber": 1,
            "text": "The full question text including any sub-parts",
            "type": "MCQ" or "ESSAY",
            "marks": number or null,
            "startPage": 1,
            "endPage": 2,
            "options": [
                {{"text": "Option A text", "isCorrect": false}},
                {{"text": "Option B text", "isCorrect": true}}
            ] or null for essay questions,
            "modelAnswer": "The model answer text if available, otherwise null"
        }}
    ]
}}

IMPORTANT:
- "startPage" must be the 1-indexed page number in the QUESTION PAPER PDF where this question FIRST appears.
- "endPage" must be the 1-indexed page number where this question ENDS (including the last sub-part). If the question is only on one page, endPage equals startPage."""

        print("[PAPER EXTRACTION] === PROMPT ===")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("=" * 60)

        # Send to Gemini
        raw_response, usage = model_loader.predict_with_parts(parts, prompt)

        # Log token usage
        token_logger.log_usage(
            operation="extract_paper",
            input_tokens=usage['prompt_token_count'],
            output_tokens=usage['candidates_token_count'],
            doc_type="paper",
            subject=subject,
            lesson=lesson,
            image_included=True,
            batch_size=1
        )

        print(f"[PAPER EXTRACTION] Raw response length: {len(raw_response)}")
        print(f"[PAPER EXTRACTION] Raw response preview: {raw_response[:200]}...")

        # Clean up the response - remove markdown code blocks if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        def fix_latex_json_escapes(text: str) -> str:
            """
            Gemini returns LaTeX math inside JSON strings, e.g.
              "text": "... $I_n = \\int_0^1 x^n dx$ ..."
            LaTeX backslash commands like \\int, \\frac, \\alpha are not valid
            JSON escape sequences and cause json.loads() to raise JSONDecodeError.

            This function walks the string character-by-character and doubles any
            backslash that is not part of a valid JSON escape sequence, so that
            json.loads() parses them as literal backslash + letter (which is the
            correct representation of LaTeX commands in JSON strings).

            Valid JSON escapes preserved as-is: \\" \\/ \\\\ \\n \\r \\t \\uXXXX
            Everything else (\\i, \\f, \\a, \\b, \\s, \\p, …) is doubled.
            Note: \\f (form-feed) and \\b (backspace) are intentionally treated as
            LaTeX because Gemini never emits them as control characters — they are
            always LaTeX commands like \\frac or \\begin.
            """
            result: list = []
            i = 0
            n = len(text)
            while i < n:
                ch = text[i]
                if ch != '\\':
                    result.append(ch)
                    i += 1
                    continue
                # We have a backslash — peek at the next character
                if i + 1 >= n:
                    result.append('\\\\')   # trailing backslash → escape it
                    i += 1
                    continue
                nxt = text[i + 1]
                if nxt in ('"', '\\', '/', 'n', 'r', 't'):
                    # Intentional JSON escape — keep exactly as-is
                    result.append(ch)
                    result.append(nxt)
                    i += 2
                elif nxt == 'u' and i + 5 <= n:
                    hex4 = text[i + 2: i + 6]
                    if len(hex4) == 4 and all(c in '0123456789abcdefABCDEF' for c in hex4):
                        # Valid \uXXXX unicode escape
                        result.append(text[i: i + 6])
                        i += 6
                    else:
                        # \u not followed by 4 hex digits — LaTeX \upsilon etc.
                        result.append('\\\\')
                        i += 1
                else:
                    # Invalid JSON escape (LaTeX command: \int, \frac, \alpha …)
                    result.append('\\\\')
                    i += 1
            return ''.join(result)

        # Parse the JSON response — with automatic LaTeX escape repair on failure
        try:
            parsed = json_module.loads(cleaned)
        except json_module.JSONDecodeError as e:
            print(f"[PAPER EXTRACTION] JSON parse error: {e} — attempting LaTeX escape fix")
            fixed = fix_latex_json_escapes(cleaned)
            try:
                parsed = json_module.loads(fixed)
                print("[PAPER EXTRACTION] JSON parsed successfully after LaTeX escape fix")
            except json_module.JSONDecodeError as e2:
                print(f"[PAPER EXTRACTION] JSON still invalid after fix: {e2}")
                print(f"[PAPER EXTRACTION] Cleaned response: {cleaned[:500]}")
                raise HTTPException(
                    status_code=500,
                    detail=f"AI returned invalid JSON. Raw response preview: {cleaned[:200]}"
                )

        # Validate and build response
        questions = []
        raw_questions = parsed.get("questions", [])
        
        for q in raw_questions:
            options = None
            if q.get("options") and q.get("type", "").upper() == "MCQ":
                options = [
                    {"text": opt.get("text", ""), "isCorrect": bool(opt.get("isCorrect", False))}
                    for opt in q["options"]
                ]
            
            questions.append(ParsedQuestion(
                questionNumber=q.get("questionNumber", len(questions) + 1),
                text=q.get("text", ""),
                type=q.get("type", "ESSAY").upper(),
                marks=q.get("marks", defaultMarks),
                options=options,
                modelAnswer=q.get("modelAnswer"),
                startPage=q.get("startPage"),
                endPage=q.get("endPage")
            ))

        print(f"[PAPER EXTRACTION] Successfully parsed {len(questions)} questions")

        # --- Helper: find where a question number starts on a page (pixel y-coordinate) ---
        def find_question_y_on_page(page_idx: int, question_num: int,
                                    min_y_px: int = 0) -> Optional[int]:
            """
            Find the y-coordinate (pixels) where question_num begins on the page.

            Strategy: use word-level extraction and restrict to the LEFT MARGIN
            (leftmost 13% of page width). Cambridge exam papers always put question
            numbers as isolated words in the left margin.

            min_y_px  — ignore any match whose pixel-y is below this threshold.
                        Used to skip page-header content (page numbers, running
                        titles) when searching for a question's own start position.

            Accepted forms: "28", "28.", "28)" — bare number + optional punctuation.
            Falls back to block-level scanning if word data is unavailable.
            Returns pixel y-coordinate of the earliest match, or None.
            """
            import re
            if page_idx >= len(page_text_data):
                return None
            info = page_text_data[page_idx]
            scale      = info["scale"]
            page_width = info.get("page_width", 595)   # PDF pts, default A4
            words      = info.get("words")

            # Accept the number optionally followed by one punctuation char
            word_re = re.compile(rf'^{question_num}[\.\)\:]?$')

            # Left-margin threshold: leftmost 13% of page width.
            # Cambridge question numbers sit at ~9–12% from the left edge;
            # option labels / inline values begin at ~13–17%.
            LEFT_MARGIN_MAX = page_width * 0.13

            best_y: Optional[int] = None

            # ── Primary: word-level search with left-margin + min-y filter ───
            if words:
                for word_tuple in words:
                    if len(word_tuple) < 5:
                        continue
                    wx0, wy0, wx1, wy1, word_text = word_tuple[:5]
                    # Must be in the left margin
                    if wx0 > LEFT_MARGIN_MAX:
                        continue
                    y_px = int(wy0 * scale)
                    # Skip header zone (page numbers, running titles, etc.)
                    if y_px < min_y_px:
                        continue
                    if word_re.match(word_text.strip()):
                        if best_y is None or y_px < best_y:
                            best_y = y_px

            # ── Fallback: block-level scan (no word data) ─────────────────────
            if best_y is None:
                inline_pats = [
                    re.compile(rf'^\s*{question_num}[\.\)\s]'),
                    re.compile(rf'^\s*\({question_num}\)'),
                    re.compile(rf'^\s*{question_num}$'),
                ]
                margin_pat = re.compile(rf'^\s*{question_num}[\.\):]?\s*$')
                for block in info["blocks"]:
                    if len(block) < 7:
                        continue
                    x0, y0, x1, y1, text, _bno, btype = block
                    if btype != 0:
                        continue
                    stripped = text.strip()
                    if margin_pat.match(stripped):
                        y_px = int(y0 * scale)
                        if best_y is None or y_px < best_y:
                            best_y = y_px
                        continue
                    for line_idx, line in enumerate(stripped.split('\n')[:5]):
                        for pat in inline_pats:
                            if pat.match(line.strip()):
                                block_h = y1 - y0
                                line_y  = y0 + block_h * (line_idx / max(len(stripped.split('\n')), 1))
                                y_px    = int(line_y * scale)
                                if best_y is None or y_px < best_y:
                                    best_y = y_px
                                break

            if best_y is not None:
                print(f"    [find_q_y] Q{question_num} on page {page_idx+1}: y={best_y}px")
            else:
                print(f"    [find_q_y] Q{question_num} on page {page_idx+1}: NOT FOUND")
            return best_y

        # --- Compact a page strip to only its content rows ---
        def compact_to_text_blocks(
            img: Image.Image,
            page_idx: int,
            crop_top_page_px: int,
            crop_bottom_page_px: int,
            excl_top_page_px: int = 0,
            excl_bot_page_px: int = 0,
        ) -> Image.Image:
            """
            Remove answer spaces from a question image using PyMuPDF's structural data.

            Three content sources:
              1. Text blocks  — question text, marks, sub-question labels
              2. Raster image blocks — embedded photos / scanned diagrams
              3. Solid vector drawings — coordinate axes, geometric shapes, curves

            Two answer-space sources that are explicitly excluded:
              A. Text blocks whose content is only repeated dots/dashes
                 (Cambridge prints answer lines as '......' text characters)
              B. Vector drawings that are dashed AND wide (> 65 % of page width)
                 (Cambridge also uses dashed PDF paths for some answer lines)
            """
            import re
            if not page_text_data or page_idx >= len(page_text_data):
                return img

            info = page_text_data[page_idx]
            scale = info["scale"]
            full_page_h  = page_pil_images[page_idx].height
            page_width_px = page_pil_images[page_idx].width

            CONTENT_PAD = 10
            MERGE_GAP   = 45   # merge bands within this px distance
            STRIP_GAP   = 14   # whitespace inserted between output strips

            # Characters that mark a text block as an answer-line placeholder
            ANSWER_LINE_CHARS = frozenset(
                '.−_·•–—…⋯―\u2026\u22ef\u00b7\u2015\u2010\u2012\u2013\u2014'
            )

            bands: list = []

            # ── 1. Text and raster-image blocks ─────────────────────────────────
            for block in info["blocks"]:
                if len(block) < 7:
                    continue
                x0, y0, x1, y1, text, _bno, btype = block

                if btype == 0:          # text block
                    if not text.strip():
                        continue
                    # Detect answer-line text: only repeated dot/dash characters
                    cleaned = re.sub(r'\s', '', text)
                    if len(cleaned) >= 10 and all(c in ANSWER_LINE_CHARS for c in cleaned):
                        continue        # all answer-line chars → skip

                elif btype == 1:        # embedded raster image (diagram, graph)
                    if (x1 - x0) * scale < 40 or (y1 - y0) * scale < 20:
                        continue        # too small to matter
                else:
                    continue

                pp0 = int(y0 * scale)
                pp1 = int(y1 * scale)

                if excl_top_page_px and pp0 < excl_top_page_px:
                    continue
                if excl_bot_page_px and pp1 > full_page_h - excl_bot_page_px:
                    continue
                if pp1 < crop_top_page_px or pp0 > crop_bottom_page_px:
                    continue

                rel0 = max(0, pp0 - crop_top_page_px - CONTENT_PAD)
                rel1 = min(img.height, pp1 - crop_top_page_px + CONTENT_PAD)
                if rel1 > rel0:
                    bands.append((rel0, rel1))

            # ── 2. Vector drawings (coordinate axes, shapes, diagram lines) ─────
            # Include solid drawings.  Skip dashed drawings that span most of the
            # page width — those are answer lines in vector form.
            # Narrow dashed lines (< 65 % page width) are kept — they're diagram
            # details like dimension indicators (e.g. the "h" line in mechanics).
            if page_drawing_data and page_idx < len(page_drawing_data):
                for drawing in page_drawing_data[page_idx]:
                    rect = drawing.get("rect")
                    if not rect:
                        continue
                    dx0, dy0, dx1, dy1 = rect
                    pp0  = int(dy0 * scale)
                    pp1  = int(dy1 * scale)
                    w_px = int((dx1 - dx0) * scale)
                    h_px = pp1 - pp0

                    if h_px < 15 or w_px < 15:
                        continue        # too small

                    dashes = drawing.get("dashes") or ""
                    is_dashed = dashes and dashes not in ("", "[] 0")
                    is_wide   = w_px > page_width_px * 0.65

                    if is_dashed and is_wide:
                        continue        # wide dashed drawing = answer line

                    if h_px < 6:
                        continue        # flat horizontal stroke = answer line element

                    if excl_top_page_px and pp0 < excl_top_page_px:
                        continue
                    if excl_bot_page_px and pp1 > full_page_h - excl_bot_page_px:
                        continue
                    if pp1 < crop_top_page_px or pp0 > crop_bottom_page_px:
                        continue

                    rel0 = max(0, pp0 - crop_top_page_px - CONTENT_PAD)
                    rel1 = min(img.height, pp1 - crop_top_page_px + CONTENT_PAD)
                    if rel1 > rel0:
                        bands.append((rel0, rel1))

            # ── 3. Merge bands and build compacted output ────────────────────────
            if not bands:
                print(f"      [compact] no content found — returning unchanged")
                return img

            bands.sort()
            merged = [list(bands[0])]
            for s, e in bands[1:]:
                if s - merged[-1][1] <= MERGE_GAP:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])

            total_content = sum(e - s for s, e in merged)
            new_h = total_content + STRIP_GAP * max(0, len(merged) - 1)

            if new_h >= img.height * 0.85:
                print(f"      [compact] only {100 - int(new_h/img.height*100)}% removable "
                      f"— returning unchanged")
                return img

            result = Image.new("RGB", (img.width, new_h), (255, 255, 255))
            y_out = 0
            for i, (s, e) in enumerate(merged):
                result.paste(img.crop((0, s, img.width, e)), (0, y_out))
                y_out += e - s
                if i < len(merged) - 1:
                    y_out += STRIP_GAP

            removed_pct = 100 - int(new_h / img.height * 100)
            print(f"      [compact] {img.height}px → {new_h}px "
                  f"({len(merged)} strips, {removed_pct}% removed)")
            return result

        # --- Generate per-question cropped + compacted images ---
        # Cambridge papers at 200 DPI:
        #   Header (page number + any running title) occupies roughly the top 160 px.
        #   Footer (© line + barcode) occupies roughly the bottom 85 px.
        HDR_EXCL_PX = 160  # raised from 45 — must clear the printed page number
        FTR_EXCL_PX = 85

        # Track the crop_bottom used for each (question_index, page_index) so that
        # the NEXT question can fall back to it when its own start cannot be found.
        prev_crop_bottom: dict = {}   # key: page_idx → last known crop_bottom on that page

        question_images_b64: List[Optional[str]] = []
        if page_pil_images:
            print(f"[PAPER EXTRACTION] Generating question images for {len(questions)} questions...")
            for qi, q in enumerate(questions):
                sp = (q.startPage or 1) - 1   # 0-indexed start page
                ep = (q.endPage or q.startPage or 1) - 1   # 0-indexed end page
                sp = max(0, min(sp, len(page_pil_images) - 1))
                ep = max(sp, min(ep, len(page_pil_images) - 1))

                next_q = questions[qi + 1] if qi + 1 < len(questions) else None
                next_q_num = next_q.questionNumber if next_q else None

                try:
                    compacted_strips = []
                    for pi in range(sp, ep + 1):
                        page_img = page_pil_images[pi]
                        crop_top    = 0
                        crop_bottom = page_img.height

                        if pi == sp:
                            # Pass HDR_EXCL_PX as min_y_px so page-number words
                            # at the top of the page are never mistaken for Q1/Q2/Q3
                            y_start = find_question_y_on_page(
                                pi, q.questionNumber, min_y_px=HDR_EXCL_PX)
                            if y_start is not None:
                                crop_top = max(0, y_start - 15)
                                print(f"  Q{q.questionNumber}: start found at y={y_start}px on page {pi+1}")
                            else:
                                # Fallback: use the previous question's crop_bottom on
                                # this same page as our crop_top.  This handles the
                                # common case where the question number is not found
                                # (e.g. Q3 on a page that also prints "3" as page-number
                                # in a different position) but we know where Q2 ended.
                                fallback = prev_crop_bottom.get(pi)
                                if fallback is not None:
                                    crop_top = fallback
                                    print(f"  Q{q.questionNumber}: start NOT found — "
                                          f"using prev boundary y={crop_top}px on page {pi+1}")
                                else:
                                    print(f"  Q{q.questionNumber}: start NOT found, no fallback")

                        if pi == ep and next_q_num:
                            # Always search for the next question on this page as a boundary.
                            y_end = find_question_y_on_page(pi, next_q_num, min_y_px=HDR_EXCL_PX)
                            if y_end is not None and y_end > crop_top + 80:
                                crop_bottom = min(y_end + 5, page_img.height)
                                print(f"  Q{q.questionNumber}: bottom boundary Q{next_q_num} at y={y_end}px")
                                print(f"  Q{q.questionNumber}: found Q{next_q_num} boundary at y={y_end}px on page {pi+1}")
                            elif y_end is not None:
                                print(f"  Q{q.questionNumber}: ignored suspicious Q{next_q_num} boundary at y={y_end}px "
                                      f"(only {y_end - crop_top}px below crop_top={crop_top}px — likely false positive)")

                        raw_strip = page_img.crop((0, crop_top, page_img.width, crop_bottom))

                        # Record crop_bottom for this page so the next question can
                        # use it as a fallback crop_top if its own start is not found.
                        prev_crop_bottom[pi] = crop_bottom

                        # Header exclusion only on pages that are NOT the start page
                        # (the start page's crop already begins below the header).
                        # Footer exclusion on every page so Cambridge © line is removed.
                        hdr = HDR_EXCL_PX if pi != sp else 0

                        compacted = compact_to_text_blocks(
                            raw_strip, pi,
                            crop_top_page_px=crop_top,
                            crop_bottom_page_px=crop_bottom,
                            excl_top_page_px=hdr,
                            excl_bot_page_px=FTR_EXCL_PX,
                        )
                        compacted_strips.append(compacted)

                    # Stitch compacted strips vertically
                    if len(compacted_strips) == 1:
                        final_img = compacted_strips[0]
                    else:
                        max_w   = max(s.width  for s in compacted_strips)
                        total_h = sum(s.height for s in compacted_strips)
                        final_img = Image.new("RGB", (max_w, total_h), (255, 255, 255))
                        y_off = 0
                        for s in compacted_strips:
                            final_img.paste(s, (0, y_off))
                            y_off += s.height

                    buf = io.BytesIO()
                    final_img.save(buf, format="PNG", optimize=True)
                    question_images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
                    print(f"  Q{q.questionNumber}: pages {sp+1}-{ep+1} → "
                          f"final {final_img.width}x{final_img.height}px")

                except Exception as q_err:
                    print(f"  Q{q.questionNumber}: failed ({q_err}), fallback to full page")
                    import traceback; traceback.print_exc()
                    try:
                        buf = io.BytesIO()
                        page_pil_images[sp].save(buf, format="PNG", optimize=True)
                        question_images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
                    except Exception:
                        question_images_b64.append(None)

            print(f"[PAPER EXTRACTION] Generated {len(question_images_b64)} question images")

        return PaperExtractionResponse(
            questions=questions,
            totalQuestions=len(questions),
            paperTitle=parsed.get("paperTitle"),
            questionImages=question_images_b64 if question_images_b64 else None
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PAPER EXTRACTION] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
