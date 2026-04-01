# Lecture Conversion Workflow

## Process for Each Lecture

### Phase 1: Extract Content (One Page at a Time)

For each page in the PDF:

**Prompt to use with Mathpix Snip or Claude/ChatGPT:**
```
I have a handwritten lecture note page from a course on AI4Science/mechanics/numerical analysis. 
Please transcribe this page into markdown format with LaTeX math notation using:
- Display math: $$...$$
- Inline math: $...$
- Preserve section headings, bullet points, and structure
- Note any diagrams/figures that need to be recreated
- Include any annotations or side notes

Focus on accuracy over speed. Flag anything unclear with [UNCLEAR: description].
```

**Steps per page:**
1. Open `OldNotes\ENM5320\Lecture Notes\Lecture_2.pdf`
2. Take screenshot of page 1 (or extract as image)
3. Upload to Mathpix Snip OR ChatGPT/Claude
4. Use above prompt
5. Copy output to `LectureConversion\Lecture02\Lecture02_draft.md` under "Page 1 Notes"
6. Review and verify equations/content
7. Repeat for each subsequent page

### Phase 2: Consolidate and Organize

This is the **most critical phase** where raw OCR output becomes polished lecture notes.

#### 2.1 Initial Organization Prompt

After all pages are transcribed:

```
I have transcribed lecture notes split by page. Please help reorganize this into a coherent 
document with proper section headings (## for main topics, ### for subtopics). 
Identify main themes and create a logical structure. Suggest a descriptive title for the lecture.
Number major sections (1, 2, 3...) and subsections (1.1, 1.2...).
```

#### 2.2 Enhancement Guidelines (Based on Lecture_2.md Exemplar)

After initial organization, enhance the document with:

**Add Overview Section (Section 0):**
- Pedagogical overview explaining the lecture's context within the course
- Connect to previous material and motivate why these topics matter
- Explain the philosophy/approach being taken
- Set expectations for what students will learn (2-4 paragraphs)

**Include Historical Context:**
- Add historical background where relevant (e.g., Manhattan Project origins of FDM)
- Cite key papers/references with proper links
- Prefer Penn library links for textbooks: `[Author et al., Chapter X](Penn library URL)`
- Mention important contributors

**Add Pedagogical Commentary:**
- Explanatory paragraphs before complex sections explaining *why* we use techniques
- Remarks connecting concepts (e.g., "finite difference stencils are essentially CNNs")
- Clarify subtle distinctions (e.g., L¹ vs L² norms, natural vs energy norms)
- Motivate mathematical tools before diving into formalism

**Structure Best Practices:**
- Use numbered sections for major topics: `## 1. Topic Name`
- Use numbered subsections: `### 1.1 Subtopic`
- Include callouts: "**Important Notes:**", "**Critical observation:**"
- Label derivations clearly: "Step 1:", "Step 2:", etc.
- Bold key terms when first introduced

**Add Summary Section:**
- End with numbered list of main topics covered
- Include "**Key Takeaway:**" highlighting the most important concept
- Connect to next lecture if appropriate

#### 2.3 Example Structure Template

```markdown
# Lecture X: Descriptive Title

**Reference:** [Primary textbook with Penn library link]

**Topics:** Topic 1, Topic 2, Topic 3

---

## 0. Overview
[2-4 paragraphs: context, philosophy, motivation, learning objectives]

## 1. First Major Topic
[Introduction paragraph explaining motivation]

### 1.1 Subtopic with Details
[Content with clear derivations]

**Important Notes:**
- Key observation 1
- Key observation 2

## 2. Second Major Topic
[Pedagogical commentary explaining relevance]

### 2.1 Mathematical Framework
[Definitions, theorems]

### 2.2 Worked Example
**Step 1:** [First step]
**Step 2:** [Second step]
...

---

## Summary
This lecture covered:
1. First topic
2. Second topic
3. Third topic

**Key Takeaway:** [One sentence capturing the essence]
```

#### 2.4 Quality Checklist

Before moving to Phase 3:
- [ ] Overview section (Section 0) provides context and motivation
- [ ] All sections numbered and logically ordered
- [ ] Step-by-step derivations clearly labeled
- [ ] Pedagogical commentary added for subtle concepts
- [ ] Historical context included if relevant
- [ ] Connections to modern ML/applications noted
- [ ] Summary section with key takeaway exists
- [ ] References properly linked (Penn library for textbooks)
- [ ] Mathematical flow is clear and complete
- [ ] All LaTeX verified correct

**Output:** Create organized `LectureXX.md` following this structure

### Phase 3: Convert to HTML

Once markdown is finalized, use this prompt:

**Conversion prompt:**
```
Convert this markdown file to HTML following the template in 
agenticinstructions/HTML_LaTeX_Conversion_Guide.md. Include:
- KaTeX support for all math
- Penn color scheme (#011F5B, #990000, #82AFD3, #F2C100)
- Definition/theorem/example boxes where appropriate
- Back link to course schedule
```

**Output:** `NewMaterial/LectureXX_MonDD/TopicName.html`

---

## Free Tool Options

### Option A: Mathpix Snip (50 snips/month free)
- Best for math equations
- Desktop app or web interface
- Can process entire pages at once

### Option B: Claude with image uploads
- Upload page screenshots
- Use transcription prompt
- Good for mixed text/math
- Free tier available

### Option C: ChatGPT with GPT-4V
- Similar to Claude
- Can handle handwritten content
- Requires Plus subscription ($20/mo) for image uploads

---

## Verification Checklist (Per Page)

- [ ] All equations transcribed correctly
- [ ] Proper LaTeX delimiters ($ for inline, $$ for display)
- [ ] Section structure preserved
- [ ] Diagrams noted for recreation
- [ ] No OCR artifacts (common: l/1, O/0, | confusion)
- [ ] Special symbols correct (∇, ∂, ∫, ∑, etc.)

---

## Template for Each Lecture Folder

```
LectureConversion/
  LectureXX/
    LectureXX_draft.md          # Page-by-page raw transcription
    LectureXX.md                # Organized final markdown
    figures/                    # Any recreated diagrams
    LectureXX_verification.txt  # Notes on unclear sections
```
