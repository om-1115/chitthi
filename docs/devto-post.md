---
title: "Chitthi: A Séance Where the Ghost Can Only Quote Itself"
published: true
description: Ask old letters a question. A local Gemma 4 answers in the writer's voice, but every quote is checked word-for-word against the original scan.
tags: devchallenge, hf26challenge, ai, opensource
cover_image: https://raw.githubusercontent.com/om-1115/chitthi/master/docs/cover.png
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

## What I Built

**Chitthi** (चिट्ठी, Hindi for "a letter") lets you ask questions of people who left letters behind and hear the answer in their own words. Literally their own words: the ghost may only quote what it actually wrote.

Most "talk to a historical figure" chatbots are confident ventriloquists. Ask one about Nehru and you get a fluent paragraph with a quote that sounds right and appears nowhere. For a séance, that is the wrong kind of spooky.

Chitthi works the other way round. It reads scanned letters page by page, indexes them on your own machine, and answers only from the passages it retrieves:

- **Archivist mode** answers in the third person and cites every claim.
- **Séance mode** answers as the writer, in the first person, bound to the same passages.
- Every quoted phrase is checked **word-for-word** against the letters. Verified quotes are highlighted; a quote that isn't in the letters is flagged.
- Every citation jumps to its passage, and every passage opens **the original scan**.
- If the letters don't cover your question, it says so instead of inventing an answer.

The first collection is Jawaharlal Nehru's *Letters from a Father to His Daughter*, written in the summer of 1928 to ten-year-old Indira in Mussoorie, together with the 1931 Hindi translation of the same letters by Premchand, *पिता के पत्र पुत्री के नाम*. Two great writers, one set of letters, two scripts.

## Demo

![Séance mode: Nehru's answer cites passage 1, highlights two verified quotes, and opens the 1929 scan](https://raw.githubusercontent.com/om-1115/chitthi/master/docs/demo.gif)

**Full 97-second walkthrough with narration:** [chitthi-demo.mp4](https://github.com/om-1115/chitthi/blob/master/docs/chitthi-demo.mp4)

I asked, in séance mode: *"Why did you write these letters, and where was Indira?"* Gemma answered as Nehru:

> *I wrote them with a hope for all who might read them. I wished that they might "gradually begin to think of this world of ours as a large family of nations". Furthermore, I hoped that readers would find "a fraction of the pleasure that I had in the writing of them".*

Both quotes were found word-for-word in the 1929 foreword, and the citation opens that very page:

![The séance answer with two highlighted, verified quotes and a "2 quotes found word-for-word" badge](https://raw.githubusercontent.com/om-1115/chitthi/master/docs/seance-answer.png)

![The cited page, the foreword of the 1929 first edition, opened from the citation](https://raw.githubusercontent.com/om-1115/chitthi/master/docs/scan.png)

The same letters in Hindi, answered from Premchand's translation, with its quote (संसाररूपी पुस्तक, "the book of the world") verified against the Hindi text:

![Archivist mode answering in Hindi, with citations and a verified Hindi quote](https://raw.githubusercontent.com/om-1115/chitthi/master/docs/hindi-answer.png)

Everything in the demo is the real app running against Gemma 4 on a 16 GB M4 Mac. The only edit is cutting the 40 to 60 seconds Gemma takes to think.

## Code

{% github om-1115/chitthi %}

Python (FastAPI and SQLite) with a single-file web UI built on Pico CSS. No build step, and `pytest` runs fully offline with a stand-in model.

## How I Built It

```
scanned PDF ─► page by page: PDF text layer?  Gemma 4 vision?  Tesseract?
            ─► passages (≤ 900 chars, never crossing a page; Hindi split on ।)
            ─► SQLite full-text search  +  EmbeddingGemma vectors, fused
            ─► Gemma 4 answers as archivist or séance, citing [n]
            ─► every quote checked word-for-word against the passages
```

**Open models, all local through Ollama.**

- **Gemma 4 (E4B)** does two jobs: it transcribes scanned pages through its vision input, and it writes every answer.
- **EmbeddingGemma** embeds each passage (with its task prefixes) for semantic search, while SQLite FTS5 handles exact words. The two result lists are merged with reciprocal rank fusion, so a question can match exact names as well as paraphrases.

**The OCR is chosen per page, and measured.** This is where most of the work went. Scans on archive.org come with text layers of very mixed quality. The Hindi PDF's text layer *looked* perfect, since every character was Devanagari, but it repeated syllables (पुस्पुतक for पुस्तक) and dropped spaces. My script-based quality check gave it a perfect 1.00. So I wrote a detector for those repeats, then compared every method against a hand-checked transcription of one page (202 words):

![Word error rate on a Hindi page: PDF text layer 23.8%, Gemma 4 vision 5.0%, Tesseract 4.0%, Tesseract corrected by the text layer 0.5%](https://raw.githubusercontent.com/om-1115/chitthi/master/docs/ocr-comparison.png)

| Method | Word error rate |
|---|---|
| PDF text layer | 23.8% |
| Gemma 4 vision | 5.0% |
| Tesseract | 4.0% |
| Tesseract, corrected by the PDF's own text | **0.5%** |

The surprise was *how* each one failed. Gemma got the Hindi vowel signs right, but on other pages it silently dropped whole sentences and swapped in plausible words: महावत (mahout) became मोहब्बत (love), पत्थर (stone) became पन्चर. For an app whose whole promise is exact quotes, a fluent wrong word is worse than an obvious typo.

Tesseract failed the opposite way: nearly all its errors were one visual slip, reading है as हे and और as ओर. The broken text layer happens to spell exactly those words right. Aligning the two and taking the layer's vowel sign only where the words otherwise match fixed all 43 of those errors across the first ten pages.

So each tool does what it is honest at: Tesseract plus the text layer for the typeset Hindi, Gemma for the 1929 English scan (which has no text layer at all) and for the answers.

**Grounding is the product.** After Gemma answers, a checker removes citation numbers that point at passages that don't exist, normalises the Unicode, and verifies every quoted span word-for-word against the retrieved passages. An answer that cites nothing is treated as silence. The UI highlights verified quotes and strikes through any that can't be found.

## Why Does Open Innovation Matter?

Letters are among the most private things a family owns. Nobody should have to upload a grandparent's letters to someone else's server just to ask them a question. Chitthi runs entirely on a laptop: open weights through Ollama, SQLite on disk, and even the stylesheet is vendored into the repo. It works with the Wi-Fi off.

Open models also made the OCR work possible. I could run the same ten Hindi pages through three methods, compare them word by word, and choose per page, as often as I liked, at no cost per page. That comparison is what showed me *where* the vision model was unsafe for quotes, and that a decades-old open-source OCR engine, corrected by the document's own text, beat it.

And an archive needs to outlive its tools. Open weights on your own disk can't be deprecated or repriced. If a better open OCR or chat model appears, swapping it in is one environment variable.

## The Spirit of It

Día de los Muertos remembers people through what they loved and left behind. A letter may be the most direct thing a person leaves: their voice, addressed to someone they loved.

The moment it landed for me was the first séance answer. I asked Nehru why he wrote these letters, and the reply began *"My dear friend, I remember writing these letters to my daughter Indira during the summer of 1928."* Then the highlighted quote appeared, and the citation opened the page of the 1929 first edition where he says the same thing in the same words, nearly a century ago.

That is the séance I wanted: a ghost that can only say what it really said.

Next up: Ghalib's Urdu letters (Nastaliq OCR is the hard part). You can already add your own scans from the command line, and they never leave your machine.

## Prize Categories

**Best Use of Gemma:** Gemma 4 transcribes the scans and writes every answer, and EmbeddingGemma powers the semantic search, all running locally through Ollama.
