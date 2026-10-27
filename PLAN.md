# Build plan: Chitthi

**Deadline:** 31 Oct 2026, 11:59 PM PDT = **Sun 1 Nov, 12:29 PM IST**. Aim to publish Sat night IST
and keep Sunday morning as buffer.

**Prize categories to list:** Gemma (featured, $200), since Gemma 4 does transcription and answers
and EmbeddingGemma does retrieval. The overall prize is judged mostly on writing.

## Day by day

| Day (IST) | Goal | Done when |
|---|---|---|
| **Tue 27 (tonight)** | Set up | Ollama is running, `gemma4:e4b` and `embeddinggemma` are pulled, `pytest` passes, Nehru English is fetched and ingested, and one real question gets a cited answer in the UI |
| **Wed 28** | OCR quality | Hindi: compare text layer, Gemma and Tesseract on 10 pages of *Pita ke Patra* and keep the best. Urdu: try Gemma on 5 pages of *Urdu-e-Mualla*. **By tonight, decide whether Urdu stays or becomes a "next steps" line.** |
| **Thu 29** | Answers you can trust | Write ~15 test questions and check citations, verified quotes and the silent answers. Test an English question that finds a Hindi passage. Tune the séance prompt. |
| **Fri 30** | Polish + demo | UI polish, a scan zoom that looks good, a 60–90s screen recording and GIF, the GitHub repo public with a README |
| **Sat 31** | Write the post | Fill the DEV template (below), add screenshots and the demo, proofread, **publish** |
| **Sun 1 (until 12:29 PM)** | Buffer | Only fixes; note any post-deadline commits in the README |

## Risks to watch

- **Urdu Nastaliq OCR** is the hardest part, and small models may struggle with it. Fallback: ship
  with Nehru (English + Hindi), the same letters in two languages, which makes a strong
  cross-language demo on its own.
- **RAM:** E4B needs about 12 GB of unified memory. With 24 GB or more, try `CHAT_MODEL=gemma4:26b` for
  better answers and keep E4B for OCR.
- **OCR speed:** Gemma vision takes seconds per page. Keep `pages =` ranges small until the pipeline is stable.
- **Rights:** check each scan's edition before you post screenshots.

## Post outline (DEV template)

- **What I Built**: suggestion: open with the idea that the ghost may only speak words it actually wrote.
- **Demo**: GIF of a séance answer, then a click on a citation that opens the original scan.
- **Code**: the repo link.
- **How I Built It**: the pipeline diagram from the README, the OCR choice per page, hybrid search
  and quote verification.
- **Why Open Innovation Matters**: letters are private and family archives shouldn't be uploaded
  anywhere. It works offline in places with poor connectivity, and open weights won't disappear the way a hosted API can.
- **The Spirit of It**: suggestion: honouring the past through words people left behind.
- **Prize Categories**: Gemma.
