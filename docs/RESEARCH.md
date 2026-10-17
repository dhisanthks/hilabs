# Related work and what we took from it

Survey done during development (2026-10-17) to check our design against similar systems and borrow ideas.

| Work | What it does | Reported result | What we took / tested |
|---|---|---|---|
| **SNOMED CT Entity Linking Challenge** (DrivenData 2024, winners repo `drivendataorg/snomed-ct-entity-linking`; JAMIA 2025 analysis) | Span detection + linking on MIMIC notes. 1st place (KIRI) = dictionary built from **training annotations + SNOMED names + OMOP synonyms**, keyed by *(section header, mention)*, rules for overlaps, SNOMED-relation post-processing. Other winners: NER + linker, LLM + FAISS retrieval + LLM classification. | macro char-IoU 0.42 (different metric, includes NER) | Confirms that **local annotation conventions** (training-term memory) beat general resources → our kNN-over-training-terms features are the strongest ranker signals. Section/context keys motivated the context-kNN (Exp10). |
| **n2c2 2019 Track 3** clinical concept normalization (SNOMED CT + RxNorm, JAMIA 2020) | Normalize pre-extracted spans in discharge summaries. Top system: **ensemble of 5 BERT models**; best rule-based multi-pass sieve 82.0%. | top 85.3%, median 77.3% accuracy | Closest public analogue to our task (spans given, map to SNOMED). Its labels are adjudicated on clean discharge summaries; ours include labs/procedures/devices and OCR noise. Ensembling several fine-tuned encoders is a known gain (candidate next step). |
| **xMEN** (Borchert et al., JAMIA 2024) | Modular toolkit: candidate generation = **char n-gram TF-IDF + SapBERT ensemble**, then a **trainable cross-encoder reranker** (softmax loss over 64 candidates, rank regularization, ~20 epochs). | SapBERT R@64 0.92 (Medline) | Almost exactly our architecture (char TF-IDF + SapBERT → reranker). Differences: we use a LightGBM LambdaRank stacker over many retrievers and feed the cross-encoder score into it (plays the role of their rank regularization: generator ranks stay in the model). Their 20-epoch reranker training suggests our 1-epoch cross-encoder is under-trained. |
| **MedCAT** (Kraljevic et al., AI in Medicine 2021) | Dictionary linking + **concept context embeddings** learned from unambiguous mentions; ambiguous names disambiguated by context similarity; similarity doubles as confidence. Design notes for v2 combine it with SapBERT, since SapBERT ignores context. | — | Implemented as **context-kNN features** (Exp10): similarity of the mention's surrounding text to contexts of training mentions of each candidate code. |
| **SapBERT** (Liu et al., NAACL 2021) | PubMedBERT trained contrastively on UMLS synonyms. | SOTA zero-shot biomedical EL | Our base encoder; won our 7-way benchmark. Fine-tuned further on our training pairs (+hard negatives). |
| **LLM reranking for ICD-10 coding** (several 2024–2026 papers: ColBERT/embedding retrieval → GPT-class reranker; RAG for ED and German oncology coding) | Retrieve top-k codes, LLM picks one. | single clean terms: 71% → 93% top-1 with LLM rerank; clinical text: 38–76% exact match; failures dominated by retrieval | Supports the cascade: high recall first, judgement model second. A GPT-class API is not allowed for this data (third-party retention), and a 4 GB GPU only fits ~3B local models, so the LLM stage is optional/low-confidence only. Exact-match on real clinical text in these papers (38–76%) is in line with our numbers. |

## Takeaways applied
1. Our architecture matches current best practice (xMEN-style hybrid retrieval + trained reranker).
2. Local conventions matter most → keep training-term memory (kNN) and fine-tune on in-domain pairs.
3. Context disambiguation (MedCAT) → context-kNN features.
4. Rerankers benefit from longer training (xMEN) → more cross-encoder epochs for the final model if time allows.
5. Ensembles of fine-tuned encoders (n2c2 winner) → second fine-tuned encoder as a candidate improvement.
6. Published exact-match accuracies on realistic clinical text (n2c2 85% on clean SNOMED-only spans; 38–76% for ICD from notes)
   frame the 95% target: it exceeds the best published results and our measured label self-consistency.

## Sources
- https://github.com/drivendataorg/snomed-ct-entity-linking
- https://drivendata.co/blog/snomed-ct-entity-linking-challenge-winners
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12361850
- https://academic.oup.com/jamia/article-abstract/27/10/1529/5910736
- https://www.atala.org/sites/default/files/TAL-61-2-2_Wang.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11671143 (xMEN)
- https://arxiv.org/pdf/2010.01165 (MedCAT)
- https://hackmd.io/@w-is-h/rJdNvgpFK (MedCAT v2 design notes)
- https://arxiv.org/pdf/2407.12849
- https://www.medrxiv.org/content/10.1101/2024.10.15.24315526.full.pdf
