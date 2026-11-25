# Literature source audit — 2026-11-26

The review is a focused narrative review. Numerical claims below refer to the inspected versions only; absence of a measurement in the reviewed sections does not establish absence from all versions or supplementary materials. No external published numbers have been inserted into the archived local experiment tables.

| Key | Source and inspected material | Numerical/technical claim | Access caveat |
|---|---|---|---|
| chen2025mrd | https://arxiv.org/html/2504.07724 sections 4.1–5.2, Table 3 | 746+130 diseases; Qwen2-7B-Instruct MRD-RAG-PMH 3.76, single round 3.26; retrieval bypass; 3 dialogue rounds, k=5 | HTML preprint; no comparable latency/QPS found; later IEEE publication record exists, not used for numerical claims |
| li2025garmle | https://arxiv.org/pdf/2506.21615, p. 8 (zero-index 8), section IV-E and abstract | hypertension; P@1 .940, MRR .964, Hits@3 1.0 at .72 semantic threshold; 568M combined embedding parameters | PDF text accessible, page screenshot attempt failed; numbers checked in extracted text; not journal-version results |
| li2025garmlejournal | https://pubmed.ncbi.nlm.nih.gov/41364573/ and https://researchnow.flinders.edu.au/en/publications/refine-medical-diagnosis-using-generation-augmented-retrieval-and/ | online journal DOI and expanded hypertension/CHD scope | indexed abstract/metadata only; PubMed direct page blocked; author list abbreviated in bibliography |
| khatibi2025cdf | https://arxiv.org/abs/2504.12560 and https://arxiv.org/pdf/2504.12560, Table 1, limitations, appendix A.3 | MedMCQA .90 LLaMA 3-8B, .94 GPT-4; PPO and verifier overhead | preprint text; PDF screenshot attempt failed; table values checked in extracted text; no comparable latency/QPS located |
| jeong2024adaptive | https://aclanthology.org/2024.naacl-long.389/ and https://arxiv.org/html/2403.14403 Table 3 | .35/3.08/27.18 seconds for predicted no/one/multi retrieval; 8.60/53.33/38.07% shares | publisher metadata and HTML preprint |
| ou2025accelerating | https://aclanthology.org/2025.findings-acl.1384/ | 2.79x prefill and 2.33x decoding acceleration | publisher abstract; no direct comparison to local CPU measurements |
| jiang2023flare | https://aclanthology.org/2023.emnlp-main.495/ | uncertainty-driven future-text retrieval | publisher abstract/metadata |
| asai2024selfrag | https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html | retrieval and self-reflection | proceedings abstract |
| jiang2023llmlingua | https://aclanthology.org/2023.emnlp-main.825/ | prompt compression | publisher abstract/metadata |
| lewis2020rag | https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html | RAG foundation | publisher abstract/metadata |
| robertson2009bm25 | https://doi.org/10.1561/1500000019 and https://dblp.org/rec/journals/ftir/RobertsonZ09.html | BM25 background | bibliographic records and indexed descriptions |
| reimers2019sbert | https://aclanthology.org/D19-1410/ | cosine-comparable sentence embeddings | publisher abstract/metadata |
| edge2024graphrag | https://arxiv.org/abs/2404.16130 | graph-oriented global summarization | preprint abstract and project metadata |
| gallifant2025tripod | https://www.nature.com/articles/s41591-024-03425-5 | reporting transparency, human oversight | publisher abstract/metadata; abbreviated authors |

## Pending before submission

Verify the final institutional citation style, complete author lists where abbreviated, journal/conference final versions and the supervisor's preferred scope. The two PDF screenshot attempts did not resolve, so inspect the numerical tables visually when original files are available. The reviewed publications address different tasks, datasets, hardware and definitions of quality. Mark unreported or unverified values as N/A, never zero, and do not rank them by a common speedup.
