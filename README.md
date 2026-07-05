# word-associations

## TODO
- [x] Add LWOW dataset
- [x] Add SWOW dataset
- [ ] ~~Implement batch processing for Claude Opus 4.5~~
- [ ] ~~Add Georgian population demographic dataset~~
- [x] ~~Contact Claude sales for research discount~~
- [ ] Ability to search SWOW/LWOW sorted by association strengths
- [x] Find overlapping/unique words for SWOW/LWOW
- [ ] Create notebook for AI persona creation
- [ ] Try couple of models with AI personas and compare results
    - [ ] 18 cue per persona
    - [ ] 60,000 persona (needs revision)
    - [ ] 100 responses per cue
    - [ ] total of 12,000 unique cues
    - [ ] [LLM Generated Persona is Promise With A Catch](https://proceedings.neurips.cc/paper_files/paper/2025/file/5fc150b75e8b509735564de16bc3527b-Paper-Position_Paper_Track.pdf)
    - [ ] [A word association network methodology for evaluating
implicit biases in LLMs compared to humans](https://arxiv.org/pdf/2510.24488)
- [x] Search SWOW/LWOW with association strengths
- [x] Find overlapping/unique words for SWOW/LWOW
- [ ] Minimize empty/malformed responses from models
- [ ] Add more datasets
    - [ ] Kimi v2.6 — Luka Nadiradze
    - [ ] GLM-5.2 — Luka Nadiradze
    - [ ] Nemotron 3 Nano 30B A3B — Luka Tabagari
    - [ ] Qwen 3.5/3.6 — Luka Tabagari
    - [ ] MiniMax M2.7 — Temo Saghinadze
    - [ ] Deepseek —
    - [ ] Gemma 4 — 
- Come up with the new similarity metric (subgraph permutations?)


## Other Datasets

### NRC Valence, Arousal, Dominance
These dimensions impact various aspects of our lives from social competence and emotion regulation to success in the work place and how we view the world. The NRC VAD Lexicon v2 has human ratings of valence, arousal, and dominance for more than 55,000 English words and phrases.
- [Paper](https://arxiv.org/abs/2503.23547)
- [Website](https://saifmohammad.com/WebPages/nrc-vad.html)

### SUBTLEX
SubtlexUS is database containing word frequencies based on English and American movies and TV series subtitles.
- [Website](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus)
- [Table](https://github.com/chrplr/openlexicon/blob/master/datasets-info/SUBTLEX-US/README-SUBTLEXus.md)

### Georgian National Corpus
The Georgian National Corpus is currently the most developed and detailed corpus of the Georgian language.
- [Website](http://gnc.gov.ge/gnc/page)
