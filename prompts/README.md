# study100 prompt set
Prompt set for the WAN pipeline-parallel throughput study of the GLM-5.2 ring (greedy decoding, 96 new tokens per prompt): `study100.jsonl` has 101 lines, one JSON object per line with `id`, `cat`, and `prompt`.
Counts: 41 code (the receipt prompt `code-000`, `def quicksort(arr):`, plus 40 new stubs: Python 16, JavaScript/TypeScript 8, Rust 4, Go 4, C/C++ 4, SQL/Bash 4), 20 prose, 20 reasoning, 20 instruct.
Ids are `<cat>-<NNN>`, where `cat` is one of `code`, `prose`, `reasoning`, `instruct` and `NNN` is a zero-padded counter starting at 001 within the category; `code-000` is reserved for the receipt prompt.
Regenerate with `python3 ../scripts/make_prompts.py`; the generator checks that every prompt is ASCII with no trailing whitespace, unique, within the word and line limits, and free of the word "quicksort" outside `code-000`.
