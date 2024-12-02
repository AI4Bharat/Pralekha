# Pralekha: An Indic Document Alignment Evaluation Benchmark

<div style="display: flex; gap: 10px;">
  <a href="https://arxiv.org/abs/2411.19096">
    <img src="https://img.shields.io/badge/arXiv-2411.19096-B31B1B" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/ai4bharat/Pralekha">
    <img src="https://img.shields.io/badge/huggingface-Pralekha-yellow" alt="HuggingFace">
  </a>
  <a href="https://github.com/AI4Bharat/Pralekha">
    <img src="https://img.shields.io/badge/github-Pralekha-blue" alt="GitHub">
  </a>
  <a href="https://creativecommons.org/licenses/by/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey" alt="License: CC BY 4.0">
  </a>
</div>

# Overview
**PRALEKHA** is a large-scale benchmark for evaluating document-level alignment techniques. It includes 2M+ documents, covering 11 Indic languages and English, with a balanced mix of aligned and unaligned pairs.

# Usage
## 1. Input Directory Structure

The pipeline expects a directory structure in the following format:

- A **main directory** containing language subdirectories named using their **3-letter ISO codes** (e.g., `eng` for English, `hin` for Hindi, `tam` for Tamil, etc.)
- Inside each **language subdirectory**, there will be a collection of `.txt` files, each representing a document. The file names should follow the pattern `{doc_id}.txt`, where `doc_id` is the unique identifier for each document.

Below is an example of the expected directory structure:
```plaintext
data/
├── eng/
│   ├── tech-innovations-2023.txt      # Document on tech innovations
│   ├── global-news-789.txt            # Global news summary
│   ├── sports-highlights-day5.txt     # Sports highlights
│   ├── press-release-456.txt          # Official press release
│   ├── ...
├── hin/
│   ├── daily-briefing-april.txt       # Daily briefing for April
│   ├── market-trends-yearend.txt      # Year-end market trends
│   ├── इंडिया-न्यूज़123.txt              # News in Hindi
│   ├── पर्यावरण-संवाद.txt               # Environmental updates
│   ├── ...
├── tam/
│   ├── kollywood-review-movie5.txt   # Movie review
│   ├── 2023-pilgrimage-guide.txt     # Pilgrimage guide for 2023
│   ├── tamil-nadu-budget-7890.txt    # Tamil Nadu budget report
│   ├── கடலோர-மாநில-செய்தி.txt          # Coastal state news in Tamil
│   ├── ...
...

# License

This dataset is released under the [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) license.


# Contact

For any questions or feedback, please contact:

- Raj Dabre ([raj.dabre@cse.iitm.ac.in](mailto:raj.dabre@cse.iitm.ac.in))  
- Sanjay Suryanarayanan ([sanj.ai@outlook.com](mailto:sanj.ai@outlook.com))  
- Haiyue Song ([haiyue.song@nict.go.jp](mailto:haiyue.song@nict.go.jp))  
- Mohammed Safi Ur Rahman Khan ([safikhan2000@gmail.com](mailto:safikhan2000@gmail.com))  

Please get in touch with us for any copyright concerns.
