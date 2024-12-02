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
### 1. Input Directory Structure

The pipeline expects a directory structure in the following format:

- A **main directory** containing language subdirectories named using their **3-letter ISO codes** (e.g., `eng` for English, `hin` for Hindi, `tam` for Tamil, etc.)
- Each language subdirectory will contain `.txt` documents named in the format `{doc_id}.txt`, where `doc_id` serves as the unique identifier for each document.

Below is an example of the expected directory structure:
```plaintext
data/
├── eng/
│   ├── tech-innovations-2023.txt                
│   ├── sports-highlights-day5.txt     
│   ├── press-release-456.txt         
│   ├── ...
├── hin/
│   ├── daily-briefing-april.txt       
│   ├── market-trends-yearend.txt      
│   ├── इंडिया-न्यूज़123.txt              
│   ├── ...
├── tam/
│   ├── kollywood-review-movie5.txt   
│   ├── 2023-pilgrimage-guide.txt       
│   ├── கடலோர-மாநில-செய்தி.txt          
│   ├── ...
...
```
### 2. Split Documents into Granular Shards

To process documents into granular shards, use the `doc2granular-shards.sh` script.

This script allows you to:
1. **Tokenize documents into sentences.**
2. **Split documents into chunks.**

Run the script:
```bash
./doc2granular-shards.sh
```
# License

This dataset is released under the [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) license.


# Contact

For any questions or feedback, please contact:

- Raj Dabre ([raj.dabre@cse.iitm.ac.in](mailto:raj.dabre@cse.iitm.ac.in))  
- Sanjay Suryanarayanan ([sanj.ai@outlook.com](mailto:sanj.ai@outlook.com))  
- Haiyue Song ([haiyue.song@nict.go.jp](mailto:haiyue.song@nict.go.jp))  
- Mohammed Safi Ur Rahman Khan ([safikhan2000@gmail.com](mailto:safikhan2000@gmail.com))  

Please get in touch with us for any copyright concerns.
