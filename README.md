# Adaptive Test Synthesis for Modern JavaScript: A RAG-Enhanced Approach

This project aims to develop a GenAI-powered solution for JavaScript testing using ECMAScript 2024, leveraging an LLM with a Retrieval-Augmented Generation (RAG) approach. The system generates high-quality test cases for JavaScript functions, ensuring comprehensive coverage of typical use cases, edge cases, and invalid inputs.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
4. [Usage Instructions](#usage-instructions)

    [//]: # (5. [Example Output]&#40;#example-output&#41;)
5. [Folder Structure](#folder-structure)
6. [Future Work](#future-work)
7. [Contributors](#contributors)

---

## Project Overview

JavaScript is one of the most widely used programming languages, but testing its complex features and edge cases is often a challenge. This project uses state-of-the-art GenAI models integrated with ECMAScript 2024 specifications to automatically generate robust test cases. By employing the RAG approach, the system combines the contextual understanding of LLMs with real-time retrieval from the ECMAScript dataset for precise, context-aware outputs.

---

## Features

- **LLM-Powered Test Case Generation:** Automatically generates JavaScript test cases, covering:
  - Typical use cases.
  - Edge cases like `Infinity`, `NaN`, and extreme values.
  - Invalid inputs such as non-numeric strings, `null`, and `undefined`.

- **Retrieval-Augmented Generation (RAG):** Utilizes ECMAScript specifications as context to enhance the relevance and accuracy of generated test cases.

- **Chroma Vector Store Integration:** Efficiently stores and retrieves ECMAScript pages to speed up query processing.

- **Parallel Context Retrieval:** Supports parallel processing for multi-query workflows.

---
## Setup Instructions

1. **Install Python 3.12.1**  
   Ensure you have Python 3.12.1 installed. You can download it from the official [Python website](https://www.python.org/downloads/release/python-3121/).

2. **Create a Virtual Environment**  
   Run the following command to create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. **Activate the Virtual Environment**  
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```bash

     ```
     
4. **Install Required Packages**  
   After activating the virtual environment, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download ECMAScript Dataset**  
   Ensure you have the ECMAScript 2024 PDF dataset in the dataset/ folder. If not, download it from the [official ECMA website](https://ecma-international.org/publications-and-standards/standards/ecma-262/).
6. **Run the Application**
    To generate test cases, run the main.py script with the desired question:
    ```bash
    python main.py
    ```
---
## Usage Instructions

1. **Generate Test Cases**  
    Specify your question in the ```main.py``` script under the ```question_to_ask``` variable:
    ```python
    question_to_ask = "Generate test cases for the function: Math.abs()"
    ```
2. **Process Output**  
   Review the generated JavaScript test cases in the terminal.

3. **Modify Prompt Template**  
To customize the test case generation style, edit the prompt template in ```constants.py```:
    ```python
    PROMPT_TEMPLATE = """Your custom template here"""
    ```
---
## Folder Structure
```graphql
Adaptive-Test-Synthesis-for-Modern-JavaScript/
│
├── dataset/                  # Folder containing ECMAScript dataset
├── storage/                  # Persistent storage for split pages and vector store
├── lib/                      # Core Python modules for the project
│   ├── __init__.py
│   ├── utils.py
├── config/                   
│   ├── __init__.py
│   ├── config.py             # Configuration file with paths and settings
├── constants/                   
│   ├── __init__.py
│   ├── constants.py          # Configuration file for constants parameters
├── log/                   
│   ├── __init__.py
│   ├── logger.py             # Log Configuration file
├── main.py                   # Entry point for running the application
├── requirements.txt          # Required Python dependencies
└── README.md                 # Project documentation
```
---
## Future Work

- **Enhance with Fine-Tuning:** Incorporate fine-tuning techniques to improve the accuracy and relevance of the generated test cases, tailored to specific use cases or projects.
- **Improve Performance for Each Module:** Optimize each system module, including PDF parsing, vector store retrieval, and LLM response time, to achieve faster execution.
- **Extend Support to Other Languages:** Expand the framework to generate test cases for additional programming languages beyond JavaScript.
---

## Contributors 
<a href="https://github.com/suriya-1403" target="_blank">
<img src="https://github.com/suriya-1403.png?size=1000" alt="Suriyakrishnan Sathish" style="border-radius: 50%; width: 100px; height: 100px;">
</a>
<a href="https://github.com/tejaskumarleelavathi" target="_blank">
<img src="https://github.com/tejaskumarleelavathi.png?size=1000" alt="Tejas Kumar Leelavathi" style="border-radius: 50%; width: 100px; height: 100px;">
</a>