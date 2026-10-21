# InsureDoc AI – Intelligent Insurance Document Processing & RAG-Based Question Answering System

### Project Overview

InsureDoc AI is an AI-powered application that automates insurance document processing and enables users to retrieve information from insurance policies using natural-language questions. The system extracts text and tables from insurance PDFs, organizes the extracted content, and uses Retrieval-Augmented Generation (RAG) to generate relevant answers with source document names and page references.

### Problem Statement

Insurance documents contain extensive information about premiums, benefits, exclusions, eligibility criteria, and policy conditions. Manually searching through lengthy policy documents is time-consuming. This project addresses the challenge by automating document processing and providing an interactive question-answering system for efficient information retrieval.

### Architecture and Workflow

Insurance documents are stored in Amazon S3. AWS Step Functions orchestrates the processing workflow, while AWS Lambda extracts and processes text and tables from PDF documents. The extracted information is stored in JSON format in S3, and Amazon DynamoDB tracks document-processing status.

The RAG application loads the extracted content, splits it into smaller text chunks, and generates embeddings using Sentence Transformers. ChromaDB stores and retrieves relevant information, while hybrid retrieval combines semantic search with BM25-based keyword search. The retrieved passages are passed to an OpenAI language model through LangChain to generate context-based answers. The Streamlit application is containerized using Docker and deployed on Amazon EC2.

### Technologies Used

* **Programming Language:** Python
* **Cloud Services:** Amazon S3, AWS Lambda, AWS Step Functions, Amazon DynamoDB, Amazon EC2
* **Document Processing:** pypdf, pdfplumber
* **Generative AI:** OpenAI API, LangChain
* **RAG and Retrieval:** ChromaDB, Sentence Transformers, BM25
* **Frontend:** Streamlit
* **Deployment:** Docker

### Key Features

* Automated insurance document processing using AWS services.
* Text and table extraction from PDF documents.
* Storage of extracted content in structured JSON format.
* Processing-status tracking using Amazon DynamoDB.
* Semantic and hybrid retrieval of relevant document passages.
* Natural-language question answering using RAG.
* Source document and page references for answer verification.
* Containerized deployment of the application on AWS EC2.

### Outcome

InsureDoc AI demonstrates how cloud computing and Generative AI can be combined to simplify insurance document analysis. It helps users retrieve policy information efficiently, reduces manual document searching, and provides answers supported by references to the original documents.
