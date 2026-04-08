---
title: "Experience"
permalink: /experience/
author_profile: false
classes:
  - wide
  - experience-page
---
This page keeps the detailed, full version of my professional and research experience.

## Applied Research Scientist, [Thomson Reuters Lab](https://www.thomsonreuters.com/en/careers/our-jobs/join-thomson-reuters-labs), India
**Period:** August 2024 - Current

**Legal AI Reasoning and Model Enhancement**
- Pioneered a legal synthetic data generation pipeline for Process Reward Models (PRMs), creating domain-specific training datasets that improved legal reasoning capabilities on LegalBench benchmark tasks.
- Evaluated different Test-Time Scaling paradigms for legal reasoning and inference-time compute optimization.
- Architected an IRAC (Issue-Rule-Application-Conclusion) Knowledge Graph framework using Thomson Reuters' Westlaw corpus and court case data, generating high-quality preference datasets that improved legal reasoning alignment in fine-tuned LLMs.

**AI-Powered Legal Document Update System**
- Built a comprehensive end-to-end LLM workflow pipeline to update Word documents with XML parsing and intelligent contextual mapping for paragraph identification and edit, achieving 70%+ success rate while preserving complete document formatting integrity.
- Engineered a human-in-the-loop validation interface with reasoning chains and alert-point mapping, resulting in a 60% reduction in manual processing time while maintaining legal accuracy through strategic human oversight and audit trails.

**DocEvolver**
- Created an MVP for a "Cursor for Word" style extension for updating and understanding MS Word files for lawyer-editors.

**Search-and-Replace Agentic System**
- Architected a multi-agent AI system for automated Word editing with a comprehensive validation pipeline (schema enforcement, content integrity, audit logging), achieving 98% accuracy and 65% reduction in manual content revision.
- Constructed error-resolving agents with function calling and multi-turn reasoning to fix XML issues, implementing few-shot learning and self-healing mechanisms. This led to adoption across teams, processing hundreds of documents monthly with sub-8 second processing time per section.

**Additional Tools**
- **Truth Social Monitor:** Created a monitoring system for Trump's Truth Social posts with sub-3 second latency, generating automated alerts for Reuters journalists.
- **Page Flipper:** Revived the Page Monitor extension for website tracking, eliminating Visual Ping subscriptions for the team.
- These tools provided Reuters with a critical competitive advantage over competitors.

## Research Intern, Microsoft Research India
**Period:** January 2024 - July 2024

**Programming with Representations (PwR)**
- Led backend development for Microsoft's PwR Studio platform, focusing on the Natural Language to Domain Specific Language (NL2DSL) translation system using GPT-3.5 and GPT-4.
- Developed a symbolic translation pipeline that generates finite state machines structured as custom DSL, achieving an 85% reduction in hallucinations.
- Formulated rubrics and evaluation loops with error correction over DSL, increasing valid DSL generation from 65% to 95%.

**Jugalbandi-Studio-Engine**
- Architected a Python-based platform that converts DSL into scalable finite-state-machine-based chatbot applications, reducing development time by 80%.
- The platform was featured in Satya Nadella's keynote talks, and I represented Microsoft Research in the pilot project, enabling 15+ non-technical organizations to build AI-powered conversational bots.

**Jugalbandi (JB) Manager**
- Established a chatbot management platform supporting WhatsApp, Telegram, and Web channels with multilingual text and voice capabilities.
- Integrated Bhashini Speech models with Azure service failover mechanisms, enabling 70% faster deployment of new chatbots.

**Open-sourced Work**
- [PwR-NL2DSL](https://github.com/microsoft/PwR-NL2DSL): Natural Language to DSL conversion.
- [PwR-Studio](https://github.com/microsoft/PwR-Studio/): Studio environment for Programming with Representations.
- [Jugalbandi Studio](https://github.com/OpenNyAI/Jugalbandi-Studio-Engine): Open-source chatbot framework.
- [Jugalbandi Manager](https://github.com/OpenNyAI/Jugalbandi-Manager): Chatbot management platform.
- The complete system was picked up by Bhashini to support chatbots across government initiatives.

**Mentors:** [Sriram Rajamani](https://www.microsoft.com/en-us/research/people/sriram/), [B. Ashok](https://www.microsoft.com/en-us/research/people/bash/), [Akash Lal](https://www.microsoft.com/en-us/research/people/akashl/), [Sameer Segal](https://www.microsoft.com/en-us/research/people/sameersegal/)

## Research Intern, [AI Institute, University of South Carolina](https://aiisc.ai/)
**Period:** December 2022 - April 2024

- [Master thesis](https://drive.google.com/file/d/1uj8zn-3BVYmetg1mKluTsJGl_0-n3UfV/view) on Knowledge Enabled Multimodal Ingredient Substitution. Built a knowledge graph incorporating 27K ingredients and 40K substitution pairs, enabling precise ingredient recommendations using multimodal and constraint-based search.
- Developed an LLM-based query module for the ingredient substitution knowledge graph and submitted this work to AAAI-25.
- Resources: [GitHub Repository](https://github.com/kanak8278/MISKG/) | [Dataset on Kaggle](https://www.kaggle.com/datasets/kanakraj/multimodal-ingredient-substitution/) | Dataset used in [UC Irvine + Stanford Health Hackathon 2024](https://www.healthunity.org/2024hackathon)
- Formulated cross-modal recipe retrieval and developed cooking action recognition for recipe analysis, achieving 95% recall and leading to [Cook-Gen](https://ieeexplore.ieee.org/abstract/document/10394432/) at IEEE SMC 2023.
- **Mentors:** [Revathy Venkataramanan](https://scholar.google.com/citations?user=nwri7HkAAAAJ&hl=en), [Dr. Amit Sheth](https://amit.aiisc.ai/)

## Visiting Researcher, Societal Computing at Saarland University [(SIC)](https://saarland-informatics-campus.de/)
**Period:** May 2023 - August 2023

- Project: Time and multispectral domain satellite image super-resolution.
- Worked on satellite image super-resolution using temporal and multispectral information.
- Leveraged high temporal frequency low-resolution data for wildlife tracking and improved disaster analysis through GAN and diffusion approaches.
- **Technologies:** GANs, Diffusion Models, Computer Vision, Remote Sensing, PyTorch
- **Mentors:** [Ingmar Weber](https://ingmarweber.de/), [Ferda Ofli](https://www.ferdaofli.com/)

## Research Intern, University of Maryland, Baltimore (US)
**Period:** October 2022 - April 2023

- Project: Personalized AI Assistant, funded under a HealthCareNLP grant.
- Developed personalized response generation models using reward scaling over BART and T5.
- Work accepted as [K-PERM](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31203) at AAAI Symposium 2024 and improved NUBIA score by 10%.
- Focused on knowledge and persona-aware loss scaling for better response generation.
- **Technologies:** NLP, Information Retrieval, Large Language Models, Conversational Models, Question Answering, Generative AI
- **Mentor:** [Manas Gaur](https://manasgaur.github.io/)

## AI Intern, [EdgeNeural.ai](https://edgeneural.ai/), Pune, India
**Period:** June 2022 - August 2022

- Project: Accelerated inference with model optimization through quantization and CPU/GPU customization.
- Developed training and optimization pipelines for automatic model training and hosting.
- **Technologies:** OCR, Object Detection (YOLO, SSD), TensorRT, GPU Optimization, OpenVINO, Docker, AWS, PyTorch, TensorFlow
- **Collaborators:** [Sarvesh Devi](https://www.linkedin.com/in/sarveshdevi/), [Chidhambararajan](https://www.linkedin.com/in/chidha1434/), [Dhanraj](https://www.linkedin.com/in/dhanraj-katkar-24357b14b/)

## Research Intern, Video Analytics Lab, IISc Bangalore
**Period:** May 2022 - August 2022

- Implemented StyleGAN-based architectures for disentangled video interpretation across domains.
- Improved image/video generation and analysis workflows using GAN-based modeling.
- **Technologies:** GANs, Recurrent Neural Networks, PyTorch, TensorFlow, Python
- **Mentor:** [Rishubh Parihar](https://www.linkedin.com/in/rishubh-parihar/)

## Research Intern, Visual Learning and Intelligence Lab, IIT Hyderabad
**Period:** November 2021 - April 2022

- Researched medical image processing with Prof. Dr. C. Krishna Mohan.
- Developed a novel architecture for improved classification of low-quality images and unbalanced datasets.
- Published SFFNet for panoramic dental X-ray segmentation at IEEE APSCON 2023.
- **Technologies:** Medical Imaging, Healthcare, Deep Learning, TensorFlow, PyTorch
- **Mentor:** [R Sai Chandra Teja](https://scholar.google.com/citations?user=gVSXe1IAAAAJ&hl=en)
- **Collaborators:** [Dhruv Makhwana](https://scholar.google.com/citations?user=enWWINgAAAAJ&hl=en), [Rohit Pawar](https://www.linkedin.com/in/rohit-pawar-a9992a1b4/)

## Computer Vision Engineer, AI Mage (WETHEKOO)
**Period:** March 2021 - April 2021

- Developed a fashion tagging engine using deep learning.
- Optimized and deployed computer vision models on edge devices to improve product/user outcomes.
- **Technologies:** TensorFlow, Computer Vision, Siamese Neural Networks, Tagging Engine, Segmentation

## Software Engineer, Rhizicube Technologies
**Period:** June 2021 - September 2021

- Oversaw server and REST API development and database design for a consumer data platform using Golang (Gin).
- Built a real-time streaming data pipeline using Apache Kafka.
- Built a LinkedIn scraper and a generalized organization website crawler using Selenium and Beautiful Soup.
- **Technologies:** Back-end Web Development, Relational Databases, Kafka, MySQL, Data Scraping, Go, Database Design, Selenium, Python
- **Collaborators:** [Udit Sarin](https://www.linkedin.com/in/uditsarin/), [Yash Goyal](https://www.linkedin.com/in/yashgoyal07/)

## Publications and Papers
1. [Transformers Remember First, Forget Last: Dual-Process Interference in LLMs](https://arxiv.org/abs/2603.00270) - arXiv 2025
2. [K-PERM: Personalized Response Generation Using Dynamic Knowledge Retrieval and Persona-Adaptive Queries](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31203) - AAAI Spring Symposium 2024
3. [Multimodal Ingredient Substitution Knowledge Graph (MISKG)](https://doi.org/10.13140/RG.2.2.20050.11205) - Dataset Release 2024
4. [Cook-Gen: Robust Generative Modeling of Cooking Actions from Recipes](https://arxiv.org/abs/2306.01805) - IEEE SMC 2023
5. [Spatial Field Fusion Network (SFFNet) for Panoramic Dental X-ray Segmentation](https://ieeexplore.ieee.org/abstract/document/10101175/) - IEEE APSCON 2023
