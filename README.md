# CaseStudyNLP

# Topic

1.8 Tokenization for Code-Switched and Mixed-Language Text

# Description

The goal of this case study is to investigate how tokenization methods handle code-switched and mixed-language text, where multiple languages appear within the same sentence or document (e.g., English–Spanish, Hindi–English, Arabic–French).

Students will explore:
- How traditional (word-level) vs. subword (BPE, WordPiece, SentencePiece) tokenizers handle intra-sentence language switching
- The impact of shared vs. language-specific vocabularies on token fragmentation and OOV rates
- Sequence length inflation caused by mixed-language morphology and script differences
- The effect of tokenization strategies on downstream tasks such as sentiment analysis, named entity recognition (NER), and machine translation

Students will design experiments comparing monolingual tokenizers, multilingual tokenizers, and hybrid approaches (e.g., language-ID-aware tokenization), and evaluate their impact on model efficiency and task performance.

## Expected Deliverables

As a result of conducting the case studies, the students are expected to provide a **link to the GitHub repository** in Moodle. 

The GitHub repository should contain:

1. One or more **Jupiter notebooks** with the source code of the conducted experiments.
  **Note**: The notebooks must be structured, clear, sufficiently commented, and contain no major errors, otherwise the grade will be decreased.
2. A **poster** in .pdf format containing the motivation for the case study, methodology, experimental setup (datasets, models, etc.), results, and conclusions. Adding visuals is recommended and may positively affect the grade.
  [Example of a poster](https://disk.yandex.ru/i/6OJMmrcJXO9dOA)

## Deadlines

1. The students fill in the case study preference form: **March 8, 23:59**
2. The instructors publish the list of case studies assigned to the students: **March 11**
3. The students submit and present the results of the case studies: **Final Exam (Datetime to be announced)**

## Grading

The case studies will be graded during the final exam. The students must present the results by providing 1) **background of the case study**, 2) **experimental results** and 3) **conclusions**. It is expected that the students accompany the explanations with the poster.

The case study presentation is awared with **50% of the grade of the Final Exam**. The case studies will be graded by how well the experiments are performed and whether the experiments indeed justify the conclusions.
