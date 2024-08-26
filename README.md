# Development of two machine-learning models to predict conversion from primary HER2-0 breast cancer to HER2-low metastases: a proof of concept study (2024)

This repository contains the R code developed to produce the results of the paper entitled as above.

## Research in context
### Evidence before this study
HER2-low expression has gained clinical relevance in breast cancer (BC) due to the availability of anti-HER2 antibody-drug conjugates (ADCs) for HER2-low metastatic BC (MBC) patients. Our research, along with others, has shown that HER2-low phenotype can evolve dynamically throughout the natural history of the disease, with a not negligible proportion of patients with HER2-0 phenotype on primary tumors evolving towards HER2-low BC at disease relapse. Capturing this phenomenon is critical as it has implications in terms of expanding available treatment options. The development of artificial intelligence-based tools could aid in identifying patients for whom a relapse/metastasis biopsy may provide impactful clinical information. However, the main areas of application of AI in the field of BC are mainly restricted, so far, to BC early detection, prediction of BC development in higher-risk populations and computational pathology, with growing interest in prognostic stratification. There is a strong need for a broader and less niche use of AI-based tools in BC research, which requires the identification of relevant clinical questions. In response to this need, we conducted a proof of principle study, developing two machine learning-based models, each addressing a different need, explainability and performance, to predict the phenomenon of HER2-low phenotype gain from primary BC to relapse.
### Added value of this study
This study is among the first to demonstrate the application of machine learning models to predict a highly relevant phenomenon in modern breast cancer oncology, and specifically, the acquisition of a druggable target, which has significant implications for drug access. We began with an explainable model, which was later integrated into an ensemble approach, allowing us to enhance performance while maintaining transparency, explainability, and intelligibility. The explainable model showed promising accuracy in predicting the acquisition of the HER2-low phenotype at relapse in cases of HER2-0 primary tumors. The ensemble model, with a sensitivity of 75%, demonstrated sufficient power to meet the clinical principle of reliability, thereby laying the groundwork for its external clinical validation.
### Implications of all the available evidence
Overall, the available evidence supports the significant instability of the HER2-low phenotype throughout the natural history of breast cancer. Our model successfully identifies the underlying drivers of this phenomenon. Additionally, we have demonstrated that applying an AI-based approach, built on readily obtainable traditional clinico-pathological features, to the clinically relevant question of predicting HER2-low phenotype gain is both feasible and reliable.
