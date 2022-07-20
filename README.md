# Software Engineering for Data Science - Final Assignment

This repository contains the code for the Final Assignment of the graduate class INF2922 (PUC-Rio).

# Assignment Work

The work done is found in the notebooks, numbered by their sequence of execution. The explanation in the notebooks are in Portuguese-BR.

The first notebook makes a problem context and definition and establishes a main user story that is developed accross all notebooks.
Notebooks from 02 to 04 are experimental steps typically built by a Data Scientist.
The last notebook (05) is actually a simulation of a training and inference pipeline, taking advantage of OOP concepts. These final steps are generally built by Machine Learning Engineers with the colaboration of Data Scientists.

# Technical Requirements

In order to re-run the notebooks, the following packages are necessary:
- Python >= 3.9.12
- Pandas >= 1.4.2
- Numpy >= 1.21.6
- Scikit-Learn >= 1.1.1
- LightGBM >= 3.3.2
- Optuna >= 2.1
- Matplotlib >= 3.5.1
- Seaborn >= 0.11.2
- Yaml >= 0.2.5

# Context

Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress.
The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.

In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more.

# Data
This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes:

Normal
Suspect
Pathological
