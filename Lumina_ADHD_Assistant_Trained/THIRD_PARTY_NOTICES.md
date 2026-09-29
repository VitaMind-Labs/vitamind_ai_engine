# Dataset attribution and notices

MASSIVE 1.1 dataset: Copyright Amazon.com Inc. or its affiliates. Licensed under Creative Commons Attribution 4.0 International. Full supplied license: `data/raw/massive/LICENSE`. License link: https://creativecommons.org/licenses/by/4.0/

Source: https://github.com/alexa/massive
Archive: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
Dataset card: https://huggingface.co/datasets/AmazonScience/massive

Citation: Jack FitzGerald and collaborators (2022), *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages*, https://arxiv.org/abs/2204.08582 . Release 1.1 adds another language; this package uses only en-US and ar-SA.

Underlying English source citation: Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski and Verena Rieser (2020), *SLURP: A Spoken Language Understanding Resource Package*, EMNLP, https://aclanthology.org/2020.emnlp-main.588/ .

Changes in this bundle: select calendar and sampled nonplanning utterances; map selected intent labels to Lumina's taxonomy; add provenance, normalized split auditing and separately marked AI-authored synthetic data. The raw language files are unchanged. The derived dataset and synthetic additions carry the CC BY 4.0 dataset notice with their origin retained. Nothing here implies endorsement by the original creators. Data is supplied without warranty; the original license terms remain in force.

The user's earlier `adhd_dataset.csv` has separate unknown provenance and is not relicensed or used as planning supervision. The previous VitaMind journal artifact is included with its original model provenance; MASSIVE did not train that safety model. Application library dependencies retain their own licenses.
