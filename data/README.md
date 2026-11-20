# Kaggle dataset provenance

Source: [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset) by Adison Goh.

Downloaded 19 November 2026 via the public Kaggle download API.
Kaggle metadata identifies version 1 and the license as CC0: Public Domain.

Raw file: **raw/all_tickets_processed_improved_v3.csv**.
SHA256: **044fdace33fa564e1e60453f2941dafc95539c99878b0d32746950394b9dd4d4**.

The source has 47,837 records with columns **Document** and **Topic_group**.
Document contains already processed text. This is the original downloaded CSV,
although its text was preprocessed by the dataset publisher.

| Source label | Records |
| --- | --- |
| Hardware | 13,617 |
| HR Support | 10,915 |
| Access | 7,125 |
| Miscellaneous | 7,060 |
| Storage | 2,777 |
| Purchase | 2,464 |
| Internal Project | 2,119 |
| Administrative rights | 1,760 |

prepare_data.py renames Document to ticket and Topic_group to category.
It removes empty texts, conflicting normalized duplicate labels, and duplicate
normalized texts before splitting. All 47,837 rows survived these checks.
The split is stratified with random_state=42 and test_size=0.2.

**tickets.csv** has 38,269 training rows. **evaluation.csv** has 9,568 held-out
rows. Both contain ticket and category columns. Original category labels are
retained. The source does not provide VPN, Password, Software, Network, or System
as standalone categories. There are no severity annotations. Category accuracy
is measurable on this dataset; severity accuracy requires separately labeled data.

The previous synthetic training and evaluation files were preserved in the
workspace work/before-kaggle folder. The active classifier uses Kaggle data.
