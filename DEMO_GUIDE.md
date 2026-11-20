# Screen-sharing explanation

1. Open http://127.0.0.1:5000/. Say “SupportPilot receives an IT support ticket,
   predicts a category, estimates severity, assigns priority and stores the result.”
2. Enter sample contact details. Describe an access request or hardware issue.
   Submit and explain the category and confidence displayed by the model.
3. Say “We use Kaggle's IT Service Ticket Classification Dataset by Adison Goh.
   It has 47,837 processed ticket documents and eight labels. We use 38,269 for
   training and 9,568 for evaluation. The categories are Hardware, HR Support,
   Access, Miscellaneous, Storage, Purchase, Internal Project and Administrative rights.”
4. Show data/raw/all_tickets_processed_improved_v3.csv and data/tickets.csv.
   Explain “Document becomes ticket and Topic_group becomes category.
   We check duplicate text before splitting, then fit TF-IDF and logistic regression.”
5. Show train_model.py and the actual evaluation output. Category accuracy is
   8,209/9,568 (85.8%), below the 90% milestone target. Explain that the
   Kaggle source has no severity labels. Severity accuracy remains unmeasured
   on this source, and the application uses the milestone PDF's keyword rules.
6. Open Tickets and reopen the saved ticket. After confirming resolution,
   click **Mark as resolved & close** and show the Closed status.

Use the displayed category for your demonstration. A VPN request can be grouped
under a broader Kaggle label. Confidence for one ticket differs from accuracy
across the held-out evaluation dataset.
