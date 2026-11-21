# Dataset: ULB Credit Card Fraud Detection

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The user-provided `creditcard.csv` is copied to this folder as-is. Dataset contains 284,807 European credit-card transactions collected over two days in September 2013, with 492 labelled fraud cases. Verify the dataset licence and attribution before redistribution.

Columns:
- `Time`: seconds elapsed since the first transaction in the dataset; not clock time.
- `Amount`: transaction amount (currency not explicitly specified in this dataset).
- `V1`–`V28`: anonymised principal-component-transformed numerical features; original meanings are not available.
- `Class`: target label, `0` = legitimate, `1` = fraud.

**Important:** Location, device, merchant identity and readable previous spending history are **not** available. Do not fabricate these features or claim the model uses them.

The `Class` column must never be sent to the prediction endpoint. Keep the dataset private if its usage terms require it; `.gitignore` excludes all CSV files from Git.
