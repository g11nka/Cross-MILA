# Data layout

Raw and processed datasets are intentionally excluded from version control.
After preprocessing/fold construction, use the following local layout:

```text
data/processed/
  IEMOCAP/
    fold_1/{train,val,test}.pkl
    fold_1/classes.json
    fold_1/manifest.json
    ...
    fold_5/
  ESD/
    fold_1/{train,val,test}.pkl
    fold_1/classes.json
    fold_1/manifest.json
    ...
    fold_5/
  MELD/
    train.pkl
    val.pkl
    test.pkl
    classes.json
```

The PKL records contain paths to local audio files. Moving the data therefore
requires regenerating the PKL files or rewriting those paths. Never commit raw
IEMOCAP, ESD, or MELD media to this repository.

