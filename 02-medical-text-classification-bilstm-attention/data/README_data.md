# Data notes

## Included file

`sample_medical_text_data.csv` contains **10 short synthetic demonstration rows** with these columns:

- `medical_specialty`
- `transcription`

The five labels are:

1. Cardiology
2. Gastroenterology
3. Neurology
4. Orthopedic
5. Radiology

There are two rows per label. The file is suitable for verifying the project pipeline and user interface, but it is **not sufficient for training or benchmarking a credible classifier**.

## Privacy and responsible use

Do not add:

- patient names,
- medical record numbers,
- dates of birth,
- addresses,
- phone numbers,
- email addresses,
- insurance identifiers,
- confidential medical records,
- protected health information,
- employer-confidential text.

Use only appropriately licensed, de-identified, non-sensitive, and legally permitted data. For public GitHub repositories, retain a small synthetic sample and keep full datasets outside version control unless redistribution is explicitly allowed.

## Replacing the sample dataset

Use a CSV containing one text column and one category column. The loader can infer common names such as:

- text: `transcription`, `clinical_text`, `medical_text`, `clinical_note`, `abstract`, `description`, or `text`
- label: `medical_specialty`, `specialty`, `category`, `label`, `target`, or `class`

Example:

```bash
python scripts/train_model.py \
  --data /path/to/deidentified_dataset.csv \
  --text-column transcription \
  --label-column medical_specialty
```

Before training, review class support, duplicates, missing values, licensing, privacy, and whether the dataset represents the intended deployment population.
