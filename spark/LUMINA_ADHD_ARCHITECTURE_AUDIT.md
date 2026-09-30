# Standalone trained architecture

```text
English / Arabic request + optional task/history context
                       |
              Strict Pydantic validation
                       |
             Existing journal safety gate
                       |
        +--------------+---------------+
        |                              |
  Intent SVM + explicit guard    Difficulty logistic model
        |                              |
        +--------------+---------------+
                       |
        Task/date rules + local calendar context
                       |
         Capacity -> priority -> one small action
                       |
           Focus / recovery / outcome patterns
                       |
        Controlled bilingual response + strict JSON
                       |
     Optional explicit commit to local SQLite calendar
```

`lumina/adhd/executive_function/classical_model.py` loads the actual model artifacts once per assistant instance, verifies checksums and performs NumPy inference. Training uses scikit-learn separately. There is no runtime training, API-key client, model download or remote fallback.

The new dataset changes classification, not the task/date grammar or template-based responses. `models/<task>/metadata.json` and `/version` identify active artifacts. The older journal safety artifact is copied with provenance and was not retrained here.

English/Arabic source transcripts and their derived mappings are under `data/`; runtime tests verify the assistant does not read the training data. SQLite stores explicit local task/outcome state. This development app binds to localhost and is not production authentication or a hospital integration.
