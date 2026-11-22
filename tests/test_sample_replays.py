import json
from pathlib import Path
from clauseforge.taxonomy import load_taxonomy_metadata

def test_real_sample_records_preserve_an_invalid_output():
    root=Path(__file__).resolve().parents[1]/"src/clauseforge/frontend"
    data=json.loads((root/"sample-replays.json").read_text(encoding="utf-8"))
    assert len(data["samples"])==5
    assert data["provenance"]["adapter_gguf_sha256"]=="e60c991d247dd2faaca8e1142a4ba34301be2972295c8ebe472b20c871aece83"
    labels={row.category_id for row in load_taxonomy_metadata()}
    outputs={row["label"]:row["raw_output"] for row in data["samples"]}
    assert outputs["Termination"]=="termination_for_convenience"
    assert outputs["Non-compete"]=="non_solicit_of_customers"
    assert outputs["Non-compete"] not in labels
    assert all(row["input_tokens"]>0 and row["duration_seconds"]>0 for row in data["samples"])
