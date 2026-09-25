from pathlib import Path
from clinical_intelligence.data.parser import parse_pubmed_rct

def test_parser(tmp_path: Path):
    p = tmp_path / "sample.txt"
    p.write_text(
        "###123\nBACKGROUND\tBackground sentence.\nOBJECTIVE\tObjective sentence.\n",
        encoding="utf-8",
    )
    df = parse_pubmed_rct(p)
    assert len(df) == 2
    assert df.iloc[0]["abstract_id"] == "123"
    assert df.iloc[1]["label"] == "OBJECTIVE"
