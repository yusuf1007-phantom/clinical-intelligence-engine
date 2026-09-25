from clinical_intelligence.evaluation.error_analysis import disagreement_table

def test_disagreement_categories():
    df = disagreement_table(
        ["a", "b"],
        ["RESULTS", "METHODS"],
        ["BACKGROUND", "METHODS"],
        ["RESULTS", "RESULTS"],
    )
    assert df.iloc[0]["outcome"] == "transformer_fixed"
    assert df.iloc[1]["outcome"] == "baseline_was_better"
