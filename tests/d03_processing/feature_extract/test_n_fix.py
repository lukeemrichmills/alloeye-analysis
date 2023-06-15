from src.d03_processing.feature_calculate.fix_sacc_calculations import n_fix


def test_n_fix():
    df = None
    object = ""
    assert n_fix(df, object) is None