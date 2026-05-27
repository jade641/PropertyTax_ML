from ml_service.model_loader import _apply_estimator_compatibility_fixes


def test_apply_estimator_compatibility_fixes_sets_multi_class_for_legacy_logistic_regression():
    legacy_estimator = type("LogisticRegression", (), {})()

    fixed_estimator = _apply_estimator_compatibility_fixes(legacy_estimator)

    assert fixed_estimator is legacy_estimator
    assert fixed_estimator.multi_class == "auto"


def test_apply_estimator_compatibility_fixes_leaves_other_estimators_unchanged():
    other_estimator = type("RandomForestClassifier", (), {})()

    fixed_estimator = _apply_estimator_compatibility_fixes(other_estimator)

    assert fixed_estimator is other_estimator
    assert not hasattr(fixed_estimator, "multi_class")