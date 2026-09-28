import numpy as np
import pytest

from detection.fusion import Detection, containment, iou, weighted_box_fusion


def det(box, score=0.8, label="caries", source="a"):
    return Detection(tuple(float(v) for v in box), score, label, source)


def test_iou_identical_and_disjoint():
    a = np.array([0, 0, 10, 10.0])
    b = np.array([[0, 0, 10, 10.0], [20, 20, 30, 30.0]])
    assert iou(a, b) == pytest.approx([1.0, 0.0])


def test_containment_nested_box():
    outer = np.array([0, 0, 100, 100.0])
    inner = np.array([[10, 10, 30, 30.0]])
    assert containment(outer, inner)[0] == pytest.approx(1.0)
    assert iou(outer, inner)[0] < 0.1


def test_boxes_from_both_models_are_merged():
    out = weighted_box_fusion([det([0, 0, 10, 10], 0.9, source="a"), det([1, 1, 11, 11], 0.7, source="b")])
    assert len(out) == 1
    assert out[0].source == "fused"
    assert out[0].score == pytest.approx(0.8)  # mean, both models agreed


def test_nested_boxes_are_merged():
    # One model boxes the whole tooth, the other just the lesion.
    out = weighted_box_fusion([det([0, 0, 100, 100], 0.6, source="a"), det([20, 30, 60, 90], 0.8, source="b")])
    assert len(out) == 1


def test_single_model_box_is_down_weighted():
    out = weighted_box_fusion([det([0, 0, 10, 10], 0.8, source="a")], num_models=2)
    assert out[0].score == pytest.approx(0.4)
    assert out[0].source == "a"


def test_separate_lesions_stay_separate():
    out = weighted_box_fusion([det([0, 0, 10, 10]), det([50, 50, 60, 60])])
    assert len(out) == 2


def test_different_labels_are_not_merged():
    out = weighted_box_fusion([det([0, 0, 10, 10], label="caries"), det([0, 0, 10, 10], label="filling")])
    assert sorted(d.label for d in out) == ["caries", "filling"]


def test_fused_box_is_score_weighted_mean():
    out = weighted_box_fusion([det([0, 0, 10, 10], 0.75, source="a"), det([1, 1, 11, 11], 0.25, source="b")])
    assert len(out) == 1
    assert out[0].box == pytest.approx((0.25, 0.25, 10.25, 10.25))


def test_empty_input():
    assert weighted_box_fusion([]) == []
