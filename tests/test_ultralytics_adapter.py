import numpy as np

from city_multimodal_detection.ultralytics_adapter import result_to_predictions


class _TensorLike:
    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._value)


class _Boxes:
    xywhn = _TensorLike([[0.5, 0.4, 0.2, 0.1]])
    cls = _TensorLike([6])
    conf = _TensorLike([0.75])


class _Result:
    boxes = _Boxes()


def test_result_to_predictions_converts_xywhn_boxes():
    predictions = result_to_predictions(_Result())

    assert len(predictions) == 1
    assert predictions[0].class_id == 6
    assert predictions[0].cx == 0.5
    assert predictions[0].confidence == 0.75
