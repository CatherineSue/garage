"""
Autopickable.
"""
import tensorflow as tf

# flake8: noqa


class PickleCall:
    def __new__(cls, to_call, args=(), kwargs=None, __pickle_target=None):
        # construct this class
        if __pickle_target is None:
            result = super(cls, PickleCall).__new__(cls)
            result.to_call = to_call
            result.args = args
            result.kwargs = kwargs or {}
            return result
        else:
            # when unpickling
            return __pickle_target(*args, **kwargs)

    def __getnewargs__(self):
        return (None, self.args, self.kwargs, self.to_call)


def build_layers(config, weights, **kwargs):
    model = tf.keras.models.model_from_json(config, custom_objects=kwargs)
    model.set_weights(weights)
    return model


class AutoPickable:
    def __getstate__(self):
        print("Calling __getstate__")
        state = self.__dict__.copy()
        for k, v in self.__dict__.items():
            if isinstance(v, tf.keras.models.Model):
                custom_objects = {}
                for c in state[k].layers:
                    if "garage" in str(type(c)):  # detect subclassed layer
                        name = type(c).__name__
                        custom_objects[name] = type(c)
                state[k] = PickleCall(build_layers,
                                      (v.to_json(), v.get_weights()),
                                      custom_objects)
        return state
