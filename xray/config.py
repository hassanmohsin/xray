import numpy as np

decay_constant = 4


class Material:
    def __init__(self):
        self._material_constant = {
            'metal': [.161, .486, .965],  # blue (0, 0, 255),
            'ooi': [0.01, 0.01, 0.01],  # black (0, 0, 0)
            'plastic': [.957, .749, .478],  # orange(255, 165, 0),
            'leather': [.318, .741, .506],  # green (0, 255, 0),
            'unk': [.957, .749, .478]  # orange(255, 165, 0)
        }

    def _normalize(self):
        # Normalize
        r = -np.log(np.array(list(self._material_constant.values())))
        r = r / np.sqrt(np.sum(r ** 2, axis=1, keepdims=True))
        self._material_constant = dict(zip(list(self._material_constant.keys()), r))

    def get_const(self, material, color_deviation=False):
        if color_deviation:
            self._material_constant[material] = np.array(self._material_constant[material]) * np.array([
                1., 1., np.random.uniform(0.6, 1.)
            ])
        self._normalize()
        return self._material_constant.get(material, self._material_constant['unk'])

    def get_material(self, s):
        return '' if s not in self._material_constant.keys() else s

    def __str__(self):
        return '\n'.join([f"{k}: {v}" for k, v in self._material_constant.items()])


if __name__ == '__main__':
    m = Material()
    print(m)
    print(m.get_const('plastic'))
    print(m.get_const('janina'))
