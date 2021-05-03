import numpy as np

decay_constant = 4


class Material:
    def __init__(self):
        self.material_constant = {
            'metal': [.161, .486, .965],  # blue (0, 0, 255)
            'ooi': [0.01, 0.01, 0.01],  # red (255, 0, 0)
            'plastic': [.957, .749, .478],  # orange(255, 165, 0)
            'cloth': [.957, .749, .478],  # red orange (255, 69, 0)
            'leather': [.318, .741, .506],  # green (0, 255, 0),
            'unk': [.957, .749, .478]  # orange(255, 165, 0)
        }

        # Normalize
        r = -np.log(np.array(list(self.material_constant.values())))
        r = r / np.sqrt(np.sum(r ** 2, axis=1, keepdims=True))
        self.material_constant = dict(zip(list(self.material_constant.keys()), r))

    def get_const(self, material):
        return self.material_constant.get(material, self.material_constant['unk'])

    def __str__(self):
        output = ""
        for k, v in self.material_constant.items():
            output += f"{k}: {v}\n"

        return output


if __name__ == '__main__':
    m = Material()
    print(m)
    print(m.get_const('plastic'))
    print(m.get_const('janina'))
