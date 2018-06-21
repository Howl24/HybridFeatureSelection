import time
from project.utils.io import save_pickle
from project.utils.io import load_pickle


def calculate_time_lapse(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    lapse = time.time() - start
    return lapse, result


class ResultMixin(object):
    """Mixin generico de resultados.
    Atributos:
        - PATH: (string) Dirección en la que se guardará el resultado.
    """

    def save(self):
        filename = self.PATH + self.__str__()
        save_pickle(filename, self)

    def load(self):
        filename = self.PATH + self.__str__()
        obj = load_pickle(filename)
        return obj if obj else self
