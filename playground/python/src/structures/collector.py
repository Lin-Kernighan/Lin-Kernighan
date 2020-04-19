from json import dump
from time import ctime, time
from typing import List

from pandas import DataFrame


class Collector:

    def __init__(self, names: List[str], info: dict):
        """
        names: названия собираеммых данных, по сути имя колонки
        info: просто доп инфа в json
        колонки time and delta (time) создаются сами
        """
        self.data = {}
        self.info = {'info': info, 'time': ctime(), 'src': self.data}
        self.dimension = len(names)
        for name in names:
            self.data.update({name: []})
        if 'time' not in self.data:
            self.data.update({'time': []})
        if 'delta' not in self.data:
            self.data.update({'delta': []})
        self.time = time()

    def update(self, new: dict) -> None:
        """ добавить данные, time and delta (time) сами обновляются """
        e_time = time()
        if len(new) != self.dimension:
            raise Exception('bad data')
        for name in new:
            self.data[name].append(new[name])
        self.data['delta'].append(e_time - self.time)
        self.time = time()
        self.data['time'].append(self.time)

    def dump(self, filename: str) -> None:
        """ слить в какую-то json """
        with open(filename, 'w') as file:
            dump(self.info, file)

    def as_frame(self) -> DataFrame:
        """ вернуть как data frame """
        return DataFrame(self.data)

    def __str__(self):
        return str(self.as_frame())

    def __repr__(self):
        return str(self)
