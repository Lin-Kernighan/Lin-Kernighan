from typing import List


class Matrix:
    matrix: List[List[float]]
    length: int

    def __str__(self):
        string = ''
        for s in self.matrix:
            for elem in s:
                string += f'{elem:0.2f}\t'
            string += '\n'
        return string

    def __len__(self) -> int:
        return self.length

    def __repr__(self):
        return str(self)

    def __getitem__(self, index: int) -> List[float]:
        return self.matrix[index]
