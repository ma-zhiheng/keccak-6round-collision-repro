"""Small GF(2) affine equation system using Python integers as bitsets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GF2System:
    nvars: int
    rows: dict[int, int] = field(default_factory=dict)
    inconsistent: bool = False

    def copy(self) -> "GF2System":
        return GF2System(self.nvars, dict(self.rows), self.inconsistent)

    def add_equation(self, coeff: int, rhs: int = 0) -> bool:
        if self.inconsistent:
            return False
        row = coeff | ((rhs & 1) << self.nvars)
        coeff_mask = (1 << self.nvars) - 1
        while row & coeff_mask:
            pivot = (row & coeff_mask).bit_length() - 1
            existing = self.rows.get(pivot)
            if existing is None:
                self.rows[pivot] = row
                return True
            row ^= existing
        if (row >> self.nvars) & 1:
            self.inconsistent = True
            return False
        return True

    def add_equations(self, equations: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
        snapshot = self.copy()
        for coeff, rhs in equations:
            if not self.add_equation(coeff, rhs):
                self.rows = snapshot.rows
                self.inconsistent = snapshot.inconsistent
                return False
        return True

    def is_consistent_with(self, equations: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
        trial = self.copy()
        return trial.add_equations(equations)

    @property
    def rank(self) -> int:
        return len(self.rows)

    @property
    def dimension(self) -> int:
        return self.nvars - self.rank

    def _rref_rows(self) -> dict[int, int]:
        rows = list(self.rows.values())
        coeff_mask = (1 << self.nvars) - 1
        pivot_cols: list[int] = []
        current = 0
        for col in range(self.nvars - 1, -1, -1):
            pivot_index = None
            for i in range(current, len(rows)):
                if (rows[i] >> col) & 1:
                    pivot_index = i
                    break
            if pivot_index is None:
                continue
            rows[current], rows[pivot_index] = rows[pivot_index], rows[current]
            pivot_row = rows[current]
            for i in range(len(rows)):
                if i != current and ((rows[i] >> col) & 1):
                    rows[i] ^= pivot_row
            pivot_cols.append(col)
            current += 1

        for row in rows:
            if (row & coeff_mask) == 0 and ((row >> self.nvars) & 1):
                raise ValueError("inconsistent system")
        return {pivot: rows[i] for i, pivot in enumerate(pivot_cols)}

    def particular_solution(self) -> int:
        if self.inconsistent:
            raise ValueError("inconsistent system")
        solution = 0
        for pivot, row in self._rref_rows().items():
            if (row >> self.nvars) & 1:
                solution |= 1 << pivot
        return solution

    def nullspace_basis(self) -> list[int]:
        if self.inconsistent:
            raise ValueError("inconsistent system")
        pivot_rows = self._rref_rows()
        pivots = set(pivot_rows)
        basis: list[int] = []
        for free in range(self.nvars):
            if free in pivots:
                continue
            vector = 1 << free
            for pivot, row in pivot_rows.items():
                if (row >> free) & 1:
                    vector |= 1 << pivot
            basis.append(vector)
        return basis
