class PolyM:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def values(self):
        return self.coeffs

    def __mul__(self, other):
        if not isinstance(other, PolyM):
            raise TypeError()
        n = len(self.coeffs)
        m = len(other.coeffs)
        result = [0] * (n + m - 1)
        self.coeffs = self.coeffs + (m - 1) * [0]
        other.coeffs = other.coeffs + (n - 1) * [0]
        for j in range(n + m - 1):
            for k in range(j + 1):
                result[j] += self.coeffs[j - k] * other.coeffs[k]
        return PolyM(result)

    def __str__(self):
        n = len(self.coeffs)
        terms = [f"{self.coeffs[i]} z^{i}" for i in range(n)]  # if self.coeffs[i] != 0]
        return " + ".join(terms)

    def __repr__(self):
        return f"PolyM({self.coeffs})"
