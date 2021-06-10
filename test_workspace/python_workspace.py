from pathlib import Path


class Car:
    def __init__(self, brand: str):
        self._brand = brand

    @property
    def brand(self): return self._brand

    def drive(self):
        print(self.brand)

    def __repr__(self):
        return repr((self._brand, self._brand))


car = Car("audi")
print(car)
