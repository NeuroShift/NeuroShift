from neuroshift.model.data.const import Const


@Const("name", "surname", "birthday")
class FirstClass:
    def __init__(
        self,
        name: str = "a",
        surname: str = "b",
        birthday: int = 1,
        age: int = 20,
        meal: str = "carrots",
    ):
        self.name = name
        self.surname = surname
        self.birthday = birthday
        self.age = age
        self.meal = meal


@Const()
class OtherClass:
    def __init__(self, name: str = "a", surname: str = "b", birthday: int = 1):
        self.name = name
        self.surname = surname
        self.birthday = birthday


@Const()
class SlotClass:
    __slots__ = ("name", "surname", "birthday")

    def __init__(self, name: str = "a", surname: str = "b", birthday: int = 1):
        self.name = name
        self.surname = surname
        self.birthday = birthday


class NonConst:
    def __init__(self) -> None:
        pass


@Const()
class NonHashable:
    def __init__(self, c: NonConst = NonConst()) -> None:
        self.c = c
