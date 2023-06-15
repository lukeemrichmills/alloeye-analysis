class Errors(Exception):
    """Base class for other exceptions"""
    pass


class ListsNotEqualLengthError(Errors):
    """Raised when two lists are not of equal length"""
    def __init__(self, list1, list2, message="Lists t same length"):
        self.list1 = list1
        self.list2 = list2
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.list1}, {self.list2} -> {self.message}'


class InvalidValue(Errors):
    """Raised value does not match expected"""
    def __init__(self, value, expected, message="Value not as expected"):
        self.value = value
        self.expected = expected
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class CannotParseError(Errors):
    """Custom parse encountered error"""
    def __init__(self, parse_input, message="Attempted parse failed"):
        self.parse_input = parse_input
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class UnknownComputer(Errors):
    """Unknown machine"""
    def __init__(self, message = "Unknown device, will need to register to define data dir"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InvalidMoveCode(Errors):
    """Move code does not match move type"""
    def __init__(self, message = "Move code is invalid for move type"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class UnmatchingValues(Errors):
    """two values should match but do not"""
    def __init__(self, message="values should match"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InvalidSQLOutput(Errors):
    """if sql output comes back in the wrong format or invalid value"""
    def __init__(self, message="sql output invalid"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InvalidInput(Errors):
    """if input wrong format or value"""
    def __init__(self, message="invalid input"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message