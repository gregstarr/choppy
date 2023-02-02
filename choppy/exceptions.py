class InvalidOperationError(Exception):
    def __init__(self) -> None:
        super().__init__("operation not 'union' or 'difference'")


class OperationFailedError(Exception):
    def __init__(self) -> None:
        super().__init__("Couldn't insert connector")


class ConnectedComponentError(Exception):
    def __init__(self, cc, msg) -> None:
        super().__init__(f"(area {cc.area:.1f}) {msg}")


class NoValidSitesError(ConnectedComponentError):
    def __init__(self, cc) -> None:
        super().__init__(cc, "no valid connector sites")


class CcTooSmallError(ConnectedComponentError):
    def __init__(self, cc) -> None:
        super().__init__(cc, "connected component area too small")


class CrossSectionError(Exception):
    variants = [
        "split lost a part",
        "plane missed part",
        "cc missing part",
        "part missing cc",
    ]

    def __init__(self, var) -> None:
        """variants =
        [
            "split lost a part",
            "plane missed part",
            "cc missing part",
            "part missing cc"
        ]
        """
        super().__init__(self.variants[var])


class LowVolumeError(Exception):
    """low volume"""


class InvalidChoppyInputError(Exception):
    def __init__(self) -> None:
        super().__init__("Input mesh already small enough to fit in printer")


class ProcessFailureError(Exception):
    def __init__(self) -> None:
        super().__init__("No valid chops found")


class ConnectorPlacerInputError(Exception):
    ...


class NoConnectorSitesFoundError(Exception):
    ...
