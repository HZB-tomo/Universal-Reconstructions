"""
Exception types for the import process. Distinct classes let calling
code catch each situation specifically.
"""


class NoVendorMatchError(Exception):
    """
    Raised when no registered plugin's signature matched the given
    file. Carries per-plugin failure detail.
    """

    def __init__(self, details: list[tuple[str, list[str]]]):
        # details: [(plugin_name, [failed_condition, ...]), ...]
        self.details = details
        message = "No vendor plugin matched.\n" + "\n".join(
            f"  - {name}: failed on {failed}" for name, failed in details
        )
        super().__init__(message)


class AmbiguousVendorMatchError(Exception):
    """
    Raised when two or more plugins matched with the same top
    specificity score. The caller must resolve this, e.g. by passing
    `plugin=` explicitly to DataImporter.import_project().
    """

    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        super().__init__(
            f"Ambiguous vendor match, tied at top specificity: {candidates}. "
            "Please specify the plugin explicitly."
        )


class ProjectImportError(Exception):
    """
    Raised when a plugin was selected, but after reading its metadata,
    one or more mandatory_hard fields are still missing.
    """

    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(f"Missing mandatory fields: {missing_fields}")
