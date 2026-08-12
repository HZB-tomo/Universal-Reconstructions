"""
Custom exception types for the import process. We use distinct classes
(rather than generic Exceptions or error strings) so calling code -- a
CLI, a GUI, or a test -- can catch each situation specifically and
respond appropriately, e.g. showing a "pick a vendor manually" dialog
only for AmbiguousVendorMatchError, not for every possible failure.
"""


class NoVendorMatchError(Exception):
    """
    Raised when NO registered plugin's signature matched the given file.
    Carries the full per-plugin failure detail so the user/developer can
    see exactly why -- e.g. "generic_hdf5 failed because no 3D dataset
    was found" -- instead of a bare "couldn't detect vendor" message.
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
    Raised when two or more plugins matched with the SAME top
    specificity score, so there's no principled way to auto-pick a
    winner. Rather than silently guessing (which could point at the
    wrong data), we surface this and let the caller decide -- e.g. by
    asking the user, or by passing `plugin=` explicitly to
    DataImporter.import_project().
    """

    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        super().__init__(
            f"Ambiguous vendor match, tied at top specificity: {candidates}. "
            "Please specify the plugin explicitly."
        )


class ProjectImportError(Exception):
    """
    Raised when a plugin was successfully selected, but after reading
    its metadata, one or more mandatory_hard fields are still missing
    (e.g. the file matched a vendor's general structure but was somehow
    missing the actual projections dataset). This is different from
    NoVendorMatchError -- here we DID find a matching vendor, the file
    just isn't usable.
    """

    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(f"Missing mandatory fields: {missing_fields}")
