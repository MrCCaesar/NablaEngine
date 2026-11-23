"""Optional parts of the package: load(package, feature, *names) imports a module (or several submodules of one package)
that a code path needs only for a given feature, and turns its absence into NotInRelease ("<feature> is not available
in this release of NablaEngine") instead of a bare ModuleNotFoundError. When the module is present, load is the same
import as `from package import name` (the same module objects, nothing else changes); an ImportError raised inside a
present module propagates unchanged."""
import importlib

__all__ = ["BAM", "FOREST", "NotInRelease", "load"]

FOREST = "the forest engine (models outside the G1 class: several trees under the world, or no free root)"
BAM = "the BAM servo model (actuator='bam')"


class NotInRelease(NotImplementedError):
    """A feature whose modules this installation of NablaEngine does not include."""


def load(package, feature, *names):
    """The module `package` (no names), the submodule package.<name> (one name) or a tuple of them (several names).
    NotInRelease when one of them (or the package) is not installed."""
    mods = [f"{package}.{n}" for n in names] if names else [package]
    out = []
    for mod in mods:
        try:
            out.append(importlib.import_module(mod))
        except ModuleNotFoundError as e:
            if e.name is None or not (mod == e.name or mod.startswith(e.name + ".")):
                raise
            raise NotInRelease(f"{feature} is not available in this release of NablaEngine") from None
    return out[0] if len(out) == 1 else tuple(out)
