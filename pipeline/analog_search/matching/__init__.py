"""Steps B3–B7: type as a signal, technical compatibility, ranking, verdict, explanation."""
from .compat import Compatibility, compare, type_relation
from .rules import Rules, load_rules

__all__ = ["Compatibility", "compare", "type_relation", "Rules", "load_rules"]
