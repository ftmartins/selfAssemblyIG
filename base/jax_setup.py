"""
Explicit, opt-in JAX configuration.

The old ``config_patchy_particle`` module enabled x64 **and**
``jax_debug_nans`` at import time.  ``jax_debug_nans`` is a large performance
hit and aborts on benign NaNs (e.g. masked-out ghost particles in GCMC), so it
is off by default here.  Runners call :func:`configure_jax` at the top of
``main()``; notebooks call it in their first cell.
"""

from jax import config as _jax_config

_CONFIGURED = False


def configure_jax(x64: bool = True, debug_nans: bool = False) -> None:
    """
    Set global JAX flags.  Idempotent — safe to call more than once as long as
    the arguments do not change.
    """
    global _CONFIGURED
    _jax_config.update("jax_enable_x64", x64)
    _jax_config.update("jax_debug_nans", debug_nans)
    _CONFIGURED = True


def is_configured() -> bool:
    return _CONFIGURED
