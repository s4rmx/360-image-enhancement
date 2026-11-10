"""Method protocol and registry."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

CoreFn = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


class Method(Protocol):
    name: str

    def __call__(self, img: np.ndarray, params: dict) -> np.ndarray: ...


METHOD_REGISTRY: dict[str, CoreFn] = {}


def register(name: str):
    def deco(fn: CoreFn) -> CoreFn:
        METHOD_REGISTRY[name] = fn
        fn.name = name  # type: ignore[attr-defined]
        return fn

    return deco


def get_method(name: str) -> CoreFn:
    ensure_registered()
    if name not in METHOD_REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {list_methods()}")
    return METHOD_REGISTRY[name]


def list_methods() -> list[str]:
    ensure_registered()
    return list(METHOD_REGISTRY.keys())


@register("original")
def original(img: np.ndarray, params: dict) -> np.ndarray:
    return img.copy()


@register("wb_only")
def wb_only(img: np.ndarray, params: dict) -> np.ndarray:
    """Identity core — shared pre/post (WB + levels) provide the effect."""
    return img.copy()


_LOADED = False


def ensure_registered() -> None:
    global _LOADED
    if _LOADED:
        return
    # Late import to register concrete methods without circular import at module load
    from enhance.methods import clahe as _clahe  # noqa: F401
    from enhance.methods import fusion as _fusion  # noqa: F401
    from enhance.methods import lime as _lime  # noqa: F401
    from enhance.methods import llf as _llf  # noqa: F401
    from enhance.methods import msrcr as _msrcr  # noqa: F401

    _LOADED = True
