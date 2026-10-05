"""Run the tests with the standard library. pytest picks up the same functions."""

import traceback

import tests.test_api as api
import tests.test_model as model


def main() -> int:
    failed = 0
    for module in (model, api):
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(module, name)()
            except Exception:
                failed += 1
                print(f"FAIL {module.__name__}.{name}")
                traceback.print_exc()
            else:
                print(f"ok   {module.__name__}.{name}")
    if failed:
        print(f"{failed} failed")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
