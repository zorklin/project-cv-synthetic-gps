"""Warning-free CLI shim for :mod:`project_cv.final.runner`."""

from project_cv.final.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
