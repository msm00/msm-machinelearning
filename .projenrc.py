import projen.python
from projen.python import PyprojectTomlFile, PythonProject

PROJECT_NAME = "msm-machinelearning"
PROJECT_VERSION = "0.1.1"
PROJECT_DESCRIPTION = "MSM test projen project"
AUTHOR_NAME = "Milan Šmíd"
AUTHOR_EMAIL = "smidmi@gmail.com"
PYTHON_RANGE = ">=3.13,<4.0"
MODULE_NAME = "msm_machinelearning"


def apply_pep621_overrides(pyproject: PyprojectTomlFile) -> None:
    for key in [
        "tool.poetry.name",
        "tool.poetry.version",
        "tool.poetry.description",
        "tool.poetry.readme",
        "tool.poetry.authors",
    ]:
        pyproject.add_deletion_override(key)

    pyproject.add_override("project.name", PROJECT_NAME)
    pyproject.add_override("project.version", PROJECT_VERSION)
    pyproject.add_override("project.description", PROJECT_DESCRIPTION)
    pyproject.add_override("project.readme", "README.md")
    pyproject.add_override("project.requires-python", PYTHON_RANGE)
    pyproject.add_override("project.dynamic", ["dependencies"])
    pyproject.add_override(
        "project.authors",
        [{"name": AUTHOR_NAME, "email": AUTHOR_EMAIL}],
    )


project = PythonProject(
    author_email=AUTHOR_EMAIL,
    author_name=AUTHOR_NAME,
    module_name=MODULE_NAME,
    name=PROJECT_NAME,
    description=PROJECT_DESCRIPTION,
    pip=False,
    poetry=True,
    venv=False,
    version=PROJECT_VERSION,
    deps=[f"python@{PYTHON_RANGE}",
          "beautifulsoup4@4.12.3",
          "markdownify@0.6.5",
          "requests@2.28.2"
          ],

)

pyproject = next(
    component
    for component in project.components
    if isinstance(component, PyprojectTomlFile),
)

apply_pep621_overrides(pyproject)

project.synth()
