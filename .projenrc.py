from projen.python import PythonProject

project = PythonProject(
    author_email="smidmi@gmail.com",
    author_name="Milan Šmíd",
    module_name="msm_machinelearning",
    name="msm-machinelearning",
    description="MSM test projen project",
    pip=False,
    poetry=True,
    venv=False,
    version="0.1.1",
    deps=["python@>=3.13,<4.0",
          "beautifulsoup4@4.12.3",
          "markdownify@0.6.5",
          "requests@2.28.2"
          ],

)

# Správný přístup přes PoetryPyproject → file → addOverride
if project.packaging_manager:
    # packagingManager je Poetry instance, ta má přístup k pyproject přes komponent
    pyproject = project.try_find_object_file("pyproject.toml")
    if pyproject:
        pyproject.add_override("project.readme", "README.md")
        pyproject.add_override("project.name", project.name)
        pyproject.add_override("project.version", project.version)

# project.pyprojectConfig = {
#     "project": {
#         "name": project.name,
#         "version": project.version,
#         "readme": "README.md",
#     }
# }

project.synth()
