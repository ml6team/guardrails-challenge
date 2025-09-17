from setuptools import setup, find_packages

setup(
    name="workshop_utils",
    version="0.1.0",
    description="Utils functions for the TechDays 2025 Workshop",
    packages=find_packages(include=["workshop_utils", "workshop_utils.*"]),
    install_requires=[
        "torch",
        "tqdm",
        "numpy",
        "scikit-learn"
    ],
    include_package_data=True,
    package_data={
        "workshop_utils": [
            "hidden_states/**/*",
            "workshop_dataset.csv",
        ],
    },
)
