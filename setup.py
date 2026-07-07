from setuptools import setup, find_packages

setup(
    name="eco-logic",
    version="2.1.0",
    description="Thermal-Aware RL Optimizer for PowerCool Data Centers",
    author="Ravada Omkar",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "requests>=2.31",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "prometheus-client>=0.19",
        "mysql-connector-python>=8.2",
    ],
    extras_require={
        "torch": ["torch>=2.1.0"],
        "dev":   ["pytest>=7.4", "black>=23.0", "isort>=5.12", "flake8>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "ecologic=src.main:main",
        ]
    },
)
