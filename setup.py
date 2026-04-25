from setuptools import setup, find_packages

setup(
    name="eeg-cp-bci",
    version="0.1.0",
    description="EEG-based Motor Intention Decoder for Children with Cerebral Palsy",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "mne>=1.6.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
    ],
)
