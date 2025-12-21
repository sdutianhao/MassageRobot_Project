from setuptools import find_packages, setup

requirements = ["wheel",
                "torch>=1.6.0",
                "smplx",
                "trimesh",
                "tqdm",
                "matplotlib",
                "moderngl-window==2.4.6"]

setup(
    name="skel",
    description="SKEL model Loader.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version=1.0,
    author="Marilyn Keller",
    packages=find_packages(),
    include_package_data=True,
    keywords=[
        "motion",
        "machine learning",
        "sequences",
        "smpl",
        "computer graphics",
        "computer vision",
        "3D",
        "meshes",
        "skel",
        "smpl"
    ],
    platforms=["any"],
    install_requires=requirements,
)
