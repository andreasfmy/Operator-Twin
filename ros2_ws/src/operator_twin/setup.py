from setuptools import setup

package_name = "operator_twin"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Andrea Moya",
    maintainer_email="amoyadel@purdue.edu",
    description="Node that computes and publishes operator awareness state",
    license="MIT",
    entry_points={
        "console_scripts": [
            "awareness_state = operator_twin.awareness_state_node:main",
        ],
    },
)