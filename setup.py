import codecs
import setuptools

with codecs.open('requirements.txt', 'r', encoding='utf-16') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="chat-assistant",
    version="0.1.1",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    description="Chat Assistant",
    author="sugarkwork",
    python_requires='>=3.10',
)
