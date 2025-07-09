import codecs
import setuptools

with codecs.open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

with codecs.open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="chat-assistant",
    version="0.2.0",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    description="Chat Assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="sugarkwork",
    url="https://github.com/sugarkwork/chat_assistant",
    project_urls={
        "Source": "https://github.com/sugarkwork/chat_assistant"
    },
    license="MIT",
    license_files=["LICENSE"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks"
    ],
    python_requires='>=3.10',
)
