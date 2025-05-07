import codecs
import setuptools

with codecs.open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="chat-assistant",
    version="0.1.5",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    description="Chat Assistant",
    author="sugarkwork",
    url="https://github.com/sugarkwork/chat_assistant",
    project_urls={
        "Source": "https://github.com/sugarkwork/chat_assistant"
    },
    python_requires='>=3.10',
)
