from setuptools import setup, find_packages

setup(
    name='SLR',  # 项目的名称
    version='0.1.0',  # 版本号
    author='Your Name',  # 作者姓名
    author_email='your.email@example.com',  # 作者邮箱
    description='A Sign Language Recognition System',  # 项目描述
    long_description=open('README.md').read(),  # 长描述，通常是从 README 文件读取
    long_description_content_type='text/markdown',  # 长描述的内容类型
    url='https://github.com/yourusername/SLR',  # 项目主页
    packages=find_packages(exclude=('tests', 'notebooks')),  # 自动发现并包含所有的包
    package_data={  # 包含的数据文件
        '': ['*.yaml', '*.pkl', '*.txt', '*.sh', '*.stm'],
    },
    include_package_data=True,  # 包括 MANIFEST.in 文件指定的数据文件
    install_requires=[  # 安装依赖
        'numpy',
        'torch',
        'scikit-learn',
        'pandas',
        'wandb',
        # 其他依赖...
    ],
    extras_require={  # 可选依赖
        'dev': [
            'pytest',
            'coverage',
            'flake8',
            'black',
            'isort',
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
        ],
    },
    classifiers=[  # 分类器
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={  # 入口点
        'console_scripts': [
            'slr-run=slr.main:main',  # 命令行入口点
        ],
    },
)
