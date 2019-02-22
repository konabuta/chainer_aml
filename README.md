# Azure Machine Learning service Chainer サンプルコード

Azure Machine Learning service におけるChainerのサンプルコードをご紹介します。

[ローカル環境で学習](../Chainer-local)

[Machine Learning Computeで学習](../Chainer-remote)

[Hyperdriveによるパラメータチューニング](../Chainer-hyperdrive)

[ChainerMN で分散学習](../Chainer-remote)

[ChainerMN + Infiniband で分散学習](../Chainer-remote)





# Azure Machine Learning service の作成と構成

## 1. Azure Machine Learning ワークスペースの作成

こちらのドキュメントを参考に、Azure Machine Learning ワークスペースを作成してください。

Azure Machine Learning Services ワークスペースを作成し、サンプルコードをAzure Notebookから実行する:

https://docs.microsoft.com/ja-jp/azure/machine-learning/service/quickstart-get-started



## 2a. Azure Notebook を利用する場合

Azure Notebook には Azure Machine Learning service Python SDK がプリインストールされているため、すぐにハンズオンを始める事ができます。

ノートブックで Azure Machine Learning service を使用する:  

https://docs.microsoft.com/ja-jp/azure/notebooks/use-machine-learning-services-jupyter-notebooks

Azure Notebook にて、GitHub にあるコンテンツをクローンします。

GitHub からプロジェクトをインポートする:  

https://docs.microsoft.com/ja-jp/azure/notebooks/create-clone-jupyter-notebooks#import-a-project-from-github


## 2b. ローカル環境 を利用する場合

ローカル環境を利用される場合には、Python SDK のインストールが必要になります。

ローカル環境のセットアップ：  
https://docs.microsoft.com/ja-jp/azure/machine-learning/service/how-to-configure-environment#local


```shell
# create a new Conda environment with Python 3.6, NumPy, and Cython
conda create -n myenv Python=3.6 cython numpy

# activate the Conda environment
conda activate myenv

# On macOS run
source activate myenv

```


次に、git clone などを用いてハンズオンコンテンツをGitHubからクローンします。

インポート元：

```
https://github.com/konabuta/chainer_aml
```



Copyright (c) Microsoft Corporation. All rights reserved.  
    
