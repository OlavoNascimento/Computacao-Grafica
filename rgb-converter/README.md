# O que é
Ferramenta que permite carregar imagens em formato RGB, convertê-la para CMYK ou HSL e salvar cada camada em uma imagem.

# Como executar
- Siga as instruções para instalar as dependências conforme descrito na seção 'Dependências'.
- Execute:
```console
$ ./rgb_converter.py $caminho_para_imagem
```
- Verifique a saída nas pastas 'CMKY' e 'HSL'.

# Dependências
- [Numpy](https://pypi.org/project/numpy/)
	- Dependência do OpenCV, é utilizado para armazenar os pixels das imagens carregadas em arrays da biblioteca.
- [OpenCV](https://pypi.org/project/opencv-python/)
	- Realiza a leitura de imagens para matrizes.
- [Pillow](https://pypi.org/project/Pillow/)
	- Utilizado para salvar as imagens das diferentes camadas utilizando os padrões RGB, CMYK e HSL.

## Como instalar
### Poetry
- Para gerenciamento de dependências foi utilizado o [Poetry](https://python-poetry.org/).
	- Instruções para instalação podem ser encontradas [aqui](https://python-poetry.org/docs/master/#installing-with-the-official-installer)
- Para instalar as dependências através do Poetry execute:
```console
$ poetry install
```
- Para utilizar o ambiente virtual criado pelo Poetry utilize:
```console
$ poetry shell
```

### PIP
- Também é possível instalar as dependências através do [PIP](https://pip.pypa.io/en/stable/), para isso execute:
```console
$ pip install -r requirements.txt
```

### Pacotes do sistema
- Por fim, também é possível instalar as dependências para todo o sistema, entretanto é a opção mais trabalhosa.
- As bibliotecas necessárias estão listadas no início desta seção, assim como nos arquivos 'pyproject.toml' e 'requirements.txt'.

