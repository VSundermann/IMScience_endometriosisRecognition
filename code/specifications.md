O codigo principal deste trabalho se encontra no arquivo EndoSeg_ImScience.ipynb, executado em ambiente do Google Colab com GPU A100 HIGH RAM, e ambiente virtual mais recente disponivel.

Os arquivos AutoSeg_Endometriosis.py e RAovSeg_tools.py possuem o mesmo codigo, mas para execucao local. O codigo entretanto esta desatualizado, e sera refeito para atender as boas praticas de projetos Deep Learning.

Codigos auxiliares para preparo e processamento inicial do dataset estao disponiveis em:
dataset_prep.py
dataset_label_organizer.py
dataset_mri_organizer.py

Os codigos de processamento serao refeitos para simplificar a logica e adicionar mais funcionalidades.

===================================================

Instrucoes para execucao local:

1. Preparo do dataset
2. Ativacao do ambiente virtual: source venv/bin/activate
3. Instalacao das dependencias: pip install -r requirements.txt
4. Execucao do codigo: python EndoSeg_ImScience.ipynb