# Trabalho_Final_FIA

Este repositório contém o projeto **"Trabalho_Final_FIA"**, desenvolvido como parte das atividades acadêmicas da disciplina de Fundamentos de Inteligência Artificial no Instituto de Computação da Universidade Federal do Amazonas (IComp/UFAM).

---

## 👥 Equipe

| Nome | E-mail |
|------|---------|
| Anna Luisa Antony Afonso | anna.antony@icomp.ufam.edu.br |
| Elaine de Castro Freire | elaine.freire@icomp.ufam.edu.br |
| Giovanna Bembom da Silva Bandeira | giovanna.bembom@icomp.ufam.edu.br |
| Luiggy Augusto Lima Alves | luiggy.alves@icomp.ufam.edu.br |
| Manuela Figueira Batista | manuela.batista@icomp.ufam.edu.br |
| Maria Flach da Costa | maria.flach@icomp.ufam.edu.br |
| Mariana Ramos André Simões | mariana.simoes@icomp.ufam.edu.br |
| Raíssa Clara Teixeira Brasil | raissa.brasil@icomp.ufam.edu.br |

---
## Neuro-Symbolic Learning (NeSy)

O **Neuro-Symbolic Learning (NeSy)** refere-se a uma classe de abordagens em Inteligência Artificial que buscam integrar técnicas de **aprendizado de máquina conexionista**, especialmente redes neurais artificiais, com **métodos simbólicos**, tradicionalmente associados à lógica formal, representação de conhecimento e raciocínio automatizado.

O principal objetivo do NeSy é combinar as vantagens complementares desses dois paradigmas. Enquanto os modelos neurais apresentam capacidade de aprendizado a partir de dados, generalização e robustez a ruídos, os métodos simbólicos oferecem **interpretabilidade**, **consistência lógica** e **capacidade de raciocínio explícito** baseada em regras e conhecimento prévio. A integração permite o desenvolvimento de sistemas capazes de aprender a partir de dados empíricos sem abrir mão de restrições lógicas e conhecimento estruturado.

As abordagens NeSy são particularmente relevantes em cenários onde dados são escassos, conhecimento especializado está disponível ou onde a explicabilidade das decisões do modelo é um requisito fundamental. Dessa forma, o NeSy representa um avanço em direção a sistemas de IA mais confiáveis, transparentes e alinhados ao raciocínio humano.

---

## Logic Tensor Networks (LTN)

As **Logic Tensor Networks (LTN)** constituem um framework específico dentro do paradigma neuro-simbólico, proposto para integrar **lógica de primeira ordem** com **redes neurais diferenciáveis**. Nesse modelo, predicados lógicos são representados como funções contínuas parametrizadas, geralmente implementadas por redes neurais, enquanto constantes e variáveis são mapeadas para vetores em espaços de características.

As fórmulas lógicas, expressas por meio de conectivos (como conjunção, disjunção e implicação) e quantificadores, são relaxadas para o domínio contínuo utilizando princípios da **lógica fuzzy**. Isso permite que o grau de satisfação das fórmulas seja mensurável e diferenciável, possibilitando o treinamento dos modelos por meio de métodos de otimização baseados em gradiente.

O aprendizado em LTNs é orientado pela maximização do grau de satisfação das restrições lógicas impostas, em conjunto com a adequação aos dados observados. Dessa forma, o modelo aprende representações que não apenas se ajustam aos dados, mas também respeitam, de forma aproximada, o conhecimento lógico previamente especificado.

As LTNs são amplamente utilizadas em tarefas que exigem **aprendizado com conhecimento prévio**, **raciocínio relacional** e **interpretação semântica**, sendo uma das abordagens mais consolidadas no campo de aprendizado neuro-simbólico.

---

## Dataset CLEVR

O **CLEVR (Compositional Language and Elementary Visual Reasoning)** é um dataset sintético utilizado para a avaliação de modelos de **raciocínio visual** e **aprendizado multimodal**. Seu principal objetivo é permitir a análise controlada da capacidade dos modelos em realizar raciocínio composicional, reduzindo vieses comuns presentes em bases de dados coletadas a partir de imagens do mundo real.

O dataset é composto por imagens sintéticas tridimensionais contendo múltiplos objetos geométricos simples, como cubos, esferas e cilindros, dispostos sobre uma superfície. Esses objetos variam sistematicamente em atributos como **forma**, **cor**, **material** e **tamanho**, o que possibilita a construção de cenários visuais bem definidos e semanticamente controlados.

Associadas a cada imagem, o CLEVR fornece **perguntas em linguagem natural** projetadas para exigir diferentes tipos de raciocínio, incluindo contagem, comparação, identificação de atributos, relações espaciais e encadeamento lógico de múltiplas etapas. As perguntas são acompanhadas de respostas objetivas, permitindo uma avaliação quantitativa precisa do desempenho dos modelos.

Uma característica central do CLEVR é a disponibilidade de **anotações estruturadas**, como grafos de cena e programas funcionais que descrevem as etapas de raciocínio necessárias para responder cada pergunta. Essas informações tornam o dataset particularmente adequado para pesquisas em **modelos neuro-simbólicos**, **raciocínio explicável** e **integração entre visão e linguagem**, sendo adotado como benchmark em estudos de raciocínio visual e inferência multimodal.

---

## Comentários sobre o Código

A partir deste ponto, o texto apresenta comentários e explicações diretamente associados ao código-fonte, com o objetivo de descrever suas definições, estrutura e funcionamento. As seções a seguir estão organizadas de forma a acompanhar a lógica do código apresentado.

### 1 — Definições
#### Bloco 1.1 — Instalação das dependências

Neste bloco são instaladas as bibliotecas necessárias para a execução do código no ambiente do Google Colab.

- **torch**: biblioteca principal para computação com tensores e redes neurais (PyTorch).
- **ltn**: biblioteca *Logic Tensor Networks*, utilizada para combinar lógica simbólica com aprendizado profundo.
- **numpy**: biblioteca fundamental para computação numérica e manipulação de arrays.
- **matplotlib**: biblioteca para visualização de dados e geração de gráficos.

O uso do comando `!pip install` garante que todas as dependências estejam disponíveis, mesmo em ambientes recém-criados.

#### Bloco 1.2 — Importação de módulos

Este bloco realiza a importação das bibliotecas e módulos que serão utilizados ao longo do código.

- **torch** e **torch.nn**: fornecem suporte a tensores, operações numéricas e definição de modelos neurais.
- **ltn**: permite a definição de predicados, funções e fórmulas lógicas diferenciáveis.
- **numpy**: utilizado para operações matemáticas auxiliares.
- **matplotlib.pyplot** e classes gráficas associadas: usadas para visualização de objetos geométricos e resultados.
- **sklearn.metrics**: fornece métricas de avaliação como *accuracy*, *precision*, *recall* e *F1-score*.

Ao final, são exibidas mensagens confirmando o sucesso da importação e a versão do PyTorch utilizada, o que auxilia na reprodutibilidade dos experimentos.


#### Bloco 1.3 — Seleção do dispositivo de execução

Neste bloco é definido o dispositivo computacional que será utilizado para executar os cálculos.

- Caso uma GPU compatível com CUDA esteja disponível, o código será executado na GPU (`cuda:0`).
- Caso contrário, o processamento será realizado na CPU.

Essa verificação automática permite melhor desempenho quando há suporte a aceleração por hardware.
O dispositivo selecionado é exibido para fins de verificação e depuração.


#### Bloco 1.4 — Definição de constantes geométricas

Este bloco define constantes que representam o tamanho de objetos geométricos utilizados no experimento.

- **SMALL_SIZE**: representa objetos classificados como pequenos, com dimensão 2 × 2.
- **LARGE_SIZE**: representa objetos classificados como grandes, com dimensão 3 × 3.

Esses valores serão utilizados para caracterizar propriedades visuais dos objetos, possivelmente associadas a rótulos lógicos ou semânticos dentro do modelo.

### 2 — Tarefa principal: taxonomia e formas

#### Bloco 2.1 — Dataset fixo para a tarefa de taxonomia e formas

Este bloco define a função `load_fixed_clevr_dataset`, responsável por construir um dataset sintético fixo, inspirado no CLEVR, para a tarefa principal de **taxonomia de objetos geométricos**. O objetivo é representar atributos visuais e semânticos de cada objeto, permitindo seu uso em modelos neurais e em *Logic Tensor Networks*.

Os dados são definidos manualmente na lista `raw_data`, onde cada instância descreve um objeto por meio de sua posição espacial `(x, y)`, cor no espaço RGB, forma geométrica codificada em *one-hot* (círculo, quadrado, cilindro, cone ou triângulo) e um atributo binário de tamanho (*small* ou *large*). Essa definição explícita garante controle total sobre a distribuição e a taxonomia das classes.

As coordenadas espaciais são normalizadas para o intervalo `[0, 1]`, tornando o dataset independente da escala absoluta do plano. Em seguida, para cada objeto é construído um vetor de características com **11 atributos**: posição normalizada (2), cor RGB (3), codificação da forma (5) e tamanho (1). Esses vetores constituem a representação numérica utilizada pelos modelos.

O atributo lógico de tamanho é convertido em uma **dimensão física real** (`SMALL_SIZE` ou `LARGE_SIZE`), armazenada separadamente, permitindo integrar informações geométricas contínuas ao raciocínio lógico. Além disso, são gerados rótulos textuais legíveis que descrevem cada objeto (forma, cor, tamanho e posição), facilitando inspeção, depuração e visualização.

Por fim, os vetores são convertidos em um tensor PyTorch do tipo `float32`, as dimensões físicas em um array NumPy, e uma validação garante que o dataset final possua formato `(25, 11)`. A função retorna o tensor de dados, os rótulos descritivos e as dimensões físicas associadas a cada objeto.

#### Bloco 2.2 — Geração de dataset CLEVR aleatório

Este bloco define a função `generate_random_clevr_dataset`, responsável por criar um **dataset sintético aleatório**, compatível com a mesma taxonomia e estrutura do dataset fixo. Essa versão é utilizada para testes, generalização e experimentos controlados por semente aleatória.

A função permite definir o número de objetos (`n_objects`) e uma semente (`seed`) para garantir reprodutibilidade. Quando fornecida, a semente sincroniza os geradores do NumPy e do PyTorch, assegurando que os mesmos dados sejam gerados em execuções distintas.

Para cada objeto, são amostradas posições contínuas `(x, y)` dentro dos limites do plano, posteriormente normalizadas para o intervalo `[0, 1]`. As cores são geradas aleatoriamente no espaço RGB, e a forma geométrica é selecionada uniformemente entre cinco categorias (círculo, quadrado, cilindro, cone e triângulo), sendo codificada em *one-hot*.

O atributo de tamanho é amostrado como uma variável binária (*small* ou *large*), mantendo compatibilidade semântica com o dataset fixo. Cada instância é representada por um vetor de **11 atributos**: posição normalizada (2), cor (3), forma (5) e tamanho (1), garantindo compatibilidade direta com os modelos neurais e lógicos utilizados.

O tamanho lógico é convertido em uma **dimensão física real** (`SMALL_SIZE` ou `LARGE_SIZE`), armazenada separadamente para uso em raciocínio geométrico ou visual. Para fins de identificação, cada objeto recebe um rótulo simples (`Obj1`, `Obj2`, …).

Ao final, os dados são convertidos para um tensor PyTorch do tipo `float32`, as dimensões físicas para um array NumPy, e a função retorna o tensor de atributos, os rótulos e as dimensões associadas a cada objeto.

#### Bloco 2.3 — Visualização do cenário 2D

Este bloco define a função `plot_scene`, responsável por **visualizar graficamente o cenário 2D** do dataset CLEVR. As coordenadas normalizadas são desnormalizadas apenas para fins de visualização, preservando a correspondência espacial original do plano.

Cada objeto é renderizado com sua **cor RGB**, **forma geométrica** e **dimensão física real** (`2×2` ou `3×3`). As formas são desenhadas como *patches* preenchidos do Matplotlib (círculo, quadrado, cilindro, cone e triângulo), sempre acompanhadas de sua **caixa delimitadora**, evidenciando o espaço físico ocupado.

Os rótulos indicam o identificador do objeto e sua dimensão, facilitando inspeção visual e depuração. O gráfico utiliza grade com coordenadas inteiras, aspecto igual e legenda semântica, garantindo uma representação fiel e interpretável do cenário espacial.


