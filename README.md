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

#### Bloco 2.4 — Conectivos lógicos fuzzy e quantificadores

Este bloco define os **operadores lógicos fuzzy** e os **quantificadores** utilizados nas *Logic Tensor Networks*. São implementados diretamente a partir dos operadores padrão da biblioteca LTN, incluindo negação, conjunção, disjunção e implicação fuzzy.

Os quantificadores universal e existencial são implementados por meio de agregadores do tipo *p-mean*, permitindo avaliar fórmulas sobre conjuntos de objetos de forma diferenciável. Um agregador global de satisfação (`SatAgg`) é definido para combinar múltiplas fórmulas lógicas em um único valor de consistência, servindo como métrica de satisfação das restrições lógicas.

#### Bloco 2.5 — Predicados neurais para classificação de formas

Este bloco define o modelo `MultiShapePredictor`, uma rede neural única responsável por aprender **predicados de forma** para cinco categorias geométricas. A arquitetura utiliza um **encoder compartilhado** para extrair representações latentes gerais a partir dos vetores de atributos, seguido por **cabeças especializadas**, uma para cada forma.

Cada cabeça retorna um valor escalar em `[0,1]`, interpretado como o grau de verdade fuzzy do predicado correspondente. Essa estrutura permite o compartilhamento de conhecimento entre formas, mantendo ao mesmo tempo especialização semântica, sendo diretamente integrável ao raciocínio lógico diferenciável das LTNs.

#### Bloco 2.6 — Wrapper LTN para predicados de forma

Este bloco define a classe `LTNShapePredicate`, que atua como um **wrapper** para tornar o modelo neural de formas compatível com a interface esperada pelas *Logic Tensor Networks*. O wrapper permite que o predicado seja utilizado como uma função lógica, encapsulando a saída do modelo em um objeto com atributo `value`.

Essa adaptação garante integração direta entre as saídas neurais contínuas e os operadores lógicos fuzzy da LTN, sem alterar a arquitetura do modelo.

#### 2.7 — Instanciação dos predicados de forma

Este bloco cria os cinco predicados lógicos de forma (`isCircle`, `isSquare`, `isCylinder`, `isCone` e `isTriangle`) a partir de um **único modelo neural compartilhado**. Cada predicado utiliza uma cabeça especializada distinta, preservando compartilhamento de representações e especialização semântica.

Essa abordagem reduz redundância, melhora eficiência de aprendizado e permite que múltiplos predicados lógicos sejam avaliados de forma consistente dentro do framework de raciocínio lógico diferenciável.

#### Bloco 2.8 — Predicados neurais de tamanho

Este bloco define o modelo `SizePredictor`, uma rede neural dedicada à identificação do **tamanho dos objetos**. O modelo aprende dois predicados fuzzy binários: `isSmall(x)` e `isBig(x)`, correspondentes às categorias *small* e *large*.

A arquitetura utiliza um **encoder compartilhado** para extrair características relevantes de tamanho a partir dos vetores de atributos, seguido por duas cabeças especializadas, uma para cada predicado. Cada cabeça retorna um valor contínuo em `[0,1]`, interpretado como o grau de verdade fuzzy do predicado, permitindo integração direta com os operadores e quantificadores das *Logic Tensor Networks*.

#### Bloco 2.9 — Wrapper LTN para predicados de tamanho

Este bloco define a classe `LTNSizePredicate`, responsável por tornar o modelo neural de tamanho compatível com a interface das *Logic Tensor Networks*. O wrapper permite que os predicados `isSmall(x)` e `isBig(x)` sejam utilizados como funções lógicas, encapsulando a saída contínua do modelo em um objeto com atributo `value`.

Essa abstração garante a integração transparente entre os predicados neurais de tamanho e os operadores lógicos fuzzy, mantendo consistência semântica e compatibilidade com o mecanismo de raciocínio diferenciável da LTN.

#### Bloco 2.10 — Instanciação dos predicados de tamanho

Este bloco instancia os predicados lógicos de tamanho `isSmall` e `isBig` a partir de um **único modelo neural compartilhado** (`SizePredictor`). Cada predicado utiliza uma cabeça especializada correspondente à sua categoria, mantendo consistência semântica e eficiência computacional.

Essa abordagem permite avaliar os predicados de tamanho de forma diferenciável e integrá-los diretamente às fórmulas lógicas fuzzy no framework de *Logic Tensor Networks*.

#### Bloco 2.11 — Extração do *ground truth*

Este bloco realiza a extração do *ground truth* a partir do **dataset fixo**, garantindo alinhamento total com a especificação definida em `load_fixed_clevr_dataset`. As informações de forma são recuperadas diretamente da codificação *one-hot* nos índices `[5–9]`, enquanto o tamanho é extraído do índice `[10]`.

Para as formas, são geradas máscaras booleanas (`circles_gt`, `squares_gt`, `cylinders_gt`, `cones_gt`, `triangles_gt`) indicando a classe verdadeira de cada objeto. Para o tamanho, são criadas máscaras correspondentes às categorias *small* (`0.0`) e *big* (`1.0`).

A função auxiliar `get_size_labels` encapsula a extração dos rótulos de tamanho, facilitando reutilização e garantindo consistência semântica entre os dados, os predicados neurais e as avaliações posteriores.

#### Bloco 2.12 — Axiomas lógicos para predicados de forma

Este bloco define a função `shape_axioms`, responsável por construir o conjunto de **axiomas lógicos fuzzy** que regulam o aprendizado e a consistência dos predicados de forma. Os axiomas são avaliados diretamente sobre o `data_tensor` e agregados em um único valor de satisfação.

São incorporados seis tipos principais de restrições:  
(i) **supervisionamento positivo forte**, reforçando ativações corretas com maior peso;  
(ii) **supervisionamento negativo seletivo**, penalizando apenas confusões entre formas semanticamente distintas;  
(iii) **exclusividade global**, incentivando que cada objeto ative predominantemente uma única forma;  
(iv) **penalização de entropia**, forçando especialização e decisões mais nítidas;  
(v) **exclusividade par-a-par seletiva**, aplicada apenas a pares de formas empiricamente problemáticos;  
(vi) **completude**, garantindo que todo objeto pertença a pelo menos uma categoria de forma.

O conjunto de axiomas é combinado por média, produzindo um escore global de satisfação lógica que pode ser utilizado diretamente como objetivo de otimização no framework de *Logic Tensor Networks*.

#### Bloco 2.13 — Axiomas lógicos para predicados de tamanho e agregação final

Este bloco define os **axiomas lógicos associados aos predicados de tamanho**, bem como a função de **agregação global** entre forma e tamanho.

A função `size_axioms` impõe duas restrições principais:  
(i) **supervisionamento positivo**, garantindo alta ativação de `isSmall` para objetos pequenos e de `isBig` para objetos grandes;  
(ii) **supervisionamento negativo**, penalizando ativações contraditórias (objetos pequenos ativando `isBig` e vice-versa).  

Devido à natureza binária do atributo tamanho, não são necessárias restrições adicionais de exclusividade ou completude, diferentemente do caso das formas.

A função `combined_axioms` integra os axiomas de **forma** e **tamanho** por meio de uma média ponderada, permitindo controlar a influência relativa de cada grupo de predicados durante o treinamento. No arranjo adotado, maior peso é atribuído às formas (0.7), refletindo sua maior complexidade semântica em comparação ao tamanho (0.3).

#### Bloco 2.14 — Função de treinamento conjunto dos predicados

Este bloco implementa o **processo de treinamento unificado** de todos os predicados lógicos definidos na Seção 2, abrangendo **formas** e **tamanhos** simultaneamente. Os parâmetros treináveis incluem tanto o modelo compartilhado de formas quanto o modelo de tamanhos, otimizados conjuntamente por meio do algoritmo Adam.

O critério de otimização é baseado na **Lógica Tensorial (LTN)**, onde a função objetivo busca **maximizar o grau de satisfação dos axiomas lógicos**. Para isso, a perda é definida como o complemento da satisfação global (`loss = 1 − sat`), permitindo o uso direto de gradiente descendente.

Durante cada época, são computadas separadamente a satisfação total, a satisfação associada às formas e a satisfação associada aos tamanhos, possibilitando monitorar o comportamento individual de cada grupo de predicados ao longo do treinamento. Essas métricas são armazenadas para posterior análise e visualização.

O processo de treinamento é inteiramente guiado por restrições lógicas, **sem uso de rótulos explícitos no formato tradicional**, reforçando o caráter neuro-simbólico e declarativo da abordagem adotada.



