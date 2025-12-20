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

#### Bloco 1.1 — Instalação das bibliotecas principais

Nesta célula realizamos a instalação das bibliotecas fundamentais do projeto:

- **torch**: framework principal para computação tensorial e redes neurais.
- **ltn**: implementação de *Logic Tensor Networks* para integração entre lógica e aprendizado profundo.
- **numpy**: suporte para operações numéricas.
- **matplotlib**: biblioteca para visualização gráfica.

Essas dependências formam a base para o desenvolvimento e experimentação do modelo.

#### Bloco 1.2 — Instalação do LTNtorch

Aqui instalamos o **LTNtorch**, uma biblioteca baseada em PyTorch que permite:

- Representar predicados lógicos como funções diferenciáveis;
- Definir regras da lógica de primeira ordem;
- Integrar conhecimento simbólico ao processo de aprendizado.

Essa biblioteca será utilizada para construir e avaliar fórmulas lógicas no modelo.

#### Bloco 1.3 — Importação dos módulos

Nesta etapa importamos todas as bibliotecas necessárias para o experimento, incluindo:

- PyTorch e seus módulos de redes neurais;
- LTN para modelagem lógica;
- NumPy para operações numéricas auxiliares;
- Matplotlib para visualização de objetos geométricos;
- Métricas de avaliação do `scikit-learn`.

Ao final, são exibidas mensagens confirmando o sucesso da importação e a versão do PyTorch utilizada.

#### Bloco — 1.4 Configuração do dispositivo de execução

Esta célula define automaticamente o dispositivo de execução do código:

- **GPU (CUDA)**, caso esteja disponível;
- **CPU**, caso contrário.

Essa abordagem garante maior eficiência computacional e compatibilidade com diferentes ambientes, como máquinas locais e Google Colab.

#### Bloco 1.5 — Definição de constantes geométricas

Aqui são definidas constantes que representam o tamanho dos objetos geométricos usados no experimento:

- `SMALL_SIZE`: objetos pequenos, com dimensão 2 × 2;
- `LARGE_SIZE`: objetos grandes, com dimensão 3 × 3.

Esses valores serão utilizados posteriormente para caracterizar propriedades geométricas e regras lógicas.

# 2 - Tarefa Principal: Taxonomia e Formas
#### Bloco 2.1 — Dataset CLEVR simplificado — instanciação determinística

Neste bloco é definida a função `load_fixed_clevr_dataset`, responsável por construir um **dataset sintético fixo**, utilizado como **referência canônica** ao longo dos experimentos.

Esse dataset possui as seguintes finalidades:
- Servir como base controlada para depuração e validação;
- Facilitar a interpretação qualitativa dos resultados;
- Permitir análises detalhadas de casos específicos.

##### Estrutura dos objetos

Cada objeto é representado por um vetor de **11 atributos**, organizados da seguinte forma:

- `[0–1]`: coordenadas espaciais `(x, y)` normalizadas no intervalo `[0, 1]`;
- `[2–4]`: componentes de cor no espaço RGB;
- `[5–9]`: codificação *one-hot* da forma geométrica (círculo, quadrado, cilindro, cone ou triângulo);
- `[10]`: atributo de tamanho (`0.0` = pequeno, `1.0` = grande).

Além do vetor numérico, cada objeto possui:
- Um rótulo textual legível para inspeção humana;
- Uma dimensão física real (`2×2` ou `3×3`) derivada do atributo de tamanho.

Esse dataset é **determinístico**, isto é, sua composição não varia entre execuções.

#### Bloco 2.2 — Dataset CLEVR simplificado — instanciação estocástica

Neste bloco é definida a função `generate_random_clevr_dataset`, que gera um **dataset sintético aleatório** seguindo **exatamente a mesma estrutura vetorial** apresentada na Seção 2.1.

A diferença entre os dois datasets reside exclusivamente no **processo de geração dos objetos**:
- No dataset fixo, os atributos são pré-definidos;
- No dataset aleatório, os atributos são amostrados estocasticamente.

##### Propriedades

- A dimensionalidade e a organização dos atributos são idênticas às do dataset determinístico;
- As formas geométricas, cores, posições e tamanhos são sorteados aleatoriamente;
- Uma *seed* opcional pode ser fornecida para garantir reprodutibilidade experimental.

Esse dataset é utilizado principalmente para:
- Avaliar a capacidade de generalização dos modelos;
- Explorar múltiplos cenários possíveis mantendo a mesma taxonomia;
- Evitar dependência excessiva de exemplos específicos.

#### Bloco 2.3 — Visualização do cenário bidimensional

Neste bloco é definida a função `plot_scene`, responsável por **visualizar graficamente instâncias do dataset CLEVR**, sejam elas provenientes do dataset determinístico (Seção 2.1) ou do dataset estocástico (Seção 2.2).

A função assume implicitamente a **mesma estrutura de dados definida anteriormente**, garantindo compatibilidade com ambos os tipos de dataset.

##### Convenções de visualização

- As coordenadas normalizadas são desnormalizadas para o plano original;
- A referência espacial adotada é:
  - o **canto inferior esquerdo** do objeto para a maioria das formas;
  - o **centro geométrico** para círculos.

##### Representação gráfica

Cada objeto é desenhado respeitando:
- Sua forma geométrica;
- Sua cor RGB;
- Sua dimensão física real (`2×2` ou `3×3`).

Caixas delimitadoras, rótulos textuais e uma legenda explicativa são incluídos para facilitar a interpretação visual e a validação do cenário.


#### Bloco 2.4 — Conectivos lógicos e quantificadores fuzzy

Neste bloco são definidos os conectivos e quantificadores fuzzy utilizados pelo framework Logic Tensor Networks (LTN), responsáveis por permitir a avaliação diferenciável de fórmulas lógicas.

São utilizados:
- **Not**: negação fuzzy padrão;
- **And**: conjunção por produto;
- **Or**: disjunção por soma probabilística;
- **Implies**: implicação de Reichenbach.

Os quantificadores definidos são:
- **Forall**: quantificador universal com agregação *p-mean error*;
- **Exists**: quantificador existencial com agregação *p-mean*.

O operador `sat_agg` agrega os graus de satisfatibilidade das fórmulas e é utilizado como objetivo de otimização.

#### Bloco 2.5 — Predicados neurais de forma

Neste bloco é definido um modelo neural para predição de formas geométricas a partir do vetor de atributos do dataset.

O modelo utiliza:
- Um **encoder compartilhado** para extração de características;
- **Cabeças de classificação independentes** para cada forma geométrica.

Cada cabeça produz uma saída contínua em `[0, 1]`, interpretada como grau de pertinência à forma correspondente.

As saídas do modelo são encapsuladas como **predicados LTN**, permitindo sua utilização direta em fórmulas lógicas fuzzy:
- `isCircle`
- `isSquare`
- `isCylinder`
- `isCone`
- `isTriangle`

#### Bloco 2.6 — Predicados neurais de tamanho

Neste bloco é definido um modelo neural para **classificação do tamanho dos objetos**, utilizando a mesma representação vetorial de 11 atributos.

O modelo é composto por:
- Um **encoder compartilhado** para extração de características;
- Duas **cabeças de classificação**, correspondentes às categorias:
  - pequeno (`small`)
  - grande (`big`)

Cada cabeça retorna um valor contínuo em `[0, 1]`, interpretado como grau de pertinência ao respectivo tamanho.

As saídas do modelo são encapsuladas como **predicados LTN**, permitindo sua utilização direta em fórmulas lógicas:
- `isSmall`
- `isBig`

#### Bloco 2.7 — Extração do ground truth

Neste bloco é realizada a extração dos **rótulos verdadeiros (ground truth)** a partir do dataset fixo definido na Seção 2.1.

Os rótulos são obtidos diretamente da codificação original dos dados:
- As **formas geométricas** são extraídas da codificação *one-hot* nos índices `[5–9]`;
- O **tamanho** é extraído do índice `[10]`, onde:
  - `0.0` representa objetos pequenos;
  - `1.0` representa objetos grandes.

Esses rótulos são utilizados para:
- Avaliação quantitativa dos modelos;
- Validação das predições neurais;
- Comparação com os graus de satisfatibilidade das fórmulas lógicas.

#### Bloco 2.8 — Axiomas lógicos para classificação de formas

Neste bloco são definidos os **axiomas lógicos associados à classificação de formas geométricas**, utilizando os predicados neurais definidos anteriormente.

Os axiomas impõem três tipos de restrições:

- **Axiomas positivos**: objetos rotulados com uma forma específica devem satisfazer o predicado correspondente (ex.: círculos satisfazem `isCircle`);
- **Exclusividade**: um objeto não pode pertencer simultaneamente a duas formas distintas;
- **Completude**: todo objeto deve pertencer a pelo menos uma das cinco formas consideradas.

Os axiomas são avaliados de forma fuzzy e agregados por meio do operador global de satisfatibilidade (`sat_agg`), resultando em um único escalar que mede o grau de consistência lógica das predições de forma.

#### Bloco 2.9 — Axiomas lógicos para classificação de tamanho

Neste bloco são definidos os **axiomas lógicos associados à classificação de tamanho**, considerando as duas categorias possíveis: pequeno e grande.

As restrições impostas são:

- Objetos pequenos devem satisfazer `isSmall` e não satisfazer `isBig`;
- Objetos grandes devem satisfazer `isBig` e não satisfazer `isSmall`.

Assim como no caso das formas, os axiomas são avaliados de maneira fuzzy e agregados em um único valor de satisfatibilidade.

#### Bloco 2.10 — Treinamento conjunto dos predicados

Neste bloco é definida a rotina de **treinamento conjunto** dos predicados neurais de **forma** e **tamanho**, utilizando as restrições lógicas como função objetivo.

O treinamento segue o paradigma de *Logic Tensor Networks*, onde:
- A função de perda é definida como `1 − sat`, sendo `sat` o grau de satisfatibilidade dos axiomas;
- Os parâmetros dos modelos de forma e tamanho são otimizados simultaneamente;
- A otimização é realizada via **Adam**.

Durante o treinamento são monitorados:
- A satisfatibilidade total dos axiomas;
- A satisfatibilidade dos axiomas de forma;
- A satisfatibilidade dos axiomas de tamanho.

O processo busca maximizar a consistência lógica global das predições em relação às taxonomias definidas.






### 3 - Tarefa: Raciocínio Espacial e Horizontal

#### Bloco 3.1 — Predicados Espaciais Aprendíveis

Este bloco define os **predicados espaciais binários aprendíveis** utilizados na tarefa de raciocínio espacial horizontal. Cada predicado é modelado por uma rede neural que recebe como entrada a **concatenação das representações de dois objetos** e produz um valor escalar em \([0,1]\), interpretado como o grau de verdade fuzzy da relação espacial entre eles.

A arquitetura empregada é um *Multilayer Perceptron (MLP)* com múltiplas camadas totalmente conectadas, funções de ativação ReLU e camadas de *dropout*, visando capturar dependências espaciais não lineares e reduzir sobreajuste. A camada final utiliza uma função sigmoide, garantindo compatibilidade com o framework de lógica fuzzy.

Instâncias independentes do modelo são criadas para cada relação espacial horizontal aprendida (`leftOf` e `rightOf`), permitindo que cada predicado adquira uma representação semântica própria a partir dos dados e das restrições lógicas impostas posteriormente.

#### Bloco 3.2 — Wrappers LTN para Predicados Espaciais

Este bloco implementa *wrappers* responsáveis por integrar os modelos neurais de relações espaciais ao framework de **Logic Tensor Networks (LTN)**. Esses *wrappers* garantem que as saídas dos predicados sejam tratadas como valores de verdade fuzzy, permitindo sua composição direta em axiomas e consultas lógicas.

A classe `LTNSpatialPredicate` encapsula um modelo neural aprendível, realizando a padronização das entradas (extração de valores, conversão para tensores e ajuste de dimensionalidade), seguida da concatenação dos objetos e da inferência do grau de verdade da relação espacial. O resultado é retornado em uma estrutura compatível com operações lógicas fuzzy.

Além disso, o predicado `closeTo` é definido como uma **função fixa**, baseada em um *kernel gaussiano* aplicado à distância euclidiana entre as posições dos objetos. Essa escolha permite modelar proximidade espacial de forma contínua e diferenciável, sem necessidade de treinamento supervisionado.

Por fim, os predicados `leftOf`, `rightOf` e `closeTo` são instanciados e disponibilizados para uso nos axiomas lógicos e nos procedimentos de raciocínio definidos nos blocos subsequentes.
#### Bloco 3.3 — Construção do Ground Truth Espacial

Este bloco define o procedimento para geração do *ground truth* das relações espaciais entre todos os pares de objetos do cenário. A partir das coordenadas normalizadas dos objetos, são calculadas matrizes booleanas que representam as relações `leftOf`, `rightOf` e `closeTo`.

As relações horizontais `leftOf` e `rightOf` são determinadas exclusivamente pela comparação das coordenadas no eixo *x*, garantindo uma definição direta e consistente dessas relações. Já a relação `closeTo` é definida com base na distância euclidiana entre objetos no plano bidimensional.

Um ponto importante deste bloco é o **ajuste do limiar de proximidade**, aumentado de 0.20 para 0.35 no espaço normalizado. Esse valor corresponde aproximadamente a 9 unidades no espaço original, permitindo uma definição mais realista de proximidade espacial e evitando um *ground truth* excessivamente esparso.

Além de calcular as matrizes de verdade, o bloco também apresenta estatísticas descritivas sobre a distribuição das relações, fornecendo uma visão quantitativa do equilíbrio entre pares positivos e negativos utilizados nos estágios posteriores de raciocínio e avaliação.

#### Bloco 3.4 — Definição dos Axiomas Espaciais Horizontais

Este bloco formaliza os **axiomas lógicos fuzzy** responsáveis por guiar o aprendizado das relações espaciais horizontais no framework LTN. O objetivo é combinar *supervisionamento explícito* com *propriedades estruturais da lógica*, produzindo uma medida única de satisfação global.

O supervisionamento é aplicado **exclusivamente aos predicados aprendidos** `leftOf` e `rightOf`, utilizando o *ground truth* para reforçar exemplos positivos e negativos. Cada tipo de supervisão contribui diretamente para a função de satisfação, incentivando altas ativações nos pares corretos e penalizando predições inconsistentes.

Além do supervisionamento, o bloco impõe **propriedades lógicas fundamentais**:
- **Irreflexividade**, garantindo que nenhum objeto esteja à esquerda ou à direita de si mesmo.
- **Inversão**, assegurando consistência entre `leftOf(x,y)` e `rightOf(y,x)`.
- **Transitividade**, reforçando a coerência relacional entre triplas de objetos.

A relação `closeTo` é explicitamente excluída do treinamento neste bloco, pois é definida como uma função fixa baseada em distância, e não como um predicado aprendido.

Por fim, todas as contribuições são agregadas e normalizadas em um único escalar de satisfação, que representa o grau global de aderência do modelo às regras espaciais horizontais definidas.

#### Bloco 3.5 — Treinamento dos Predicados Espaciais Horizontais

Este bloco define o treinamento conjunto dos predicados espaciais aprendidos `leftOf` e `rightOf`. Os parâmetros de ambos os modelos são otimizados simultaneamente utilizando o otimizador Adam.

A função de perda é formulada como o complemento da satisfação agregada dos axiomas espaciais horizontais, incentivando o modelo a maximizar a coerência lógica definida no Bloco 3.4.

Durante o treinamento, a satisfação lógica e a perda são registradas a cada época, permitindo acompanhar a convergência do raciocínio espacial.

#### Bloco 3.6 — Avaliação dos Predicados Espaciais

Este bloco realiza a avaliação quantitativa dos predicados espaciais `leftOf`, `rightOf` e `closeTo` sobre todos os pares de objetos distintos. As predições fuzzy são binarizadas com limiar 0.5 e comparadas ao *ground truth* correspondente.

São calculadas as métricas clássicas de classificação — acurácia, precisão, recall e F1-score — para cada relação espacial. No caso de `closeTo`, a avaliação utiliza diretamente a função baseada em kernel gaussiano, sem modelo treinável.

Além das métricas individuais, é reportada a satisfação agregada dos axiomas espaciais horizontais (*SatAgg*), permitindo relacionar desempenho estatístico e coerência lógica.

#### Bloco 3.7 — Queries e Relações Compostas

Este bloco implementa consultas lógicas e relações compostas sobre objetos:

1. **inBetween(x, y, z)**  
   Determina se o objeto `x` está entre `y` e `z`, horizontal ou verticalmente.  
   - Horizontal: `(leftOf(y,x) ∧ rightOf(z,x)) ∨ (leftOf(z,x) ∧ rightOf(y,x))`  
   - Vertical: `(below(y,x) ∧ above(z,x)) ∨ (below(z,x) ∧ above(y,x))`  
   A saída é um valor fuzzy (`0-1`) combinando ambos os casos via disjunção.

2. **query_leftmost_object(data_tensor)**  
   Identifica o objeto mais à esquerda segundo o predicado `leftOf`.  
   Utiliza um quantificador universal fuzzy (`pMeanError`) sobre todos os pares, retornando o objeto com maior score.

3. **query_rightmost_object(data_tensor)**  
   De forma análoga, retorna o objeto mais à direita considerando `rightOf` e quantificador universal fuzzy.

Estes métodos permitem extrair informações lógicas complexas e composições espaciais a partir dos predicados treinados.

#### Bloco 3.8 — Execução Completa do Raciocínio Espacial Horizontal

Este bloco integra todas as etapas da Tarefa 2:

1. **Ground Truth**  
   Calcula os pares de objetos para `leftOf`, `rightOf` e `closeTo` com threshold ajustado (0.35).

2. **Treinamento**  
   Chama `train_spatial_predicates` para otimizar os predicados `leftOf` e `rightOf` usando axiomas horizontais.

3. **Avaliação**  
   Executa `evaluate_spatial_predicates` para medir acurácia, precisão, recall, F1-score e satisfação dos axiomas (`SatAgg`).

4. **Consultas Lógicas**  
   - Identifica objeto mais à esquerda e mais à direita via `query_leftmost_object` e `query_rightmost_object`.  
   - Realiza um teste de relação composta `inBetween(C2, Q1, Q3)` para verificar posição relativa entre objetos.

Retorna o histórico de treinamento e os resultados de avaliação para análise posterior.

### 4 - Tarefa: Raciocínio Espacial Vertical com Abordagem Neuro-Simbólica

Esta tarefa apresenta um sistema de **Inteligência Artificial Neuro-Simbólica** que integra **redes neurais profundas** com **Lógica de Primeira Ordem Suave**, implementada por meio de **Logic Tensor Networks (LTN)**.  
O objetivo central é capacitar o modelo a **aprender, representar e raciocinar** sobre relações espaciais verticais e restrições físicas elementares, de forma consistente, interpretável e logicamente fundamentada.

Diferentemente de abordagens puramente conexionistas, o modelo não se limita à minimização de um erro estatístico. Em vez disso, o aprendizado é guiado pela **maximização da satisfação de axiomas lógicos**, que codificam propriedades geométricas e físicas invariantes do domínio.

As relações consideradas nesta tarefa incluem:
- **Relações espaciais verticais**: `below(x, y)` e `above(x, y)`
- **Relações físicas de empilhamento**: `canStack(x, y)`

---

#### Bloco 4.1 — Arquitetura Neural para Inferência Relacional Vertical  
#### `VerticalRelationPredictor`

Este bloco define o componente neural responsável por modelar relações binárias entre pares de objetos. O modelo adotado é um **Perceptron Multicamadas (MLP)**, projetado para operar como um **predicado lógico fuzzy**, produzindo graus contínuos de verdade.

#### Estrutura da Rede

A rede recebe como entrada um vetor de dimensão 22, resultante da concatenação dos atributos de dois objetos distintos. Cada objeto é descrito por um conjunto fixo de características geométricas, espaciais e semânticas.

A arquitetura é composta por:
- Camadas lineares com 64, 32 e 16 neurônios
- Função de ativação ReLU em todas as camadas ocultas
- Camadas de Dropout com taxas de 0.3 e 0.2, introduzidas para reduzir sobreajuste
- Camada de saída com ativação Sigmoid

A função Sigmoid é fundamental no contexto das LTN, pois garante que a saída do modelo pertença ao intervalo contínuo \([0, 1]\), interpretável como um **grau de verdade lógico**.

Formalmente, a saída do modelo representa:
\[
\mu_{below}(x, y) \in [0,1] \quad \text{ou} \quad \mu_{above}(x, y) \in [0,1]
\]

---

#### Bloco 4.2 — Construção do Ground Truth Vertical

Antes do processo de aprendizado, é necessário estabelecer uma referência objetiva de verdade baseada na geometria do ambiente. Para isso, define-se um **ground truth determinístico** a partir das coordenadas verticais dos objetos.

#### Definição Formal

Seja \(y_i\) a coordenada vertical do objeto \(i\):

- Relação *below*:
\[
below(i, j) \iff y_i < y_j
\]

- Relação *above*:
\[
above(i, j) \iff y_i > y_j
\]

A função `compute_vertical_ground_truth` avalia todos os pares ordenados de objetos e constrói duas matrizes booleanas \(N \times N\), que funcionam como **matrizes de adjacência direcionadas** para as relações verticais.

Essas matrizes são utilizadas tanto para **supervisão direta** quanto para validação empírica dos predicados aprendidos.

---

#### Bloco 4.3 — Definição do Predicado Físico `canStack(x, y)`

O predicado `canStack` modela conhecimento físico básico e não é aprendido por uma rede neural. Em vez disso, ele é definido explicitamente por meio de **regras lógicas fuzzy**, refletindo princípios elementares de estabilidade e equilíbrio.

#### Condições para Empilhamento

Um objeto \(x\) pode ser empilhado sobre um objeto \(y\) se, e somente se, as seguintes condições forem satisfeitas:

#### Base Estável
O objeto inferior \(y\) não pode possuir geometria pontiaguda, como cones ou triângulos, pois essas formas comprometem a estabilidade estrutural.

#### Condições de Equilíbrio
O empilhamento é considerado viável se existir:
- Compatibilidade dimensional entre \(x\) e \(y\), ou
- Alinhamento horizontal suficiente entre seus centroides

#### Operadores Lógicos Suaves

- Conjunção lógica (∧): multiplicação
- Disjunção lógica (∨): operador máximo
- Negação (¬): complemento

O resultado final é um valor contínuo que expressa o **grau de plausibilidade física** do empilhamento.

---

#### Bloco 4.4 — Formulação dos Axiomas Lógicos Verticais  
#### `vertical_axioms`

Este bloco constitui o núcleo simbólico do sistema, no qual são definidos axiomas universais que devem ser respeitados pelas predições neurais.

#### Axioma 1 – Relação Inversa
\[
\forall x, y \; (below(x, y) \leftrightarrow above(y, x))
\]

Esse axioma garante consistência semântica entre os dois predicados verticais.

---

#### Axioma 2 – Transitividade
\[
\forall i, j, k \; (below(i, j) \land below(j, k) \rightarrow below(i, k))
\]

Esse princípio assegura coerência global das relações espaciais inferidas.

---

#### Axioma 3 – Supervisão Positiva e Negativa

As predições dos modelos `below_model` e `above_model` são explicitamente penalizadas quando divergem do ground truth geométrico, reforçando o alinhamento entre aprendizado estatístico e realidade física.

---

#### Axioma 4 – Consistência do Predicado `canStack`

As inferências relacionadas ao empilhamento são avaliadas de acordo com as regras físicas definidas, garantindo que o sistema não produza conclusões fisicamente implausíveis.

---

#### Bloco 4.5 — Processo de Treinamento Baseado em Satisfatibilidade  
#### `train_vertical_predicates`

O treinamento do sistema não visa minimizar uma função de erro tradicional, mas sim **maximizar o grau médio de satisfação dos axiomas lógicos**.

Seja \(Sat \in [0,1]\) a satisfação agregada dos axiomas. A função de perda é definida como:
\[
\mathcal{L} = 1 - Sat
\]

A otimização é realizada com o algoritmo Adam, ajustando simultaneamente os parâmetros dos modelos `below` e `above`.

---

#### Bloco 4.6 — Avaliação Quantitativa e Verificação Semântica  
#### `evaluate_vertical_predicates`

A avaliação do sistema combina métricas estatísticas tradicionais com testes de coerência lógica.

#### Métricas Quantitativas
- Acurácia
- Precisão
- Recall
- F1-score

#### Verificações Semânticas
- Identificação correta dos objetos mais alto e mais baixo
- Validação da propriedade inversa entre `below` e `above`
- Análise qualitativa do predicado `canStack`

Essa abordagem garante que o sistema não apenas obtenha bom desempenho numérico, mas também mantenha **consistência conceitual**.

---

#### Bloco 4.7 — Coordenação do Processo Neuro-Simbólico de Aprendizado e Inferência  
#### `run_task3_complete`

Este bloco atua como o **mecanismo de coordenação global** do sistema. Ele encapsula e integra todas as etapas do processo neuro-simbólico, incluindo:

1. Construção do ground truth
2. Inicialização e treinamento dos predicados neurais
3. Avaliação quantitativa e lógica do sistema
4. Retorno estruturado dos resultados experimentais

Esse encapsulamento assegura reprodutibilidade, modularidade e clareza metodológica.
