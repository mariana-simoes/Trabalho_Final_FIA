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

#### Bloco 3.1 – Definição de Predicados para Raciocínio Espacial

Este bloco é responsável por estabelecer a **base do raciocínio geométrico** do sistema. Nele são definidos os mecanismos que permitem ao modelo interpretar relações espaciais fundamentais, tais como *“estar à esquerda de”*, *“estar à direita de”* e *“estar perto de”*, a partir de vetores de características que descrevem os objetos.

Diferentemente de abordagens puramente simbólicas, essas relações são representadas como **predicados fuzzy**, capazes de assumir valores contínuos no intervalo \([0,1]\), compatíveis com o formalismo das **Logic Tensor Networks (LTN)**.


##### Arquitetura Neural: `SpatialRelationPredictor`

A classe `SpatialRelationPredictor` define uma **Rede Neural do tipo Multi-Layer Perceptron (MLP)**, que atua como o componente aprendível responsável por modelar relações espaciais entre pares de objetos.

**Características principais:**
- **Camadas:**  
  A arquitetura é composta por três camadas lineares intercaladas com:
  - Funções de ativação **ReLU**, que introduzem não-linearidade
  - Camadas de **Dropout**, utilizadas para reduzir *overfitting* e promover melhor generalização
- **Camada de saída:**  
  A rede finaliza com uma função **Sigmoid**, garantindo que a saída esteja no intervalo \([0,1]\).  
  Esse valor é interpretado como o **grau de verdade** (lógica fuzzy) da relação espacial avaliada.
- **Tratamento de dados no `forward`:**  
  O método `forward` inclui verificações e ajustes de dimensionalidade para assegurar que tanto entradas individuais quanto lotes (*batches*) sejam corretamente processados pelo PyTorch.

Esse modelo é utilizado como base para predicados aprendíveis, como *leftOf* e *rightOf*.

---

##### Wrapper de Predicado: `SpatialPredicate`

A classe `SpatialPredicate` atua como uma **ponte entre o domínio simbólico da lógica e o domínio subsimbólico das redes neurais**.

**Funções principais:**
- **Concatenação de pares:**  
  Para avaliar uma relação como *leftOf(A, B)*, o modelo precisa considerar simultaneamente os atributos de ambos os objetos.  
  Esta classe:
  - Extrai os valores de entrada
  - Ajusta as dimensões usando `unsqueeze` e `expand`, quando necessário
  - Concatena os vetores de características de \(A\) e \(B\) em um único vetor de entrada
- **Compatibilidade com LTN:**  
  Permite que o LTN avalie pares de objetos diretamente dentro de fórmulas lógicas, mantendo o fluxo diferenciável necessário para o treinamento.

---

##### Predicados Espaciais Aprendíveis: `leftOf` e `rightOf`

leftOf = ltn.Predicate(SpatialPredicate(leftOf_model))
rightOf = ltn.Predicate(SpatialPredicate(rightOf_model))

---

#### Lógica Geométrica Fixa: `CloseToFunc`

Diferentemente dos predicados espaciais aprendíveis, o predicado `CloseToFunc` **não depende de pesos neurais nem de aprendizado supervisionado**. Em vez disso, ele modela a relação *“estar perto de”* por meio de uma **formulação matemática explícita**, baseada em princípios geométricos contínuos.

#### Extração de Coordenadas
O predicado opera exclusivamente sobre os índices `0:2` do vetor de características, os quais correspondem às coordenadas cartesianas \((x, y)\) dos objetos no plano.

#### Distância Euclidiana
A proximidade entre dois objetos é calculada a partir do **quadrado da distância euclidiana** entre seus centroides, capturando de forma precisa a separação espacial entre eles.

#### Kernel Gaussiano
Sobre a distância calculada, é aplicada uma função exponencial do tipo kernel gaussiano:
\[
e^{-\text{scale} \cdot \text{distância}}
\]

Essa formulação possui as seguintes propriedades:
- Quando a distância é zero, o valor retornado é **1**, indicando que a relação *closeTo* é totalmente verdadeira
- À medida que a distância aumenta, o valor decresce suavemente em direção a **0**, refletindo uma diminuição progressiva do grau de proximidade

#### Escala Aprendível
O parâmetro `self.scale` é **aprendível**, permitindo que o modelo ajuste automaticamente o quão restritiva é a noção de *“perto”*.  
Valores maiores de `scale` tornam o predicado mais rigoroso, enquanto valores menores resultam em uma interpretação mais permissiva da proximidade espacial.

**Em síntese**, `CloseToFunc` fornece um predicado geométrico contínuo, diferenciável e semanticamente bem definido, que complementa os predicados aprendíveis no raciocínio espacial baseado em Logic Tensor Networks.


### Bloco 3.2 – Geração de *Ground Truth* Espacial

Enquanto o bloco anterior definiu os modelos responsáveis por **aprender** as relações espaciais, este bloco estabelece as **regras matemáticas absolutas** que determinam o que é considerado *verdadeiro* no espaço físico, com base direta nas coordenadas \((x, y)\) dos objetos.

---

#### Gestão de Memória e Dispositivos (CPU/GPU)

if data_tensor.is_cuda:
    data_cpu = data_tensor.cpu()

#### Por que isso é necessário?
O **PyTorch** frequentemente realiza operações na GPU com o objetivo de acelerar o processamento numérico. Entretanto, a biblioteca **NumPy**, utilizada para a construção das matrizes de *ground truth*, opera exclusivamente na CPU e não é compatível com tensores alocados em dispositivos CUDA.

#### Ação
O código verifica se o tensor de entrada está armazenado na GPU e, em caso afirmativo, realiza uma cópia explícita para a memória principal (CPU) antes da conversão para o formato NumPy.  

Esse procedimento previne erros de execução do tipo `RuntimeError` e assegura a correta interoperabilidade entre as bibliotecas PyTorch e NumPy.

#### Extração de Coordenadas Espaciais

O código assume que os dados de entrada (`data_tensor`) contêm informações espaciais organizadas nas primeiras colunas do vetor de características:

- **`x_coords` (Índice 0)**: representa a posição horizontal do objeto  
- **`y_coords` (Índice 1)**: representa a posição vertical do objeto  

Essa suposição permite que as relações espaciais sejam inferidas diretamente a partir de comparações geométricas simples, sem a necessidade de processamento adicional.

---

#### Matrizes de Relação \((N \times N)\)

São inicializadas matrizes booleanas de dimensão \(N \times N\), onde \(N\) corresponde ao número total de objetos presentes no cenário.

- Cada célula \([i, j]\) nessas matrizes representa a relação espacial entre o objeto \(i\) e o objeto \(j\).
- Por exemplo:
  - `gt_leftOf[2, 5] = True` indica que o objeto 2 está fisicamente à esquerda do objeto 5.

Essa estrutura matricial é particularmente adequada para a representação de relações binárias e para a posterior integração com mecanismos de raciocínio lógico.

---

#### Lógica de Comparação por Pares

O cálculo do *ground truth* é realizado por meio de dois laços aninhados (`for i in range(N)` e `for j in range(N)`), que percorrem todas as combinações possíveis de pares de objetos.

#### Identidade (\(i = j\))
- Comparações de um objeto consigo mesmo são ignoradas (`continue`), pois relações espaciais reflexivas não são semanticamente válidas.

#### Relações Esquerda/Direita (*leftOf* / *rightOf*)
- A definição dessas relações baseia-se em uma comparação escalar direta:
\[
x_i < x_j \Rightarrow i \text{ está à esquerda de } j
\]
- A relação *rightOf* é inferida de forma complementar.

#### Relação de Proximidade (*closeTo*)
- A proximidade espacial é avaliada por meio do cálculo da **distância euclidiana** entre dois objetos, conforme a expressão:
\[
\text{dist}(i, j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
\]
- O valor da distância calculada é comparado com o parâmetro `threshold_close`, definido como um limiar fixo de **0.35**.
- Caso a distância entre os objetos seja inferior a esse limiar, a relação *closeTo* é considerada **verdadeira**; caso contrário, é marcada como **falsa**.

---

#### Retorno dos Dados

A função retorna um **dicionário contendo três matrizes booleanas**, correspondentes às relações espaciais:

- *leftOf*
- *rightOf*
- *closeTo*

Esse dicionário é utilizado posteriormente no cálculo da **função de perda (*loss*)** do modelo.  
Durante o processo de treinamento, o sistema busca minimizar a discrepância entre:

- As **predições** produzidas pelos predicados aprendíveis definidos no **Bloco 3.1**
- As **verdades geométricas absolutas** calculadas neste bloco de *ground truth* (**Bloco 3.2**)

Esse mecanismo garante que o aprendizado neural seja guiado por relações espaciais consistentes e semanticamente bem definidas.

### Bloco 3.3 – Axiomas para Raciocínio Espacial Horizontal

Este bloco define um **conjunto de axiomas lógicos** que impõem restrições estruturais ao modelo, obrigando-o a internalizar propriedades fundamentais do espaço horizontal. Diferentemente do *ground truth* geométrico, esses axiomas expressam **leis universais** que devem ser respeitadas por qualquer configuração de objetos, independentemente dos exemplos específicos observados.

---

#### Supervisionamento (*Conhecimento Base*)

O processo se inicia com a incorporação explícita do *ground truth* espacial calculado no **Bloco 3.2**, fornecendo ao modelo uma base supervisionada para o aprendizado inicial das relações horizontais.

- **Supervisionamento Positivo**  
  Quando a geometria indica que um objeto \(x\) está à esquerda de um objeto \(y\), o axioma força o predicado:
  \[
  \text{leftOf}(x, y)
  \]
  a assumir um alto valor de verdade.

- **Supervisionamento Negativo**  
  Quando a geometria indica que \(x\) **não** está à esquerda de \(y\), o axioma impõe:
  \[
  \neg \text{leftOf}(x, y)
  \]

Esse mecanismo é aplicado de forma simétrica tanto para o predicado *leftOf* quanto para *rightOf*, garantindo coerência básica entre as relações aprendidas e a realidade geométrica.

---

#### Propriedades Estruturais da Lógica Espacial

Nesta etapa, o modelo é forçado a aprender **propriedades lógicas universais** que caracterizam relações espaciais horizontais, válidas para qualquer par de objetos no domínio.

#### Irreflexividade
Um objeto não pode manter uma relação espacial consigo mesmo:
\[
\neg \text{leftOf}(x, x)
\]

#### Assimetria
Se um objeto \(x\) está à esquerda de \(y\), então \(y\) não pode estar à esquerda de \(x\):
\[
\text{leftOf}(x, y) \rightarrow \neg \text{leftOf}(y, x)
\]

#### Relação Inversa
As relações *leftOf* e *rightOf* são logicamente inversas:
\[
\text{leftOf}(x, y) \leftrightarrow \text{rightOf}(y, x)
\]

#### Transitividade
Esta é a propriedade mais expressiva do conjunto de axiomas. Ela garante consistência global do espaço horizontal:
\[
(\text{leftOf}(x, y) \land \text{leftOf}(y, z)) \rightarrow \text{leftOf}(x, z)
\]

Essa regra impede contradições topológicas e força o modelo a aprender uma noção ordenada do espaço.

---

#### Eficiência Computacional e Amostragem (*Sampling*)

A avaliação exaustiva de todas as combinações possíveis de objetos teria custo computacional elevado (\(N^2\) para pares e \(N^3\) para trios, no caso da transitividade). Para mitigar esse problema, o código adota:

- **Amostragem Aleatória**  
  Apenas subconjuntos de pares e trios de objetos são selecionados aleatoriamente para a aplicação dos axiomas, reduzindo significativamente o custo computacional sem comprometer a qualidade do aprendizado.

- **Variáveis LTN**  
  O uso de `ltn.Variable` permite mapear subconjuntos do tensor de dados diretamente para o domínio lógico, viabilizando a avaliação eficiente das fórmulas.

---

#### Processamento Crítico e Agregação de Satisfação (`sat_agg`)

A etapa final trata da consolidação técnica e semântica das avaliações lógicas:

- **Uniformização de Saídas**  
  Como diferentes axiomas podem produzir tensores com formas distintas, o código garante que todos os valores de verdade sejam reduzidos a **escalares** no intervalo \([0, 1]\).

- **Média de Satisfação**  
  Para cada axioma, é utilizada a operação `val.mean()` a fim de obter um valor médio de satisfação.

- **Agregação Global (`sat_agg`)**  
  Todos os valores de satisfação individuais são combinados em um único escalar que representa o **Nível Global de Satisfação dos Axiomas**.

**Objetivo do Treinamento:**  
Maximizar esse valor global de satisfação, idealmente aproximando-o de **1.0**, garantindo que o modelo neural respeite simultaneamente os dados empíricos (*ground truth*) e as leis lógicas universais do espaço horizontal.


### Bloco 3.4 – Ciclo de Treinamento Baseado em Lógica

Nesta etapa, o **Logic Tensor Network (LTN)** utiliza um procedimento de otimização contínua para **reduzir a insatisfação lógica** dos axiomas definidos anteriormente. Em vez de minimizar apenas um erro estatístico clássico, o treinamento busca transformar **violações lógicas** em **conhecimento internalizado** pelos modelos neurais, alinhando aprendizado numérico e raciocínio simbólico.

---

#### Preparação do Otimizador e dos Parâmetros

parameters = list(leftOf_model.parameters()) + list(rightOf_model.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)

#### Fusão de Parâmetros e Estratégia de Otimização

- **Fusão de Parâmetros**  
  O código agrupa os pesos internos dos modelos **leftOf** e **rightOf** em um único conjunto de parâmetros treináveis.  
  É importante notar que o predicado **closeTo** não é incluído nesse processo, pois ele foi definido como uma **função geométrica fixa** (ou possui apenas um parâmetro de escala independente), não sendo ajustado diretamente neste ciclo de treinamento específico.

- **Algoritmo Adam**  
  O uso do otimizador **Adam** é particularmente adequado nesse contexto, pois:
  - Lida bem com gradientes ruidosos e não estacionários;
  - Ajusta dinamicamente a taxa de aprendizado para cada parâmetro;
  - É robusto frente à natureza contínua e fuzzy da **satisfação lógica**, que pode variar de forma não linear ao longo do treinamento.

---

#### Função de Perda (*Loss*) Diferenciada

Ao contrário do *Deep Learning* tradicional, que utiliza funções como **Cross-Entropy** ou **MSE**, no contexto de **Logic Tensor Networks** a noção de erro é substituída pela ideia de **insatisfação lógica**.

- **Cálculo da Satisfação (`sat`)**  
  A função definida no **Bloco 3.3** é chamada para avaliar o quanto os axiomas lógicos estão sendo respeitados.  
  O valor retornado pertence ao intervalo:

  \[
  \text{sat} \in [0, 1]
  \]

  onde:
  - `0` representa contradição total;
  - `1` representa verdade lógica plena.

- **Minimização da Insatisfação**  
  A função de perda é definida como:

  \[
  \text{Loss} = 1.0 - \text{Satisfação}
  \]

  Interpretação:
  - Se `sat = 1.0`, então `loss = 0.0` → o modelo já satisfaz completamente os axiomas.
  - Se `sat = 0.2`, então `loss = 0.8` → há forte violação lógica, exigindo ajustes significativos nos pesos.

**Essência conceitual:**  
O treinamento busca **maximizar a coerência lógica**, e não apenas minimizar um erro estatístico.

---

#### Mecanismos de Robustez Numérica e Computacional

Para garantir estabilidade e evitar falhas silenciosas durante o treinamento, o código inclui diversas salvaguardas:

- **Clamping da Satisfação** (torch.clamp(sat, 0.0, 1.0): Este mecanismo assegura que, mesmo na presença de **erros numéricos de ponto flutuante** inerentes à computação em precisão finita, o valor de satisfação permaneça estritamente dentro do **intervalo lógico permitido \([0, 1]\)**. Isso é fundamental para preservar a interpretação semântica dos valores de verdade no contexto da **lógica fuzzy**, evitando estados inválidos que poderiam comprometer tanto o treinamento quanto a avaliação do modelo.
- **Verificação de Gradiente (`requires_grad`)**: Este é um ponto crítico do processo de treinamento. Caso a função de perda não esteja computacionalmente conectada aos pesos do modelo — por exemplo, se os dados não forem efetivamente propagados através da rede neural — o PyTorch não conseguirá calcular os gradientes necessários para a otimização. O código inclui verificações explícitas para detectar essa situação e emitir alertas, prevenindo o fenômeno conhecido como **“treinamento fantasma”**, no qual o processo de aprendizagem aparenta ocorrer, mas nenhum parâmetro é, de fato, atualizado.

#### Fluxo de Execução por Época

- **Zeragem dos Gradientes** (optimizer.zero_grad()): Este comando limpa os gradientes acumulados do cálculo anterior, garantindo que a atualização dos parâmetros na época atual seja baseada exclusivamente nas informações recém-computadas. Essa etapa é essencial para evitar que contribuições residuais de iterações passadas distorçam o processo de otimização.
- **Avaliação Lógica**: Nesta etapa, o modelo avalia os **axiomas lógicos** definidos no sistema, verificando o grau de verdade das relações espaciais em relação aos dados atuais. Em termos conceituais, o sistema “lê” o conjunto de regras lógicas e mede o quanto suas inferências satisfazem essas leis, produzindo um valor contínuo de satisfação que orienta o processo de aprendizado.
- **Retropropagação dos Gradientes** (loss.backward()): Nesta etapa, o mecanismo de diferenciação automática do PyTorch calcula como cada peso da rede neural contribuiu para a insatisfação lógica (isto é, para a violação dos axiomas). A partir desse cálculo, são determinados os gradientes que indicam como e em que direção os parâmetros devem ser ajustados, de modo a reduzir a insatisfação e aproximar o modelo de um estado de maior coerência lógica.
- **Atualização dos Parâmetros** (optimizer.step()): Este comando aplica as atualizações calculadas durante a retropropagação, ajustando os pesos da rede neural de acordo com o algoritmo de otimização escolhido. Como resultado, os parâmetros passam a refletir melhor as restrições impostas pelos axiomas lógicos, aumentando o grau de satisfação das relações espaciais na próxima iteração de treinamento.

#### Monitoramento e Histórico de Aprendizado

O dicionário **`history`** é utilizado para registrar a evolução do modelo ao longo do processo de treinamento, armazenando métricas relevantes a cada época, como a **perda (loss)** e o **grau de satisfação lógica**.

Em sistemas de **IA simbólica e neuro-simbólica**, o gráfico de **Satisfação dos Axiomas** constitui o principal indicador de desempenho. Diferentemente de métricas puramente estatísticas, ele evidencia se o sistema está, de fato, **internalizando e respeitando as regras espaciais** impostas, revelando a progressão do modelo de um estado de inconsistência lógica para um raciocínio progressivamente mais coerente e semanticamente fundamentado.

### Bloco 3.5: Avaliação dos Predicados Espaciais

Esta função corresponde ao **estágio final de avaliação** do modelo neuro-simbólico. Seu objetivo é traduzir o comportamento da **lógica nebulosa (fuzzy logic)** aprendida pelo LTN em **métricas estatísticas clássicas**, amplamente utilizadas no aprendizado de máquina tradicional.

---

#### Desativação do Gradiente e Avaliação da Coesão Lógica

with torch.no_grad(): sat_axioms = spatial_axioms_horizontal(data_tensor, ground_truth)
    
- **`torch.no_grad()`**: Indica ao PyTorch que o modelo está em modo de inferência, e não mais em treinamento. Nesse contexto, o framework deixa de rastrear operações para o cálculo de derivadas, o que reduz significativamente o consumo de memória e o custo computacional, tornando a avaliação mais eficiente.

- **Consistência Lógica (SatAgg)**: O primeiro passo da avaliação consiste na análise do valor de **Satisfação Agregada (SatAgg)**. Esse indicador quantifica o grau em que o modelo treinado respeita, sobre os dados de teste, as leis formais da geometria espacial — como transitividade, assimetria e irreflexividade — indo além da simples correção estatística das predições.

#### O Papel Crítico das `ltn.Variable` na Inferência

Um erro recorrente ao utilizar *Logic Tensor Networks* (LTN) consiste em tentar fornecer tensores brutos do PyTorch diretamente aos predicados durante a fase de avaliação.

**Por que utilizar `ltn.Variable`?**  
Os predicados em LTN não operam apenas sobre valores numéricos, mas sobre elementos de um **domínio lógico**. Mesmo durante a inferência, os dados precisam ser encapsulados em objetos do tipo `ltn.Variable`, de modo que o sistema possa:

- Associar corretamente cada tensor ao seu papel semântico no domínio lógico;
- Manter a consistência dimensional esperada pelos predicados;
- Permitir que as fórmulas lógicas sejam avaliadas de forma coerente e bem definida.

No código, esse encapsulamento é realizado **objeto por objeto**, conforme ilustrado abaixo: 
x_var = ltn.Variable(..., data_device[i:i+1]).

#### Extração e Normalização de Valores

Como os predicados em LTN produzem **valores de verdade fuzzy** no intervalo contínuo \([0, 1]\), o código precisa tratar cuidadosamente os diferentes formatos de saída gerados durante a inferência.

- **Conversão de Tensor para Escalar (`float`)**  
  O valor de verdade retornado pelo predicado é inicialmente um tensor do PyTorch. O código extrai o valor numérico correspondente utilizando o método `.item()`.  
  Caso o tensor possua dimensões adicionais (por exemplo, devido a operações em lote), essas dimensões são previamente reduzidas por meio da média, garantindo a obtenção de um único valor escalar representativo.

- **Limiar de Decisão (Thresholding)**  
  Para converter o valor fuzzy em uma decisão binária, é aplicado um limiar fixo:

  \[
  \text{Predição} =
  \begin{cases}
  1, & \text{se } \text{valor} > 0.5 \\
  0, & \text{caso contrário}
  \end{cases}
  \]

  Esse procedimento traduz o grau de certeza do modelo em uma classificação discreta (*Sim/Não*), permitindo a comparação direta com o *ground truth* e o cálculo de métricas de desempenho tradicionais.

#### Métricas de Desempenho

Para avaliar quantitativamente o comportamento dos predicados espaciais aprendidos (`leftOf`, `rightOf` e `closeTo`), o código recorre à biblioteca **scikit-learn (sklearn)**, amplamente utilizada em aprendizado de máquina. São calculadas quatro métricas fundamentais:

- **Acurácia (Accuracy)**  
  Mede a proporção total de pares de objetos corretamente classificados pelo modelo, considerando tanto os casos positivos quanto os negativos.

- **Precisão (Precision)**  
  Indica a confiabilidade das predições positivas. Em outras palavras, quando o modelo afirma que *“A está à esquerda de B”*, essa métrica expressa a probabilidade dessa afirmação estar correta.

- **Recall (Revocação)**  
  Avalia a capacidade do modelo em identificar corretamente todas as relações que são verdadeiras no *ground truth*. Por exemplo, de todos os pares que realmente satisfazem a relação espacial, quantos foram corretamente detectados.

- **F1-Score**  
  Representa a média harmônica entre precisão e recall, fornecendo uma medida balanceada do desempenho global. Essa métrica é especialmente útil em cenários com desbalanceamento entre classes ou quando se deseja um compromisso entre exatidão e cobertura.

Conjuntamente, essas métricas permitem avaliar não apenas a correção estatística das predições, mas também a robustez do aprendizado dos predicados espaciais.

#### Interpretação do SatAgg Final

O **SatAgg (Satisfação Agregada)** é apresentado ao final do processo de avaliação como um indicador global do grau de consistência lógica do modelo.

- Um **SatAgg elevado** indica que o conjunto de axiomas lógicos definidos (por exemplo, transitividade, assimetria e irreflexividade) está sendo amplamente satisfeito pelos predicados aprendidos.
- Por outro lado, se a **acurácia estatística for alta**, mas o **SatAgg for baixo**, isso sugere que o modelo conseguiu aprender padrões de rotulagem locais, mas **falhou em internalizar a estrutura lógica subjacente**.  
  Um exemplo típico dessa situação ocorre quando o modelo afirma que \(A < B\) e \(B < C\), mas nega que \(A < C\), violando a transitividade.

O objetivo central de uma **Logic Tensor Network** é alcançar simultaneamente:
- **Alto desempenho estatístico** (acurácia, precisão, recall, F1-score); e
- **Alta satisfação lógica** (SatAgg próximo de 1).

A coexistência desses dois fatores caracteriza um modelo que não apenas prediz corretamente, mas também **razona de forma logicamente consistente**, alinhando aprendizado numérico e conhecimento simbólico.

### Bloco 3.6: Predicados Derivados e Consultas Lógicas

Este bloco evidencia a **flexibilidade e expressividade das Logic Tensor Networks (LTN)** ao demonstrar como **novos conceitos semânticos podem ser definidos exclusivamente por meio de fórmulas lógicas**, sem a necessidade de treinar modelos neurais adicionais. Trata-se de um passo fundamental do paradigma neuro-simbólico: reutilizar conhecimento aprendido para construir raciocínios mais complexos.

---

#### Predicado Composto: `inBetween(x, y, z)`

O predicado `inBetween` formaliza o conceito geométrico de **“estar entre” no eixo horizontal**, utilizando apenas relações previamente aprendidas (`leftOf` e `rightOf`).

#### Intuição Lógica

Para que um objeto \( x \) esteja entre \( y \) e \( z \), uma das duas condições deve ser satisfeita:

1. \( y \) está à esquerda de \( x \) **e** \( z \) está à direita de \( x \);  
2. \( z \) está à esquerda de \( x \) **e** \( y \) está à direita de \( x \).

Essa definição garante **simetria** em relação a \( y \) e \( z \), refletindo corretamente o significado geométrico do conceito.

#### Formulação Formal

\[
\text{inBetween}(x, y, z) \iff 
(\text{leftOf}(y, x) \land \text{rightOf}(z, x)) 
\;\lor\; 
(\text{leftOf}(z, x) \land \text{rightOf}(y, x))
\]

#### Implementação no LTN

- O código utiliza os operadores `And` e `Or` do LTN, que implementam **lógica fuzzy**, permitindo graus intermediários de verdade no intervalo \([0, 1]\).
- O predicado resultante não retorna um valor booleano rígido, mas sim um **grau de confiança** de que \( x \) está entre \( y \) e \( z \).
- Isso permite que o sistema lide naturalmente com **incerteza, ruído e ambiguidades espaciais**.

---

#### Consulta Lógica: `query_leftmost_object` (Objeto Mais à Esquerda)

Esta função implementa uma **consulta existencial-universal**, buscando o objeto que melhor satisfaz a propriedade de ser o mais à esquerda no conjunto.

#### Definição Lógica

O objeto mais à esquerda é aquele que está à esquerda de **todos os outros objetos**:

\[
\text{isLeftmost}(x) = \forall y \; (y \neq x \rightarrow \text{leftOf}(x, y))
\]

#### Estratégia de Avaliação

- Para cada objeto \( x_i \), o código avalia a relação `leftOf(x_i, x_j)` para todos os outros objetos \( x_j \).
- Cada avaliação retorna um **valor de verdade fuzzy**.
- A quantificação universal (\( \forall \)) é aproximada computacionalmente por meio da **média dos valores de verdade**, resultando em um `universal_score`.

#### Resultado

- O objeto com o **maior valor médio de satisfação** é considerado o mais à esquerda.
- Essa abordagem é robusta mesmo quando as relações não são perfeitamente verdadeiras (ex.: pequenas imprecisões geométricas).

---

#### Consulta Lógica: `query_rightmost_object` (Objeto Mais à Direita)

Esta consulta é **estruturalmente simétrica** à anterior, substituindo apenas o predicado `leftOf` por `rightOf`.

#### Definição Lógica

\[
\text{isRightmost}(x) = \forall y \; (y \neq x \rightarrow \text{rightOf}(x, y))
\]

#### Interpretação

- O objeto mais à direita é aquele que maximiza a satisfação média da relação `rightOf(x, y)` em relação a todos os outros objetos.
- A simetria entre as duas consultas reforça a **consistência lógica do modelo**, especialmente quando axiomas de inversão (`leftOf(x,y) \leftrightarrow rightOf(y,x)`) foram corretamente aprendidos.

---

#### Detalhes Técnicos de Implementação

Alguns aspectos técnicos são cruciais para o funcionamento correto dessas consultas:

- **Tratamento de Tensores e Device**  
  Os dados são explicitamente movidos para o dispositivo correto (CPU ou GPU), garantindo compatibilidade com os modelos neurais subjacentes.

- **Ajuste de Dimensionalidade (Batch)**  
  O uso de `unsqueeze` assegura que cada objeto seja tratado como um batch unitário, conforme esperado pelo PyTorch e pelo LTN.

- **Uso de `ltn.Variable`**  
  As variáveis são criadas dinamicamente dentro dos loops, pois o LTN exige que os dados estejam encapsulados em objetos simbólicos para realizar inferência lógica corretamente.

- **Extração de Valores Numéricos**  
  O uso de `.value` e `.item()` converte o resultado lógico (tensor) em um escalar Python, permitindo:
  - Comparações numéricas;
  - Ordenação dos objetos via `scores.sort()`.

---

#### Importância Conceitual do Bloco

Este bloco marca a transição do sistema de:
- **Predição de relações locais**,  
para:
- **Raciocínio simbólico estruturado de alto nível**.

Ele demonstra que o modelo:
- Reutiliza conhecimento aprendido;
- Combina predicados básicos em conceitos mais complexos;
- Realiza consultas globais sobre o conjunto de objetos.

Em termos neuro-simbólicos, este é o ponto em que o sistema passa a operar como um **mecanismo de inferência lógica espacial**, indo além da simples classificação e aproximando-se de um **agente cognitivo explicável e consistente**.

### Bloco 3.7: Orquestração e Execução do Pipeline Espacial

Este bloco descreve a função **`run_task2_complete`**, que atua como o **ponto central de orquestração** de todo o pipeline de raciocínio espacial horizontal. Nela, os componentes **neurais** (redes treináveis) e **simbólicos** (axiomas e consultas lógicas) são integrados em um fluxo único e coerente, desde a construção do conhecimento de base até inferências de ordem superior.

---

#### Inicialização do Ground Truth Espacial

ground_truth = compute_spatial_ground_truth(data_tensor, threshold_close=0.35)

A primeira etapa não envolve aprendizado por IA, mas sim **geometria determinística**. O sistema analisa diretamente as coordenadas reais dos objetos e, a partir delas, constrói as **matrizes de referência (ground truth)** que representam o gabarito geométrico do cenário. Esse gabarito é fundamental, pois fornece um **sinal supervisório explícito** que orienta o ajuste dos pesos dos modelos neurais na etapa subsequente, funcionando como um critério objetivo de verdade espacial.

#### Treinamento e Evolução dos Predicados Espaciais

O objeto `history` registra a **trajetória de aprendizado** do sistema ao longo das 500 épocas de treinamento. Nesse processo, o modelo **SpatialRelationPredictor** evolui de uma inicialização aleatória para uma representação estruturada do espaço, passando a capturar que **diferenças relativas nas coordenadas \(X\)** dos objetos são determinantes para as relações espaciais `leftOf` e `rightOf`. Essa evolução é guiada não apenas por exemplos, mas também pelas **restrições lógicas impostas pelos axiomas**, o que favorece um aprendizado semanticamente consistente.

---

#### Avaliação Estatística do Modelo

A chamada à função `evaluate_spatial_predicates` converte o comportamento lógico-nebuloso do LTN em **métricas quantitativas amplamente utilizadas em aprendizado de máquina**, tais como **Acurácia, Precisão, Recall e F1-Score**. Nesta etapa, o sistema gera relatórios individuais para cada predicado espacial, permitindo verificar se o modelo **generalizou adequadamente os conceitos aprendidos** e se mantém coerência entre desempenho estatístico e consistência lógica.

#### Aplicação Prática: Consultas de Extremos Espaciais

Nesta etapa, evidencia-se de forma concreta o **potencial do raciocínio neuro-simbólico**, no qual predicados aprendidos e lógica formal são combinados para produzir inferências semanticamente ricas.

- **Busca de Âncoras Espaciais**  
  O sistema emprega as funções `query_leftmost_object` e `query_rightmost_object` para identificar, respectivamente, os objetos que ocupam as posições mais à esquerda e mais à direita do cenário. Essas consultas exploram quantificação universal e agregação fuzzy para determinar os extremos espaciais de maneira robusta.

- **Saída Interpretável ao Usuário**  
  Em vez de retornar apenas um índice numérico, o código utiliza a expressão `labels[idx].split(':')[0]` para mapear o resultado a um rótulo semântico compreensível (por exemplo, *Cubo* ou *Esfera*). Essa escolha reforça a **interpretabilidade do sistema**, tornando suas conclusões acessíveis a usuários humanos e adequadas a aplicações práticas como robótica e sistemas de apoio à decisão.

#### Teste de Raciocínio de Ordem Superior: Predicado `inBetween`

O estágio final do pipeline consiste na verificação de um **predicado de ordem superior**, especificamente um predicado de **aridade 3**, que envolve simultaneamente três objetos do cenário espacial.

pred = inBetween(data_tensor[19], data_tensor[0], data_tensor[2])

**O que ocorre nesta etapa:**  
O sistema avalia se o objeto identificado pelo índice `19` encontra-se **posicionado horizontalmente entre** os objetos de índices `0` e `2`, utilizando exclusivamente inferência lógica.

**Diferencial do método:**  
Observe que o modelo **não foi treinado explicitamente** com rótulos referentes ao conceito de “estar entre”. Em vez disso, essa relação é **deduzida por composição lógica**, a partir do conhecimento previamente aprendido sobre os predicados básicos `leftOf` e `rightOf`.  

Caso o valor retornado seja próximo de **1.0**, isso indica que a inferência foi bem-sucedida, evidenciando que o sistema neuro-simbólico conseguiu **generalizar e raciocinar corretamente**, indo além do aprendizado supervisionado direto.

