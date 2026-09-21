# Ingestão e Busca Semântica com LangChain e PostgreSQL

Aplicação RAG em Python que extrai texto de um PDF, divide o conteúdo em chunks,
gera embeddings com a OpenAI e persiste os vetores em PostgreSQL com pgVector.
Depois da ingestão, um chat no terminal recupera os 10 trechos semanticamente mais
próximos e pede à LLM que responda apenas com base neles.

## Arquitetura

```text
document.pdf
    │ PyPDFLoader
    ▼
chunks de 1.000 caracteres (overlap 150)
    │ OpenAIEmbeddings
    ▼
PostgreSQL + pgVector
    ▲ similarity_search_with_score(k=10)
    │
pergunta no CLI ──► prompt com contexto ──► ChatOpenAI ──► resposta
```

## Pré-requisitos

- Python 3.11 ou superior
- Docker com Docker Compose
- chave da API da OpenAI com créditos disponíveis

Os modelos padrão são `text-embedding-3-small` para embeddings e
`gpt-4.1-mini` para respostas. Ambos podem ser alterados no `.env`.

## Instalação

Clone este repositório e, na raiz do projeto, crie o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows PowerShell, a ativação é feita com:

```powershell
venv\Scripts\Activate.ps1
```

Crie a configuração local:

```bash
cp .env.example .env
```

Edite `.env` e substitua `sk-sua-chave-aqui` por sua chave real. Nunca publique
o arquivo `.env`; ele já está ignorado pelo Git.

## Execução

1. Inicie o PostgreSQL e aguarde o healthcheck:

   ```bash
   docker compose up -d
   docker compose ps
   ```

2. Faça a ingestão do PDF:

   ```bash
   python src/ingest.py
   ```

3. Inicie o chat:

   ```bash
   python src/chat.py
   ```

Exemplo de uso:

```text
Faça sua pergunta sobre o PDF (digite 'sair' para encerrar).

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Use `sair`, `exit`, `quit` ou `:q` para encerrar.

## Configuração

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `OPENAI_API_KEY` | obrigatório | Chave da API da OpenAI |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Modelo de chat |
| `POSTGRES_CONNECTION` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | Conexão SQLAlchemy/psycopg |
| `PGVECTOR_COLLECTION` | `pdf_documents` | Nome lógico da collection |
| `PDF_PATH` | `document.pdf` | PDF relativo à raiz ou caminho absoluto |
| `RECREATE_COLLECTION` | `true` | Substitui a collection durante a ingestão |

Por padrão, cada ingestão recria somente a collection configurada. Isso evita
duplicação ao executar o comando novamente. Use `RECREATE_COLLECTION=false`
quando quiser acrescentar documentos à collection existente.

## Troca do modelo de embeddings

Modelos de embeddings podem produzir vetores com dimensões diferentes. Antes de
trocar `OPENAI_EMBEDDING_MODEL`, apague os dados anteriores e ingira novamente:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

O comando `down -v` remove o volume local do banco e todos os dados nele.

## Solução de problemas

- **`OPENAI_API_KEY nao foi definida`**: confira se `.env` existe na raiz e
  contém uma chave válida.
- **`connection refused`**: confirme com `docker compose ps` se `postgres` está
  saudável e se a porta 5432 está livre.
- **erro de dimensão do vetor**: o modelo de embeddings foi alterado após a
  primeira ingestão; recrie o volume conforme a seção anterior.
- **PDF sem texto**: PDFs formados apenas por imagens precisam passar por OCR
  antes da ingestão.

Para acompanhar o banco:

```bash
docker compose logs -f postgres
```

Para encerrar sem perder os vetores:

```bash
docker compose down
```

## Testes

Os testes não acessam a API da OpenAI e podem ser executados com a biblioteca
padrão do Python:

```bash
python -m unittest discover -s tests -v
```

## Estrutura

```text
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── tests/
├── document.pdf
└── README.md
```

## Segurança da resposta

O prompt instrui a LLM a usar somente os trechos recuperados e a responder com
uma mensagem fixa quando a informação não estiver explícita. Isso reduz
alucinações, mas não constitui uma garantia formal; aplicações críticas devem
adicionar avaliações, limiar de relevância e observabilidade.
