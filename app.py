from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.azure_client import AzureLLM
from src.config import Settings
from src.rag import LocalRAG

st.set_page_config(page_title="Copiloto Assistente", layout="wide")
st.title("Copiloto Assistente")
st.caption("Simulação viva de atendimento com RAG, sugestões em tempo real e rastreabilidade completa")

settings = Settings()


@st.cache_resource(show_spinner="Carregando embeddings e indexando documentos...")
def get_rag(model: str, chunk_size: int, overlap: int) -> LocalRAG:
    return LocalRAG(
        embedding_model=model,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )


rag = get_rag(settings.embedding_model, settings.chunk_size, settings.chunk_overlap)


def build_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "Nenhuma fonte recuperada."
    return "\n\n".join(
        f"FONTE: {item['source']}\n"
        f"CHUNK: {item['chunk_id']}\n"
        f"SCORE: {item['score']:.3f}\n"
        f"CONTEÚDO:\n{item['text']}"
        for item in results
    )


def conversation_text() -> str:
    return "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in st.session_state.conversation
    )


def retrieval_query() -> str:
    recent = st.session_state.conversation[-6:]
    return "\n".join(item["content"] for item in recent)


def extract_suggested_answer(markdown: str) -> str:
    marker = "### Resposta sugerida"
    if marker not in markdown:
        return markdown.strip()
    content = markdown.split(marker, 1)[1]
    for next_marker in ["### Próxima ação", "### Atenção", "### Fontes"]:
        if next_marker in content:
            content = content.split(next_marker, 1)[0]
    return content.strip()


def run_rag() -> None:
    query = retrieval_query()
    results = rag.search(query, settings.top_k)
    context = build_context(results)
    history = conversation_text()

    st.session_state.last_query = query
    st.session_state.sources = results
    st.session_state.context_sent = context

    llm = AzureLLM(settings)
    system = """Você é um copiloto para um operador humano.
Use somente os trechos recuperados para afirmar preços, políticas, prazos, requisitos e procedimentos.
Considere o histórico completo da conversa.
Produza uma resposta objetiva, segura e pronta para uso.
Quando não houver evidência suficiente, informe isso claramente.
Responda em português do Brasil com exatamente estas seções:

### Resposta sugerida
### Próxima ação
### Atenção
### Fontes
"""
    user = f"HISTÓRICO COMPLETO:\n{history}\n\nCONTEXTO RAG:\n{context}"
    suggestion = llm.complete(system, user)

    st.session_state.suggestion = suggestion
    st.session_state.suggested_answer = extract_suggested_answer(suggestion)
    st.session_state.events.append(
        {
            "event": "rag_execution",
            "query": query,
            "retrieved_chunks": len(results),
            "sources": [item["source"] for item in results],
            "scores": [round(item["score"], 4) for item in results],
        }
    )


def add_customer_message(message: str) -> None:
    clean = message.strip()
    if not clean:
        st.warning("Digite a mensagem do cliente.")
        return
    st.session_state.conversation.append({"role": "cliente", "content": clean})
    st.session_state.events.append({"event": "customer_message", "content": clean})
    run_rag()


def add_operator_message(message: str, origin: str) -> None:
    clean = message.strip()
    if not clean:
        st.warning("Digite ou selecione uma resposta do atendente.")
        return
    st.session_state.conversation.append({"role": "atendente", "content": clean})
    st.session_state.events.append(
        {"event": "operator_message", "origin": origin, "content": clean}
    )


def generate_full_answer() -> None:
    results = rag.search(conversation_text(), settings.top_k)
    context = build_context(results)
    llm = AzureLLM(settings)
    system = """Você é um assistente de atendimento.
Receba o histórico completo da conversa e o contexto recuperado pelo RAG.
Gere a melhor resposta final possível para o atendente enviar agora.
Use somente as fontes fornecidas para fatos, preços, condições, prazos e procedimentos.
Responda somente com a mensagem final ao cliente, sem explicações adicionais.
"""
    user = f"CONVERSA COMPLETA:\n{conversation_text()}\n\nCONTEXTO RAG:\n{context}"
    st.session_state.full_answer = llm.complete(system, user).strip()
    st.session_state.events.append(
        {
            "event": "full_context_generation",
            "retrieved_chunks": len(results),
            "sources": [item["source"] for item in results],
        }
    )


for key, default in {
    "conversation": [],
    "suggestion": "",
    "suggested_answer": "",
    "full_answer": "",
    "sources": [],
    "last_query": "",
    "context_sent": "",
    "events": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

sim_tab, docs_tab, detail_tab, architecture_tab = st.tabs(
    ["Simulação viva", "Base documental", "Detalhes da execução", "Arquitetura"]
)

with sim_tab:
    st.subheader("Simulação de atendimento em tempo real")
    st.write(
        "Você controla os dois lados da conversa. Ao enviar uma mensagem como cliente, "
        "o RAG é executado imediatamente e a sugestão aparece no painel central."
    )

    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Documentos", len({chunk.source for chunk in rag.chunks}))
    top_b.metric("Chunks", len(rag.chunks))
    top_c.metric("Top-k", settings.top_k)
    top_d.metric("Turnos", len(st.session_state.conversation))

    client_col, rag_col, operator_col = st.columns([1, 1.2, 1], gap="large")

    with client_col:
        st.markdown("## Cliente")
        st.caption("Digite como se fosse o cliente real.")
        with st.form("customer_form", clear_on_submit=True):
            customer_input = st.text_area(
                "Mensagem do cliente",
                height=130,
                placeholder="Ex.: Qual o preço dessa mesa?",
            )
            customer_submit = st.form_submit_button(
                "Enviar como cliente",
                type="primary",
                use_container_width=True,
            )
        if customer_submit:
            try:
                with st.spinner("Executando busca vetorial e gerando sugestão..."):
                    add_customer_message(customer_input)
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao processar mensagem: {exc}")

        st.markdown("### Histórico do cliente")
        for item in st.session_state.conversation:
            if item["role"] == "cliente":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(item["content"])

    with rag_col:
        st.markdown("## RAG e sugestão")
        st.caption("Atualizado após cada nova mensagem do cliente.")

        if st.session_state.suggestion:
            st.markdown(st.session_state.suggestion)
        else:
            st.info("Envie uma mensagem como cliente para executar o RAG.")

        if st.button(
            "Enviar todo o contexto ao GPT",
            use_container_width=True,
            disabled=not st.session_state.conversation,
        ):
            try:
                with st.spinner("Gerando resposta com histórico completo..."):
                    generate_full_answer()
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao gerar resposta completa: {exc}")

        if st.session_state.full_answer:
            st.markdown("### Resposta com contexto completo")
            st.success(st.session_state.full_answer)

        with st.expander("Chunks recuperados", expanded=True):
            if not st.session_state.sources:
                st.write("Nenhum chunk recuperado ainda.")
            for item in st.session_state.sources:
                st.markdown(
                    f"**{item['source']}**  \n"
                    f"Chunk: `{item['chunk_id']}`  \n"
                    f"Score: `{item['score']:.3f}`"
                )
                st.caption(item["text"])
                st.divider()

    with operator_col:
        st.markdown("## Atendente")
        st.caption("Escreva manualmente ou reutilize a resposta sugerida.")

        default_operator = st.session_state.get("operator_draft", "")
        with st.form("operator_form", clear_on_submit=True):
            operator_input = st.text_area(
                "Resposta do atendente",
                value=default_operator,
                height=130,
                placeholder="Digite a resposta que será enviada ao cliente...",
            )
            manual_submit = st.form_submit_button(
                "Enviar como atendente",
                type="primary",
                use_container_width=True,
            )

        if manual_submit:
            add_operator_message(operator_input, "manual")
            st.session_state.operator_draft = ""
            st.rerun()

        if st.button(
            "Usar resposta da IA",
            use_container_width=True,
            disabled=not st.session_state.suggested_answer,
        ):
            add_operator_message(st.session_state.suggested_answer, "ai_suggestion")
            st.rerun()

        if st.button(
            "Usar resposta completa",
            use_container_width=True,
            disabled=not st.session_state.full_answer,
        ):
            add_operator_message(st.session_state.full_answer, "full_context")
            st.rerun()

        st.markdown("### Histórico do atendente")
        for item in st.session_state.conversation:
            if item["role"] == "atendente":
                with st.chat_message("assistant", avatar="🎧"):
                    st.markdown(item["content"])

    st.divider()
    st.markdown("## Linha do tempo completa")
    for item in st.session_state.conversation:
        role = "user" if item["role"] == "cliente" else "assistant"
        avatar = "👤" if item["role"] == "cliente" else "🎧"
        with st.chat_message(role, avatar=avatar):
            st.markdown(f"**{item['role'].title()}:** {item['content']}")

    reset_col, export_col = st.columns(2)
    if reset_col.button("Limpar simulação", use_container_width=True):
        for key in [
            "conversation",
            "suggestion",
            "suggested_answer",
            "full_answer",
            "sources",
            "last_query",
            "context_sent",
            "events",
        ]:
            st.session_state[key] = [] if key in {"conversation", "sources", "events"} else ""
        st.rerun()

    export_payload = json.dumps(
        {
            "conversation": st.session_state.conversation,
            "events": st.session_state.events,
            "last_query": st.session_state.last_query,
            "last_sources": st.session_state.sources,
        },
        ensure_ascii=False,
        indent=2,
    )
    export_col.download_button(
        "Exportar evidências em JSON",
        data=export_payload,
        file_name="simulacao_copiloto_assistente.json",
        mime="application/json",
        use_container_width=True,
    )

with docs_tab:
    st.subheader("Base documental e indexação")
    st.write(
        "Envie documentos `.md` ou `.txt`. O sistema divide os arquivos em chunks, "
        "gera embeddings locais e reconstrói o índice vetorial."
    )
    uploads = st.file_uploader(
        "Adicionar documentos",
        type=["md", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Salvar e reindexar", disabled=not uploads):
        docs_dir = Path("data/docs")
        docs_dir.mkdir(parents=True, exist_ok=True)
        for uploaded in uploads or []:
            safe_name = Path(uploaded.name).name
            (docs_dir / safe_name).write_bytes(uploaded.getvalue())
        get_rag.clear()
        st.success("Arquivos salvos e índice invalidado. Recarregando...")
        st.rerun()

    for source in sorted({chunk.source for chunk in rag.chunks}):
        source_chunks = [chunk for chunk in rag.chunks if chunk.source == source]
        with st.expander(f"{source} — {len(source_chunks)} chunks"):
            for chunk in source_chunks:
                st.markdown(f"**{chunk.chunk_id}**")
                st.write(chunk.text)
                st.divider()

with detail_tab:
    st.subheader("O que ocorreu por trás da simulação")
    st.markdown("### 1. Consulta enviada ao retriever")
    st.code(st.session_state.last_query or "Nenhuma consulta executada.")

    st.markdown("### 2. Contexto montado para o GPT")
    st.code(st.session_state.context_sent or "Nenhum contexto montado.")

    st.markdown("### 3. Eventos registrados")
    if st.session_state.events:
        st.json(st.session_state.events)
    else:
        st.info("Nenhum evento registrado.")

    st.markdown("### 4. Fluxo explicado")
    st.write(
        "1. A mensagem do cliente é adicionada ao histórico.\n"
        "2. Os últimos turnos formam a consulta semântica.\n"
        "3. A consulta vira embedding.\n"
        "4. A matriz NumPy compara a consulta com todos os chunks.\n"
        "5. Os chunks com maior similaridade são selecionados.\n"
        "6. O contexto recuperado e o histórico são enviados ao GPT-4o mini.\n"
        "7. O copiloto gera uma sugestão rastreável.\n"
        "8. O operador pode escrever, usar a IA ou pedir uma resposta com todo o contexto."
    )

with architecture_tab:
    st.subheader("Arquitetura técnica do RAG")
    st.write(
        "Esta aba mostra exatamente o que está implementado no MVP atual. "
        "Os embeddings não estão em um banco persistente: eles ficam em uma matriz NumPy na memória do processo Streamlit."
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Chunk size", f"{settings.chunk_size} caracteres")
    metric_2.metric("Overlap", f"{settings.chunk_overlap} caracteres")
    metric_3.metric("Top-k", settings.top_k)
    metric_4.metric("Chunks indexados", len(rag.chunks))

    st.markdown("## Resumo dos componentes")
    st.markdown(
        f"""
| Componente | Implementação atual |
|---|---|
| Documentos | `.md` e `.txt` em `data/docs/` |
| Chunking | Por caracteres, com cortes preferenciais em parágrafos, linhas e frases |
| Tamanho | `{settings.chunk_size}` caracteres por chunk |
| Overlap | `{settings.chunk_overlap}` caracteres |
| Embedding | `{settings.embedding_model}` |
| Execução | Local, com `sentence-transformers` |
| Normalização | Vetores normalizados antes da busca |
| Banco vetorial | **Nenhum banco persistente neste MVP** |
| Armazenamento | Matriz NumPy `float32` em memória |
| Similaridade | Produto escalar entre vetores normalizados, equivalente ao cosseno |
| Recuperação | Top-k global de `{settings.top_k}` chunks |
| Threshold | Não implementado |
| Reranking | Não implementado |
| Consulta | Últimos 6 turnos da conversa |
| LLM | Azure OpenAI, deployment `{settings.azure_deployment}` |
"""
    )

    st.info(
        "Índice vetorial local, neste projeto, significa uma matriz de vetores em memória. "
        "Não significa Elastic, Azure AI Search, Qdrant, Milvus, Weaviate ou pgvector."
    )

    st.markdown("## 1. Fluxo de ingestão e indexação")
    st.markdown(
        """
```mermaid
flowchart LR
    A[Documento MD ou TXT] --> B[Leitura UTF-8]
    B --> C[Normalização das linhas]
    C --> D[Janela de até CHUNK_SIZE caracteres]
    D --> E{Existe separador natural?}
    E -->|Sim| F[Corta em parágrafo, linha, frase ou ponto e vírgula]
    E -->|Não| G[Corta no limite da janela]
    F --> H[Aplica overlap]
    G --> H
    H --> I[Gera chunk_id, fonte, start e end]
    I --> J[SentenceTransformer.encode]
    J --> K[Normaliza embeddings]
    K --> L[Matriz NumPy float32 em memória]
```
"""
    )

    st.markdown("### Tipo de chunk")
    st.write(
        "O chunking é baseado em caracteres. Ele procura um corte natural na segunda metade da janela, "
        "priorizando parágrafo, quebra de linha, final de frase e ponto e vírgula. "
        "Não é chunking semântico, por tokens ou hierárquico nesta versão."
    )

    st.markdown("### Por que existe overlap")
    st.write(
        "O overlap repete parte do final do chunk anterior no início do próximo. "
        "Isso diminui o risco de uma informação importante ficar dividida exatamente entre dois chunks."
    )

    st.markdown("## 2. Fluxo de recuperação")
    st.markdown(
        """
```mermaid
flowchart LR
    A[Nova mensagem do cliente] --> B[Histórico da conversa]
    B --> C[Seleciona os últimos 6 turnos]
    C --> D[Monta consulta semântica]
    D --> E[Gera embedding normalizado]
    E --> F[Compara com todos os chunks]
    F --> G[Produto escalar]
    G --> H[Ordena scores do maior para o menor]
    H --> I[Seleciona top-k]
    I --> J[Retorna fonte, chunk_id, score e conteúdo]
```
"""
    )

    st.markdown("### Cálculo de similaridade")
    st.code("scores = embeddings_dos_chunks @ embedding_da_consulta", language="python")
    st.write(
        "Os embeddings dos documentos e da consulta são normalizados. Por isso, o produto escalar "
        "representa a similaridade de cosseno. Quanto maior o score, maior a proximidade semântica."
    )

    st.markdown("## 3. Montagem do contexto e geração")
    st.markdown(
        """
```mermaid
sequenceDiagram
    participant C as Cliente
    participant H as Histórico
    participant E as Embeddings
    participant M as Matriz NumPy
    participant G as GPT-4o mini
    participant O as Operador

    C->>H: Envia mensagem
    H->>E: Entrega os últimos 6 turnos
    E->>M: Gera e compara embedding da consulta
    M->>M: Ordena scores e seleciona top-k
    M->>G: Envia chunks, fontes, IDs e scores
    H->>G: Envia histórico completo
    G->>O: Gera resposta sugerida, ação, alertas e fontes
    O->>C: Responde manualmente
    O->>C: Ou usa a resposta da IA
    O->>G: Opcionalmente envia todo o histórico
    G->>O: Gera resposta consolidada
```
"""
    )

    st.markdown("### Estrutura do contexto")
    st.code(
        """FONTE: catalogo.md
CHUNK: catalogo.md::3
SCORE: 0.842
CONTEÚDO:
Texto recuperado do documento..."""
    )

    st.markdown("## 4. O que está em memória")
    memory_col_1, memory_col_2 = st.columns(2)
    with memory_col_1:
        st.markdown("### Chunks")
        st.write(
            "Lista Python contendo `chunk_id`, fonte, texto e posições inicial e final de cada trecho."
        )
    with memory_col_2:
        st.markdown("### Embeddings")
        st.write(
            "Matriz NumPy `float32`, com uma linha para cada chunk e uma coluna para cada dimensão do embedding."
        )

    st.warning(
        "Ao reiniciar o aplicativo, os embeddings são recalculados. Não existe persistência em banco ou arquivo vetorial."
    )

    st.markdown("## 5. Limitações atuais")
    st.markdown(
        """
- somente documentos `.md` e `.txt`;
- chunking por caracteres, não semântico;
- índice reconstruído a cada reinício;
- busca exaustiva sobre todos os chunks;
- nenhum threshold mínimo de similaridade;
- nenhum BM25 ou busca híbrida;
- nenhum reranker;
- nenhum filtro por metadados;
- adequado para demonstração e bases pequenas, não para produção em larga escala.
"""
    )

    st.markdown("## 6. Evolução recomendada")
    st.markdown(
        """
```mermaid
flowchart LR
    A[MVP atual: NumPy em memória] --> B[Persistência vetorial]
    B --> C[Elastic ou Azure AI Search]
    C --> D[Busca híbrida BM25 + vetorial]
    D --> E[Reranking]
    E --> F[Threshold e filtros de metadados]
    F --> G[Observabilidade e avaliação]
```
"""
    )
