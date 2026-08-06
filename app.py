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
st.caption(
    "Simulação viva com Naive RAG ou GraphRAG, sugestões em tempo real e rastreabilidade completa"
)

settings = Settings()


@st.cache_resource(show_spinner="Carregando embeddings e indexando documentos...")
def get_rag(model: str, chunk_size: int, overlap: int) -> LocalRAG:
    return LocalRAG(
        embedding_model=model,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )


rag = get_rag(settings.embedding_model, settings.chunk_size, settings.chunk_overlap)

MODE_LABELS = {
    "Naive RAG — busca vetorial direta": "naive",
    "GraphRAG — busca vetorial + expansão no grafo": "graph",
}


def build_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "Nenhuma fonte recuperada."
    parts: list[str] = []
    for item in results:
        relations = ", ".join(item.get("relations", [])) or "sem expansão"
        parts.append(
            f"FONTE: {item['source']}\n"
            f"SEÇÃO: {item.get('section') or 'Sem seção'}\n"
            f"TIPO: {item.get('kind', 'text')}\n"
            f"CHUNK: {item['chunk_id']}\n"
            f"SCORE: {item['score']:.3f}\n"
            f"ORIGEM: {item.get('retrieval_origin', 'similaridade vetorial')}\n"
            f"RELAÇÕES: {relations}\n"
            f"CONTEÚDO:\n{item['text']}"
        )
    return "\n\n---\n\n".join(parts)


def conversation_text() -> str:
    if not st.session_state.conversation:
        return "Nenhuma mensagem registrada."
    return "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in st.session_state.conversation
    )


def customer_questions() -> list[str]:
    return [
        item["content"]
        for item in st.session_state.conversation
        if item["role"] == "cliente"
    ]


def customer_questions_text() -> str:
    questions = customer_questions()
    if not questions:
        return "Nenhuma pergunta do cliente."
    return "\n".join(f"{index}. {question}" for index, question in enumerate(questions, 1))


def retrieval_query() -> str:
    """Cria uma consulta centrada na pergunta atual, sem perder continuidade."""
    questions = customer_questions()
    latest_question = questions[-1] if questions else ""
    previous_questions = questions[-4:-1]
    latest_operator = next(
        (
            item["content"]
            for item in reversed(st.session_state.conversation)
            if item["role"] == "atendente"
        ),
        "",
    )

    parts = [f"PERGUNTA ATUAL DO CLIENTE: {latest_question}"]
    if previous_questions:
        parts.append(
            "PERGUNTAS ANTERIORES DO CLIENTE:\n"
            + "\n".join(f"- {question}" for question in previous_questions)
        )
    if latest_operator:
        parts.append(f"ÚLTIMA RESPOSTA DO ATENDENTE: {latest_operator}")
    return "\n\n".join(parts)


def selected_mode() -> str:
    return MODE_LABELS.get(st.session_state.get("rag_mode_label", ""), "naive")


def extract_suggested_answer(markdown: str) -> str:
    marker = "### Resposta sugerida"
    if marker not in markdown:
        return markdown.strip()
    content = markdown.split(marker, 1)[1]
    for next_marker in ["### Próxima ação", "### Atenção", "### Fontes"]:
        if next_marker in content:
            content = content.split(next_marker, 1)[0]
    return content.strip()


def clear_last_retrieval() -> None:
    """Evita que a interface mostre fontes de uma pergunta anterior."""
    st.session_state.suggestion = ""
    st.session_state.suggested_answer = ""
    st.session_state.full_answer = ""
    st.session_state.sources = []
    st.session_state.last_query = ""
    st.session_state.context_sent = ""
    st.session_state.prompt_sent = ""


def run_rag() -> None:
    mode = selected_mode()
    query = retrieval_query()
    results = rag.search(query, settings.top_k, mode=mode)
    rag_context = build_context(results)
    history = conversation_text()
    questions = customer_questions_text()
    latest_question = customer_questions()[-1] if customer_questions() else ""

    prompt = (
        f"PERGUNTA ATUAL DO CLIENTE:\n{latest_question}\n\n"
        f"TODAS AS PERGUNTAS DO CLIENTE:\n{questions}\n\n"
        f"HISTÓRICO COMPLETO DA CONVERSA:\n{history}\n\n"
        f"CONTEXTO RECUPERADO PELO {mode.upper()}:\n{rag_context}"
    )

    st.session_state.last_query = query
    st.session_state.sources = results
    st.session_state.context_sent = rag_context
    st.session_state.prompt_sent = prompt
    st.session_state.current_customer_question = latest_question
    st.session_state.last_mode = mode

    llm = AzureLLM(settings)
    system = """Você é um copiloto para um operador humano.
A pergunta atual do cliente é o foco principal. Use as perguntas anteriores e o histórico para resolver referências como "essa mesa", "esse plano" ou "e o prazo?".
Use somente os trechos recuperados para afirmar preços, políticas, prazos, requisitos e procedimentos.
Não reutilize uma resposta anterior quando a pergunta atual tratar de outro assunto.
Quando não houver evidência suficiente, informe claramente o que precisa ser confirmado.
Responda em português do Brasil com exatamente estas seções:

### Resposta sugerida
### Próxima ação
### Atenção
### Fontes
"""
    suggestion = llm.complete(system, prompt)

    st.session_state.suggestion = suggestion
    st.session_state.suggested_answer = extract_suggested_answer(suggestion)
    st.session_state.events.append(
        {
            "event": "rag_execution",
            "mode": mode,
            "current_customer_question": latest_question,
            "query": query,
            "retrieved_chunks": len(results),
            "chunk_ids": [item["chunk_id"] for item in results],
            "sources": [item["source"] for item in results],
            "scores": [round(item["score"], 4) for item in results],
            "origins": [item.get("retrieval_origin") for item in results],
        }
    )


def add_customer_message(message: str) -> None:
    clean = message.strip()
    if not clean:
        st.warning("Digite a mensagem do cliente.")
        return
    clear_last_retrieval()
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
    mode = selected_mode()
    full_query = (
        f"PERGUNTA ATUAL:\n{st.session_state.current_customer_question}\n\n"
        f"PERGUNTAS DO CLIENTE:\n{customer_questions_text()}\n\n"
        f"CONVERSA:\n{conversation_text()}"
    )
    results = rag.search(full_query, settings.top_k, mode=mode)
    context = build_context(results)
    llm = AzureLLM(settings)
    system = """Você é um assistente de atendimento.
Receba a pergunta atual, todas as perguntas do cliente, o histórico completo e o contexto recuperado.
Gere a melhor resposta final possível para o atendente enviar agora.
Use somente as fontes fornecidas para fatos, preços, condições, prazos e procedimentos.
Responda somente com a mensagem final ao cliente, sem explicações adicionais.
"""
    user = (
        f"PERGUNTA ATUAL:\n{st.session_state.current_customer_question}\n\n"
        f"TODAS AS PERGUNTAS DO CLIENTE:\n{customer_questions_text()}\n\n"
        f"CONVERSA COMPLETA:\n{conversation_text()}\n\n"
        f"CONTEXTO {mode.upper()}:\n{context}"
    )
    st.session_state.full_answer = llm.complete(system, user).strip()
    st.session_state.events.append(
        {
            "event": "full_context_generation",
            "mode": mode,
            "current_customer_question": st.session_state.current_customer_question,
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
    "prompt_sent": "",
    "current_customer_question": "",
    "last_mode": "",
    "events": [],
    "rag_mode_label": "Naive RAG — busca vetorial direta",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

sim_tab, docs_tab, detail_tab, architecture_tab = st.tabs(
    ["Simulação viva", "Base documental", "Detalhes da execução", "Arquitetura"]
)

with sim_tab:
    st.subheader("Simulação de atendimento em tempo real")
    st.write(
        "Você controla cliente e atendente. Cada nova mensagem do cliente limpa a recuperação anterior, "
        "executa uma nova busca e atualiza a sugestão com a pergunta atual e o histórico."
    )

    control_left, control_right = st.columns([2, 1])
    with control_left:
        st.selectbox(
            "Estratégia de recuperação",
            options=list(MODE_LABELS),
            key="rag_mode_label",
            help=(
                "Naive RAG usa somente similaridade vetorial. GraphRAG começa pela busca vetorial "
                "e expande para chunks ligados por seção, sequência ou entidades compartilhadas."
            ),
        )
    with control_right:
        if st.button(
            "Reprocessar pergunta atual",
            use_container_width=True,
            disabled=not customer_questions(),
        ):
            try:
                clear_last_retrieval()
                with st.spinner("Refazendo a recuperação com a estratégia selecionada..."):
                    run_rag()
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao reprocessar: {exc}")

    top_a, top_b, top_c, top_d, top_e = st.columns(5)
    top_a.metric("Documentos", len({chunk.source for chunk in rag.chunks}))
    top_b.metric("Chunks", len(rag.chunks))
    top_c.metric("Tabelas", sum(chunk.kind == "table" for chunk in rag.chunks))
    top_d.metric("Top-k", settings.top_k)
    top_e.metric("Turnos", len(st.session_state.conversation))

    client_col, rag_col, operator_col = st.columns([1, 1.25, 1], gap="large")

    with client_col:
        st.markdown("## Cliente")
        st.caption("A mensagem enviada aqui se torna a pergunta principal da nova recuperação.")
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
                with st.spinner("Limpando resultado anterior e executando nova recuperação..."):
                    add_customer_message(customer_input)
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao processar mensagem: {exc}")

        st.markdown("### Perguntas do cliente")
        for item in st.session_state.conversation:
            if item["role"] == "cliente":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(item["content"])

    with rag_col:
        st.markdown("## RAG e sugestão")
        active_mode = st.session_state.last_mode or selected_mode()
        st.caption(f"Recuperação atual: `{active_mode}`. Atualizada a cada pergunta do cliente.")

        if st.session_state.current_customer_question:
            st.markdown("**Pergunta usada como foco:**")
            st.info(st.session_state.current_customer_question)

        if st.session_state.suggestion:
            st.markdown(st.session_state.suggestion)
        else:
            st.info("Envie uma mensagem como cliente para executar a recuperação.")

        if st.button(
            "Enviar todo o contexto ao GPT",
            use_container_width=True,
            disabled=not st.session_state.conversation,
        ):
            try:
                with st.spinner("Gerando resposta com perguntas, histórico e contexto recuperado..."):
                    generate_full_answer()
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao gerar resposta completa: {exc}")

        if st.session_state.full_answer:
            st.markdown("### Resposta com contexto completo")
            st.success(st.session_state.full_answer)

        with st.expander("Chunks recuperados nesta pergunta", expanded=True):
            if not st.session_state.sources:
                st.write("Nenhum chunk recuperado para a pergunta atual.")
            for item in st.session_state.sources:
                type_label = "Tabela preservada" if item.get("kind") == "table" else item.get("kind", "text")
                st.markdown(
                    f"**{item['source']}**  \n"
                    f"Seção: `{item.get('section') or 'Sem seção'}`  \n"
                    f"Tipo: `{type_label}`  \n"
                    f"Chunk: `{item['chunk_id']}`  \n"
                    f"Score: `{item['score']:.3f}`  \n"
                    f"Origem: `{item.get('retrieval_origin', 'vetorial')}`"
                )
                if item.get("relations"):
                    st.caption("Relações do grafo: " + ", ".join(item["relations"]))
                st.code(item["text"] if item.get("kind") == "table" else item["text"])
                st.divider()

    with operator_col:
        st.markdown("## Atendente")
        st.caption("Escreva manualmente ou reutilize a resposta da IA referente à pergunta atual.")

        with st.form("operator_form", clear_on_submit=True):
            operator_input = st.text_area(
                "Resposta do atendente",
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

        st.markdown("### Respostas do atendente")
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
            "prompt_sent",
            "current_customer_question",
            "last_mode",
            "events",
        ]:
            st.session_state[key] = [] if key in {"conversation", "sources", "events"} else ""
        st.rerun()

    export_payload = json.dumps(
        {
            "retrieval_mode": st.session_state.last_mode,
            "conversation": st.session_state.conversation,
            "customer_questions": customer_questions(),
            "events": st.session_state.events,
            "last_query": st.session_state.last_query,
            "last_sources": st.session_state.sources,
            "prompt_sent": st.session_state.prompt_sent,
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
    st.subheader("Base documental e indexação estruturada")
    st.write(
        "Envie documentos `.md` ou `.txt`. O indexador respeita títulos e seções, preserva tabelas "
        "Markdown como chunks únicos e gera embeddings locais para todos os chunks."
    )
    st.warning(
        "Para preservar corretamente tabelas de preços, mantenha-as no Markdown com cabeçalho, "
        "linha separadora e linhas usando `|`. Uma tabela pode ultrapassar o tamanho normal do chunk."
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
        clear_last_retrieval()
        st.success("Arquivos salvos. Cache do índice limpo e base reindexada.")
        st.rerun()

    docs_a, docs_b, docs_c = st.columns(3)
    docs_a.metric("Documentos", len({chunk.source for chunk in rag.chunks}))
    docs_b.metric("Chunks totais", len(rag.chunks))
    docs_c.metric("Tabelas atômicas", sum(chunk.kind == "table" for chunk in rag.chunks))

    for source in sorted({chunk.source for chunk in rag.chunks}):
        source_chunks = [chunk for chunk in rag.chunks if chunk.source == source]
        with st.expander(f"{source} — {len(source_chunks)} chunks"):
            for chunk in source_chunks:
                st.markdown(
                    f"**{chunk.chunk_id}** · tipo `{chunk.kind}` · seção `{chunk.section or 'Sem seção'}`"
                )
                if chunk.kind == "table":
                    st.code(chunk.text)
                else:
                    st.write(chunk.text)
                if chunk.entities:
                    st.caption("Entidades: " + ", ".join(chunk.entities[:12]))
                st.divider()

with detail_tab:
    st.subheader("O que ocorreu por trás da simulação")

    st.markdown("### 1. Pergunta atual do cliente")
    st.code(st.session_state.current_customer_question or "Nenhuma pergunta enviada.")

    st.markdown("### 2. Todas as perguntas do cliente")
    st.code(customer_questions_text())

    st.markdown("### 3. Consulta enviada ao retriever")
    st.code(st.session_state.last_query or "Nenhuma consulta executada.")

    st.markdown("### 4. Chunks recuperados")
    if st.session_state.sources:
        st.json(
            [
                {
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "section": item.get("section"),
                    "kind": item.get("kind"),
                    "score": round(item["score"], 4),
                    "origin": item.get("retrieval_origin"),
                    "relations": item.get("relations", []),
                }
                for item in st.session_state.sources
            ]
        )
    else:
        st.info("Nenhum chunk recuperado para a pergunta atual.")

    st.markdown("### 5. Contexto documental montado")
    st.code(st.session_state.context_sent or "Nenhum contexto montado.")

    st.markdown("### 6. Prompt completo enviado ao GPT")
    st.code(st.session_state.prompt_sent or "Nenhum prompt enviado.")

    st.markdown("### 7. Eventos registrados")
    if st.session_state.events:
        st.json(st.session_state.events)
    else:
        st.info("Nenhum evento registrado.")

    st.markdown("### 8. Regra de atualização")
    st.write(
        "Ao chegar uma nova pergunta, o sistema apaga da interface a sugestão, a resposta completa, "
        "os chunks e o contexto anteriores. Em seguida, destaca a nova pergunta, monta uma nova consulta, "
        "executa novamente o retriever escolhido e só então chama o GPT-4o mini. O cache do Streamlit "
        "é usado apenas para o modelo e o índice documental; resultados de perguntas não são armazenados nele."
    )

with architecture_tab:
    graph_stats = rag.graph_stats()
    st.subheader("Arquitetura técnica do RAG")
    st.write(
        "O MVP oferece duas estratégias sobre o mesmo conjunto de chunks e embeddings. "
        "O GraphRAG atual é estrutural e local: não utiliza Neo4j nem o pipeline completo da Microsoft."
    )

    arch_a, arch_b, arch_c, arch_d, arch_e = st.columns(5)
    arch_a.metric("Chunk size textual", settings.chunk_size)
    arch_b.metric("Overlap textual", settings.chunk_overlap)
    arch_c.metric("Tabelas atômicas", sum(chunk.kind == "table" for chunk in rag.chunks))
    arch_d.metric("Nós de chunk", graph_stats["chunk_nodes"])
    arch_e.metric("Arestas do grafo", graph_stats["edges"])

    st.markdown("## Chunking estruturado")
    st.markdown(
        """
- **Títulos e seções:** os cabeçalhos Markdown definem o contexto estrutural de cada chunk.
- **Texto comum:** parágrafos da mesma seção são agrupados até o limite configurado.
- **Textos longos:** são quebrados preferencialmente em parágrafos, linhas, frases ou ponto e vírgula.
- **Overlap:** é aplicado somente aos textos longos que precisaram ser divididos.
- **Tabelas Markdown:** cada tabela inteira vira um único chunk atômico, mesmo que ultrapasse o tamanho normal.
- **Blocos de código:** também são preservados integralmente.
"""
    )

    st.markdown("## Naive RAG")
    st.markdown(
        """
```mermaid
flowchart LR
    Q[Pergunta atual + continuidade] --> E[Embedding da consulta]
    E --> M[Matriz NumPy de embeddings]
    M --> S[Similaridade de cosseno]
    S --> K[Top-k chunks]
    K --> C[Contexto para GPT-4o mini]
```
"""
    )
    st.write(
        "É a opção mais simples e previsível. Recomendada para perguntas diretas, como localizar "
        "o preço de um produto em uma tabela bem identificada."
    )

    st.markdown("## GraphRAG estrutural local")
    st.markdown(
        """
```mermaid
flowchart LR
    D[Documento] --> S[Seções]
    S --> C[Chunks]
    C --> N[Chunk seguinte e anterior]
    C --> T[Chunks da mesma seção]
    C --> E[Entidades e valores compartilhados]

    Q[Pergunta] --> V[Sementes por similaridade vetorial]
    V --> G[Expansão no grafo]
    N --> G
    T --> G
    E --> G
    G --> K[Top-k combinado]
    K --> X[Contexto expandido]
```
"""
    )
    st.write(
        "O GraphRAG começa com chunks semanticamente relevantes e busca vizinhos relacionados. "
        "As arestas representam sequência no documento, mesma seção e entidades compartilhadas, "
        "como nomes de produtos, códigos e valores. Isso pode ajudar quando preço, condição e descrição "
        "estão em chunks diferentes ou quando a pergunta depende de relações entre informações."
    )

    st.markdown("## O que vai para o GPT")
    st.markdown(
        """
```mermaid
sequenceDiagram
    participant C as Cliente
    participant Q as Consulta
    participant R as Naive RAG ou GraphRAG
    participant P as Montagem do prompt
    participant G as GPT-4o mini
    participant O as Atendente

    C->>Q: Nova pergunta
    Q->>Q: Destaca pergunta atual e continuidade
    Q->>R: Executa nova recuperação
    R->>P: Chunks, fontes, tipos, scores e relações
    C->>P: Todas as perguntas do cliente
    C->>P: Histórico completo
    P->>G: Pergunta atual + perguntas + histórico + contexto
    G->>O: Sugestão referente à pergunta atual
```
"""
    )

    st.markdown("## Armazenamento atual")
    st.info(
        "Embeddings e grafo ficam em memória durante a execução do Streamlit. "
        "A matriz NumPy armazena os vetores; dicionários de adjacência armazenam as arestas do grafo. "
        "Ainda não existe banco vetorial ou banco de grafos persistente."
    )

    st.markdown("## Quando usar cada opção")
    st.markdown(
        """
| Cenário | Opção inicial recomendada |
|---|---|
| Pergunta direta de preço em uma tabela | Naive RAG |
| Preço relacionado a condições, planos ou categorias | GraphRAG |
| Documento pequeno e bem estruturado | Naive RAG |
| Informação espalhada entre seções ou documentos | GraphRAG |
| Necessidade de máxima previsibilidade e menor expansão | Naive RAG |
| Perguntas relacionais ou com referências indiretas | GraphRAG |
"""
    )

    st.warning(
        "GraphRAG não substitui a preservação das tabelas. Para preços, o primeiro controle de qualidade "
        "é manter a tabela inteira e seus cabeçalhos no mesmo chunk. O grafo é uma opção adicional para "
        "relacionar informações, não uma correção automática para documentos mal extraídos."
    )
