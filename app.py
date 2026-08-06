from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.azure_client import AzureLLM
from src.config import Settings
from src.rag import LocalRAG

st.set_page_config(page_title="Copiloto Assistente", layout="wide")
st.title("Copiloto Assistente")
st.caption("Cliente sintético vivo + operador humano + sugestões RAG em tempo real")

settings = Settings()


@st.cache_resource(show_spinner="Carregando modelo de embeddings e indexando documentos...")
def get_rag(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> LocalRAG:
    return LocalRAG(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


rag = get_rag(
    settings.embedding_model,
    settings.chunk_size,
    settings.chunk_overlap,
)


def build_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "Nenhuma fonte recuperada."
    return "\n\n".join(
        f"FONTE: {item['source']} | CHUNK: {item['chunk_id']} | "
        f"SCORE: {item['score']:.3f}\n{item['text']}"
        for item in results
    )


def conversation_text(conversation: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{item['role'].upper()}: {item['content']}" for item in conversation
    )


def retrieval_query(conversation: list[dict[str, str]]) -> str:
    recent = conversation[-6:]
    return "\n".join(item["content"] for item in recent)


def generate_assistance(
    conversation: list[dict[str, str]],
) -> tuple[str, list[dict[str, Any]]]:
    query = retrieval_query(conversation)
    results = rag.search(query, settings.top_k)
    context = build_context(results)
    llm = AzureLLM(settings)

    system = """Você é um copiloto de atendimento que ajuda um operador humano em tempo real.
Use somente as fontes recuperadas para afirmar políticas, valores, prazos, requisitos e procedimentos.
Considere toda a conversa. Não responda como cliente e não finja ser o operador.
A resposta deve ser curta, prática e fácil de copiar durante um atendimento.
Quando faltar evidência, diga claramente o que precisa ser confirmado.
Responda em português do Brasil com estas seções:

### Resposta sugerida
Texto que o operador poderia enviar agora.

### Próxima ação
Uma ação objetiva.

### Atenção
Riscos, dados ainda necessários ou limites da informação.

### Fontes
Liste documento, chunk e score usados.
"""

    user = (
        f"CONVERSA:\n{conversation_text(conversation)}\n\n"
        f"TRECHOS RECUPERADOS:\n{context}"
    )
    return llm.complete(system, user), results


def extract_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("O modelo não devolveu um cenário JSON válido.")
    return json.loads(match.group(0))


def generate_scenario(difficulty: str, topic_hint: str) -> dict[str, Any]:
    if not rag.chunks:
        raise ValueError("Adicione documentos .md ou .txt antes de iniciar.")

    sample_chunks = random.sample(rag.chunks, min(len(rag.chunks), 12))
    corpus = "\n\n".join(
        f"FONTE: {chunk.source}\n{chunk.text}" for chunk in sample_chunks
    )
    llm = AzureLLM(settings)

    system = """Crie um cenário realista de atendimento baseado exclusivamente nos documentos fornecidos.
O usuário será o operador. Você será o cliente durante a simulação.
Escolha um problema que possa ser solucionado ou orientado com os documentos.
Não copie respostas completas do documento para a fala inicial.
Retorne somente JSON válido, sem markdown, neste formato:
{
  "title": "título curto",
  "persona": "descrição do cliente",
  "objective": "objetivo oculto do cliente",
  "facts": ["fatos que o cliente sabe"],
  "hidden_information": ["informações que só revela quando perguntado"],
  "behavior": "como reage ao atendimento",
  "success_criteria": ["critérios objetivos"],
  "initial_message": "primeira fala natural do cliente"
}
"""

    user = (
        f"DIFICULDADE: {difficulty}\n"
        f"PREFERÊNCIA DE ASSUNTO: {topic_hint or 'qualquer assunto dos documentos'}\n\n"
        f"DOCUMENTOS:\n{corpus}"
    )
    return extract_json(llm.complete(system, user))


def generate_customer_reply() -> str:
    llm = AzureLLM(settings)
    scenario = st.session_state.scenario
    conversation = st.session_state.conversation

    system = """Você é o cliente sintético de um treinamento de atendimento.
Responda somente como cliente, em português do Brasil.
Reaja diretamente à última mensagem do operador e mantenha coerência com o cenário.
Não revele objetivo oculto, roteiro, critérios de sucesso, fontes ou que é uma IA.
Revele informações escondidas apenas quando o operador perguntar de forma adequada.
Não invente novas políticas da empresa.
Se o problema estiver resolvido, confirme a resolução e encerre naturalmente.
Produza apenas uma fala curta e natural do cliente.
"""

    user = (
        f"CENÁRIO INTERNO:\n{json.dumps(scenario, ensure_ascii=False)}\n\n"
        f"CONVERSA:\n{conversation_text(conversation)}"
    )
    return llm.complete(system, user).strip()


def start_training(difficulty: str, topic_hint: str) -> None:
    scenario = generate_scenario(difficulty, topic_hint)
    first_message = str(scenario["initial_message"]).strip()
    conversation = [{"role": "cliente", "content": first_message}]
    suggestion, sources = generate_assistance(conversation)

    st.session_state.training_active = True
    st.session_state.training_finished = False
    st.session_state.scenario = scenario
    st.session_state.conversation = conversation
    st.session_state.suggestion = suggestion
    st.session_state.sources = sources


def send_operator_message(message: str) -> None:
    clean_message = message.strip()
    if not clean_message:
        st.warning("Digite sua resposta como operador.")
        return

    st.session_state.conversation.append(
        {"role": "operador", "content": clean_message}
    )
    customer_reply = generate_customer_reply()
    st.session_state.conversation.append(
        {"role": "cliente", "content": customer_reply}
    )
    suggestion, sources = generate_assistance(st.session_state.conversation)
    st.session_state.suggestion = suggestion
    st.session_state.sources = sources


def finish_training() -> None:
    llm = AzureLLM(settings)
    results = rag.search(
        retrieval_query(st.session_state.conversation),
        settings.top_k,
    )
    system = """Você é um avaliador de treinamento de atendimento.
Avalie apenas o desempenho do operador humano. Considere o cenário, a conversa e as fontes.
Dê notas de 0 a 10 para: investigação, clareza, aderência às fontes, resolução e segurança.
Mostre acertos, falhas, informações que deveriam ter sido perguntadas e uma resposta melhor para o pior turno.
"""
    user = (
        f"CENÁRIO:\n{json.dumps(st.session_state.scenario, ensure_ascii=False)}\n\n"
        f"CONVERSA:\n{conversation_text(st.session_state.conversation)}\n\n"
        f"FONTES:\n{build_context(results)}"
    )
    st.session_state.evaluation = llm.complete(system, user)
    st.session_state.training_finished = True


for key, default in {
    "training_active": False,
    "training_finished": False,
    "conversation": [],
    "scenario": {},
    "suggestion": "",
    "sources": [],
    "evaluation": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

training_tab, docs_tab, architecture_tab = st.tabs(
    ["Treinamento vivo", "Base documental", "Como funciona"]
)

with training_tab:
    if not st.session_state.training_active:
        st.subheader("Criar atendimento a partir da base documental")
        col_a, col_b = st.columns(2)
        with col_a:
            difficulty = st.selectbox(
                "Dificuldade",
                ["Fácil", "Média", "Difícil"],
                index=1,
            )
        with col_b:
            topic_hint = st.text_input(
                "Assunto opcional",
                placeholder="Deixe vazio para o sistema escolher qualquer assunto",
            )

        st.info(
            f"Base atual: {len(rag.chunks)} chunks em "
            f"{len({chunk.source for chunk in rag.chunks})} documentos."
        )
        if st.button("Iniciar conversa", type="primary", use_container_width=True):
            try:
                with st.spinner("Criando cliente e primeira sugestão..."):
                    start_training(difficulty, topic_hint)
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao iniciar treinamento: {exc}")
    else:
        header_left, header_right = st.columns([4, 1])
        with header_left:
            st.subheader(st.session_state.scenario.get("title", "Atendimento"))
            st.caption("Você é o operador. A IA representa somente o cliente.")
        with header_right:
            if st.button("Novo cenário", use_container_width=True):
                st.session_state.training_active = False
                st.session_state.training_finished = False
                st.rerun()

        chat_col, assist_col = st.columns([1.45, 1], gap="large")

        with chat_col:
            st.markdown("### Conversa")
            chat_box = st.container(height=510)
            with chat_box:
                for item in st.session_state.conversation:
                    avatar = "👤" if item["role"] == "cliente" else "🎧"
                    role = "user" if item["role"] == "cliente" else "assistant"
                    with st.chat_message(role, avatar=avatar):
                        st.markdown(item["content"])

            if not st.session_state.training_finished:
                with st.form("operator_form", clear_on_submit=True):
                    operator_message = st.text_area(
                        "Sua resposta como operador",
                        height=110,
                        placeholder="Digite o que você responderia ao cliente...",
                    )
                    send_col, finish_col = st.columns([3, 1])
                    submitted = send_col.form_submit_button(
                        "Enviar ao cliente",
                        type="primary",
                        use_container_width=True,
                    )
                    finished = finish_col.form_submit_button(
                        "Encerrar",
                        use_container_width=True,
                    )

                if submitted:
                    try:
                        with st.spinner("Cliente respondendo e RAG atualizando..."):
                            send_operator_message(operator_message)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Falha ao continuar conversa: {exc}")
                if finished:
                    try:
                        with st.spinner("Avaliando seu atendimento..."):
                            finish_training()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Falha ao avaliar: {exc}")

        with assist_col:
            st.markdown("### Copiloto em tempo real")
            st.markdown(st.session_state.suggestion or "Aguardando conversa.")

            with st.expander("Trechos recuperados pelo RAG", expanded=True):
                if not st.session_state.sources:
                    st.warning("Nenhum trecho relevante recuperado.")
                for item in st.session_state.sources:
                    st.markdown(
                        f"**{item['source']}** · `{item['chunk_id']}` · "
                        f"score `{item['score']:.3f}`"
                    )
                    st.caption(item["text"])
                    st.divider()

        if st.session_state.training_finished:
            st.divider()
            st.subheader("Avaliação do operador")
            st.markdown(st.session_state.evaluation)

with docs_tab:
    st.subheader("Documentos usados pelo RAG")
    st.write(
        "Envie arquivos `.md` ou `.txt`. Após salvar, reindexe a base para criar "
        "novos chunks e embeddings."
    )

    uploaded_files = st.file_uploader(
        "Adicionar documentos",
        type=["md", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Salvar e reindexar", disabled=not uploaded_files):
        docs_dir = Path("data/docs")
        docs_dir.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files or []:
            safe_name = Path(uploaded_file.name).name
            (docs_dir / safe_name).write_bytes(uploaded_file.getvalue())
        get_rag.clear()
        st.success("Documentos salvos. A página será recarregada com o novo índice.")
        st.rerun()

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Documentos", len({chunk.source for chunk in rag.chunks}))
    metric_b.metric("Chunks", len(rag.chunks))
    metric_c.metric("Top-k", settings.top_k)

    for source in sorted({chunk.source for chunk in rag.chunks}):
        source_chunks = [chunk for chunk in rag.chunks if chunk.source == source]
        with st.expander(f"{source} — {len(source_chunks)} chunks"):
            for chunk in source_chunks:
                st.markdown(f"**{chunk.chunk_id}**")
                st.write(chunk.text)
                st.divider()

with architecture_tab:
    st.markdown(
        """
```mermaid
sequenceDiagram
    participant D as Documentos
    participant E as Embeddings
    participant C as Cliente sintético
    participant U as Operador humano
    participant R as RAG
    participant A as Copiloto

    D->>E: Chunking + embeddings
    E->>R: Índice vetorial
    C->>U: Mensagem do cliente
    C->>R: Consulta com histórico
    R->>A: Top-k chunks e fontes
    A->>U: Sugestão ao lado da conversa
    U->>C: Resposta digitada pelo operador
    C->>C: Reage ao que o operador disse
    C->>R: Nova mensagem + histórico
    R->>A: Nova recuperação
    A->>U: Sugestão atualizada
```
"""
    )
